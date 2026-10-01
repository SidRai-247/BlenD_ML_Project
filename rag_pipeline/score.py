"""Step 4 (laptop): score model answers with BLEnD's OFFICIAL scorer.

    python rag_pipeline/score.py --run baseline                              # everything
    python rag_pipeline/score.py --run baseline --languages Spanish,Amharic  # only some languages
    python rag_pipeline/score.py --run baseline --models gemma3-4b           # only some models

Reads results/<run>/raw/<model>.jsonl (+ jobs/<run>.jsonl). For every (model, country, language,
prompt) group it writes, as soon as that group is done (re-runs skip finished groups):
  results/<run>/csv/{model}-{Country}_{Language}_{prompt}_result.csv     BLEnD's response-file format
  results/<run>/groups/{model}-{Country}_{Language}_{prompt}.csv         per-question 0/1 scores
Then it combines ALL finished groups (also ones scored elsewhere, e.g. in WSL) into:
  results/<run>/per_question.csv   one row per (model, country, language, prompt, question)
  results/<run>/scores.csv         one row per group: n_scored, SEM-B, SEM-W
  results/<run>/table.csv          SEM-B averaged over inst-4 and pers-3 (the number the paper reports)

How the official code is reused: BLEnD/evaluation/exact_match.py imports tools for all 13 BLEnD
languages at the top, several of which are not in the repo. We register harmless stand-ins for the
languages we don't use, then import the file as-is and call its soft_exact_match().
Spanish and Amharic use Spark NLP. Spark on native Windows needs Hadoop's winutils, so we score
those two in WSL (rag_pipeline/setup_scorer_wsl.sh).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import types

import pandas as pd

from config import BLEND_DIR, JOBS_DIR, RESULTS_DIR
from data import load_annotations

AZ_STEMMER_DIR = os.environ.get("BLEND_AZ_STEMMER", r"C:\Users\Siddharth\blend_scorer\az_stemmer")
SPARK_LANGUAGES = {"Spanish", "Amharic"}  # BLEnD lemmatizes these two with Spark NLP

# Spark on Windows needs Java plus Hadoop's winutils.exe/hadoop.dll (HADOOP_HOME). Use our install if present.
_JAVA_HOME = r"C:\Program Files\Java\jdk-21.0.10"
_HADOOP_HOME = r"C:\Users\Siddharth\blend_scorer\hadoop"
if not os.environ.get("JAVA_HOME") and os.path.isdir(_JAVA_HOME):
    os.environ["JAVA_HOME"] = _JAVA_HOME
if not os.environ.get("HADOOP_HOME") and os.path.isfile(os.path.join(_HADOOP_HOME, "bin", "winutils.exe")):
    os.environ["HADOOP_HOME"] = _HADOOP_HOME
    os.environ["PATH"] = os.path.join(_HADOOP_HOME, "bin") + os.pathsep + os.environ["PATH"]


def _import_official_scorer(need_spark):
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    class NotUsed:
        def __init__(self, *a, **k):
            raise RuntimeError("this language's tool is not installed (language not in our 5 countries)")

    # Tools for languages outside our 5 countries (Hausa, Indonesian, Persian, Arabic, Greek,
    # Sundanese, Chinese, Assamese) -> stand-ins. They are only called for those languages.
    stub("hausastemmer")
    stub("nlp_id"); stub("nlp_id.lemmatizer", Lemmatizer=NotUsed)
    stub("hazm", Lemmatizer=NotUsed)
    stub("qalsadi"); stub("qalsadi.lemmatizer", Lemmatizer=NotUsed)
    stub("cltk", NLP=NotUsed)
    stub("SUSTEM"); stub("SUSTEM.SUSTEM_S")
    stub("jieba")
    stub("indicnlp"); stub("indicnlp.common"); stub("indicnlp.loader")
    stub("indicnlp.tokenize"); stub("indicnlp.tokenize.indic_tokenize")
    stub("matplotlib"); stub("matplotlib.pyplot")
    # BLEnD/utils.py pulls in every API SDK (openai, anthropic, torch, ...). The scorer only needs
    # these few names from it.
    import csv, re
    import numpy as np
    from tqdm.auto import tqdm
    stub("utils", os=os, re=re, json=json, sys=sys, csv=csv, pd=pd, np=np, tqdm=tqdm)
    # Azerbaijani stemmer is imported as `stemmer.stemmer`
    pkg = stub("stemmer")
    pkg.__path__ = [AZ_STEMMER_DIR]
    # Spark NLP (Spanish + Amharic lemmatizers). If it can't start, those two languages are skipped.
    spark_ok = False
    try:
        import sparknlp
        if need_spark:
            try:  # installed is not enough: on Windows Spark also needs HADOOP_HOME/winutils to start
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    sparknlp.start()
                spark_ok = True
            except Exception:
                pass
    except ImportError:
        for name in ["sparknlp", "sparknlp.base", "sparknlp.annotator"]:
            stub(name)
        stub("sparknlp.pretrained", PretrainedPipeline=NotUsed)

    sys.path.insert(0, str(BLEND_DIR / "evaluation"))
    import exact_match

    if spark_ok:
        # Speed-up only, scores unchanged: the official lemma_check re-lemmatizes the same response
        # once per gold answer (twice per text for Amharic), each a Python->JVM round trip. The
        # lemmatizer is deterministic, so we memoize fullAnnotate per text.
        class CachedLightPipeline(exact_match.LightPipeline):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._cache = {}

            def fullAnnotate(self, target, *a, **k):
                if a or k or not isinstance(target, str):
                    return super().fullAnnotate(target, *a, **k)
                if target not in self._cache:
                    self._cache[target] = super().fullAnnotate(target)
                return self._cache[target]

        exact_match.LightPipeline = CachedLightPipeline
    return exact_match, spark_ok


def load_run(run):
    with open(JOBS_DIR / f"{run}.jsonl", encoding="utf-8") as f:
        jobs = {j["job_id"]: j for j in map(json.loads, f)}
    rows = []
    for path in sorted((RESULTS_DIR / run / "raw").glob("*.jsonl")):
        model = path.stem
        with open(path, encoding="utf-8") as f:
            for rec in map(json.loads, f):
                job = jobs[rec["job_id"]]
                rows.append({"model": model, "country": job["country"], "language": job["language"],
                             "prompt_no": job["prompt_no"], "ID": job["qid"], "topic": job["topic"],
                             "prompt": job["prompt"], "response": rec["response"]})
    return pd.DataFrame(rows)


def score_group(em, out_dir, model, country, language, prompt_no, g):
    res = g[["ID", "prompt", "response"]].reset_index(drop=True)
    res.to_csv(out_dir / "csv" / f"{model}-{country}_{language}_{prompt_no}_result.csv",
               index=False, encoding="utf-8")
    # Score only questions this run answered (a pilot covers a subset; a full run covers all 500).
    ann = {k: v for k, v in load_annotations(country).items() if k in set(res["ID"])}
    cwd = os.getcwd()
    os.chdir(AZ_STEMMER_DIR)  # the AZ stemmer reads words.txt/suffix.txt from the working dir
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sem_b, sem_w, scored = em.soft_exact_match(
                country=country, language=language, annotation_dict=ann, response_df=res,
                id_col="ID", r_col="response", annotations_key="annotations")
    finally:
        os.chdir(cwd)
    scored = scored.merge(g[["ID", "topic"]], on="ID")
    out = pd.DataFrame({"model": model, "country": country, "language": language, "prompt_no": prompt_no,
                        "ID": scored["ID"], "topic": scored["topic"], "response": scored["response"],
                        "score": scored["binary_score"], "weight": scored["weight_score"]})
    return out, sem_b


def combine(out_dir):
    """Merge all finished groups into per_question.csv, scores.csv and table.csv."""
    parts = sorted((out_dir / "groups").glob("*.csv"))
    pq = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    pq.to_csv(out_dir / "per_question.csv", index=False, encoding="utf-8")
    valid = pq[pq["score"].notna()]  # questions the official filter scores
    keys = ["model", "country", "language", "prompt_no"]
    scores = valid.groupby(keys).agg(n_scored=("score", "size"), **{"SEM-B": ("score", "mean"),
                                                                   "SEM-W": ("weight", "mean")}).reset_index()
    scores[["SEM-B", "SEM-W"]] = (scores[["SEM-B", "SEM-W"]] * 100).round(2)
    scores.to_csv(out_dir / "scores.csv", index=False)
    # The paper's number: average of the inst-4 and pers-3 scores
    final = scores.groupby(["model", "country", "language"])["SEM-B"].mean().reset_index()
    final["pair"] = final["country"] + "-" + final["language"]
    table = final.pivot(index="model", columns="pair", values="SEM-B").round(1)
    table.to_csv(out_dir / "table.csv")
    print(f"\nSEM-B, averaged over inst-4 and pers-3 ({len(parts)} groups):\n")
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--languages", default=None, help="only these languages, e.g. Spanish,Amharic")
    ap.add_argument("--models", default=None, help="only these models, e.g. gemma3-4b,mistral-7b")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / args.run
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    (out_dir / "groups").mkdir(exist_ok=True)

    df = load_run(args.run)
    if args.languages:
        df = df[df["language"].isin(args.languages.split(","))]
    if args.models:
        df = df[df["model"].isin(args.models.split(","))]
    keys = ["model", "country", "language", "prompt_no"]
    todo = [(k, g) for k, g in df.groupby(keys, sort=True)
            if not (out_dir / "groups" / "{}-{}_{}_{}.csv".format(*k)).exists()]
    print(f"{len(todo)} groups to score", flush=True)

    if todo:
        need_spark = any(k[2] in SPARK_LANGUAGES for k, _ in todo)
        em, spark_ok = _import_official_scorer(need_spark)
        if need_spark and not spark_ok:
            print("WARNING: Spark NLP unavailable -> skipping Spanish and Amharic (local-language) groups\n")
            todo = [(k, g) for k, g in todo if k[2] not in SPARK_LANGUAGES]
        for (model, country, language, prompt_no), g in todo:
            out, sem_b = score_group(em, out_dir, model, country, language, prompt_no, g)
            out.to_csv(out_dir / "groups" / f"{model}-{country}_{language}_{prompt_no}.csv",
                       index=False, encoding="utf-8")
            print(f"{model:12} {country:12} {language:12} {prompt_no}  n={out['score'].notna().sum():3}"
                  f"  SEM-B={sem_b:6.2f}", flush=True)
    combine(out_dir)


if __name__ == "__main__":
    main()
