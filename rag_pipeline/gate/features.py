"""RAG v3, step 1 (laptop): one feature row per answer that RAG v2 actually changed the prompt for.

    python rag_pipeline/gate/features.py

A row = (model, country, language, prompt, question) where v2 retrieval passed >= 1 card. For such a row we already
have two officially scored answers: the baseline answer (no cards) and the RAG v2 answer (with cards). The gate's job
is to pick one of them, using only things known at inference time (never the gold answers):
  retrieval   r_max, r_mean, r_min, r_gap, n_cards       (cross-encoder scores of the cards that passed tau)
  behaviour   base_agree   baseline inst-4 and pers-3 answers are the same (model is self-consistent)
              base_in_cards / rag_in_cards   share of the answer's words that occur in the cards (+ their keywords)
              changed, rag_base_overlap      did the cards change the answer, and how much
              base_len, rag_len              answer lengths in words
  context     model, English-or-local language, prompt, topic
  judge       judge_yes   P(YES) from the LLM answer-check judge (rag_pipeline/gate/judge.py), if it has been run
Label (only used for training/evaluation): help = RAG v2 score > baseline score; ties are kept with help = NaN.

Writes results/gate/features.csv.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, RESULTS_DIR

KEYS = ["model", "country", "language", "prompt_no", "ID"]
STOP = set("the a an and or of in on to for with is are was be by at as from that this it its their his her "
           "they them which what who most common popular usually people often also one some".split())


def words(text):
    return {w for w in re.findall(r"\w+", str(text).lower()) if len(w) > 2 and w not in STOP}


def norm(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def share_in(answer, ref):
    a = words(answer)
    return len(a & ref) / len(a) if a else 0.0


def load_cards():
    """(country, qid) -> list of {id, text_en, rerank} for the questions v2 sent cards for."""
    keywords = {}
    for path in (ROOT / "KB_Articles").glob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for card in map(json.loads, f):
                keywords[card["id"]] = " ".join(card.get("keywords") or [])
    cards = {}
    with open(RESULTS_DIR / "retrieval" / "v2_rerank_top3.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["cards"]:
                for c in r["cards"]:
                    c["keywords"] = keywords.get(c["id"], "")
                cards[(r["country"], r["qid"])] = r["cards"]
    return cards


def main():
    cards = load_cards()
    base = pd.read_csv(RESULTS_DIR / "baseline" / "per_question.csv")
    rag = pd.concat([pd.read_csv(RESULTS_DIR / run / "per_question.csv") for run in ["rag_v2", "rag_v2_dev"]])
    df = base.merge(rag[KEYS + ["response", "score"]], on=KEYS, suffixes=("_b", "_v"))
    df = df[[(c, q) in cards for c, q in zip(df["country"], df["ID"])]].copy()

    # baseline self-consistency: same answer under both prompts
    piv = base.pivot_table(index=["model", "country", "language", "ID"], columns="prompt_no",
                           values="response", aggfunc="first")
    agree = {k: norm(r["inst-4"]) == norm(r["pers-3"]) for k, r in piv.iterrows()}

    rows = []
    for r in df.itertuples(index=False):
        cs = cards[(r.country, r.ID)]
        sc = sorted((c["rerank"] for c in cs), reverse=True)
        ref = words(" ".join(c["text_en"] + " " + c["keywords"] for c in cs))
        rows.append({
            "r_max": sc[0], "r_mean": sum(sc) / len(sc), "r_min": sc[-1],
            "r_gap": sc[0] - (sc[1] if len(sc) > 1 else 0.0), "n_cards": len(sc),
            "base_agree": int(agree.get((r.model, r.country, r.language, r.ID), False)),
            "base_in_cards": share_in(r.response_b, ref), "rag_in_cards": share_in(r.response_v, ref),
            "changed": int(norm(r.response_b) != norm(r.response_v)),
            "rag_base_overlap": share_in(r.response_v, words(r.response_b)),
            "base_len": len(str(r.response_b).split()), "rag_len": len(str(r.response_v).split()),
        })
    feats = pd.concat([df[KEYS + ["topic", "score_b", "score_v"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    feats["is_english"] = (feats["language"] == "English").astype(int)
    feats["template"] = feats["ID"]  # the same question template is asked in every country -> CV group
    feats["help"] = (feats["score_v"] > feats["score_b"]).astype(float)
    tie_or_unscored = (feats["score_v"] == feats["score_b"]) | feats["score_b"].isna() | feats["score_v"].isna()
    feats.loc[tie_or_unscored, "help"] = float("nan")

    judge_path = RESULTS_DIR / "gate" / "judge.jsonl"
    if judge_path.exists():
        with open(judge_path, encoding="utf-8") as f:
            judge = {(j["country"], j["qid"]): j["p_yes"] for j in map(json.loads, f)}
        feats["judge_yes"] = [judge.get((c, q)) for c, q in zip(feats["country"], feats["ID"])]

    out = RESULTS_DIR / "gate" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False, encoding="utf-8")
    scored = feats["score_b"].notna() & feats["score_v"].notna()
    print(f"{len(feats)} rows ({scored.sum()} scored; {(feats['help'] == 1).sum()} help, "
          f"{(feats['help'] == 0).sum()} hurt) -> {out}")


if __name__ == "__main__":
    main()
