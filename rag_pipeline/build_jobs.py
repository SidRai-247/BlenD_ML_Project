"""Step 1 (laptop): turn BLEnD questions + prompt templates into a jobs file, one prompt per line.

    python rag_pipeline/build_jobs.py --run pilot --limit 20
    python rag_pipeline/build_jobs.py --run baseline
    python rag_pipeline/build_jobs.py --run rag --retrieval results/retrieval/oracle_en_top3.jsonl
    python rag_pipeline/build_jobs.py --run rag_v2 --pc-run rag_v2_pc --prompt-version v2 \
        --retrieval results/retrieval/v2_rerank_top3.jsonl --countries Azerbaijan Ethiopia

Output: jobs/<run>.jsonl. With --retrieval, each prompt is the unchanged baseline prompt with a block of
retrieved KB cards in front of it (same cards for the local-language and English version of a question).
"""
import argparse
import json
import random

from config import COUNTRY_LANG, JOBS_DIR, PROMPT_NOS, country_language_pairs
from data import is_scorable, load_annotations, load_prompt_template, load_questions, question_text


COUNTRY_NAME = {"US": "the US", "Spain": "Spain", "South_Korea": "South Korea",
                "Azerbaijan": "Azerbaijan", "Ethiopia": "Ethiopia"}


def load_retrieval(path):
    with open(path, encoding="utf-8") as f:
        return {(r["country"], r["qid"]): r["cards"] for r in map(json.loads, f)}


def context_block(country, cards, version="v1"):
    if version == "v1":
        lines = [f"Background information about {COUNTRY_NAME[country]} (may or may not be relevant):"]
    else:  # v2: explicit fallback to own knowledge, no refusals
        lines = [f"Background information about {COUNTRY_NAME[country]}. Use it only if it directly helps to answer "
                 "the question; otherwise ignore it and answer from your own knowledge. Always give an answer and "
                 "never say that the information is not provided."]
    lines += [f"[{i}] {c['text_en']}" for i, c in enumerate(cards, 1)]
    return "\n".join(lines) + "\n\n"


def pick_ids(limit, seed):
    """All question IDs, or a fixed random sample of IDs that are scorable in every country."""
    ids = list(load_questions("US")["ID"])
    if limit is None:
        return set(ids)
    anns = {c: load_annotations(c) for c in COUNTRY_LANG}
    ok = [i for i in ids if all(is_scorable(anns[c][i]) for c in COUNTRY_LANG)]
    return set(random.Random(seed).sample(ok, limit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. pilot or baseline")
    ap.add_argument("--limit", type=int, default=None, help="questions per country (default: all 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieval", default=None, help="retrieval file from retrieve.py -> RAG prompts")
    ap.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--countries", nargs="*", default=None, help="restrict to these countries")
    ap.add_argument("--pc-run", default=None,
                    help="also write jobs/<pc-run>.jsonl with only the prompts that have >=1 card (to send to the PC); "
                         "the others reuse the baseline answer (identical prompt, greedy decoding)")
    args = ap.parse_args()

    ids = pick_ids(args.limit, args.seed)
    retrieved = load_retrieval(args.retrieval) if args.retrieval else None
    JOBS_DIR.mkdir(exist_ok=True)
    out = JOBS_DIR / f"{args.run}.jsonl"
    pc = open(JOBS_DIR / f"{args.pc_run}.jsonl", "w", encoding="utf-8") if args.pc_run else None
    n = n_pc = 0
    with open(out, "w", encoding="utf-8") as f:
        for country, language in country_language_pairs():
            if args.countries and country not in args.countries:
                continue
            questions = load_questions(country)
            questions = questions[questions["ID"].isin(ids)]
            for prompt_no in PROMPT_NOS:
                template = load_prompt_template(country, prompt_no, language)
                for _, row in questions.iterrows():
                    prompt = template.replace("{q}", question_text(row, country, language))
                    cards = retrieved[(country, row["ID"])] if retrieved is not None else []
                    if cards:
                        prompt = context_block(country, cards, args.prompt_version) + prompt
                    job = {
                        "job_id": f"{country}|{language}|{prompt_no}|{row['ID']}",
                        "country": country, "language": language, "prompt_no": prompt_no,
                        "qid": row["ID"], "topic": row["Topic"],
                        "prompt": prompt,
                    }
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
                    n += 1
                    if pc and cards:
                        pc.write(json.dumps(job, ensure_ascii=False) + "\n")
                        n_pc += 1
    print(f"{n} prompts ({len(ids)} questions/country) -> {out}")
    if pc:
        pc.close()
        print(f"{n_pc} prompts with context -> {pc.name}")


if __name__ == "__main__":
    main()
