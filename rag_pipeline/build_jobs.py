"""Step 1 (laptop): turn BLEnD questions + prompt templates into a jobs file, one prompt per line.

    python rag_pipeline/build_jobs.py --run pilot --limit 20
    python rag_pipeline/build_jobs.py --run baseline
    python rag_pipeline/build_jobs.py --run rag --retrieval results/retrieval/oracle_en_top3.jsonl

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


def context_block(country, cards):
    lines = [f"Background information about {COUNTRY_NAME[country]} (may or may not be relevant):"]
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
    args = ap.parse_args()

    ids = pick_ids(args.limit, args.seed)
    retrieved = load_retrieval(args.retrieval) if args.retrieval else None
    JOBS_DIR.mkdir(exist_ok=True)
    out = JOBS_DIR / f"{args.run}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for country, language in country_language_pairs():
            questions = load_questions(country)
            questions = questions[questions["ID"].isin(ids)]
            for prompt_no in PROMPT_NOS:
                template = load_prompt_template(country, prompt_no, language)
                for _, row in questions.iterrows():
                    prompt = template.replace("{q}", question_text(row, country, language))
                    if retrieved is not None:
                        prompt = context_block(country, retrieved[(country, row["ID"])]) + prompt
                    job = {
                        "job_id": f"{country}|{language}|{prompt_no}|{row['ID']}",
                        "country": country, "language": language, "prompt_no": prompt_no,
                        "qid": row["ID"], "topic": row["Topic"],
                        "prompt": prompt,
                    }
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} prompts ({len(ids)} questions/country) -> {out}")


if __name__ == "__main__":
    main()
