"""Step R (laptop): region-aware hybrid retrieval (BM25 + bge-m3) over each country's English KB.

    python rag_pipeline/retrieve.py                 # all 500 questions x 5 countries, top-3
    python rag_pipeline/retrieve.py --k 5 --alpha 0.5 --out results/retrieval/oracle_en_top5.jsonl

Design (PLAN.md, "Always-RAG: exact design"):
  * one index per country: a question only retrieves cards of its own country
  * document = card text_en + keywords;  query = BLEnD's English version of the question
  * BM25Okapi (k1=1.5, b=0.75) on lowercase \\w+ tokens minus English stop-words
  * dense = cosine of L2-normalised BAAI/bge-m3 embeddings
  * hybrid = alpha * minmax(dense) + (1 - alpha) * minmax(BM25), per query over the country's cards
Output: one JSON line per (country, question) with the top-k cards and their scores (raw + normalised),
which the RAG prompts and the Adaptive-RAG gate (PS-2) both use. Also prints a retrieval diagnostic
(answer-hit@k) that uses the gold answers ONLY to measure coverage. It is never used to tune anything.
"""
import argparse
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import COUNTRY_LANG, ROOT, RESULTS_DIR
from data import is_scorable, load_annotations, load_questions

KB_DIR = ROOT / "KB_Articles"
KB_FILES = {
    "US": "US_KB_150.jsonl",
    "Spain": "Spain_KB_150.jsonl",
    "South_Korea": "South_Korea_KB_150.jsonl",
    "Azerbaijan": "Azerbaijan_KB_150.jsonl",
    "Ethiopia": "Ethiopia_KB_partial.jsonl",
}
EMBEDDER = os.environ.get("BLEND_EMBEDDER", r"C:\Users\Siddharth\blend_rag\models\bge-m3")


def load_kb(country):
    with open(KB_DIR / KB_FILES[country], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def answer_hit(cards, ann):
    """Diagnostic only: does any of these cards mention one of the gold English answers?"""
    text = " ".join(c["text_en"].lower() for c in cards)
    golds = {g.lower() for agg in ann["annotations"] for g in agg["en_answers"] if len(g) >= 3}
    return any(re.search(r"\b" + re.escape(g) + r"\b", text) for g in golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the dense score")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or RESULTS_DIR / "retrieval" / f"oracle_en_top{args.k}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER, device="cpu")
    model.max_seq_length = 256  # cards and questions are short

    with open(out, "w", encoding="utf-8") as f:
        for country in COUNTRY_LANG:
            cards = load_kb(country)
            docs = [c["text_en"] + " " + " ".join(c.get("keywords") or []) for c in cards]
            bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75)
            doc_emb = model.encode(docs, normalize_embeddings=True, batch_size=16)

            questions = load_questions(country)
            q_emb = model.encode(list(questions["q_en"]), normalize_embeddings=True, batch_size=32)
            anns = load_annotations(country)
            hits = [0, 0]  # hit@1, hit@k over scorable questions
            n_scorable = 0
            for i, row in enumerate(questions.itertuples()):
                dense = doc_emb @ q_emb[i]
                lex = bm25.get_scores(tokenize(row.q_en))
                hybrid = args.alpha * minmax(dense) + (1 - args.alpha) * minmax(lex)
                top = np.argsort(-hybrid)[: args.k]
                rec = {
                    "country": country, "qid": row.ID, "topic": row.Topic, "query": row.q_en,
                    "max_dense": round(float(dense.max()), 4), "max_bm25": round(float(lex.max()), 4),
                    "cards": [{"id": cards[j]["id"], "category": cards[j]["category"],
                               "text_en": cards[j]["text_en"], "dense": round(float(dense[j]), 4),
                               "bm25": round(float(lex[j]), 4), "hybrid": round(float(hybrid[j]), 4)}
                              for j in top],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if is_scorable(anns[row.ID]):
                    n_scorable += 1
                    hits[0] += answer_hit(rec["cards"][:1], anns[row.ID])
                    hits[1] += answer_hit(rec["cards"], anns[row.ID])
            print(f"{country:12} cards={len(cards):3}  answer-hit@1={100 * hits[0] / n_scorable:5.1f}%  "
                  f"answer-hit@{args.k}={100 * hits[1] / n_scorable:5.1f}%  (n={n_scorable})", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
