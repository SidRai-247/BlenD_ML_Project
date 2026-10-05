"""RAG v2 retrieval (laptop): hybrid candidates -> cross-encoder rerank -> relevance cut-off.

    python rag_pipeline/retrieve_v2.py

1. Candidates: alpha*dense(bge-m3) + (1-alpha)*BM25 (min-max per query) over the country's KB
   (v1 cards + KB_Articles/<Country>_KB_v2_additions.jsonl if present); top-20.
2. Rerank the 20 candidates with the cross-encoder cross-encoder/ms-marco-MiniLM-L-12-v2 (relevance score in 0..1).
3. Keep at most 3 cards with score >= tau (so 0-3 cards per question; 0 -> the baseline prompt is used).

alpha and tau are chosen ONLY on the development countries (US, Spain, South_Korea), using answer-hit
(does a card mention a gold answer) as the relevance label; they are then frozen for Azerbaijan/Ethiopia.
"""
import json
import os

import numpy as np

from config import COUNTRY_LANG, RESULTS_DIR
from data import is_scorable, load_annotations, load_questions
from retrieve import EMBEDDER, KB_DIR, BM25Okapi, load_kb, minmax, tokenize
import re

# bge-reranker-v2-m3 ran at ~1 pair/s on the laptop CPU (~12 h for all countries); queries and cards are English,
# so the much smaller English cross-encoder ms-marco-MiniLM-L-12-v2 is used instead.
RERANKER = os.environ.get("BLEND_RERANKER", r"C:\Users\Siddharth\blend_rag\models\ms-marco-MiniLM-L12-v2")
DEV = ["US", "Spain", "South_Korea"]
N_CAND, K = 20, 3
ALPHAS = [0.8, 1.0]
TAUS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def load_kb_v2(country):
    cards = load_kb(country)
    extra = KB_DIR / f"{country}_KB_v2_additions.jsonl"
    if extra.exists():
        with open(extra, encoding="utf-8") as f:
            cards += [json.loads(line) for line in f if line.strip()]
    return cards


def golds(ann):
    return {g.lower() for agg in ann["annotations"] for g in agg["en_answers"] if len(g) >= 3}


def mentions(text, gold):
    return any(re.search(r"\b" + re.escape(g) + r"\b", text) for g in gold)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="rerank only these countries this time (scores are cached; run once more without it at the end)")
    args = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    emb = SentenceTransformer(EMBEDDER, device="cpu")
    emb.max_seq_length = 256
    rr = CrossEncoder(RERANKER, device="cpu", max_length=256)  # question + one card fit easily

    # ---- stage 1: hybrid scores for every question, both alphas ----
    data = {}
    for country in COUNTRY_LANG:
        cards = load_kb_v2(country)
        docs = [c["text_en"] + " " + " ".join(c.get("keywords") or []) for c in cards]
        bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75)
        d_emb = emb.encode(docs, normalize_embeddings=True, batch_size=16)
        qs = load_questions(country)
        q_emb = emb.encode(list(qs["q_en"]), normalize_embeddings=True, batch_size=32)
        anns = load_annotations(country)
        rows = []
        for i, row in enumerate(qs.itertuples()):
            dense, lex = d_emb @ q_emb[i], bm25.get_scores(tokenize(row.q_en))
            rows.append({"qid": row.ID, "topic": row.Topic, "query": row.q_en, "ann": anns[row.ID],
                         "dn": minmax(dense), "ln": minmax(lex)})
        data[country] = (cards, rows)
        print(f"{country}: {len(cards)} cards", flush=True)

    # ---- choose alpha on DEV: recall@20 of a gold-mentioning card ----
    def recall20(alpha):
        hit = n = 0
        for c in DEV:
            cards, rows = data[c]
            for r in rows:
                if not is_scorable(r["ann"]):
                    continue
                top = np.argsort(-(alpha * r["dn"] + (1 - alpha) * r["ln"]))[:N_CAND]
                n += 1
                hit += any(mentions(cards[j]["text_en"].lower(), golds(r["ann"])) for j in top)
        return hit / n
    rec = {a: recall20(a) for a in ALPHAS}
    alpha = max(ALPHAS, key=lambda a: rec[a])
    print("dev recall@20:", {a: round(100 * v, 1) for a, v in rec.items()}, "-> alpha =", alpha, flush=True)

    # ---- stage 2: rerank top-20 for every question (cached per country, resumable) ----
    cache_dir = RESULTS_DIR / "retrieval" / "v2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for country in COUNTRY_LANG:
        cards, rows = data[country]
        for r in rows:
            r["cand"] = [int(j) for j in np.argsort(-(alpha * r["dn"] + (1 - alpha) * r["ln"]))[:N_CAND]]
        cache = cache_dir / f"{country}_{os.path.basename(RERANKER)}_alpha{alpha}_cand{N_CAND}.npy"
        if cache.exists():
            scores = np.load(cache)
        elif args.only and country not in args.only:
            missing.append(country)
            continue
        else:
            pairs = [(r["query"], cards[j]["text_en"]) for r in rows for j in r["cand"]]
            chunks = []
            for s in range(0, len(pairs), 1000):
                chunks.append(np.asarray(rr.predict(pairs[s:s + 1000], batch_size=32, show_progress_bar=False),
                                         dtype=float))
                print(f"  {country}: {s + len(chunks[-1])}/{len(pairs)} pairs reranked", flush=True)
            scores = np.concatenate(chunks)
            if scores.min() < 0 or scores.max() > 1:  # raw logits -> probabilities
                scores = 1 / (1 + np.exp(-scores))
            np.save(cache, scores)
        for i, r in enumerate(rows):
            r["rr"] = scores[i * N_CAND:(i + 1) * N_CAND]
        print(f"{country}: rerank scores ready", flush=True)
    if missing:
        print("not reranked yet:", missing, "-> run again with --only for them")
        return

    # ---- choose tau on DEV: F0.5 (precision-weighted) of "card mentions a gold answer" among sent cards ----
    def f05(tau):
        tp = fp = fn = 0
        for c in DEV:
            cards, rows = data[c]
            for r in rows:
                if not is_scorable(r["ann"]):
                    continue
                g = golds(r["ann"])
                order = np.argsort(-r["rr"])
                sent = [k for k in order[:K] if r["rr"][k] >= tau]
                lab = [mentions(cards[r["cand"][k]]["text_en"].lower(), g) for k in range(N_CAND)]
                tp += sum(lab[k] for k in sent)
                fp += sum(not lab[k] for k in sent)
                fn += any(lab) and not any(lab[k] for k in sent)
        p = tp / (tp + fp) if tp + fp else 0.0
        rcl = tp / (tp + fn) if tp + fn else 0.0
        return 1.25 * p * rcl / (0.25 * p + rcl) if p + rcl else 0.0
    f = {t: f05(t) for t in TAUS}
    tau = max(TAUS, key=lambda t: f[t])
    print("dev F0.5 by tau:", {t: round(v, 3) for t, v in f.items()}, "-> tau =", tau, flush=True)

    # ---- write output + diagnostics ----
    out = RESULTS_DIR / "retrieval" / "v2_rerank_top3.jsonl"
    with open(out, "w", encoding="utf-8") as fo:
        for country in COUNTRY_LANG:
            cards, rows = data[country]
            n = n_ctx = hit = 0
            for r in rows:
                order = np.argsort(-r["rr"])
                sent = [k for k in order[:K] if r["rr"][k] >= tau]
                rec_cards = [{"id": cards[r["cand"][k]]["id"], "text_en": cards[r["cand"][k]]["text_en"],
                              "rerank": round(float(r["rr"][k]), 4)} for k in sent]
                fo.write(json.dumps({"country": country, "qid": r["qid"], "topic": r["topic"], "query": r["query"],
                                     "alpha": alpha, "tau": tau, "cards": rec_cards}, ensure_ascii=False) + "\n")
                if is_scorable(r["ann"]):
                    n += 1
                    n_ctx += bool(rec_cards)
                    hit += mentions(" ".join(c["text_en"].lower() for c in rec_cards), golds(r["ann"]))
            print(f"{country:12} questions with context={100 * n_ctx / n:5.1f}%  answer-hit={100 * hit / n:5.1f}%"
                  f"  (n={n})", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
