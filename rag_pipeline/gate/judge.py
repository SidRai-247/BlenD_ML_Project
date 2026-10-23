"""RAG v3, judge feature (laptop or PC): does the retrieved background actually STATE the answer?

    python rag_pipeline/gate/judge.py                       # Ollama on localhost, mistral
    python rag_pipeline/gate/judge.py --model qwen2.5:3b --host http://localhost:11434

One call per (country, question) that v2 retrieval sent cards for (~1,450 calls; the judge reads the English
question + English cards, so it is the same for every model, language and prompt). It never sees gold answers or
model answers. The reranker scores topic relevance; this asks the narrower question "is the answer in here?"
(the retrieval-evaluator idea of CRAG, Yan et al. 2024).

p_yes = P(YES) / (P(YES) + P(NO)) from the first token's log-probabilities (needs Ollama >= 0.12); if logprobs are
unavailable it falls back to 1.0 / 0.0 from the text. Resumable: appends to results/gate/judge.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

PROMPT = """You check whether background notes answer a question about everyday life in {country}.

Question: {question}

Background notes:
{notes}

Do the background notes explicitly state an answer to this question (not just mention the topic)?
Reply with only YES or NO."""


def p_yes(resp):
    lps = (resp.get("logprobs") or [{}])[0].get("top_logprobs") or []
    yes = sum(math.exp(t["logprob"]) for t in lps if t["token"].strip().upper().startswith("YES"))
    no = sum(math.exp(t["logprob"]) for t in lps if t["token"].strip().upper().startswith("NO"))
    if yes + no > 0:
        return yes / (yes + no), "logprobs"
    return float(resp["response"].strip().upper().startswith("YES")), "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (speed test)")
    args = ap.parse_args()

    out = RESULTS_DIR / "gate" / "judge.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {(j["country"], j["qid"]) for j in map(json.loads, f)}
    with open(RESULTS_DIR / "retrieval" / "v2_rerank_top3.jsonl", encoding="utf-8") as f:
        todo = [r for r in map(json.loads, f) if r["cards"] and (r["country"], r["qid"]) not in done]
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(done)} done, {len(todo)} to judge with {args.model}", flush=True)

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            notes = "\n".join(f"- {c['text_en']}" for c in r["cards"])
            prompt = PROMPT.format(country=r["country"].replace("_", " "), question=r["query"], notes=notes)
            resp = requests.post(f"{args.host}/api/generate", timeout=600, json={
                "model": args.model, "prompt": prompt, "stream": False, "logprobs": True, "top_logprobs": 10,
                "options": {"temperature": 0, "num_predict": 1}}).json()
            p, how = p_yes(resp)
            f.write(json.dumps({"country": r["country"], "qid": r["qid"], "p_yes": round(p, 4), "how": how,
                                "raw": resp["response"], "model": args.model}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)}  {el / i:.1f}s/q  ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
