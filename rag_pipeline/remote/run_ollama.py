"""Step 2 (GPU PC): send every prompt in a jobs file to a local Ollama model and save the answers.

Needs only python3 + requests (already on the PC). Resumable: job_ids already in the output are skipped.

    python3 run_ollama.py --jobs jobs/pilot.jsonl --model gemma3:4b --out results/pilot/gemma3-4b.jsonl
"""
import argparse
import json
import os
import time

import requests


def ask(host, model, prompt, num_predict, retries=3):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # greedy decoding as in the BLEnD paper; 512 new tokens max like their code
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict},
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry after error: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. gemma3:4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-predict", type=int, default=512)
    args = ap.parse_args()

    with open(args.jobs, encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["job_id"] for line in f if line.strip()}
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"{args.model}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, job in enumerate(todo, 1):
            t = time.time()
            response = ask(args.host, args.model, job["prompt"], args.num_predict)
            rec = {"job_id": job["job_id"], "model": args.model, "response": response,
                   "seconds": round(time.time() - t, 3)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - start:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
