"""Adaptive-RAG merge (laptop): build results/<run>/raw/ for scoring.

    python rag_pipeline/merge_gated.py --run rag_v2 --pc-run rag_v2_pc

For every job in jobs/<run>.jsonl: if the retriever sent >=1 card, the job was run on the PC (results/<pc-run>/raw);
otherwise the prompt is byte-identical to the baseline prompt and, with greedy decoding, the answer is the baseline
answer, so it is copied from results/baseline/raw. Each record gets "source": "rag" or "baseline".
"""
import argparse
import json

from config import JOBS_DIR, MODELS, RESULTS_DIR


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["job_id"]: r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pc-run", required=True)
    ap.add_argument("--baseline-run", default="baseline")
    args = ap.parse_args()

    with open(JOBS_DIR / f"{args.run}.jsonl", encoding="utf-8") as f:
        jobs = [json.loads(line)["job_id"] for line in f]
    pc_jobs = set(read(JOBS_DIR / f"{args.pc_run}.jsonl"))
    out_dir = RESULTS_DIR / args.run / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        rag = read(RESULTS_DIR / args.pc_run / "raw" / f"{model}.jsonl")
        base = read(RESULTS_DIR / args.baseline_run / "raw" / f"{model}.jsonl")
        missing = [j for j in jobs if (j in pc_jobs and j not in rag) or (j not in pc_jobs and j not in base)]
        assert not missing, f"{model}: {len(missing)} answers missing, e.g. {missing[:3]}"
        with open(out_dir / f"{model}.jsonl", "w", encoding="utf-8") as f:
            for j in jobs:
                rec = dict(rag[j], source="rag") if j in pc_jobs else dict(base[j], source="baseline")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_rag = sum(j in pc_jobs for j in jobs)
        print(f"{model}: {len(jobs)} answers ({n_rag} RAG, {len(jobs) - n_rag} baseline) -> {out_dir / (model + '.jsonl')}")


if __name__ == "__main__":
    main()
