"""Laptop-side helper that drives the GPU PC over SSH (steps 1b, 2, 3).

    python rag_pipeline/pc.py push   --run pilot                 # copy runner + jobs file to the PC
    python rag_pipeline/pc.py start  --run pilot [--models a,b]  # start inference in the background on the PC
    python rag_pipeline/pc.py status --run pilot                 # progress (lines written per model) + log tail
    python rag_pipeline/pc.py pull   --run pilot                 # copy answers back to results/<run>/raw/

Everything on the PC stays inside ~/Desktop/Siddharth_ML_Project. SSH asks for the password,
or set BLEND_SSH_ASKPASS to a script that prints it.
Run from Git Bash so `ssh`/`scp` are OpenSSH from Git.
"""
import argparse
import os
import shlex
import subprocess

from config import JOBS_DIR, MODELS, REMOTE_DIR, REMOTE_HOST, RESULTS_DIR, ROOT


def _env():
    env = dict(os.environ)
    if env.get("BLEND_SSH_ASKPASS"):
        env.update(SSH_ASKPASS=env["BLEND_SSH_ASKPASS"], SSH_ASKPASS_REQUIRE="force", DISPLAY=":0")
    return env


def ssh(cmd):
    print(f"[ssh] {cmd}")
    subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, env=_env(), stdin=subprocess.DEVNULL)


def scp(src, dst):
    print(f"[scp] {src} -> {dst}")
    subprocess.run(["scp", "-q", src, dst], check=True, env=_env(), stdin=subprocess.DEVNULL)


def push(run):
    ssh(f"mkdir -p {REMOTE_DIR}/code {REMOTE_DIR}/jobs {REMOTE_DIR}/results/{run}")
    scp(str(ROOT / "rag_pipeline" / "remote" / "run_ollama.py"), f"{REMOTE_HOST}:{REMOTE_DIR}/code/")
    scp(str(JOBS_DIR / f"{run}.jsonl"), f"{REMOTE_HOST}:{REMOTE_DIR}/jobs/")


def start(run, models):
    # one model after another (they share one 12 GB GPU); setsid -f detaches it from the SSH session,
    # so this returns at once and the run survives disconnects
    steps = " && ".join(
        f"python3 code/run_ollama.py --jobs jobs/{run}.jsonl --model {shlex.quote(MODELS[m])} "
        f"--out results/{run}/{m}.jsonl" for m in models)
    ssh(f"cd {REMOTE_DIR} && setsid -f sh -c {shlex.quote(steps)} >> results/{run}/log.txt 2>&1 < /dev/null")


def status(run):
    ssh(f"cd {REMOTE_DIR} && wc -l jobs/{run}.jsonl results/{run}/*.jsonl 2>/dev/null; "
        f"echo --- log; tail -n 5 results/{run}/log.txt; "
        f"echo --- running; pgrep -af run_ollama.py || echo none")


def pull(run):
    raw = RESULTS_DIR / run / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    scp(f"{REMOTE_HOST}:{REMOTE_DIR}/results/{run}/*.jsonl", str(raw))
    scp(f"{REMOTE_HOST}:{REMOTE_DIR}/results/{run}/log.txt", str(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "start", "status", "pull"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated short names")
    args = ap.parse_args()
    if args.action == "push":
        push(args.run)
    elif args.action == "start":
        start(args.run, args.models.split(","))
    elif args.action == "status":
        status(args.run)
    else:
        pull(args.run)


if __name__ == "__main__":
    main()
