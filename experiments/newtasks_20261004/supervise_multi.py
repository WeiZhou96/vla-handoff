# -*- coding: utf-8 -*-
"""Generic scheduler: launch up to --per-gpu worker processes per selected GPU, requeue unfinished jobs of crashed workers, and exit once every job has a result.

Launch condition (checked every 60 s): free GPU memory >= --min-free-mib; if this scheduler has no process on that GPU yet, utilisation must also be <= --max-util.
Only GPUs listed in --order are used (default 2,3,4,5). Workers must accept --run --jobs --gpu and claim jobs via the claims/ and --out directory convention.
Usage: nohup python -u supervise_multi.py --run <dir> --jobs <jobs.json> --worker worker.py [--out results] [--extra "..."] > logs/x.log 2>&1 &
"""
import os  # path configuration for the public release; see README, "Environment variables"
import argparse, json, os, shlex, subprocess, time
from pathlib import Path

PY = os.environ.get("HANDOFF_PYTHON", "python")


def gpu_table():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return {int(a): dict(free=int(b) - int(c), util=int(d))
            for a, b, c, d in (map(str.strip, l.split(",")) for l in out.strip().splitlines())}


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--extra", default="")
    ap.add_argument("--min-free-mib", type=int, default=14000)
    ap.add_argument("--max-util", type=int, default=50)
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--order", default="2,3,4,5")
    a = ap.parse_args()
    run, jobs_path = Path(a.run).resolve(), Path(a.jobs).resolve()
    worker = Path(a.worker).resolve()
    ids = [j["id"] for j in json.loads(jobs_path.read_text())["jobs"]]
    RES = a.out; CLM = "claims" if a.out == "results" else "claims_" + a.out
    for d in ("logs", CLM, RES):
        (run / d).mkdir(exist_ok=True)
    tag = jobs_path.stem + ("" if a.out == "results" else "_" + a.out)
    order = [int(x) for x in a.order.split(",")]
    assert set(order) <= {2, 3, 4, 5}, "only GPUs 2-5 may be used"
    workers = []   # (gpu, Popen)
    while True:
        done = sum((run / RES / f"{i}.json").exists() for i in ids)
        live = {p.pid for _, p in workers if p.poll() is None}
        for i in ids:
            c = run / CLM / f"{i}.json"
            if c.exists() and not (run / RES / f"{i}.json").exists():
                try:
                    pid = json.loads(c.read_text())["pid"]
                except Exception:
                    continue
                if pid not in live and not alive(pid):
                    c.unlink(missing_ok=True)
                    print(f"{time.strftime('%F %T')} requeue {i} (worker {pid} gone)", flush=True)
        for g, p in list(workers):
            if p.poll() is not None:
                print(f"{time.strftime('%F %T')} worker gpu{g} pid {p.pid} exited code {p.returncode}", flush=True)
                workers.remove((g, p))
        if done == len(ids):
            print(f"{time.strftime('%F %T')} ALL DONE {done}/{len(ids)}", flush=True)
            break
        claimed = sum((run / CLM / f"{i}.json").exists() and not (run / RES / f"{i}.json").exists() for i in ids)
        pending = len(ids) - done - claimed
        if pending > 0:
            t = gpu_table()
            for g in order:
                mine = sum(1 for gg, _ in workers if gg == g)
                if mine >= a.per_gpu or pending <= 0:
                    continue
                if t[g]["free"] >= a.min_free_mib and (mine > 0 or t[g]["util"] <= a.max_util):
                    log = open(run / "logs" / f"{tag}_gpu{g}_{int(time.time())}.log", "w")
                    cmd = [PY, "-u", "-B", str(worker), "--run", str(run), "--jobs", str(jobs_path), "--gpu", str(g)]
                    if a.out != "results":
                        cmd += ["--out", RES]
                    cmd += shlex.split(a.extra)
                    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    workers.append((g, p)); pending -= 1
                    print(f"{time.strftime('%F %T')} launch gpu{g} pid {p.pid} free={t[g]['free']} util={t[g]['util']} (slot {mine + 1})", flush=True)
                    time.sleep(45)
                    t = gpu_table()
        (run / "logs" / f"status_{tag}.json").write_text(json.dumps(dict(time=time.strftime('%F %T'), done=done, total=len(ids),
                                                                       workers=[(g, p.pid) for g, p in workers])))
        time.sleep(60)


if __name__ == "__main__":
    main()
