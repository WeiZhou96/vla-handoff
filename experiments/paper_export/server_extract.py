# -*- coding: utf-8 -*-
"""Read-only extraction of per-handoff and per-chain records for the paper figures.

Runs on the experiment server under $HANDOFF_RUN_ROOT/paper_draft_20261006/ and only reads existing result files.
Outputs (CSV, small):
  handoffs_j8.csv        one row per measured CALVIN handoff: task, source, sample id, parent (initial-state) id, k, n, wins
  round1_per_chain.csv   first-round closed loop, all 12 arms (results + results_extra + results_extra2)
  round1_first_target.csv first target handoff per chain and arm (task, success, repair)
  round2_per_chain.csv   second-round closed loop, 9 arms
  scene_per_chain.csv    cross-scene closed loop (A, B, C), 6 arms
  libero_j8.csv          LIBERO-Goal handoffs: k per handoff (from the J8 jsonl files)
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import csv, glob, json
from pathlib import Path
V = Path(f"{RUN_ROOT}")
OUT = V / "paper_draft_20261006"


def w(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
    print(name, len(rows))


# ---- CALVIN J8 handoffs ----
rows = []
for g in sorted((V / "fix_20260928/runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            rows.append([r["task_b"], "dev2", r["sample_id"], r["parent_id"], r["k"], r["n"], "".join(str(int(x["success"])) for x in r["repeats"])])
for run, src in ((V / "multitask_vocab_20261002", "mt"), (V / "newtasks_20261004", "nt")):
    for f in sorted((run / "results").glob("j8_*.json")):
        d = json.loads(f.read_text())
        if d.get("reached") and d.get("n") == 8 and "restore_failed" not in d:
            rows.append([d["task_b"], src, d["sample_id"], d["parent_id"], d["k"], d["n"], "".join(str(int(x["success"])) for x in d["repeats"])])
w("handoffs_j8.csv", ["task", "source", "sample_id", "parent_id", "k", "n", "wins"], rows)


def chain_rows(results, arms, extra_dirs=()):
    out, ft = [], []
    for f in results:
        r = json.loads(Path(f).read_text())
        for sub in extra_dirs:
            e = json.loads((Path(f).parent.parent / sub / Path(f).name).read_text())
            r["arms"].update(e["arms"])
        same = len({v["init_physics_sha"] for v in r["arms"].values()}) == 1
        row = [r["job"]["id"], r["job"]["base"], r["job"].get("stratum", ""), int(same)]
        for a in arms:
            v = r["arms"][a]
            reps = sum(1 for t in v["tasks"] if t.get("repair"))
            rtr = sum(max(0, len(t["attempts"]) - 1) for t in v["tasks"])
            row += [int(v["success"]), v["completed"], v["steps"], reps, rtr]
            first = next((t for t in v["tasks"] if t["target"]), None)
            if first is not None:
                ft.append([r["job"]["id"], r["job"]["base"], a, first["task"], first["pos"], int(first["success"]), first.get("repair") or "none",
                           int(bool(first["attempts"]) and first["attempts"][0]["success"])])
        out.append(row)
    hdr = ["chain", "base", "stratum", "same_initial_state"] + [f"{a}_{m}" for a in arms for m in ("success", "completed", "steps", "repairs", "retries")]
    return hdr, out, ft


A1 = ["none", "retry", "fixed", "risk", "benefit", "full", "inverse", "prior", "benefit_noretry", "home_all", "noloc", "benefit_m5"]
h, rows, ft = chain_rows(sorted(glob.glob(str(V / "closedloop_20261004/results/cl_*.json"))), A1, ("results_extra", "results_extra2"))
w("round1_per_chain.csv", h, rows)
w("round1_first_target.csv", ["chain", "base", "arm", "task", "pos", "success", "repair", "first_try_success"], ft)
A2 = ["none", "retry", "inverse", "prior", "home_all", "benefit", "benefit_noretry", "noloc", "benefit_alt"]
h, rows, ft = chain_rows(sorted(glob.glob(str(V / "closedloop2_20261005/results/cl2_*.json"))), A2)
w("round2_per_chain.csv", h, rows)
w("round2_first_target.csv", ["chain", "base", "arm", "task", "pos", "success", "repair", "first_try_success"], ft)
AS = ["none", "retry", "inverse", "prior", "benefit", "benefit_m5"]
allrows = []
for tag in ("scA", "scB", "scC"):
    h, rows, _ = chain_rows(sorted(glob.glob(str(V / f"crossscene_20261005/results/{tag}_*.json"))), AS)
    allrows += [[tag] + r for r in rows]
w("scene_per_chain.csv", ["scene"] + h, allrows)

# ---- LIBERO J8 ----
rows = []
for f in sorted((V / "libero_feasibility_20261004/runs/gate_j8/J8").glob("*.jsonl")):
    for line in f.read_text().splitlines():
        rows.append(json.loads(line))
keys = sorted({k for r in rows for k in r})
print("libero J8 keys:", keys[:30])
(OUT / "libero_j8_rows.json").write_text(json.dumps(rows[:3], indent=1, default=str))
