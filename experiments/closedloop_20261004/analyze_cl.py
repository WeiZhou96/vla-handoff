# -*- coding: utf-8 -*-
"""Analysis of the closed-loop confirmation experiment (CPU, read-only).
Unit of analysis: base initial state (the 4 chains sharing a base initial state form one cluster); 5000 paired bootstrap resamples.
Metrics: chain success rate, mean number of consecutively completed tasks, mean steps, utility (completed/5 - lambda*steps/1800, lambda=0.5), numbers of repairs and retries;
target-task success and steps consumed at the first target handoff (where all arms are in exactly the same state)."""
import json, glob, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ARMS = ("none", "retry", "fixed", "risk", "benefit", "full", "inverse", "prior")
LAM = 0.5
res = [json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE / "results/cl_*.json")))]
extra_ok = 0
for r in res:
    ef = HERE / "results_extra" / (r["job"]["id"] + ".json")
    if not ef.exists():
        raise SystemExit("missing extra arms for " + r["job"]["id"])
    e = json.loads(ef.read_text())
    h0 = {v["init_physics_sha"] for v in r["arms"].values()} | {v["init_physics_sha"] for v in e["arms"].values()}
    extra_ok += int(len(h0) == 1)
    r["arms"].update(e["arms"])
paired_ok = sum(r["paired_initial_state_identical"] for r in res)
rows = []
for r in res:
    row = dict(base=r["job"]["base"], id=r["job"]["id"])
    for a in ARMS:
        v = r["arms"][a]
        rep = sum(1 for t in v["tasks"] if t.get("repair"))
        rtr = sum(max(0, len(t["attempts"]) - 1) for t in v["tasks"])
        row[a] = dict(success=int(v["success"]), completed=v["completed"], steps=v["steps"],
                      U=v["completed"] / 5 - LAM * v["steps"] / 1800, repairs=rep, retries=rtr)
        first = next((t for t in v["tasks"] if t["target"]), None)
        if first is not None:
            st = sum(x["steps"] for x in first["attempts"]) + (first.get("repair_steps") or 0)
            row[a]["first_target"] = dict(task=first["task"], success=int(first["success"]), steps=st,
                                          repaired=int(bool(first.get("repair"))), q_none=first.get("q_none"))
    rows.append(row)


def boot(fn, n=5000, seed=17):
    bases = sorted({r["base"] for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r["base"]].append(r)
    rng = np.random.RandomState(seed)
    vals = [fn([x for b in rng.choice(bases, len(bases)) for x in by[b]]) for _ in range(n)]
    return float(fn(rows)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rep = dict(chains=len(rows), bases=len({r["base"] for r in rows}), paired_initial_state_identical=f"{paired_ok}/{len(res)}",
           extra_arms_same_initial_state=f"{extra_ok}/{len(res)}")
print(json.dumps(rep))
print(f"\n{'arm':8s} {'chain succ':>10s} {'avg len':>8s} {'steps':>7s} {'utility':>8s} {'repairs/chain':>13s} {'retries/chain':>13s}")
summ = {}
for a in ARMS:
    s = {k: float(np.mean([r[a][k] for r in rows])) for k in ("success", "completed", "steps", "U", "repairs", "retries")}
    summ[a] = s
    print(f"{a:8s} {s['success']:10.3f} {s['completed']:8.3f} {s['steps']:7.0f} {s['U']:8.3f} {s['repairs']:13.2f} {s['retries']:13.2f}")
rep["summary"] = summ
comps = [("benefit", "inverse"), ("benefit", "prior"), ("inverse", "retry"), ("prior", "retry"), ("benefit", "risk"), ("benefit", "fixed"), ("benefit", "retry"), ("full", "benefit"), ("full", "retry"), ("full", "none"),
         ("risk", "retry"), ("fixed", "retry"), ("retry", "none")]
rep["paired"] = {}
print("\npaired differences (A − B), 95% CI by base-initial-state cluster bootstrap:")
for A, Bm in comps:
    d = {k: boot(lambda rs, k=k: float(np.mean([x[A][k] - x[Bm][k] for x in rs]))) for k in ("success", "completed", "steps", "U")}
    rep["paired"][f"{A}-{Bm}"] = d
    print(f"  {A:8s}-{Bm:8s} " + "  ".join(f"{k}={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for k, v in d.items()))
# first target handoff
ft =[r for r in rows if all("first_target" in r[a] for a in ARMS)]
print(f"\nfirst target handoff (identical state across arms), n={len(ft)}:")
rep["first_target"] = {}
for a in ARMS:
    s = float(np.mean([r[a]["first_target"]["success"] for r in ft])); st = float(np.mean([r[a]["first_target"]["steps"] for r in ft]))
    rp = float(np.mean([r[a]["first_target"]["repaired"] for r in ft]))
    rep["first_target"][a] = dict(success=s, steps=st, repaired=rp)
    print(f"  {a:8s} success={s:.3f} steps={st:6.1f} repaired={rp:.2f}")
for A, Bm in (("benefit", "inverse"), ("benefit", "prior"), ("benefit", "risk"), ("benefit", "fixed"), ("benefit", "retry"), ("full", "retry")):
    d = boot(lambda rs: float(np.mean([x[A]["first_target"]["success"] - x[Bm]["first_target"]["success"] for x in rs if "first_target" in x[A] and "first_target" in x[Bm]])))
    ds = boot(lambda rs: float(np.mean([x[A]["first_target"]["steps"] - x[Bm]["first_target"]["steps"] for x in rs if "first_target" in x[A] and "first_target" in x[Bm]])))
    rep["first_target"][f"{A}-{Bm}"] = dict(success=d, steps=ds)
    print(f"  {A}-{Bm}: success {d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]  steps {ds[0]:+.1f}[{ds[1]:+.1f},{ds[2]:+.1f}]")
# by task (first target handoff)
bytask = collections.defaultdict(list)
for r in ft:
    bytask[r["none"]["first_target"]["task"]].append(r)
rep["first_target_by_task"] = {}
print("\nfirst target handoff by task:")
for t, rs in sorted(bytask.items()):
    line = {a: float(np.mean([r[a]["first_target"]["success"] for r in rs])) for a in ARMS}
    rep["first_target_by_task"][t] = dict(n=len(rs), **line)
    print(f"  {t:26s} n={len(rs):3d} " + " ".join(f"{a}={line[a]:.3f}" for a in ARMS))
    for A, Bm in (("benefit", "inverse"), ("benefit", "prior")):
        bs = sorted({r["base"] for r in rs}); byb = collections.defaultdict(list)
        for r in rs: byb[r["base"]].append(r)
        rng = np.random.RandomState(3)
        vals = [np.mean([x[A]["first_target"]["success"] - x[Bm]["first_target"]["success"] for b in rng.choice(bs, len(bs)) for x in byb[b]]) for _ in range(3000)]
        est = float(np.mean([x[A]["first_target"]["success"] - x[Bm]["first_target"]["success"] for x in rs]))
        rep["first_target_by_task"][t][f"{A}-{Bm}"] = [est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        print(f"      {A}-{Bm}: {est:+.3f} [{np.percentile(vals, 2.5):+.3f}, {np.percentile(vals, 97.5):+.3f}]")
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / "cl_report.json").write_text(json.dumps(rep, indent=1))
(HERE / "analysis" / "cl_rows.json").write_text(json.dumps(rows))
print("\nsaved", HERE / "analysis" / "cl_report.json")
