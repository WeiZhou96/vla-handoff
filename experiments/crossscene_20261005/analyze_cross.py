# -*- coding: utf-8 -*-
"""Cross-scene closed-loop analysis (CPU, read-only). Usage: python analyze_cross.py [scA|scB|scC]. Same statistics as round 1: cluster bootstrap over base initial configurations, 5000 resamples."""
import json, glob, collections, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "locked_config.json").read_text())
ARMS = cfg["arms"]
LAM = 0.5
TAG = sys.argv[1] if len(sys.argv) > 1 else "scA"
res = [json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE / f"results/{TAG}_*.json")))]
rows = []
for r in res:
    row = dict(base=r["job"]["base"])
    for a in ARMS:
        v = r["arms"][a]
        row[a] = dict(success=int(v["success"]), completed=v["completed"], steps=v["steps"], U=v["completed"] / 5 - LAM * v["steps"] / 1800)
        first = next((t for t in v["tasks"] if t["target"]), None)
        if first is not None:
            row[a]["ft"] = dict(task=first["task"], success=int(first["success"]), repair=first.get("repair"))
    rows.append(row)


def boot(rs, fn, n=5000, seed=17):
    bases = sorted({r["base"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["base"]].append(r)
    rng = np.random.RandomState(seed)
    vals = [fn([x for b in rng.choice(bases, len(bases)) for x in by[b]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


fmt = lambda d: f"{d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]"
rep = dict(scene=res[0].get("scene") if res else None, tag=TAG, chains=len(rows), paired=f"{sum(r['paired_initial_state_identical'] for r in res)}/{len(res)}",
           perturb_first_try=f"{sum(r['perturb_tries'] == 0 for r in res)}/{len(res)}", summary={}, paired_diff={}, first_target={})
print(json.dumps({k: rep[k] for k in ("scene", "chains", "paired", "perturb_first_try")}))
for a in ARMS:
    s = {k: float(np.mean([r[a][k] for r in rows])) for k in ("success", "completed", "steps", "U")}
    rep["summary"][a] = s
    print(f"{a:12s} succ={s['success']:.3f} len={s['completed']:.3f} steps={s['steps']:.0f} U={s['U']:.3f}")
comps = [("benefit", "retry"), ("benefit", "none"), ("benefit", "prior"), ("benefit", "inverse"), ("benefit_m5", "prior"),
         ("benefit_m5", "benefit"), ("benefit_m5", "inverse"), ("prior", "retry"), ("inverse", "retry"), ("retry", "none")]
print("\npaired differences:")
for A, B in comps:
    d = {k: boot(rows, lambda rs, k=k: float(np.mean([x[A][k] - x[B][k] for x in rs]))) for k in ("success", "steps", "U")}
    rep["paired_diff"][f"{A}-{B}"] = d
    print(f"  {A:12s}-{B:10s} " + "  ".join(f"{k}={fmt(v)}" for k, v in d.items()))
ft = [r for r in rows if all("ft" in r[a] for a in ARMS)]
print(f"\nfirst target handoff by task (n={len(ft)}):")
for t in sorted({r["none"]["ft"]["task"] for r in ft}) + ["ALL"]:
    rs = [r for r in ft if t == "ALL" or r["none"]["ft"]["task"] == t]
    line = {a: float(np.mean([r[a]["ft"]["success"] for r in rs])) for a in ARMS}
    ds = {f"{A}-{B}": boot(rs, lambda x, A=A, B=B: float(np.mean([y[A]["ft"]["success"] - y[B]["ft"]["success"] for y in x])), n=2000)
          for A, B in (("benefit", "prior"), ("benefit_m5", "prior"), ("benefit", "inverse"))}
    rep["first_target"][t] = dict(n=len(rs), **line, **ds,
                                  choices={a: collections.Counter(r[a]["ft"]["repair"] for r in rs).most_common(3) for a in ("benefit", "benefit_m5", "prior")})
    print(f"  {t:26s} n={len(rs):3d} " + " ".join(f"{a}={line[a]:.3f}" for a in ARMS) + " | " + " ".join(f"{k}={fmt(v)}" for k, v in ds.items()))
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / f"{TAG}_report.json").write_text(json.dumps(rep, indent=1, default=str))
print("\nsaved", HERE / "analysis" / f"{TAG}_report.json")
