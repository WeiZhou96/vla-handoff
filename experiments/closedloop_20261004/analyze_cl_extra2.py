# -*- coding: utf-8 -*-
"""Analysis of supplementary arms on the round-1 closed-loop chains (CPU, read-only).
Supplementary arms (added after the round-1 results were reported; all are post-hoc controls; models are trained on development data only):
  benefit_noretry  same repair decision as benefit, no retry (requires no failure detection)
  home_all         run the scripted home repair before every target task
  noloc            selection and benefit gate with the no-localiser model M7n (no simulator ground-truth supervision)
  benefit_m5       selection and benefit gate with state model M5v2 (no prior values)
Merged with results/ and results_extra/; per chain, the initial physics-state hashes of all arms are checked for equality."""
import json, glob, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ARMS = ("none", "retry", "fixed", "risk", "benefit", "full", "inverse", "prior", "benefit_noretry", "home_all", "noloc", "benefit_m5")
LAM = 0.5
res = [json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE / "results/cl_*.json")))]
ok = 0
for r in res:
    for sub in ("results_extra", "results_extra2"):
        e = json.loads((HERE / sub / (r["job"]["id"] + ".json")).read_text())
        r["arms"].update(e["arms"])
    ok += int(len({v["init_physics_sha"] for v in r["arms"].values()}) == 1)
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
rep = dict(chains=len(rows), all_arms_same_initial_state=f"{ok}/{len(res)}", summary={}, paired={}, first_target={})
print(json.dumps({k: rep[k] for k in ("chains", "all_arms_same_initial_state")}))
for a in ARMS:
    s = {k: float(np.mean([r[a][k] for r in rows])) for k in ("success", "completed", "steps", "U")}
    rep["summary"][a] = s
    print(f"{a:16s} succ={s['success']:.3f} len={s['completed']:.3f} steps={s['steps']:.0f} U={s['U']:.3f}")
comps = [("benefit_noretry", "none"), ("benefit_noretry", "retry"), ("benefit", "benefit_noretry"), ("home_all", "retry"),
         ("benefit", "home_all"), ("noloc", "benefit"), ("noloc", "prior"), ("benefit_m5", "benefit"), ("benefit_m5", "prior"),
         ("benefit_m5", "inverse")]
print("\npaired differences:")
for A, B in comps:
    d = {k: boot(rows, lambda rs, k=k: float(np.mean([x[A][k] - x[B][k] for x in rs]))) for k in ("success", "steps", "U")}
    rep["paired"][f"{A}-{B}"] = d
    print(f"  {A:16s}-{B:16s} " + "  ".join(f"{k}={fmt(v)}" for k, v in d.items()))
ft = [r for r in rows if all("ft" in r[a] for a in ARMS)]
print(f"\nfirst target handoff by task (n={len(ft)}):")
for t in sorted({r["none"]["ft"]["task"] for r in ft}):
    rs = [r for r in ft if r["none"]["ft"]["task"] == t]
    line = {a: float(np.mean([r[a]["ft"]["success"] for r in rs])) for a in ARMS}
    d = boot(rs, lambda x: float(np.mean([y["benefit_m5"]["ft"]["success"] - y["benefit"]["ft"]["success"] for y in x])), n=2000)
    ch = collections.Counter(r["benefit_m5"]["ft"]["repair"] for r in rs).most_common(3)
    chn = collections.Counter(r["noloc"]["ft"]["repair"] for r in rs).most_common(3)
    rep["first_target"][t] = dict(n=len(rs), **line, m5_minus_benefit=d, m5_choices=ch, noloc_choices=chn)
    print(f"  {t:26s} n={len(rs):3d} " + " ".join(f"{a}={line[a]:.3f}" for a in ARMS) + f" | m5-benefit={fmt(d)} m5 choices={ch} noloc={chn}")
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / "cl_extra2_report.json").write_text(json.dumps(rep, indent=1, default=str))
print("\nsaved", HERE / "analysis" / "cl_extra2_report.json")
