# -*- coding: utf-8 -*-
"""Analysis of the round-2 closed-loop confirmation experiment (CPU, read-only). Comparisons follow the preregistration file prereg_round2.md.
Unit of analysis: 5000 cluster bootstrap resamples over base initial states. Metrics: chain success rate, mean completed tasks, steps, utility (completed/5 - 0.5*steps/1800).
First target handoff: all arms are in exactly the same state up to this point; summarised by task and by task family (original 5 tasks / extended with inverse skill / extended without inverse skill)."""
import json, glob, collections, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "locked_config.json").read_text())
ARMS = cfg["arms"]
LAM = 0.5
OLD5 = {"lift_red_block_slider", "push_into_drawer", "push_red_block_right", "push_blue_block_right", "rotate_pink_block_right"}
NEW_INV = {"push_pink_block_right", "rotate_red_block_left"}
fam = lambda t: "old5" if t in OLD5 else ("new_inverse" if t in NEW_INV else "new_noninverse")
res = [json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE / "results/cl2_*.json")))]
paired_ok = sum(r["paired_initial_state_identical"] for r in res)
rows = []
for r in res:
    row = dict(base=r["job"]["base"], id=r["job"]["id"], stratum=r["job"]["stratum"])
    for a in ARMS:
        v = r["arms"][a]
        row[a] = dict(success=int(v["success"]), completed=v["completed"], steps=v["steps"],
                      U=v["completed"] / 5 - LAM * v["steps"] / 1800,
                      repairs=sum(1 for t in v["tasks"] if t.get("repair")),
                      retries=sum(max(0, len(t["attempts"]) - 1) for t in v["tasks"]))
        first = next((t for t in v["tasks"] if t["target"]), None)
        if first is not None:
            st = sum(x["steps"] for x in first["attempts"]) + (first.get("repair_steps") or 0)
            row[a]["first_target"] = dict(task=first["task"], success=int(first["success"]), steps=st,
                                          first_try=int(bool(first["attempts"]) and first["attempts"][0]["success"]),
                                          repair=first.get("repair"))
        firstnew = next((t for t in v["tasks"] if t["target"] and t["task"] not in OLD5), None)
        row[a]["first_new"] = None if firstnew is None else dict(task=firstnew["task"], pos=firstnew["pos"], success=int(firstnew["success"]),
                                                                 first_try=int(bool(firstnew["attempts"]) and firstnew["attempts"][0]["success"]),
                                                                 repair=firstnew.get("repair"))
    rows.append(row)


def boot(rs, fn, n=5000, seed=17):
    bases = sorted({r["base"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["base"]].append(r)
    rng = np.random.RandomState(seed)
    vals = [fn([x for b in rng.choice(bases, len(bases)) for x in by[b]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


fmt = lambda d: f"{d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]"
rep = dict(chains=len(rows), bases=len({r["base"] for r in rows}), paired_initial_state_identical=f"{paired_ok}/{len(res)}")
print(json.dumps(rep))
print(f"\n{'arm':16s} {'chain succ':>10s} {'avg len':>8s} {'steps':>7s} {'utility':>8s} {'repairs':>8s} {'retries':>8s}")
rep["summary"] = {}
for a in ARMS:
    s = {k: float(np.mean([r[a][k] for r in rows])) for k in ("success", "completed", "steps", "U", "repairs", "retries")}
    rep["summary"][a] = s
    print(f"{a:16s} {s['success']:10.3f} {s['completed']:8.3f} {s['steps']:7.0f} {s['U']:8.3f} {s['repairs']:8.2f} {s['retries']:8.2f}")
comps = [("benefit", "prior"), ("benefit", "inverse"), ("benefit", "retry"), ("benefit", "none"), ("benefit", "home_all"),
         ("noloc", "benefit"), ("benefit_noretry", "none"), ("benefit_noretry", "retry"), ("prior", "retry"), ("inverse", "retry"),
         ("home_all", "retry"), ("retry", "none"), ("benefit", "benefit_alt"), ("benefit_alt", "prior")]
rep["paired"] = {}
print("\npaired differences (A − B), chain level, 95% CI (base cluster bootstrap):")
for A, Bm in comps:
    d = {k: boot(rows, lambda rs, k=k: float(np.mean([x[A][k] - x[Bm][k] for x in rs]))) for k in ("success", "completed", "steps", "U")}
    rep["paired"][f"{A}-{Bm}"] = d
    print(f"  {A:16s}-{Bm:9s} " + "  ".join(f"{k}={fmt(v)}" for k, v in d.items()))
print("\nby stratum (chain success):")
rep["by_stratum"] = {}
for s in sorted({r["stratum"] for r in rows}):
    rs = [r for r in rows if r["stratum"] == s]
    line = {a: float(np.mean([r[a]["success"] for r in rs])) for a in ARMS}
    d1 = boot(rs, lambda x: float(np.mean([y["benefit"]["success"] - y["prior"]["success"] for y in x])), n=2000)
    d2 = boot(rs, lambda x: float(np.mean([y["benefit"]["success"] - y["inverse"]["success"] for y in x])), n=2000)
    rep["by_stratum"][s] = dict(n=len(rs), **line, benefit_minus_prior=d1, benefit_minus_inverse=d2)
    print(f"  {s:26s} n={len(rs):3d} " + " ".join(f"{a}={line[a]:.3f}" for a in ARMS) + f" | b-prior={fmt(d1)} b-inv={fmt(d2)}")


def handoff_block(key, label):
    ft = [r for r in rows if all(r[a].get(key) for a in ARMS)]
    if key == "first_new":
        # Count only chains where all arms reach the same first extended-task handoff (same task, same position); states before it may differ across arms, so this is descriptive
        ft = [r for r in ft if len({(r[a]["first_new"]["task"], r[a]["first_new"]["pos"]) for a in ARMS}) == 1]
    out = dict(n=len(ft))
    print(f"\n{label}, n={len(ft)}:")
    for a in ARMS:
        out[a] = float(np.mean([r[a][key]["success"] for r in ft]))
    print("  all: " + " ".join(f"{a}={out[a]:.3f}" for a in ARMS))
    for grp_name, grp in (("family", lambda r: fam(r["none"][key]["task"])), ("task", lambda r: r["none"][key]["task"])):
        out[grp_name] = {}
        for g in sorted({grp(r) for r in ft}):
            rs = [r for r in ft if grp(r) == g]
            line = {a: float(np.mean([r[a][key]["success"] for r in rs])) for a in ARMS}
            ds = {f"benefit-{b}": boot(rs, lambda x, b=b: float(np.mean([y["benefit"][key]["success"] - y[b][key]["success"] for y in x])), n=2000)
                  for b in ("prior", "inverse", "none", "home_all")}
            ds["noloc-benefit"] = boot(rs, lambda x: float(np.mean([y["noloc"][key]["success"] - y["benefit"][key]["success"] for y in x])), n=2000)
            out[grp_name][g] = dict(n=len(rs), **line, **ds)
            print(f"  {g:26s} n={len(rs):3d} " + " ".join(f"{a}={line[a]:.3f}" for a in ARMS)
                  + " | " + " ".join(f"{k}={fmt(v)}" for k, v in ds.items()))
    choice = collections.defaultdict(collections.Counter)
    for r in ft:
        for a in ("benefit", "prior", "noloc"):
            choice[(r["none"][key]["task"], a)][r[a][key]["repair"]] += 1
    out["choices"] = {f"{t}|{a}": c.most_common(4) for (t, a), c in choice.items()}
    print("  most frequent repairs (task | arm):")
    for (t, a), c in sorted(choice.items()):
        print(f"    {t:26s} {a:8s} {c.most_common(3)}")
    return out


rep["first_target"] = handoff_block("first_target", "first target handoff (identical state across arms)")
rep["first_new"] = handoff_block("first_new", "first new-target handoff (same task and position across arms; descriptive)")
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / "cl2_report.json").write_text(json.dumps(rep, indent=1, default=str))
(HERE / "analysis" / "cl2_rows.json").write_text(json.dumps(rows))
print("\nsaved", HERE / "analysis" / "cl2_report.json")
