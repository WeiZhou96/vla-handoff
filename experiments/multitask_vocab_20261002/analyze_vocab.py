# -*- coding: utf-8 -*-
"""Final analysis of the instruction-vocabulary repairs (CPU, reads results only).

Each labelled handoff has: none (no repair, re-executed 8 times with paired seeds), the top 4 candidates by post-repair score, and 2 candidates drawn at random by a fixed hash.
Report (per task; for the two original tasks the states are low-scoring under the discriminator, for the three new tasks they are states with J8 success at most 4/8):
  no repair p_none; the discriminator's first choice p_top1 and its paired gain over p_none; mean of random candidates p_rand;
  the post-hoc best among the labelled branches (split-half unbiased: pick on the first 4 runs, evaluate on the last 4, and vice versa, averaged);
  pairwise ranking agreement between score and observed success rate across the branches of the same handoff;
  the most frequent repair instructions chosen first.
Confidence intervals: cluster bootstrap over initial configurations, 2000 resamples.
G3 criterion (fixed before the analysis): on at least 2 tasks, vocabulary repair raises the success rate of low-scoring states by more than 15 percentage points.
"""
import json, collections, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

sel = {}
for f in (HERE / "analysis").glob("*_selection.json"):
    sel.update(json.loads(f.read_text()))
rows = []
for f in sorted((HERE / "results").glob("label_*.json")):
    d = json.loads(f.read_text())
    s = sel.get(d["sample_id"])
    if s is None:
        continue
    arms = {a["candidate"]: a for a in d["arms"] if "invalid" not in a and a.get("n") == 8}
    if "none" not in arms or len(arms) < 3:
        continue
    top = [c for c, _ in s["top"] if c in arms]
    rnd = [c for c, _ in s["random"] if c in arms]
    wins = {c: np.array([int(r["success"]) for r in a["rollouts"]]) for c, a in arms.items()}
    p = {c: w.mean() for c, w in wins.items()}
    halves = []
    for a, b in ((slice(0, 4), slice(4, 8)), (slice(4, 8), slice(0, 4))):
        best = max(wins, key=lambda c: wins[c][a].mean())
        halves.append(wins[best][b].mean())
    pairs = []
    cands = [c for c in arms if c != "none"]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if p[a] != p[b]:
                ds = s["all_scores"][a] - s["all_scores"][b]
                pairs.append(1.0 if np.sign(ds) == np.sign(p[a] - p[b]) else (0.5 if ds == 0 else 0.0))
    job = json.loads((HERE / "results" / f.name).read_text())
    rows.append(dict(sample_id=d["sample_id"], task=d["task_b"], parent=None, p_none=p["none"],
                     p_top1=p[top[0]] if top else np.nan, top1=top[0] if top else None,
                     p_rand=float(np.mean([p[c] for c in rnd])) if rnd else np.nan,
                     oracle_split=float(np.mean(halves)), oracle=max(p.values()), conc=pairs))
# initial configurations: taken from the label jobs files
parent = {}
for jf in HERE.glob("jobs_phase*_label*.json"):
    for j in json.loads(jf.read_text())["jobs"]:
        parent[j["sample_id"]] = j["parent_id"]
for r in rows:
    r["parent"] = parent.get(r["sample_id"], r["sample_id"])


def boot(rs, fn, reps=2000, seed=11):
    ps = sorted({r["parent"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["parent"]].append(r)
    rng = np.random.RandomState(seed)
    vals = [fn([x for q in rng.choice(ps, len(ps)) for x in by[q]]) for _ in range(reps)]
    return fn(rs), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


report = {}
groups = sorted({r["task"] for r in rows})
print(f"{'task':26s} {'n':>3s} {'p_none':>7s} {'p_top1':>7s} {'gain_top1 [95% CI]':>24s} {'p_rand':>7s} {'oracle½':>8s} {'conc':>6s}  top1 most common")
for t in groups + ["ALL"]:
    rs = [r for r in rows if t == "ALL" or r["task"] == t]
    if not rs:
        continue
    g = boot(rs, lambda x: float(np.nanmean([r["p_top1"] - r["p_none"] for r in x])))
    gr = boot(rs, lambda x: float(np.nanmean([r["p_rand"] - r["p_none"] for r in x])))
    conc = [c for r in rs for c in r["conc"]]
    rec = dict(n=len(rs), p_none=float(np.mean([r["p_none"] for r in rs])), p_top1=float(np.nanmean([r["p_top1"] for r in rs])),
               gain_top1=g, p_rand=float(np.nanmean([r["p_rand"] for r in rs])), gain_rand=gr,
               oracle_split=float(np.mean([r["oracle_split"] for r in rs])), concordance=float(np.mean(conc)) if conc else None,
               concordance_pairs=len(conc), top1_counts=dict(collections.Counter(r["top1"] for r in rs).most_common(5)))
    report[t] = rec
    print(f"{t:26s} {rec['n']:3d} {rec['p_none']:7.3f} {rec['p_top1']:7.3f} {g[0]:+8.3f} [{g[1]:+.3f}, {g[2]:+.3f}] {rec['p_rand']:7.3f} "
          f"{rec['oracle_split']:8.3f} {(rec['concordance'] or float('nan')):6.3f}  {list(rec['top1_counts'].items())[:3]}")
passed = [t for t in groups if report[t]["gain_top1"][0] >= 0.15 and report[t]["gain_top1"][1] > 0]
report["G3"] = dict(criterion="at least 2 tasks with top-1 vocabulary repair gain >= 15pp and CI lower bound > 0",
                    tasks_passing=passed, passed=len(passed) >= 2)
print("G3:", report["G3"])
(HERE / "analysis" / "vocab_repair_report.json").write_text(json.dumps(report, indent=1, default=float))
