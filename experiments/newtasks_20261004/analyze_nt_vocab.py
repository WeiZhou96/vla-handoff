# -*- coding: utf-8 -*-
"""Analysis of the vocabulary-repair labels for the extended tasks (CPU, read-only).
Label branches per handoff: none, the 4 candidates with the highest post-repair verifier score (first choice = highest score), 2 random candidates by a fixed hash, and the rule candidate (inverse skill or home).
Metrics: no-repair success rate, first choice, mean of random candidates, rule candidate, post-hoc best (split-half unbiased: choose on the first 4 runs, evaluate on the last 4, swap and average).
95% confidence intervals of paired differences by cluster bootstrap over initial configurations (2000 resamples); summarised per task and per group (low / natural)."""
import json, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
jobs = {j["sample_id"]: j for j in json.loads((HERE / "jobs_nt3_label.json").read_text())["jobs"]}
sel = json.loads((HERE / "analysis/jobs_nt3_label_selection.json").read_text())
rows = []
for sid, job in jobs.items():
    p = HERE / "results" / f"{job['id']}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    arms = {a["candidate"]: a for a in d["arms"] if "invalid" not in a and a.get("n") == 8}
    if "none" not in arms:
        continue
    pr = {c: a["k"] / 8 for c, a in arms.items()}
    wins = {c: np.array([int(x["success"]) for x in a["rollouts"]]) for c, a in arms.items() if c != "none"}
    s = sel[sid]
    top1 = s["top"][0][0]
    rnd = [c for c, _ in s["random"] if c in pr]
    halves = []
    for a, b in ((slice(0, 4), slice(4, 8)), (slice(4, 8), slice(0, 4))):
        bst = max(wins, key=lambda c: wins[c][a].mean()); halves.append(wins[bst][b].mean())
    rows.append(dict(task=job["task_b"], group=job["group"], parent=job["parent_id"], none=pr["none"],
                     top1=pr.get(top1, np.nan), rand=float(np.mean([pr[c] for c in rnd])) if rnd else np.nan,
                     rule=pr.get(job["rule_candidate"], np.nan), oracle=float(np.mean(halves)), top1_name=top1, rule_name=job["rule_candidate"],
                     invalid=sum("invalid" in a for a in d["arms"])))


def boot(rs, fn, n=2000, seed=9):
    ps = sorted({r["parent"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["parent"]].append(r)
    rng = np.random.RandomState(seed)
    v = [fn([x for q in rng.choice(ps, len(ps)) for x in by[q]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


fmt = lambda d: f"{d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]"
out = dict(n=len(rows), invalid_arms=sum(r["invalid"] for r in rows), by={})
print(f"states {len(rows)}, invalid arms {out['invalid_arms']}")
print(f"{'scope':38s} {'n':>3s} {'none':>6s} {'top1':>6s} {'rand':>6s} {'rule':>6s} {'orac½':>6s} | top1-none  rule-none  top1-rule")
scopes = [(t, g) for t in sorted({r["task"] for r in rows}) for g in ("low", "natural")] + [(t, None) for t in sorted({r["task"] for r in rows})] + [("ALL", "low"), ("ALL", "natural"), ("ALL", None)]
for t, g in scopes:
    rs = [r for r in rows if (t == "ALL" or r["task"] == t) and (g is None or r["group"] == g)]
    if not rs:
        continue
    m = {k: float(np.nanmean([r[k] for r in rs])) for k in ("none", "top1", "rand", "rule", "oracle")}
    d1 = boot(rs, lambda x: float(np.mean([r["top1"] - r["none"] for r in x])))
    d2 = boot(rs, lambda x: float(np.mean([r["rule"] - r["none"] for r in x if not np.isnan(r["rule"])])))
    d3 = boot(rs, lambda x: float(np.mean([r["top1"] - r["rule"] for r in x if not np.isnan(r["rule"])])))
    key = f"{t}|{g or 'all'}"
    out["by"][key] = dict(n=len(rs), **m, top1_minus_none=d1, rule_minus_none=d2, top1_minus_rule=d3,
                          top1_choices=collections.Counter(r["top1_name"] for r in rs).most_common(3))
    print(f"{key:38s} {len(rs):3d} {m['none']:6.3f} {m['top1']:6.3f} {m['rand']:6.3f} {m['rule']:6.3f} {m['oracle']:6.3f} | {fmt(d1)} {fmt(d2)} {fmt(d3)}")
(HERE / "analysis" / "vocab_repair_newtasks.json").write_text(json.dumps(out, indent=1, default=float))
print("saved", HERE / "analysis" / "vocab_repair_newtasks.json")
