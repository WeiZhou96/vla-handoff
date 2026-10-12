# -*- coding: utf-8 -*-
"""Run after phase 1 of the extended tasks has finished (CPU, reads results only):
1) Three-way variance decomposition over 11 target tasks (J8 of the two development tasks, multi-task J8, extended-task J8).
2) Vocabulary screening plan for the extended tasks (jobs_nt2_screen.json), two groups per new task:
   low     handoffs with at most 4 successes out of 8, at most 25 chosen by a fixed hash;
   natural 25 of the remaining handoffs chosen by a fixed hash (approximately the natural distribution, regardless of outcome).
   Selection depends only on the fixed hash and the J8 outcomes; the "no repair" branch of the labelling stage is re-executed with new seeds, so the paired comparison is unaffected by selection.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import json, hashlib, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))
PER_GROUP = 25

recs = []
for g in sorted(Path(f"{RUN_ROOT}/fix_20260928/runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            recs.append(dict(task=r["task_b"], k=r["k"], n=r["n"], wins=[int(x["success"]) for x in r["repeats"]], src="dev2"))
new, reach, total, restore_fail, audits = [], collections.Counter(), collections.Counter(), 0, []
for run, src in ((MT, "mt"), (HERE, "nt")):
    for f in sorted((run / "results").glob("j8_*.json")):
        d = json.loads(f.read_text())
        t = d["job"]["target"]
        if src == "nt":
            total[t] += 1
        if "restore_failed" in d:
            restore_fail += src == "nt"; continue
        if "determinism_audit" in d and src == "nt":
            audits.append(d["determinism_audit"])
        if d.get("reached") and d.get("n") == 8:
            r = dict(task=t, k=d["k"], n=d["n"], wins=[int(x["success"]) for x in d["repeats"]], src=src,
                     sample_id=d["sample_id"], parent=d["parent_id"], obs=d["observation"], job=d["job"])
            recs.append(r)
            if src == "nt":
                reach[t] += 1; new.append(r)


def decompose(rs):
    K = np.array([r["k"] for r in rs], float); N = np.array([r["n"] for r in rs], float)
    T = np.array([r["task"] for r in rs])
    pbar = K.sum() / N.sum(); tot = pbar * (1 - pbar)
    within = np.sum(K * (N - K) / (N - 1)) / N.sum()
    tm = {t: K[T == t].sum() / N[T == t].sum() for t in set(T)}
    between = np.sum(N * (np.array([tm[t] for t in T]) - pbar) ** 2) / N.sum()
    tot_pairs = conc = 0.0
    for t in set(T):
        pos, neg = [], []
        for r in (x for x in rs if x["task"] == t):
            w = r["wins"]
            for j, y in enumerate(w):
                s = (sum(w) - y) / (len(w) - 1)
                (pos if y else neg).append(s)
        pos, neg = np.array(pos), np.array(neg)
        if len(pos) and len(neg):
            conc += (pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum(); tot_pairs += len(pos) * len(neg)
    p = K / N
    return dict(n=len(rs), success=float(pbar), task_share=float(between / tot) if tot else None,
                state_share=float((tot - within - between) / tot) if tot else None, noise_share=float(within / tot) if tot else None,
                oracle_within_auroc=float(conc / tot_pairs) if tot_pairs else None,
                mixed=float(np.mean((p > 0) & (p < 1))), all_fail=float(np.mean(p == 0)), all_success=float(np.mean(p == 1)))


def boot_shares(rs, n=1000, seed=5):
    """Confidence interval of the state share, bootstrapped over handoffs."""
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        s = [rs[i] for i in rng.randint(0, len(rs), len(rs))]
        d = decompose(s); v.append(d["state_share"] if d["state_share"] is not None else np.nan)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


out = dict(collection=dict(chains_by_target=dict(total), reached_by_target=dict(reach), restore_failed=restore_fail,
                           determinism_audits_passed=f"{sum(audits)}/{len(audits)}"),
           pooled_all=decompose(recs), pooled_new6=decompose(new) if new else None,
           by_task={t: dict(**decompose([r for r in recs if r["task"] == t]), state_share_ci=boot_shares([r for r in recs if r["task"] == t]))
                    for t in sorted({r["task"] for r in recs})})
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / "decomposition_11tasks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["collection"]))
print(f"{'task':26s} {'n':>4s} {'succ':>6s} {'task%':>6s} {'state%':>7s} {'noise%':>7s} {'oracleAUC':>9s} {'mixed':>6s} {'allfail':>7s}")
for name, d in [("POOLED (11 tasks)", out["pooled_all"]), ("POOLED (6 new)", out["pooled_new6"])] + list(out["by_task"].items()):
    if not d:
        continue
    f = lambda v: "  n/a" if v is None else f"{100*v:6.1f}"
    print(f"{name:26s} {d['n']:4d} {d['success']:6.3f} {f(d['task_share'])} {f(d['state_share']):>7s} {f(d['noise_share']):>7s} "
          f"{(d['oracle_within_auroc'] or float('nan')):9.3f} {d['mixed']:6.3f} {d['all_fail']:7.3f}")

# ---------- screening plan ----------
jobs = []
for t in sorted({r["task"] for r in new}):
    rs = [r for r in new if r["task"] == t]
    low = sorted([r for r in rs if r["k"] <= 4], key=lambda r: hashlib.sha256(("nt-low:" + r["sample_id"]).encode()).hexdigest())[:PER_GROUP]
    used = {r["sample_id"] for r in low}
    natural = sorted([r for r in rs if r["sample_id"] not in used],
                     key=lambda r: hashlib.sha256(("nt-nat:" + r["sample_id"]).encode()).hexdigest())[:PER_GROUP]
    for grp, pool in (("low", low), ("natural", natural)):
        for r in pool:
            jobs.append(dict(type="screen", id="screen_" + hashlib.sha256(r["sample_id"].encode()).hexdigest()[:24], group=grp,
                             sample_id=r["sample_id"], index=r["job"]["index"], sequence_id=r["sample_id"].split(":")[0],
                             task_b=t, npz=str(HERE / "boundaries" / r["obs"]), parent_id=r["parent"], k_j8=r["k"],
                             candidates=[v for v in VOCAB if v != t] + ["retreat", "home"]))
# interleave tasks
by_t = collections.defaultdict(list)
for j in jobs:
    by_t[j["task_b"]].append(j)
inter = []
for i in range(max(map(len, by_t.values())) if by_t else 0):
    for t in sorted(by_t):
        if i < len(by_t[t]):
            inter.append(by_t[t][i])
(HERE / "jobs_nt2_screen.json").write_text(json.dumps(dict(
    purpose="new-target vocabulary screening: per task 25 low (k<=4) + 25 natural (hash order)", jobs=inter), indent=1))
print("screen jobs:", collections.Counter((j["task_b"], j["group"]) for j in inter))
