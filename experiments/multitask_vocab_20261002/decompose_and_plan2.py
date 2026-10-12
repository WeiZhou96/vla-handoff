# -*- coding: utf-8 -*-
"""Run after phase 1 has finished (CPU, reads results only):
1) Multi-task variance decomposition (G2): development J8 of the two tasks (304 handoffs) + J8 of the new tasks; reports, per task and pooled,
   the variance shares of between-task / within-task across-state / within-state stochasticity, the within-task AUROC of the state ceiling (mean of the other 7 runs as score),
   the fraction of handoffs with both successes and failures, the all-fail fraction, and the mean success rate.
2) Vocabulary screening plan for the new tasks (phase2 screen): handoffs of the new tasks with at most 4 successes out of 8, at most 40 per task chosen by a fixed hash.
   Selection uses the earlier execution outcomes, so the "no repair" branch of the phase-2 label jobs is re-executed with new seeds; the paired comparison is unaffected by selection.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import json, hashlib, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))

recs = []
for g in sorted(Path(f"{RUN_ROOT}/fix_20260928/runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            recs.append(dict(task=r["task_b"], k=r["k"], n=r["n"], wins=[int(x["success"]) for x in r["repeats"]], src="dev2"))
new = []
reach = collections.Counter(); total = collections.Counter(); restore_fail = 0; audits = []
for f in sorted((HERE / "results").glob("j8_*.json")):
    d = json.loads(f.read_text())
    t = d["job"]["target"]; total[t] += 1
    if "restore_failed" in d:
        restore_fail += 1; continue
    if "determinism_audit" in d:
        audits.append(d["determinism_audit"])
    if d.get("reached") and d.get("n") == 8:
        reach[t] += 1
        r = dict(task=t, k=d["k"], n=d["n"], wins=[int(x["success"]) for x in d["repeats"]], src="new",
                 sample_id=d["sample_id"], parent=d["parent_id"], obs=d["observation"], job=d["job"])
        recs.append(r); new.append(r)


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


out = dict(collection=dict(chains_by_target=dict(total), reached_by_target=dict(reach), restore_failed=restore_fail,
                           determinism_audits_passed=f"{sum(audits)}/{len(audits)}"),
           pooled_all=decompose(recs), pooled_new=decompose(new) if new else None,
           by_task={t: decompose([r for r in recs if r["task"] == t]) for t in sorted({r["task"] for r in recs})})
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / "decomposition.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["collection"]))
print(f"{'task':26s} {'n':>4s} {'succ':>6s} {'task%':>6s} {'state%':>7s} {'noise%':>7s} {'oracleAUC':>9s} {'mixed':>6s} {'allfail':>7s}")
for name, d in [("POOLED (5 tasks)", out["pooled_all"]), ("POOLED (3 new)", out["pooled_new"])] + list(out["by_task"].items()):
    if not d:
        continue
    f = lambda v: "  n/a" if v is None else f"{100*v:6.1f}"
    print(f"{name:26s} {d['n']:4d} {d['success']:6.3f} {f(d['task_share'])} {f(d['state_share']):>7s} {f(d['noise_share']):>7s} "
          f"{(d['oracle_within_auroc'] or float('nan')):9.3f} {d['mixed']:6.3f} {d['all_fail']:7.3f}")

# ---------- phase2 screen plan (new tasks) ----------
jobs = []
for t in sorted({r["task"] for r in new}):
    pool = sorted([r for r in new if r["task"] == t and r["k"] <= 4],
                  key=lambda r: hashlib.sha256(("mt-low:" + r["sample_id"]).encode()).hexdigest())[:40]
    for r in pool:
        jobs.append(dict(type="screen", id="screen_" + hashlib.sha256(r["sample_id"].encode()).hexdigest()[:24],
                         sample_id=r["sample_id"], index=r["job"]["index"], sequence_id=r["sample_id"].split(":")[0],
                         task_b=t, npz=str(HERE / "boundaries" / r["obs"]), parent_id=r["parent"], k_j8=r["k"],
                         candidates=[v for v in VOCAB if v != t] + ["retreat", "home"]))
(HERE / "jobs_phase2_screen_new.json").write_text(json.dumps(dict(
    purpose="phase2 vocabulary screening on new-task states with J8 k<=4 (max 40 per task, hash order)", jobs=jobs), indent=1))
print("phase2 new-task screen jobs:", collections.Counter(j["task_b"] for j in jobs))
