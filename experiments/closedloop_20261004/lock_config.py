# -*- coding: utf-8 -*-
"""Freeze the decision configuration of the closed-loop confirmation experiment (run before any confirmation data exist).

Method: observation at the handoff (before repair) -> state features v2 -> M7 predicts success q for "no repair" and for each candidate repair.
  Candidates: CALVIN instructions (excluding the target task itself) + retreat + home.
  Selection: u* = argmax_u q(u).
  Benefit gate (proposed method): execute u* if q(u*) - q(none) - lambda*c_bar(task,u*)/360 > delta.
  Risk gate (control): execute u* if q(none) < tau.
  Fixed per-task repair (control): the single choice with the highest mean utility for that task on development data (may be "no repair").
lambda is fixed at 0.5 (a representative point of the difference between the two gates in the development analysis). delta and tau are chosen by mean utility on development handoffs (gate + hand, cross-fitted scores).
c_bar(task,u): mean execution steps of that repair for that task in the vocabulary screening (for scripted repairs, the mean measured steps).
Selective retry (full method): after a target-task failure, score the failure observation with M2v2 and retry only if >= tau_r; tau_r = 0.30 fixed a priori.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import json, collections, glob
from pathlib import Path
import numpy as np
B = Path(f"{RUN_ROOT}/benefit_model_20261003")
OUT = Path(__file__).resolve().parent
LAM = 0.5
rows = json.loads((B / "analysis/v2_rows_with_gating.json").read_text())

# mean repair steps
steps = collections.defaultdict(list)
for d in (f"{RUN_ROOT}/multitask_vocab_20261002", f"{RUN_ROOT}/gating_vocab_20261003"):
    for f in glob.glob(d + "/results/screen_*.json"):
        r = json.loads(open(f).read())
        for c in r["candidates"]:
            if "invalid" not in c and c.get("steps") is not None:
                steps[(r["task_b"], c["candidate"])].append(c["steps"])
cbar = {f"{t}|{c}": float(np.mean(v)) for (t, c), v in steps.items()}
cbar_global = {}
for (t, c), v in steps.items():
    cbar_global.setdefault(c, []).extend(v)
cbar_global = {c: float(np.mean(v)) for c, v in cbar_global.items()}

dev = [r for r in rows if r["src"] in ("gate", "hand") and all(v is not None for v in r["rsteps"].values())]
bn = collections.defaultdict(list)
for r in rows:
    if r.get("bsteps_none") is not None:
        bn[r["task"]].append(r["bsteps_none"])
for r in dev:
    if r.get("bsteps_none") is None:
        r["bsteps_none"] = float(np.mean(bn[r["task"]]))


def util(rs, rule, m="M7"):
    v = []
    for r in rs:
        sc = r["scores"][m]; u = max(sc, key=sc.get); q0 = r["none_scores"][m]
        rep = rule(r, u, sc[u], q0)
        s_, st = (r["p"][u], r["rsteps"][u] + r["bsteps"][u]) if rep else (r["p0"], r["bsteps_none"])
        v.append(s_ - LAM * st / 360)
    return float(np.mean(v))


TAU = [-np.inf] + list(np.linspace(0, 1, 41)) + [np.inf]
DEL = [-np.inf] + list(np.round(np.arange(-0.4, 0.61, 0.025), 3)) + [np.inf]
cb = lambda t, u: cbar.get(f"{t}|{u}", cbar_global.get(u, 100.0))
tau = max(TAU, key=lambda x: util(dev, lambda r, u, qu, q0: q0 < x))
delta = max(DEL, key=lambda d: util(dev, lambda r, u, qu, q0: qu - q0 - LAM * cb(r["task"], u) / 360 > d))
# Fixed per-task policy: on development gate handoffs, pick per task the single choice with the highest mean utility (including no repair); only repairs labelled for that task are considered
fixed = {}
for t in sorted({r["task"] for r in dev}):
    rs = [r for r in dev if r["task"] == t]
    opts = collections.defaultdict(list)
    for r in rs:
        opts["none"].append(r["p0"] - LAM * r["bsteps_none"] / 360)
        for u, p in r["p"].items():
            opts[u].append(p - LAM * (r["rsteps"][u] + r["bsteps"][u]) / 360)
    cand = {u: (np.mean(v), len(v)) for u, v in opts.items() if len(v) >= 5}
    fixed[t] = max(cand, key=lambda u: cand[u][0])
cfg = dict(created="locked before any confirmation data", lam=LAM, tau_risk=float(tau), delta_benefit=float(delta),
           tau_retry=0.30, model_file=str(B / "models/benefit_models_v2.joblib"), scorer="M7", verifier="M2v2",
           fixed_repair_per_task=fixed, cbar=cbar, cbar_global=cbar_global,
           dev_utility=dict(risk=util(dev, lambda r, u, qu, q0: q0 < tau),
                            benefit=util(dev, lambda r, u, qu, q0: qu - q0 - LAM * cb(r["task"], u) / 360 > delta),
                            never=util(dev, lambda r, u, qu, q0: False), always=util(dev, lambda r, u, qu, q0: True)))
(OUT / "locked_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False, default=float))
print(json.dumps({k: v for k, v in cfg.items() if k not in ("cbar", "cbar_global")}, indent=1, default=float))
