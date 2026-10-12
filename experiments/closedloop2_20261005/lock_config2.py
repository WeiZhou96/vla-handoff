# -*- coding: utf-8 -*-
"""Freeze the decision configuration of the round-2 closed-loop confirmation experiment (extended target tasks); run before any round-2 confirmation data exist.

Relation to round 1: lambda, delta, tau, tau_r are carried over unchanged from the frozen round-1 values (closedloop_20261004/locked_config.json) and are not re-tuned on new data;
the scoring model is replaced by v3 trained on development data of 11 tasks (benefit_models_v3_all.joblib). The scorer is fixed to M7 (with prior values), because:
  supplementary controls on the round-1 closed-loop chains showed M5v2 with a closed-loop success rate 9.2 percentage points below M7 (offline selection compares only among labelled candidates, whereas closed loop must choose among all candidates,
  and M5v2 without prior anchoring extrapolates unreliably to unlabelled candidates). M5v2 is kept as the benefit_alt arm; the no-localiser control arm uses M7n.
  (The previously fixed rule "choose by offline selection success" was replaced by this rule before any round-2 data were produced; the reason and time are recorded in prereg_round2.md.)
c_bar(task,u) includes the screening steps of the extended tasks. Target tasks = all 11 tasks covered by the model.
The inverse-skill rule (inverse arm) is fixed a priori from task semantics and covers 5 push/rotate target tasks; the arm does not repair other tasks.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import json, collections, glob, time, sys
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")
from pathlib import Path
import numpy as np
import joblib
OUT = Path(__file__).resolve().parent
B = Path(f"{RUN_ROOT}/benefit_model_20261003")
v1 = json.loads(Path(f"{RUN_ROOT}/closedloop_20261004/locked_config.json").read_text())
model_file = B / "models/benefit_models_v3_all.joblib"
m = joblib.load(model_file)
steps = collections.defaultdict(list)
for d in ("multitask_vocab_20261002", "gating_vocab_20261003", "newtasks_20261004"):
    for f in glob.glob(f"{RUN_ROOT}/{d}/results/screen_*.json"):
        r = json.loads(open(f).read())
        for c in r["candidates"]:
            if "invalid" not in c and c.get("steps") is not None:
                steps[(r["task_b"], c["candidate"])].append(c["steps"])
cbar = {f"{t}|{c}": float(np.mean(v)) for (t, c), v in steps.items()}
glob_ = collections.defaultdict(list)
for (t, c), v in steps.items():
    glob_[c].extend(v)
cbar_global = {c: float(np.mean(v)) for c, v in glob_.items()}
rep = json.loads((B / "analysis/v3_report_all.json").read_text())["selection"]["all|ALL"]
scorer = "M7"
noloc = {"M5v2": "M5n", "M7": "M7n"}[scorer]
alt = {"M5v2": "M7", "M7": "M5v2"}[scorer]
INVERSE = {"push_red_block_right": "push_red_block_left", "push_blue_block_right": "push_blue_block_left",
           "rotate_pink_block_right": "rotate_pink_block_left", "push_pink_block_right": "push_pink_block_left",
           "rotate_red_block_left": "rotate_red_block_right"}
cfg = dict(created=time.strftime("%F %T") + " (before any second-round confirmation data)",
           carried_from_round1=dict(lam=v1["lam"], tau_risk=v1["tau_risk"], delta_benefit=v1["delta_benefit"], tau_retry=v1["tau_retry"]),
           lam=v1["lam"], tau_risk=v1["tau_risk"], delta_benefit=v1["delta_benefit"], tau_retry=v1["tau_retry"],
           model_file=str(model_file), aux_model_file=str(model_file), scorer=scorer, scorer_noloc=noloc, verifier="M2v2",
           scorer_rule=dict(M7=rep["M7"], M5v2=rep["M5v2"], M6=rep["M6"], n=rep["n"]),
           arm_models={"benefit": ["main", scorer], "benefit_noretry": ["main", scorer], "risk": ["main", scorer],
                       "full": ["main", scorer], "noloc": ["main", noloc], "benefit_alt": ["main", alt]},
           targets=list(m["tasks"]), inverse_rule=INVERSE, model_data=m.get("data"),
           arms=["none", "retry", "inverse", "prior", "home_all", "benefit", "benefit_noretry", "noloc", "benefit_alt"],
           cbar=cbar, cbar_global=cbar_global)
(OUT / "locked_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False, default=float))
print(json.dumps({k: v for k, v in cfg.items() if k not in ("cbar", "cbar_global")}, indent=1, ensure_ascii=False, default=float))
