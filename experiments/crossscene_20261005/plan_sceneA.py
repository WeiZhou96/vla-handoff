# -*- coding: utf-8 -*-
"""Fixed plan and configuration for the cross-scene test (scene A), written before running.
Scene A has the same furniture layout as D but a different appearance (table texture and colour, block sizes); pi0.5 (ABC->D) saw scene A during training,
while the scoring models were trained only on scene-D development data. Tests how the round-1 frozen method transfers unchanged to a scene with new appearance.
Chains: the 391 chains of the round-1 closed loop, with perturbation seeds redrawn (different from round 1).
Arms: none, retry, inverse, prior, benefit (round-1 frozen M7), benefit_m5 (M5v2 trained only on scene-D development data).
Configuration: round-1 frozen values (lambda, delta, tau, c-bar, prior table) unchanged; the model file of benefit_m5 is given by aux_model_file."""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
CL1 = Path(f"{RUN_ROOT}/closedloop_20261004")
h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
jobs = []
for j in json.loads((CL1 / "jobs_cl.json").read_text())["jobs"]:
    jobs.append(dict(j, id="scA_" + j["id"][3:], perturb_seed=int(h("scA-perturb", j["index"])[:8], 16)))
(HERE / "jobs_scA.json").write_text(json.dumps(dict(purpose="cross-scene (calvin_scene_A) closed loop, round-1 frozen method", jobs=jobs), indent=1))
cfg = json.loads((CL1 / "locked_config.json").read_text())
cfg.update(json.loads((CL1 / "addon_extra2.json").read_text()))
cfg.update(scene="calvin_scene_A", arms=["none", "retry", "inverse", "prior", "benefit", "benefit_m5"],
           note_scene="round-1 frozen configuration transferred unchanged to scene A; benefit_m5 uses M5v2 trained only on scene-D development data.")
(HERE / "locked_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
print("jobs", len(jobs))
