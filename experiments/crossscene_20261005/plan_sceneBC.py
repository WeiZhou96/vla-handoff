# -*- coding: utf-8 -*-
"""Fixed plan and configuration for the cross-scene test (scenes B and C: changed furniture layout), written before running.
In scenes B and C the sliding cabinet moves to the right side of the table, and the button, switch and drawer positions also change; initial-configuration coordinates are mapped by scene_map.py (semantics unchanged).
Chains: the chains of the first 50 base initial configurations in the round-1 closed-loop plan (at most 4 each, about 200), with independent perturbation seeds per scene.
Arms and configuration as in scene A: none, retry, inverse, prior, benefit (round-1 frozen M7), benefit_m5 (M5v2 trained on scene-D development data)."""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import json, hashlib, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CL1 = Path(f"{RUN_ROOT}/closedloop_20261004")
h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
src = json.loads((CL1 / "jobs_cl.json").read_text())["jobs"]
bases = []
for j in src:
    if j["base"] not in bases:
        bases.append(j["base"])
keep = set(bases[:50])
for sc, tag in (("calvin_scene_B", "scB"), ("calvin_scene_C", "scC")):
    jobs = [dict(j, id=f"{tag}_" + j["id"][3:], perturb_seed=int(h(tag + "-perturb", j["index"])[:8], 16)) for j in src if j["base"] in keep]
    (HERE / f"jobs_{tag}.json").write_text(json.dumps(dict(purpose=f"cross-scene ({sc}) closed loop, round-1 frozen method", scene=sc, jobs=jobs), indent=1))
    print(tag, len(jobs))
