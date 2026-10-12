# -*- coding: utf-8 -*-
"""Fixed plan of the closed-loop confirmation experiment: from the 192 base initial states in the candidate pool, take 100 by hash and 4 chains each (at least one target task at positions 1-4),
with an independent perturbation seed per chain. The selection uses no execution results."""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import json, hashlib, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
TARGETS = {"lift_red_block_slider", "push_blue_block_right", "push_into_drawer", "push_red_block_right", "rotate_pink_block_right"}
h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
cat = json.load(open(CATALOG))
by = collections.defaultdict(list)
for i, (ini, tasks) in enumerate(cat):
    if any(t in TARGETS for t in tasks[1:5]):
        by[json.dumps(ini, sort_keys=True)].append(i)
bases = sorted(by, key=lambda k: h("cl-base", k))[:100]
jobs = []
for k in bases:
    for i in sorted(by[k], key=lambda i: h("cl-chain", i))[:4]:
        jobs.append(dict(id=f"cl_{i}", index=i, base=h("base", k)[:16], tasks=cat[i][1],
                         perturb_seed=int(h("cl-perturb", i)[:8], 16)))
(HERE / "jobs_cl.json").write_text(json.dumps(dict(purpose="closed-loop confirmation on perturbed initial states", jobs=jobs), indent=1))
cnt = collections.Counter(t for j in jobs for t in j["tasks"][1:5] if t in TARGETS)
print("jobs", len(jobs), "bases", len(bases), "target handoffs (planned positions 1-4):", dict(cnt))
