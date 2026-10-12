# -*- coding: utf-8 -*-
"""Extended target tasks, phase-1 plan (fixed before running): J8 collection for 6 new target tasks.

New target tasks were chosen by their first-attempt success rate at non-target handoffs in the none arm of the full-chain closed-loop confirmation experiment (no repair outcomes involved), in two classes:
  no inverse skill available (repair must depend on the state): lift_blue_block_slider, lift_pink_block_slider, lift_pink_block_drawer, stack_block
  inverse skill available (tests the inverse-skill rule and the model on unseen tasks): push_pink_block_right, rotate_red_block_left
For each task, 120 chains are selected from a fixed candidate pool by rotating over initial-configuration hashes, with the target at zero-based position 1 to 4 (first occurrence). Selection depends only on the fixed hash.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import hashlib, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
NEW_TARGETS = ["lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer", "stack_block",
               "push_pink_block_right", "rotate_red_block_left"]
PER_TARGET = 120
h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
cat = json.load(open(CATALOG))
per = {}
for t in NEW_TARGETS:
    by_parent = collections.defaultdict(list)
    for i, (ini, tasks) in enumerate(cat):
        pos = next((p for p in range(1, 5) if tasks[p] == t), None)
        if pos is not None:
            by_parent[json.dumps(ini, sort_keys=True)].append((h("nt-chain", t, i), i, pos))
    parents = sorted(by_parent, key=lambda k: h("nt-parent", t, k))
    for k in parents:
        by_parent[k].sort()
    chosen, rnd = [], 0
    while len(chosen) < PER_TARGET and any(len(by_parent[k]) > rnd for k in parents):
        for k in parents:
            if len(by_parent[k]) > rnd and len(chosen) < PER_TARGET:
                _, i, pos = by_parent[k][rnd]
                chosen.append(dict(type="j8", id=f"j8_{t}_{i}", index=i, target=t, stop_after=pos + 1))
        rnd += 1
    per[t] = chosen
# interleave so that all tasks progress together
jobs = []
for r in range(PER_TARGET):
    for t in NEW_TARGETS:
        if r < len(per[t]):
            jobs.append(per[t][r])
plan = dict(purpose="new-target phase1: J8 decomposition on 6 additional target tasks", new_targets=NEW_TARGETS,
            per_target=PER_TARGET, catalog=CATALOG, jobs=jobs)
(HERE / "jobs_nt1_j8.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
print(len(jobs), collections.Counter(j["target"] for j in jobs))
