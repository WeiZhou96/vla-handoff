# -*- coding: utf-8 -*-
"""Fixed plan of the round-2 closed-loop confirmation experiment (stratified by extended target task).
The 6 extended target tasks are processed in turn: candidate chains = chains with that task at positions 1-4 that were neither used in the round-1 closed loop nor already chosen by this plan;
chains are grouped by base initial state and taken round-robin across groups in hash order, 67 per task. Each chain gets an independent perturbation seed. The selection uses no execution results.
Perturbed initial states share the discrete configurations of the development data (as in round 1); all chain indices used in the round-1 closed loop are excluded."""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import json, hashlib, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
CV = Path(f"{RUN_ROOT}")
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
NEW = ["lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer", "stack_block",
       "push_pink_block_right", "rotate_red_block_left"]
PER_TASK = 67
h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
cat = json.load(open(CATALOG))
used = {int(j["index"]) for j in json.loads((CV / "closedloop_20261004/jobs_cl.json").read_text())["jobs"]}
chosen, jobs = set(), []
for t in NEW:
    by = collections.defaultdict(list)
    for i, (ini, tasks) in enumerate(cat):
        if i in used or i in chosen:
            continue
        if t in tasks[1:5]:
            by[json.dumps(ini, sort_keys=True)].append(i)
    bases = sorted(by, key=lambda k: h("cl2-base", t, k))
    for k in bases:
        by[k].sort(key=lambda i: h("cl2-chain", i))
    take, rnd = [], 0
    while len(take) < PER_TASK and any(len(by[k]) > rnd for k in bases):
        for k in bases:
            if len(by[k]) > rnd and len(take) < PER_TASK:
                take.append((by[k][rnd], k))
        rnd += 1
    for i, k in take:
        chosen.add(i)
        jobs.append(dict(id=f"cl2_{i}", index=i, base=h("base", k)[:16], stratum=t, tasks=cat[i][1],
                         perturb_seed=int(h("cl2-perturb", i)[:8], 16)))
# interleave strata
by_s = collections.defaultdict(list)
for j in jobs:
    by_s[j["stratum"]].append(j)
inter = [by_s[t][r] for r in range(PER_TASK) for t in NEW if r < len(by_s[t])]
(HERE / "jobs_cl2.json").write_text(json.dumps(dict(purpose="second-round closed-loop confirmation stratified by new target, perturbed initial states",
                                                    per_task=PER_TASK, excluded_round1=len(used), jobs=inter), indent=1))
cnt = collections.Counter(t for j in inter for t in j["tasks"][1:5] if t in NEW)
print("jobs", len(inter), "bases", len({j['base'] for j in inter}), "| strata", dict(collections.Counter(j["stratum"] for j in inter)))
print("new-target handoffs at positions 1-4:", dict(cnt))
