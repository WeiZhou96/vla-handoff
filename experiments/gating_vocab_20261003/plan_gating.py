# -*- coding: utf-8 -*-
"""Data for the gating test: vocabulary-repair screening plan for natural-distribution handoffs (fixed before running).
From the natural handoffs of 5 tasks (J8 of the two development tasks + multi-task J8), handoffs that already have repair labels are excluded
(181 low-scoring vocabulary handoffs, 82 hand-chosen repair handoffs), and 30 per task are chosen by a fixed hash. Selection uses no execution outcome or model score,
so these handoffs approximately follow the natural distribution, include both high- and low-scoring states, and are used to test the benefit gate and the risk gate."""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import json, hashlib, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIX = Path(f"{RUN_ROOT}/fix_20260928")
MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
RP = Path(f"{RUN_ROOT}/repair_20261001")
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))
PER_TASK = 30

used = set()
for jf in ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"):
    used |= {j["sample_id"] for j in json.loads((MT / jf).read_text())["jobs"]}
used |= {s["sample_id"] for s in json.loads((RP / "plan.json").read_text())["states"]}
nat = {}
for g in sorted((FIX / "runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            nat[r["sample_id"]] = dict(sid=r["sample_id"], task=r["task_b"], index=r["index"], sequence_id=r["sequence_id"],
                                       parent=r["parent_id"], npz=str(g.parent / "boundaries" / r["observation"]), k=r["k"])
for f in sorted((MT / "results").glob("j8_*.json")):
    d = json.loads(f.read_text())
    if d.get("reached") and d.get("n") == 8 and "restore_failed" not in d:
        nat.setdefault(d["sample_id"], dict(sid=d["sample_id"], task=d["task_b"], index=d["job"]["index"],
                                            sequence_id=d["sample_id"].split(":")[0], parent=d["parent_id"],
                                            npz=str(MT / "boundaries" / d["observation"]), k=d["k"]))
jobs = []
for t in sorted({r["task"] for r in nat.values()}):
    pool = sorted([r for r in nat.values() if r["task"] == t and r["sid"] not in used],
                  key=lambda r: hashlib.sha256(("gating-v1:" + r["sid"]).encode()).hexdigest())[:PER_TASK]
    for r in pool:
        jobs.append(dict(type="screen", id="screen_" + hashlib.sha256(r["sid"].encode()).hexdigest()[:24], sample_id=r["sid"],
                         index=r["index"], sequence_id=r["sequence_id"], task_b=t, npz=r["npz"], parent_id=r["parent"],
                         k_j8=r["k"], candidates=[v for v in VOCAB if v != t] + ["retreat", "home"]))
(HERE / "jobs_screen.json").write_text(json.dumps(dict(
    purpose="gating data: vocabulary screening on natural-distribution handoffs (30 per task, hash order, excluding labeled)",
    jobs=jobs), indent=1))
cnt = collections.Counter(j["task_b"] for j in jobs)
k = collections.defaultdict(list)
for j in jobs:
    k[j["task_b"]].append(j["k_j8"] / 8)
print("screen jobs", len(jobs), dict(cnt))
print("mean J8 success of selected states (sanity, not used for selection):", {t: round(sum(v) / len(v), 3) for t, v in k.items()})
