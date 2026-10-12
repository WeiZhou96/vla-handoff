# -*- coding: utf-8 -*-
"""Export real static-camera frames for the case figure (read-only on the experiment data).

Selection rule (fixed before looking at any image or outcome):
  for each task in TASKS, take the labelled low-score handoffs of the five-task vocabulary-repair study
  (multitask_vocab_20261002, label jobs), sort them by sha256("paper-case|" + sample_id), and keep the first one
  whose label job has all branches valid (8 executions each). All labelled branches of that handoff are exported:
  the handoff frame itself (no repair) and the frame after each labelled repair, with the measured k/8.
The frames are the stored 200x200 static-camera observations; nothing is re-rendered.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
OUT = Path(f"{RUN_ROOT}/paper_draft_20261006/case_frames")
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["push_red_block_right", "lift_red_block_slider", "rotate_pink_block_right"]
jobs = {}
for jf in ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"):
    for j in json.loads((MT / jf).read_text())["jobs"]:
        jobs[j["sample_id"]] = j
sel = {}
for f in (MT / "analysis").glob("*_selection.json"):
    sel.update(json.loads(f.read_text()))
meta = {}
for t in TASKS:
    cands = sorted([s for s, j in jobs.items() if j["task_b"] == t], key=lambda s: hashlib.sha256(("paper-case|" + s).encode()).hexdigest())
    for sid in cands:
        job = jobs[sid]
        p = MT / "results" / f"{job['id']}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        arms = {a["candidate"]: a for a in d["arms"]}
        if any("invalid" in a or a.get("n") != 8 for a in arms.values()):
            continue
        key = hashlib.sha256(sid.encode()).hexdigest()[:12]
        rec = dict(task=t, sample_id=sid, parent_id=job["parent_id"], key=key, branches=[])
        with np.load(job["npz"]) as z:
            Image.fromarray(z["image_static"]).save(OUT / f"{key}__none.png")
        for c, a in arms.items():
            if c != "none":
                with np.load(MT / "screen" / job["screen_npz"][c]) as z:
                    Image.fromarray(z["image_static"]).save(OUT / f"{key}__{c}.png")
            role = ("none" if c == "none" else "top" if c in [x for x, _ in sel[sid]["top"]] else "random" if c in [x for x, _ in sel[sid]["random"]] else "other")
            rec["branches"].append(dict(candidate=c, k=a["k"], n=a["n"], role=role, wins="".join(str(int(x["success"])) for x in a["rollouts"]),
                                        verifier_score=sel[sid]["all_scores"].get(c)))
        meta[t] = rec
        print(t, sid, [(b["candidate"], b["k"]) for b in rec["branches"]])
        break
(OUT / "cases.json").write_text(json.dumps(meta, indent=1))
