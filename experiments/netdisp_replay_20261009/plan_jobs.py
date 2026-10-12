# -*- coding: utf-8 -*-
"""Net-displacement replay: generates the job list, smoke-test jobs and runtime estimate.
Reads only the two closed-loop round directories (jobs, results*, locked_config.json); writes only jobs.json, jobs_smoke.json, plan_summary.json in this directory.
Usage: PYTHONDONTWRITEBYTECODE=1 python -B plan_jobs.py
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import hashlib, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUNDS = {
    1: dict(dir=f"{RUN_ROOT}/closedloop_20261004", jobs="jobs_cl.json",
            res=("results", "results_extra", "results_extra2"),
            tasks=("push_red_block_right", "push_blue_block_right", "rotate_pink_block_right")),
    2: dict(dir=f"{RUN_ROOT}/closedloop2_20261005", jobs="jobs_cl2.json", res=("results",),
            tasks=("push_pink_block_right", "rotate_red_block_left")),
}
EXPECTED = {"push_red_block_right": 86, "push_blue_block_right": 47, "rotate_pink_block_right": 62,
            "push_pink_block_right": 57, "rotate_red_block_left": 56}
ARMS = ("none", "inverse", "prior", "benefit", "retry")
MAIN_ARMS = ("none", "inverse", "prior", "benefit")
SETTLE = 30


def summarize(arm_rec, first_pos):
    """Summary of one (chain, arm) record up to the end of the first target task; shared by original and replayed results."""
    pre, tgt = [], None
    for t in arm_rec["tasks"]:
        att = [[bool(x["success"]), int(x["steps"])] for x in t["attempts"]]
        if t["pos"] < first_pos:
            pre.append(dict(pos=t["pos"], task=t["task"], attempts=att, success=bool(t["success"])))
        elif t["pos"] == first_pos:
            tgt = dict(task=t["task"], target_flag=bool(t.get("target")), repair=t.get("repair"),
                       repair_steps=t.get("repair_steps"), repair_task_success=t.get("repair_task_success"),
                       ustar=t.get("ustar"), q_none=t.get("q_none"), q_ustar=t.get("q_ustar"),
                       attempts=att, success=bool(t["success"]))
    steps = sum(s for p in pre for _, s in p["attempts"])
    if tgt:
        steps += (tgt["repair_steps"] or 0) + sum(s for _, s in tgt["attempts"])
    return dict(init_physics_sha=arm_rec.get("init_physics_sha"), prefix=pre, target=tgt, steps_to_end=steps)


def model_targets(cfg):
    try:
        sys.path.insert(0, f"{RUN_ROOT}/closedloop_20261004")
        import worker_cl2  # noqa: F401  sets the same sys.path as the original worker, so that the featurizer can be unpickled
        import joblib
        return set(joblib.load(cfg["model_file"])["tasks"]), "model_file.tasks"
    except Exception as e:
        return None, f"model load failed: {e!r}"


def sec_per_step(d):
    vals = []
    for p in (d / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        arms = r.get("arms") or {}
        tot = sum(v.get("steps", 0) + SETTLE for v in arms.values()) + SETTLE * (1 + int(r.get("perturb_tries") or 0))
        if r.get("seconds") and tot:
            vals.append(r["seconds"] / tot)
    return (statistics.median(vals), len(vals)) if vals else (None, 0)


def main():
    S = dict(created=time.strftime("%F %T"), rounds={}, counts={}, warnings=[], eta={})
    W = S["warnings"]
    jobs_out = []
    for rnd, R in ROUNDS.items():
        d = Path(R["dir"]); cfg_path = d / "locked_config.json"
        txt = cfg_path.read_text(); cfg = json.loads(txt)
        targets, tsrc = (set(cfg["targets"]), "config.targets") if cfg.get("targets") else model_targets(cfg)
        src_jobs = json.loads((d / R["jobs"]).read_text())["jobs"]
        observed, loaded = set(), {}
        for job in src_jobs:
            arms, srcs = {}, {}
            for sub in R["res"]:
                p = d / sub / f"{job['id']}.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                for arm, v in (r.get("arms") or {}).items():
                    observed.update(t["task"] for t in v.get("tasks", []) if t.get("target"))
                    if arm not in ARMS:
                        continue
                    if arm in arms:
                        W.append(f"r{rnd} {job['id']}: arm {arm} appears in several directories, using {srcs[arm]['dir']}, ignoring {sub}")
                        continue
                    arms[arm] = v
                    srcs[arm] = dict(dir=sub, gpu=r.get("gpu"), seconds=r.get("seconds"),
                                     scene0=r.get("scene0"), perturb_tries=r.get("perturb_tries"))
            loaded[job["id"]] = (job, arms, srcs)
        if targets is None:
            W.append(f"r{rnd}: {tsrc}; target set falls back to the target tasks seen in the original results")
            targets, tsrc = observed, "observed"
        if not observed <= targets:
            W.append(f"r{rnd}: target tasks {sorted(observed - targets)} of the original results are not in the target set")
        S["rounds"][rnd] = dict(dir=str(d), config=str(cfg_path), config_sha256=hashlib.sha256(txt.encode()).hexdigest(),
                                model_file=cfg.get("model_file"), targets=sorted(targets), targets_source=tsrc,
                                n_source_jobs=len(src_jobs))
        for jid, (job, arms, srcs) in loaded.items():
            fp = next((p for p, t in enumerate(job["tasks"]) if p > 0 and t in targets), None)
            if fp is None or job["tasks"][fp] not in R["tasks"]:
                continue
            orig = {}
            for arm in ARMS:
                if arm not in arms:
                    continue
                v = arms[arm]
                flagged = next((t["pos"] for t in v["tasks"] if t.get("target")), None)
                if flagged is not None and flagged != fp:
                    W.append(f"r{rnd} {jid} {arm}: first target position in the original results {flagged} != plan {fp}")
                s = summarize(v, fp); s["source"] = srcs[arm]; orig[arm] = s
            sc = {json.dumps(o["source"]["scene0"]) for o in orig.values() if o["source"]["scene0"] is not None}
            if len(sc) > 1:
                W.append(f"r{rnd} {jid}: scene0 differs between source directories")
            base = job.get("base")
            if base is None:
                W.append(f"r{rnd} {jid}: no base field, using index instead"); base = f"idx{job['index']}"
            tries = max([int(o["source"]["perturb_tries"] or 0) for o in orig.values()] or [0])
            jobs_out.append(dict(id=f"r{rnd}_{jid}", orig_id=jid, round=rnd, index=job["index"], base=base,
                                 tasks=job["tasks"], perturb_seed=job["perturb_seed"], first_pos=fp,
                                 first_task=job["tasks"][fp], config=str(cfg_path),
                                 arms=[a for a in ARMS if a in orig], missing_arms=[a for a in ARMS if a not in orig],
                                 orig=orig, est_steps={a: SETTLE + o["steps_to_end"] for a, o in orig.items()},
                                 est_perturb_steps=SETTLE * (1 + tries)))
        S["rounds"][rnd]["sec_per_step_median"], S["rounds"][rnd]["sec_per_step_n"] = sec_per_step(d)
    for t, exp in EXPECTED.items():
        js = [j for j in jobs_out if j["first_task"] == t]
        S["counts"][t] = dict(n=len(js), expected=exp,
                              reached_in_orig_none=sum(1 for j in js if "none" in j["orig"] and j["orig"]["none"]["target"]),
                              arm_coverage={a: sum(a in j["orig"] for j in js) for a in ARMS})
        if len(js) != exp:
            W.append(f"count mismatch: {t} planned {len(js)}, expected {exp}")
    spp = S["rounds"][2]["sec_per_step_median"] or S["rounds"][1]["sec_per_step_median"]
    main_steps = sum(sum(j["est_steps"][a] for a in MAIN_ARMS if a in j["est_steps"]) + j["est_perturb_steps"] for j in jobs_out)
    retry_steps = sum(j["est_steps"].get("retry", 0) + j["est_perturb_steps"] for j in jobs_out)
    S["eta"] = dict(sec_per_step=spp, main_steps=main_steps, retry_steps=retry_steps,
                    main_hours={w: (main_steps * spp / w / 3600 if spp else None) for w in (8, 12)},
                    retry_hours={w: (retry_steps * spp / w / 3600 if spp else None) for w in (8, 12)},
                    note="excludes model loading (once per process); settling steps after perturbation are approximate; estimated from the per-process throughput of the original runs")
    smoke = []
    for rnd, task in ((1, "push_red_block_right"), (1, "rotate_pink_block_right"), (2, "push_pink_block_right")):
        c = [j for j in jobs_out if j["round"] == rnd and j["first_task"] == task and all(a in j["orig"] for a in MAIN_ARMS)
             and j["orig"]["none"]["target"] is not None and (j["orig"]["inverse"]["target"] or {}).get("repair")]
        if c:
            smoke.append(min(c, key=lambda j: (sum(j["est_steps"][a] for a in MAIN_ARMS), j["id"])))
        else:
            W.append(f"smoke: r{rnd} {task} has no eligible chain")
    jobs_out.sort(key=lambda j: (-sum(j["est_steps"].values()), j["id"]))
    purpose = "net-displacement replay: first target handoff, push/rotate targets"
    (HERE / "jobs.json").write_text(json.dumps(dict(purpose=purpose, created=S["created"], jobs=jobs_out)))
    (HERE / "jobs_smoke.json").write_text(json.dumps(dict(purpose=purpose + " (smoke)", created=S["created"], jobs=smoke)))
    S["n_jobs"], S["smoke_ids"] = len(jobs_out), [j["id"] for j in smoke]
    (HERE / "plan_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("counts", "eta", "n_jobs", "smoke_ids")}, indent=1))
    for r, v in S["rounds"].items():
        print(f"round {r}: targets={v['targets_source']} {v['targets']} s/step={v['sec_per_step_median']} (n={v['sec_per_step_n']})")
    print(f"WARNINGS ({len(W)}):"); [print("  " + w) for w in W[:50]]


if __name__ == "__main__":
    main()
