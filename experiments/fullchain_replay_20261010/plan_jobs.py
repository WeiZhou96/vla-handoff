# -*- coding: utf-8 -*-
"""Full-chain dual-criterion replay: job list, smoke-test list and runtime estimate.
Reads only: the two closed-loop round directories (jobs, results*, locked_config.json, addon_extra2.json); results of the net-displacement replay directory (only to estimate time per step).
Writes only: jobs.json, jobs_smoke.json, plan_summary.json in this directory.
Job = (round, original chain, tier). Tier 1 is the most important and runs first; within a tier, jobs are ordered by expected steps, longest first (load balancing).
Usage: PYTHONDONTWRITEBYTECODE=1 python -B plan_jobs.py
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import json, statistics, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT = Path(f"{RUN_ROOT}/fullchain_replay_20261010").resolve()
NETDISP = Path(f"{RUN_ROOT}/netdisp_replay_20261009")
R1 = f"{RUN_ROOT}/closedloop_20261004"
R2 = f"{RUN_ROOT}/closedloop2_20261005"
ROUNDS = {
    1: dict(dir=R1, jobs="jobs_cl.json", n_expected=391, sources={
        "results": dict(worker="worker_cl.py", configs=("locked_config.json",),
                        arms=("none", "retry", "fixed", "risk", "benefit", "full")),
        "results_extra": dict(worker="worker_cl2.py", configs=("locked_config.json",), arms=("inverse", "prior")),
        "results_extra2": dict(worker="worker_cl2.py", configs=("locked_config.json", "addon_extra2.json"),
                               arms=("benefit_noretry", "home_all", "noloc", "benefit_m5"))}),
    2: dict(dir=R2, jobs="jobs_cl2.json", n_expected=402, sources={
        "results": dict(worker="worker_cl2.py", configs=("locked_config.json",),
                        arms=("none", "retry", "inverse", "prior", "home_all", "benefit", "benefit_noretry", "noloc", "benefit_alt"))}),
}
TIERS = {1: {1: ("none", "retry", "benefit", "inverse", "prior"),
             2: ("none", "retry", "benefit", "inverse", "prior", "benefit_noretry", "noloc")},
         2: {1: ("benefit_noretry", "noloc", "benefit_m5", "home_all"), 2: ("home_all", "benefit_alt")},
         3: {1: ("fixed", "risk", "full")}}
PR = ("push_red_block_right", "push_blue_block_right", "push_pink_block_right", "rotate_pink_block_right", "rotate_red_block_left")
NOINV = ("lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer", "stack_block")
SETTLE = 30
B_FACTOR = 1.05      # assumed ratio of mode-B steps to mode-A steps (target attempts may run longer; chains may also stop earlier)


def summarize(rec):
    """Comparable summary of one (chain, arm) full-chain record; shared by original and replayed results."""
    tasks = []
    for t in rec.get("tasks", []):
        tasks.append(dict(pos=t["pos"], task=t["task"], target=bool(t.get("target")), repair=t.get("repair"),
                          repair_steps=t.get("repair_steps"), repair_task_success=t.get("repair_task_success"),
                          attempts=[[bool(x["success"]), int(x["steps"])] for x in t.get("attempts", [])],
                          success=bool(t["success"])))
    return dict(init_physics_sha=rec.get("init_physics_sha"), success=bool(rec["success"]),
                completed=int(rec.get("completed", sum(t["success"] for t in tasks))), steps=int(rec["steps"]), tasks=tasks)


def model_targets(cfg):
    try:
        sys.path.insert(0, R1)
        import worker_cl2  # noqa: F401  same sys.path as the original worker, so that the featurizer can be unpickled
        import joblib
        return set(joblib.load(cfg["model_file"])["tasks"]), "model_file.tasks"
    except Exception as e:  # noqa: BLE001
        return None, f"model load failed: {e!r}"


def sps_from(paths):
    vals = []
    for p in paths:
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        arms = r.get("arms") or {}
        tot = sum(v.get("steps", 0) + SETTLE for v in arms.values() if isinstance(v, dict)) + SETTLE * (1 + int(r.get("perturb_tries") or 0))
        if r.get("seconds") and tot:
            vals.append(r["seconds"] / tot)
    return (statistics.median(vals), len(vals)) if vals else (None, 0)


def main():
    if HERE != EXPECT:
        raise SystemExit(f"this script must be located at {EXPECT} (got {HERE})")
    S = dict(created=time.strftime("%F %T"), rounds={}, warnings=[], missing_arm_counts={}, eta={})
    W = S["warnings"]; jobs_out = []; miss = Counter(); arm_steps = {}
    for rnd, R in ROUNDS.items():
        d = Path(R["dir"])
        cfg = json.loads((d / "locked_config.json").read_text())
        targets, tsrc = (set(cfg["targets"]), "config.targets") if cfg.get("targets") else model_targets(cfg)
        src_jobs = json.loads((d / R["jobs"]).read_text())["jobs"]
        if len(src_jobs) != R["n_expected"]:
            W.append(f"r{rnd}: {len(src_jobs)} original jobs, expected {R['n_expected']}")
        keys = {}
        for sub, spec in R["sources"].items():
            ps = [d / c for c in spec["configs"]]
            absent = [str(p) for p in ps if not p.exists()]
            keys[sub] = None if absent else "+".join(str(p) for p in ps)
            if absent:
                W.append(f"r{rnd} {sub}: missing configuration {absent}; arms {spec['arms']} of this directory are not included")
        observed, per_chain = set(), {}
        for job in src_jobs:
            orig = {}
            for sub, spec in R["sources"].items():
                p = d / sub / f"{job['id']}.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                for arm, v in (r.get("arms") or {}).items():
                    observed.update(t["task"] for t in v.get("tasks", []) if t.get("target"))
                    if arm not in spec["arms"]:
                        continue
                    if arm in orig:
                        W.append(f"r{rnd} {job['id']}: {arm} duplicated, keeping {orig[arm]['source']['dir']}"); continue
                    s = summarize(v)
                    s["source"] = dict(dir=sub, worker=spec["worker"], config_key=keys[sub], gpu=r.get("gpu"),
                                       seconds=r.get("seconds"), scene0=r.get("scene0"), perturb_tries=r.get("perturb_tries"))
                    orig[arm] = s
                    arm_steps.setdefault((rnd, arm), []).append(s["steps"])
            per_chain[job["id"]] = (job, orig)
        if targets is None:
            W.append(f"r{rnd}: {tsrc}; target set falls back to the target tasks seen in the original results")
            targets, tsrc = observed, "observed"
        if not observed <= targets:
            W.append(f"r{rnd}: target tasks {sorted(observed - targets)} of the original results are not in the target set")
        S["rounds"][rnd] = dict(dir=str(d), targets=sorted(targets), targets_source=tsrc, n_source_jobs=len(src_jobs),
                                config_keys=keys,
                                sec_per_step_orig={sub: sps_from(sorted((d / sub).glob("*.json"))) for sub in R["sources"]})
        for jid, (job, orig) in per_chain.items():
            tp = [p for p, t in enumerate(job["tasks"]) if p > 0 and t in targets]
            if not tp:
                W.append(f"r{rnd} {jid}: no target handoff")
            base = job.get("base")
            if base is None:
                base = f"idx{job['index']}"; W.append(f"r{rnd} {jid}: no base, using index instead")
            tries = max([int(o["source"]["perturb_tries"] or 0) for o in orig.values()] or [0])
            if len({json.dumps(o["source"]["scene0"]) for o in orig.values() if o["source"]["scene0"] is not None}) > 1:
                W.append(f"r{rnd} {jid}: scene0 differs between source directories")
            for tier, by_round in TIERS.items():
                want = by_round.get(rnd, ())
                arms = [a for a in want if a in orig and orig[a]["source"]["config_key"]]
                for a in want:
                    if a not in arms:
                        miss[f"r{rnd}:{a}"] += 1
                if not arms:
                    continue
                jobs_out.append(dict(
                    id=f"r{rnd}_{jid}_t{tier}", orig_id=jid, round=rnd, tier=tier, index=job["index"], base=base,
                    tasks=job["tasks"], perturb_seed=job["perturb_seed"], targets=sorted(targets), target_positions=tp,
                    first_pos=tp[0] if tp else None, first_task=job["tasks"][tp[0]] if tp else None,
                    pr_chain=any(job["tasks"][p] in PR for p in tp), arms=arms,
                    config_key={a: orig[a]["source"]["config_key"] for a in arms}, orig={a: orig[a] for a in arms},
                    est_steps={a: SETTLE + orig[a]["steps"] for a in arms}, est_perturb_steps=SETTLE * (1 + tries)))
    S["missing_arm_counts"] = dict(miss)
    S["mean_orig_steps"] = {f"r{r}:{a}": round(statistics.mean(v), 1) for (r, a), v in sorted(arm_steps.items())}
    # runtime estimate
    nd, nd_n = sps_from(sorted((NETDISP / "results").glob("*.json")) + sorted((NETDISP / "results_retry").glob("*.json")))
    allsec = sum(o["source"]["seconds"] or 0 for j in jobs_out for o in j["orig"].values()) or None
    S["eta"]["sec_per_step"] = dict(netdisp_median=nd, netdisp_n=nd_n, central_estimate=0.052, low=0.033, high=0.075)
    for tier in sorted(TIERS):
        js = [j for j in jobs_out if j["tier"] == tier]
        steps = sum(sum(j["est_steps"].values()) * (1 + B_FACTOR) + j["est_perturb_steps"] for j in js)
        S["eta"][f"tier{tier}"] = dict(n_jobs=len(js), steps_AB=int(steps), hours_12w={
            k: (round(steps * v / 12 / 3600, 2) if v else None)
            for k, v in (("netdisp", nd), ("central", 0.052), ("low", 0.033), ("high", 0.075))})
    S["eta"]["note"] = "excludes model loading; B_FACTOR is an assumption; total seconds of the original results (reference only) = " + str(allsec)
    # smoke test
    def pr_rep_ok(j):
        o = j["orig"].get("benefit")
        return bool(o) and any(t["target"] and t["task"] in PR and t["repair"] and t["success"] for t in o["tasks"])

    def two_att(j, arm):
        o = j["orig"].get(arm)
        return bool(o) and any(len(t["attempts"]) == 2 for t in o["tasks"])

    specs = [
        ("r1 tier 1: >=2 targets (push/rotate and other), original benefit arm succeeded after a repair on push/rotate", 1, 1,
         lambda j: len(j["target_positions"]) >= 2 and j["pr_chain"]
         and any(j["tasks"][p] not in PR for p in j["target_positions"]) and pr_rep_ok(j)),
        ("r2 tier 1: contains a new task without inverse skill and a push/rotate target", 2, 1,
         lambda j: j["pr_chain"] and any(j["tasks"][p] in NOINV for p in j["target_positions"])),
        ("r1 tier 2 (addon_extra2 configuration)", 1, 2, lambda j: len(j["target_positions"]) >= 2),
        ("r1 tier 3 (fixed/risk/full), original full arm retried", 1, 3, lambda j: j["pr_chain"] and two_att(j, "full")),
    ]
    smoke = []
    for desc, rnd, tier, extra in specs:
        pool = [j for j in jobs_out if j["round"] == rnd and j["tier"] == tier]
        c = [j for j in pool if extra(j)] or pool
        if not [j for j in pool if extra(j)]:
            W.append(f"smoke spec \"{desc}\": no fully eligible chain, relaxing conditions")
        if c:
            smoke.append(min(c, key=lambda j: (sum(j["est_steps"].values()), j["id"])))
    jobs_out.sort(key=lambda j: (j["tier"], -sum(j["est_steps"].values()), j["id"]))
    purpose = "full-chain replay, modes A (original criterion) and B (handoff-referenced criterion B_def)"
    (HERE / "jobs.json").write_text(json.dumps(dict(purpose=purpose, created=S["created"], jobs=jobs_out)))
    (HERE / "jobs_smoke.json").write_text(json.dumps(dict(purpose=purpose + " (smoke)", created=S["created"], jobs=smoke)))
    S["n_jobs"] = len(jobs_out); S["n_jobs_by_tier"] = dict(Counter(j["tier"] for j in jobs_out))
    S["smoke_ids"] = [j["id"] for j in smoke]
    (HERE / "plan_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("n_jobs", "n_jobs_by_tier", "smoke_ids", "eta", "missing_arm_counts", "mean_orig_steps")},
                     indent=1, ensure_ascii=False))
    for r, v in S["rounds"].items():
        print(f"round {r}: targets[{v['targets_source']}]={v['targets']}  s/step(orig)={v['sec_per_step_orig']}")
    print(f"WARNINGS ({len(W)}):")
    for w in W[:60]:
        print("  " + w)


if __name__ == "__main__":
    main()
