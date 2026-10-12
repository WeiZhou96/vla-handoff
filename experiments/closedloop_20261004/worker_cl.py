# -*- coding: utf-8 -*-
"""Worker for the whole-chain closed-loop confirmation experiment. Each job = one chain, running 6 arms in turn (same perturbed initial state, same seeds, paired).

Arms (intervene only at handoffs of target tasks; target tasks = the 5 tasks covered by the classifier; never before task 0)
  none     no repair, no retry
  retry    no repair; retry once after a target-task failure
  fixed    run the fixed per-task repair (selected on development data); retry once on failure
  risk     M7 selects repair u*; repair if the predicted no-repair success q0 < tau; retry once on failure
  benefit  M7 selects u*; repair if q(u*) - q0 - lambda*c_bar/360 > delta; retry once on failure
  full     same repair decision as benefit; after a failure, score the failure observation with M2v2 and retry only if >= tau_r
Total chain budget 2160 steps (repairs, attempts and retries all count); at most 360 steps per attempt, 120 steps per language repair.
Perturbed initial state: on the base initial state, blocks are shifted by +-3 cm and rotated by +-0.5 rad about the vertical axis, sliding door and drawer by +-2 cm;
after settling for 30 steps, check block displacement < 1 cm and height drop not > 1 cm; otherwise resample with a new seed (at most 20 times, halved magnitude from the 10th).
All arms start from the same perturbed initial state in a freshly built environment, and the hash of the initial physics-state vector is checked for equality.
Usage: python -B worker_cl.py --run <dir> --jobs jobs_cl.json --gpu <2..5>
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")
sys.path.insert(0, os.path.dirname(RHO_ROOT))
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")
sys.path.insert(0, f"{RUN_ROOT}/multitask_vocab_20261002")
RHO = Path(f"{RHO_ROOT}")
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
ARMS = ("none", "retry", "fixed", "risk", "benefit", "full")
# Supplementary control (added after the confirmation data were produced, to rule out simpler explanations; the rule is fixed a priori from task semantics and does not use confirmation data)
INVERSE = {"push_red_block_right": "push_red_block_left", "push_blue_block_right": "push_blue_block_left",
           "rotate_pink_block_right": "rotate_pink_block_left"}
BUDGET, MAX_STEPS, LANG_MAX = 2160, 360, 120


def sseed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


class Brain:
    """Online features and scoring: DINOv2 (CPU) -> state features v2 -> M7 / M2v2."""
    def __init__(self, cfg):
        import joblib, numpy as np, torch
        torch.set_num_threads(8)
        self.np = np
        self.cfg = cfg
        self.m = joblib.load(cfg["model_file"])
        self.tasks, self.cands = self.m["tasks"], self.m["cands"]
        import yaml
        self.vocab = sorted(yaml.safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))

    def feats(self, obs):
        from features_patch import encode
        np = self.np
        enc = encode([obs["images"]["static"]], modes=("grid8", "grid16"))
        return self.m["featurizer"].transform(enc["grid16"].astype(np.float32), enc["grid8"].astype(np.float32),
                                              np.asarray(obs["state"], np.float32)[None])

    def prior(self, t, c):
        pt = self.m["prior_table"]
        if f"{t}|{c}" in pt:
            return pt[f"{t}|{c}"]
        return pt.get(f"{t}|none", 0.5) if c == "none" else self.m["task_mean"][t]

    def score_all(self, F, task):
        np = self.np
        opts = ["none"] + [v for v in self.vocab if v != task] + ["retreat", "home"]
        X = np.hstack([np.repeat(F, len(opts), 0), np.array([[float(task == u) for u in self.tasks]] * len(opts)),
                       np.array([[float(o == c) for c in self.cands] for o in opts]),
                       np.array([[self.prior(task, o)] for o in opts])])
        q = self.m["M7"].predict(X)
        return dict(zip(opts, map(float, q)))

    def verify(self, F, task):
        np = self.np
        X = np.hstack([F, np.array([[float(task == u) for u in self.tasks]])])
        return float(self.m["M2v2"].predict(X)[0])

    def cbar(self, t, u):
        return self.cfg["cbar"].get(f"{t}|{u}", self.cfg["cbar_global"].get(u, 100.0))


def perturb(adapter, seq, seed, recorder):
    """Return a perturbed initial state (robot, scene) that passes the settling check, and the number of attempts."""
    import numpy as np
    from rho.s04_replay import fresh_env
    robot0 = np.asarray(seq["initial"]["robot"], np.float64); scene0 = np.asarray(seq["initial"]["scene_obs"], np.float64)
    for attempt in range(20):
        rng = np.random.RandomState((seed + attempt * 7919) % (2**32))
        k = 1.0 if attempt < 10 else 0.5
        sc = scene0.copy()
        sc[0] = np.clip(sc[0] + rng.uniform(-0.02, 0.02) * k, 0.0, 0.28)
        sc[1] = np.clip(sc[1] + rng.uniform(-0.02, 0.02) * k, 0.0, 0.22)
        for b in (6, 12, 18):
            sc[b] += rng.uniform(-0.03, 0.03) * k; sc[b + 1] += rng.uniform(-0.03, 0.03) * k
            sc[b + 5] += rng.uniform(-0.5, 0.5) * k
        env = fresh_env(adapter, 0, recorder)
        env.reset(robot_obs=robot0, scene_obs=sc)
        before = np.asarray(env.get_obs()["scene_obs"])
        hold = np.zeros(7); hold[6] = 1.0 if robot0[14] > 0 else -1.0
        cams, env.cameras = env.cameras, []
        try:
            for _ in range(30):
                env.step(hold.copy())
        finally:
            env.cameras = cams
        after = np.asarray(env.get_obs()["scene_obs"])
        ok = True
        for b in (6, 12, 18):
            if np.linalg.norm(after[b:b + 2] - before[b:b + 2]) > 0.01 or before[b + 2] - after[b + 2] > 0.01:
                ok = False
        if ok:
            return robot0, sc, attempt
    raise RuntimeError("no valid perturbation in 20 attempts")


def run_arm(adapter, brain, cfg, seq, robot0, scene0, arm, chain_id, recorder):
    import numpy as np
    from rho.s04_replay import fresh_env, physics_vector, run_task, seed_all
    env = fresh_env(adapter, 0, recorder)
    env.reset(robot_obs=robot0, scene_obs=scene0)
    hold = np.zeros(7); hold[6] = 1.0 if robot0[14] > 0 else -1.0
    cams, env.cameras = env.cameras, []
    try:
        for _ in range(30):
            env.step(hold.copy())
    finally:
        env.cameras = cams
    phys_hash = hashlib.sha256(np.round(physics_vector(env), 10).tobytes()).hexdigest()
    used, tasks_out, success = 0, [], True
    targets = set(brain.tasks)
    lam, tau, delta, tau_r = cfg["lam"], cfg["tau_risk"], cfg["delta_benefit"], cfg["tau_retry"]
    for pos, task in enumerate(seq["tasks"]):
        rec = dict(pos=pos, task=task, target=(pos > 0 and task in targets))
        if rec["target"] and arm in ("fixed", "risk", "benefit", "full", "inverse", "prior"):
            obs = adapter.observe()
            F = brain.feats(obs)
            q = brain.score_all(F, task)
            reps = {u: v for u, v in q.items() if u != "none"}
            ustar = max(reps, key=reps.get)
            if arm == "fixed":
                u = cfg["fixed_repair_per_task"].get(task, "none")
                do = u != "none"
            elif arm == "inverse":
                u = INVERSE.get(task, "none")
                do = u != "none"
            elif arm == "prior":
                pri = {c: brain.prior(task, c) for c in reps}
                u = max(pri, key=pri.get)
                do = True
            elif arm == "risk":
                u, do = ustar, q["none"] < tau
            else:
                u, do = ustar, (reps[ustar] - q["none"] - lam * brain.cbar(task, ustar) / 360) > delta
            rec.update(q_none=q["none"], ustar=ustar, q_ustar=reps[ustar], repair=u if do else None)
            if do and used < BUDGET:
                seed_all(sseed("cl-repair", chain_id, pos, u))
                n0 = len(recorder.actions)
                if u in ("retreat", "home"):
                    from worker import scripted_repair
                    st = scripted_repair(adapter.env, u, seq)
                    ok_r = None
                else:
                    ok_r, st = run_task(adapter, u, min(LANG_MAX, BUDGET - used))
                used += st
                rec.update(repair_steps=st, repair_task_success=ok_r)
        attempts = []
        for att in range(2):
            if used >= BUDGET:
                break
            seed_all(sseed("cl-attempt", chain_id, pos, att))
            ok, st = run_task(adapter, task, min(MAX_STEPS, BUDGET - used))
            used += st
            attempts.append(dict(attempt=att, success=ok, steps=st))
            if ok:
                break
            # whether to retry
            if not rec["target"] or arm == "none" or used >= BUDGET:
                break
            if arm == "full":
                v = brain.verify(brain.feats(adapter.observe()), task)
                attempts[-1]["retry_score"] = v
                if v < tau_r:
                    break
        rec["attempts"] = attempts
        rec["success"] = bool(attempts and attempts[-1]["success"])
        tasks_out.append(rec)
        if not rec["success"]:
            success = False
            break
    return dict(arm=arm, success=success, completed=sum(t["success"] for t in tasks_out), steps=used,
                init_physics_sha=phys_hash, tasks=tasks_out)


def main():
    import yaml
    from rho.envs.s02_calvin import CalvinAdapter
    from rho.s04_replay import Recorder
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    cfg = json.loads((run / "locked_config.json").read_text())
    jobs = json.loads(Path(a.jobs).read_text())["jobs"]
    arms_run = a.arms.split(",")
    res_dir, claim_dir = run / a.out, run / ("claims" if a.out == "results" else "claims_" + a.out)
    for d in (res_dir, claim_dir):
        d.mkdir(exist_ok=True)
    paths = yaml.safe_load((RHO / "configs/paths.yaml").read_text())
    settings = yaml.safe_load((RHO / "configs/calvin.yaml").read_text())
    settings.update(max_steps=MAX_STEPS, sequence_catalog=CATALOG)
    adapter = CalvinAdapter(paths, settings); recorder = Recorder()
    brain = Brain(cfg)
    done = 0
    try:
        adapter.open(a.gpu, 0)
        for job in jobs:
            if a.limit is not None and done >= a.limit:
                break
            jid = job["id"]
            if (res_dir / f"{jid}.json").exists():
                continue
            try:
                fd = os.open(claim_dir / f"{jid}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                json.dump(dict(pid=os.getpid(), time=time.time()), f)
            t0 = time.time()
            seq = adapter.sequences(job["index"], 1)[0]
            robot0, scene0, tries = perturb(adapter, seq, job["perturb_seed"], recorder)
            out = dict(job=job, perturb_tries=tries, scene0=scene0.tolist(), arms={})
            for arm in arms_run:
                out["arms"][arm] = run_arm(adapter, brain, cfg, seq, robot0, scene0, arm, jid, recorder)
            hashes = {r["init_physics_sha"] for r in out["arms"].values()}
            out["paired_initial_state_identical"] = len(hashes) == 1
            out["seconds"] = round(time.time() - t0, 1); out["gpu"] = a.gpu
            tmp = res_dir / f"{jid}.tmp"
            tmp.write_text(json.dumps(out)); os.replace(tmp, res_dir / f"{jid}.json")
            done += 1
            print(f"[gpu{a.gpu}] {jid} " + " ".join(f"{k}={int(v['success'])}/{v['completed']}/{v['steps']}" for k, v in out["arms"].items())
                  + f" paired={out['paired_initial_state_identical']} {out['seconds']}s", flush=True)
        print("WORKER DONE", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
