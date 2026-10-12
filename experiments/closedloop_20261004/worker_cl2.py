# -*- coding: utf-8 -*-
"""Whole-chain closed-loop worker v2: adds control arms to worker_cl.py and allows choosing the task set and model files.

Arms (intervene only at handoffs of target tasks; never before task 0; target set = `targets` in the config, default: all tasks covered by the scoring model)
  none            no repair, no retry
  retry           no repair; retry once after a target-task failure
  inverse         before push/rotate target tasks run the inverse skill (INVERSE table), no repair otherwise; retry once on failure
  prior           run the repair with the highest mean success for this task in the development-data prior table; retry once on failure
  home_all        run the scripted home repair before every target task; retry once on failure
  risk            M7 selects u*; repair if q0 < tau; retry once on failure
  benefit         M7 selects u*; repair if q(u*) - q0 - lambda*c_bar/360 > delta; retry once on failure (proposed method)
  benefit_noretry same repair decision as benefit, no retry (requires no failure detection)
  noloc           selection and benefit gate with the no-localiser model (same lambda, delta, c_bar as benefit); retry once on failure
  benefit_m5      selection and benefit gate with state model M5v2 (no prior values); retry once on failure
  benefit_alt     selection and benefit gate with an alternative scorer (set in the config); retry once on failure
  Models per arm are set by config arm_models (default: M7 for benefit/risk/full/benefit_noretry, M7n for noloc, M5v2 for benefit_m5).
  full            repair decision of benefit; after a failure, retry only if the M2v2 verification score >= tau_r
Budget, perturbation and pairing are the same as in worker_cl.py (the perturbation function is reused).
Usage: python -B worker_cl2.py --run <dir> --jobs <jobs.json> --gpu <2..5> --arms a,b,c [--out results] [--config cfg.json]
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
sys.path.insert(0, f"{RUN_ROOT}/closedloop_20261004")
RHO = Path(f"{RHO_ROOT}")
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
INVERSE = {"push_red_block_right": "push_red_block_left", "push_blue_block_right": "push_blue_block_left",
           "rotate_pink_block_right": "rotate_pink_block_left", "push_pink_block_right": "push_pink_block_left",
           "rotate_red_block_left": "rotate_red_block_right"}
BUDGET, MAX_STEPS, LANG_MAX = 2160, 360, 120
REPAIR_ARMS = ("inverse", "prior", "home_all", "risk", "benefit", "benefit_noretry", "noloc", "benefit_m5", "benefit_alt", "full")


def sseed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


WITH_PRIOR = {"M7", "M7n", "M7a"}
NOLOC = {"M7n", "M5n"}
DEFAULT_ARM_MODELS = {"risk": ["main", "M7"], "benefit": ["main", "M7"], "benefit_noretry": ["main", "M7"], "full": ["main", "M7"],
                      "noloc": ["aux", "M7n"], "benefit_m5": ["aux", "M5v2"]}


class Brain:
    """DINOv2 (CPU) -> state features v2 / no-localiser features -> per-arm scoring model; M2v2 verification.
    Model files: main = cfg["model_file"] (M2v2 and the default scorer), aux = cfg["aux_model_file"] (optional).
    Arm -> model is set by cfg["arm_models"] (default: DEFAULT_ARM_MODELS); each model uses the featuriser from its own file."""
    def __init__(self, cfg):
        import joblib, numpy as np, torch, yaml
        torch.set_num_threads(4)
        self.np, self.cfg = np, cfg
        self.files = {"main": joblib.load(cfg["model_file"])}
        if cfg.get("aux_model_file"):
            self.files["aux"] = (self.files["main"] if cfg["aux_model_file"] == cfg["model_file"] else joblib.load(cfg["aux_model_file"]))
        self.arm_models = dict(DEFAULT_ARM_MODELS, **cfg.get("arm_models", {}))
        self.m = self.files["main"]
        self.tasks, self.cands = self.m["tasks"], self.m["cands"]
        self.targets = set(cfg.get("targets") or self.tasks)
        self.vocab = sorted(yaml.safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))

    def encode(self, obs):
        from features_patch import encode
        np = self.np
        enc = encode([obs["images"]["static"]], modes=("grid8", "grid16"))
        return dict(g16=enc["grid16"].astype(np.float32), g8=enc["grid8"].astype(np.float32),
                    rob=np.asarray(obs["state"], np.float32)[None])

    def feats(self, E):
        return self.m["featurizer"].transform(E["g16"], E["g8"], E["rob"])

    @staticmethod
    def _prior(m, t, c):
        pt = m["prior_table"]
        if f"{t}|{c}" in pt:
            return pt[f"{t}|{c}"]
        return pt.get(f"{t}|none", 0.5) if c == "none" else m["task_mean"].get(t, 0.5)

    def prior(self, t, c):
        return self._prior(self.m, t, c)

    def options(self, task):
        return ["none"] + [v for v in self.vocab if v != task] + ["retreat", "home"]

    def score_arm(self, E, task, arm):
        np = self.np
        src, key = self.arm_models[arm]
        m = self.files[src]
        F = (m["featurizer_nl"].transform(E["g8"], E["rob"]) if key in NOLOC
             else m["featurizer"].transform(E["g16"], E["g8"], E["rob"]))
        opts = self.options(task)
        parts = [np.repeat(F, len(opts), 0), np.array([[float(task == u) for u in m["tasks"]]] * len(opts)),
                 np.array([[float(o == c) for c in m["cands"]] for o in opts])]
        if key in WITH_PRIOR:
            parts.append(np.array([[self._prior(m, task, o)] for o in opts]))
        return dict(zip(opts, map(float, m[key].predict(np.hstack(parts)))))

    def verify(self, E, task):
        np = self.np
        X = np.hstack([self.feats(E), np.array([[float(task == u) for u in self.tasks]])])
        return float(self.m["M2v2"].predict(X)[0])

    def cbar(self, t, u):
        return self.cfg["cbar"].get(f"{t}|{u}", self.cfg["cbar_global"].get(u, 100.0))


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
    lam, tau, delta, tau_r = cfg["lam"], cfg["tau_risk"], cfg["delta_benefit"], cfg["tau_retry"]
    for pos, task in enumerate(seq["tasks"]):
        rec = dict(pos=pos, task=task, target=(pos > 0 and task in brain.targets))
        if rec["target"] and arm in REPAIR_ARMS:
            u, do = "none", False
            if arm == "inverse":
                u = INVERSE.get(task, "none"); do = u != "none"
            elif arm == "home_all":
                u, do = "home", True
            elif arm == "prior":
                pri = {c: brain.prior(task, c) for c in brain.options(task) if c != "none"}
                u = max(pri, key=pri.get); do = True
            else:
                E = brain.encode(adapter.observe())
                q = brain.score_arm(E, task, arm)
                reps = {k: v for k, v in q.items() if k != "none"}
                ustar = max(reps, key=reps.get)
                if arm == "risk":
                    u, do = ustar, q["none"] < tau
                else:
                    u, do = ustar, (reps[ustar] - q["none"] - lam * brain.cbar(task, ustar) / 360) > delta
                rec.update(q_none=q["none"], ustar=ustar, q_ustar=reps[ustar])
            rec["repair"] = u if do else None
            if do and used < BUDGET:
                seed_all(sseed("cl-repair", chain_id, pos, u))
                if u in ("retreat", "home"):
                    from worker import scripted_repair
                    st = scripted_repair(adapter.env, u, seq); ok_r = None
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
            if not rec["target"] or arm in ("none", "benefit_noretry") or used >= BUDGET:
                break
            if arm == "full":
                v = brain.verify(brain.encode(adapter.observe()), task)
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
    from worker_cl import perturb
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None, help="default: <run>/locked_config.json")
    ap.add_argument("--addon", default=None, help="optional: extra config (e.g. aux_model_file, arm_models) that overrides keys of the same name")
    ap.add_argument("--scene", default=None, help="optional: calvin_scene_A/B/C, replaces environment construction (policy and models unchanged)")
    a = ap.parse_args()
    if a.scene and a.scene != "calvin_scene_D":
        import rho.s04_replay as rp

        def fresh_scene(adapter, seed, recorder, scene=a.scene):
            from rlinf.envs.calvin import make_env
            if adapter.env is not None:
                try:
                    adapter.env.close()
                finally:
                    adapter.env.cid = -1
            env = make_env(scene=scene); env.seed(seed)
            adapter.env = env; recorder.actions = []; recorder.attach(env)
            return env
        rp.fresh_env = fresh_scene
    run = Path(a.run).resolve()
    cfg = json.loads(Path(a.config or (run / "locked_config.json")).read_text())
    if a.addon:
        cfg.update(json.loads(Path(a.addon).read_text()))
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
            if a.scene and a.scene != "calvin_scene_D":
                sys.path.insert(0, f"{RUN_ROOT}/crossscene_20261005")
                from scene_map import map_scene_obs
                seq["initial"]["scene_obs"] = map_scene_obs(seq["initial"]["scene_obs"], a.scene).tolist()
            robot0, scene0, tries = perturb(adapter, seq, job["perturb_seed"], recorder)
            out = dict(job=job, scene=a.scene or "calvin_scene_D", perturb_tries=tries, scene0=scene0.tolist(), arms={})
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
