# -*- coding: utf-8 -*-
"""Net-displacement replay worker: reuses the arm logic of worker_cl2.py and runs only up to the first target task (including all its attempts).

Identical to the original experiment: task chains (index/tasks/perturb_seed), perturbation and settling (worker_cl.perturb), arm logic (run_arm_probe is a
line-by-line copy of worker_cl2.run_arm), random seeds (sseed keyed by the original chain id), and each round's locked_config.json with the model it specifies.
The probe only reads simulator state (env.get_info() and the info returned by env.step); it neither changes physical state nor consumes random numbers.
Recorded: poses of the three blocks at the handoff (before repair), after repair, and at the start/end of every attempt;
judgements of the same oracle under three references: (a) start of the attempt = adapter.start_info (original experiment); (b) handoff; (c) post-repair.
Usage: python -u -B replay_worker.py --run <this directory> --jobs jobs.json --gpu <2..5> [--arms none,inverse,prior,benefit] [--out results]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, copy, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path

ROOT = Path(f"{RUN_ROOT}/netdisp_replay_20261009")
sys.path.insert(0, f"{RUN_ROOT}/closedloop_20261004")
import worker_cl2  # noqa: E402,F401  applies the same os.environ / sys.path setup as the original worker
from worker_cl2 import BUDGET, CATALOG, INVERSE, LANG_MAX, MAX_STEPS, REPAIR_ARMS, RHO, Brain, sseed  # noqa: E402


def snap(adapter):
    return copy.deepcopy(adapter.env.get_info())


def poses(info):
    """All numeric fields of length 3/4 (current_pos, current_orn, ...) and the contact count of every movable object."""
    import numpy as np
    out = {}
    for name, d in info["scene_info"]["movable_objects"].items():
        e = {}
        for k, v in d.items():
            if k == "contacts":
                e["n_contacts"] = len(v)
                continue
            try:
                arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 1 and arr.size in (3, 4) and np.isfinite(arr).all():
                e[k] = [float(x) for x in arr]
        out[str(name)] = e
    return out


class Probe:
    """Wraps env.step: when active, keeps the info of every step and checks whether the oracle was ever satisfied under each reference."""
    def __init__(self, adapter):
        self.adapter, self.active, self.last, self.n = adapter, False, None, 0
        self.task, self.refs, self.hit = None, {}, {}

    def attach(self, env):
        inner = env.step

        def step(action, _inner=inner):
            out = _inner(action)
            if self.active:
                self._see(out[3])
            return out
        env.step = step

    def _ok(self, ref, info):
        return bool(self.adapter.oracle.get_task_info_for_set(ref, info, {self.task}))

    def _see(self, info):
        self.last, self.n = info, self.n + 1
        for k, ref in self.refs.items():
            if not self.hit[k] and self._ok(ref, info):
                self.hit[k] = True

    def begin(self, task, refs):
        self.task, self.refs = task, refs
        self.hit = {k: False for k in refs}
        self.last, self.n, self.active = None, 0, True

    def finish(self, att, ok, st, a_ref):
        self.active = False
        last = self.last if self.last is not None else snap(self.adapter)
        return dict(attempt=att, success=bool(ok), steps=int(st), steps_seen=self.n,
                    a_end=self._ok(a_ref, last), b_end=self._ok(self.refs["handoff"], last),
                    c_end=self._ok(self.refs["post_repair"], last),
                    b_any=self.hit["handoff"], c_any=self.hit["post_repair"],
                    start_poses=poses(a_ref), end_poses=poses(last))


def run_arm_probe(adapter, brain, cfg, seq, robot0, scene0, arm, chain_id, recorder, first_pos):
    """Copy of worker_cl2.run_arm; differs only in the probe-related lines and in stopping after the first target task."""
    import numpy as np
    from rho.s04_replay import fresh_env, physics_vector, run_task, seed_all
    env = fresh_env(adapter, 0, recorder)
    probe = Probe(adapter)
    probe.attach(env)
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
    pr = None
    for pos, task in enumerate(seq["tasks"]):
        rec = dict(pos=pos, task=task, target=(pos > 0 and task in brain.targets))
        probing = pos == first_pos
        if probing:
            if not rec["target"]:
                raise RuntimeError(f"pos {pos} ({task}) is not a target task, inconsistent with the plan")
            h_info = snap(adapter)
            pr = dict(pos=pos, task=task, used_before=used, handoff=poses(h_info), repair_executed=False)
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
                if probing:
                    pr["repair_executed"] = True
        if probing:
            r_info = snap(adapter)
            pr["post_repair"], pr["attempts"] = poses(r_info), []
        attempts = []
        for att in range(2):
            if used >= BUDGET:
                break
            seed_all(sseed("cl-attempt", chain_id, pos, att))
            if probing:
                probe.begin(task, dict(handoff=h_info, post_repair=r_info))
            ok, st = run_task(adapter, task, min(MAX_STEPS, BUDGET - used))
            if probing:
                pr["attempts"].append(probe.finish(att, ok, st, adapter.start_info))
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
        if probing:
            break
    return dict(arm=arm, success=success, completed=sum(t["success"] for t in tasks_out), steps=used,
                init_physics_sha=phys_hash, tasks=tasks_out, probe=pr, truncated_after_pos=first_pos)


def run_job(adapter, recorder, perturb, job, cfg, brain, arms):
    seq = adapter.sequences(job["index"], 1)[0]
    if list(seq["tasks"]) != list(job["tasks"]):
        raise RuntimeError(f"task chain does not match jobs: {seq['tasks']} vs {job['tasks']}")
    fp = next((p for p, t in enumerate(seq["tasks"]) if p > 0 and t in brain.targets), None)
    if fp != job["first_pos"]:
        raise RuntimeError(f"first target position {fp} does not match the plan {job['first_pos']}")
    robot0, scene0, tries = perturb(adapter, seq, job["perturb_seed"], recorder)
    out = dict(job={k: v for k, v in job.items() if k != "orig"}, perturb_tries=tries,
               scene0=[float(x) for x in scene0], arms={})
    for arm in arms:
        out["arms"][arm] = run_arm_probe(adapter, brain, cfg, seq, robot0, scene0, arm, job["orig_id"], recorder, fp)
    out["paired_initial_state_identical"] = len({r["init_physics_sha"] for r in out["arms"].values()}) == 1
    return out


def main():
    import yaml
    import torch
    from rho.envs.s02_calvin import CalvinAdapter
    from rho.s04_replay import Recorder
    from worker_cl import perturb
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="none,inverse,prior,benefit"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.gpu not in (2, 3, 4, 5):
        raise SystemExit("only GPUs 2-5 may be used")
    run = Path(a.run).resolve()
    if run != ROOT.resolve():  # the run directory may be a symlink: compare resolved paths
        raise SystemExit(f"--run must be {ROOT}")
    jobs = json.loads(Path(a.jobs).read_text())["jobs"]
    arms_req = a.arms.split(",")
    suffix = "" if a.out == "results" else "_" + a.out
    res_dir, claim_dir, err_dir = run / a.out, run / ("claims" + suffix), run / ("errors" + suffix)
    for d in (res_dir, claim_dir, err_dir):
        d.mkdir(exist_ok=True)
    paths = yaml.safe_load((RHO / "configs/paths.yaml").read_text())
    settings = yaml.safe_load((RHO / "configs/calvin.yaml").read_text())
    settings.update(max_steps=MAX_STEPS, sequence_catalog=CATALOG)
    adapter = CalvinAdapter(paths, settings); recorder = Recorder()
    cfgs, brains = {}, {}
    for p in sorted({j["config"] for j in jobs}):    # as in the original process: Brain is built before adapter.open
        txt = Path(p).read_text()
        cfgs[p] = (json.loads(txt), hashlib.sha256(txt.encode()).hexdigest())
        brains[p] = Brain(cfgs[p][0])
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
            arms = [x for x in arms_req if x in job["arms"]]
            t0 = time.time()
            try:
                cfg, csha = cfgs[job["config"]]
                out = run_job(adapter, recorder, perturb, job, cfg, brains[job["config"]], arms)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                ef = err_dir / f"{jid}.json"
                n = (json.loads(ef.read_text()).get("n", 0) if ef.exists() else 0) + 1
                ef.write_text(json.dumps(dict(n=n, gpu=a.gpu, time=time.time(), error=tb)))
                if n >= 2:   # second failure: write a result carrying the error and do not requeue
                    tmp = res_dir / f"{jid}.tmp"
                    tmp.write_text(json.dumps(dict(job={k: v for k, v in job.items() if k != "orig"}, error=tb, arms={})))
                    os.replace(tmp, res_dir / f"{jid}.json")
                else:
                    (claim_dir / f"{jid}.json").unlink(missing_ok=True)
                raise
            out.update(seconds=round(time.time() - t0, 1), gpu=a.gpu, config_sha256=csha, worker="replay_worker.py",
                       max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
            try:
                out["max_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            except Exception:
                pass
            tmp = res_dir / f"{jid}.tmp"
            tmp.write_text(json.dumps(out)); os.replace(tmp, res_dir / f"{jid}.json")
            done += 1
            print(f"[gpu{a.gpu}] {jid} " + " ".join(
                f"{k}={int(v['success'])}/{v['steps']}/{'T' if v['probe'] else '-'}" for k, v in out["arms"].items())
                + f" paired={out['paired_initial_state_identical']} {out['seconds']}s rss={out['max_rss_mb']}MB", flush=True)
        print("WORKER DONE", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
