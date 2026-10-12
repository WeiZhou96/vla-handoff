# -*- coding: utf-8 -*-
"""Full-chain dual-criterion replay worker (fullchain_replay_20261010).

Job = (round, original chain, tier). From the same perturbed initial state, every arm of the tier is run in turn:
  Mode A: line-by-line reproduction of the original closed-loop arm logic (worker_cl2.run_arm; the fixed arm reproduces worker_cl.py: score first, then take the fixed per-task repair);
          the original criterion (each attempt referenced to its own start) drives attempt termination, retries and chain stopping -- must match the original results bit for bit;
  Mode B: same arm logic, but success of target subtasks is driven by the handoff criterion B_def (an attempt ends when B_def first holds; otherwise retry/stop per the arm's rules).
          Non-target subtasks still use the original criterion (no repair or retry precedes them, so the original criterion is already referenced to the handoff).
The probe only reads simulator state; it neither changes physical state nor consumes random numbers. At every target handoff it records poses at the handoff (before repair), after repair, and at the start/end of every attempt;
the step at which each of three judgements first holds: a = original criterion (start = start of this attempt), borc = same oracle referenced to the handoff, bdef = explicit handoff criterion;
the physics-state hash when a first holds and at the end of the attempt (to verify that mode B matches mode A before the divergence point); cumulative yaw for rotate targets (diagnostic).
B_def: push_object / lift_object: displacement relative to the handoff; contact/source-surface conditions taken from the start of this attempt, as in CALVIN;
       rotate_object: same oracle referenced to the handoff (no start condition);
       push_object_into: this attempt succeeds under the original predicate, and the block was not in the drawer at the handoff;
       stack_objects: original predicate, and the stacked block touches no other block both at the handoff and at the start of this attempt.
Usage: python -u -B fc_worker.py --run <this directory> --jobs jobs.json --gpu <2..5> [--modes A,B] [--arms ...] [--out results]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, copy, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path

ROOT = Path(f"{RUN_ROOT}/fullchain_replay_20261010")
sys.path.insert(0, f"{RUN_ROOT}/closedloop_20261004")
import worker_cl2  # noqa: E402,F401  same os.environ / sys.path setup as the original worker
from worker_cl2 import BUDGET, CATALOG, INVERSE, LANG_MAX, MAX_STEPS, REPAIR_ARMS, RHO, Brain, sseed  # noqa: E402

REPAIR = tuple(REPAIR_ARMS) + ("fixed",)       # the fixed arm exists only in worker_cl.py
NO_RETRY = ("none", "benefit_noretry")


def snap(adapter):
    return copy.deepcopy(adapter.env.get_info())


def psha(env):
    import numpy as np
    from rho.s04_replay import physics_vector
    return hashlib.sha256(np.round(physics_vector(env), 10).tobytes()).hexdigest()


def poses(info):
    """Numeric fields of length 3/4 (current_pos, current_orn, ...) and the contact count of every movable object."""
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


def rel_yaw(q0, q1):
    from scipy.spatial.transform import Rotation as R
    return float((R.from_quat(q1) * R.from_quat(q0).inv()).as_euler("xyz", degrees=True)[2])


def task_spec(oracle, task):
    fn = oracle.tasks[task]          # functools.partial(Tasks.<base>, *yaml_args)
    return fn.func.__name__, tuple(fn.args)


def _objs(x):
    return [str(x)] if isinstance(x, str) else [str(o) for o in x]


def _cs(info, ob):
    return {(c[2], c[4]) for c in info["scene_info"]["movable_objects"][ob]["contacts"]}


def flags(kind, args, info):
    """Descriptive flags related to predicate preconditions; not used in any judgement."""
    try:
        so = info["scene_info"]; rid = info["robot_info"]["uid"]
        tab = so["fixed_objects"]["table"]; tu, L = tab["uid"], tab["links"]
        if kind in ("push_object", "rotate_object", "lift_object"):
            c = _cs(info, args[0])
            out = dict(nonrobot_contacts=len({x for x in c if x[0] != rid}), robot_contact=any(x[0] == rid for x in c))
            if kind == "lift_object" and len(args) >= 4 and args[3] is not None:
                out["on_source_surface"] = (tu, L[args[3]]) in c
            return out
        if kind == "push_object_into":
            obs = _objs(args[0])
            return dict(in_drawer={o: (tu, L["drawer_link"]) in _cs(info, o) for o in obs},
                        on_table_only={o: _cs(info, o) == {(tu, L["base_link"])} for o in obs})
        if kind == "stack_objects":
            M = so["movable_objects"]; uids = {M[n]["uid"] for n in M}
            return dict(touching_block={str(n): bool({c[2] for c in M[n]["contacts"]} & uids) for n in M})
        return {}
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e))


def hybrid_start(kind, args, h, s):
    """Reference state for B_def: a copy of s (start of this attempt) in which the reference-point quantities are replaced by their values at handoff h."""
    if kind not in ("push_object", "lift_object", "rotate_object", "stack_objects"):
        return h
    y = copy.deepcopy(s)
    M, Mh = y["scene_info"]["movable_objects"], h["scene_info"]["movable_objects"]
    if kind in ("push_object", "lift_object"):
        M[args[0]]["current_pos"] = copy.deepcopy(Mh[args[0]]["current_pos"])
    elif kind == "rotate_object":
        for k in ("current_pos", "current_orn"):
            M[args[0]][k] = copy.deepcopy(Mh[args[0]][k])
    else:  # stack: start contacts = union of s and h ⇒ the start condition requires no contact with other blocks at either time
        for n in M:
            M[n]["contacts"] = list(M[n]["contacts"]) + list(Mh[n]["contacts"])
    return y


class Judge:
    """The three judgements for one attempt at one target subtask."""
    def __init__(self, oracle, task, h, s):
        self.o, self.task, self.h, self.s = oracle, task, h, s
        self.kind, self.args = task_spec(oracle, task)
        self.hyb = hybrid_start(self.kind, self.args, h, s)

    def _ok(self, start, end):
        return bool(self.o.get_task_info_for_set(start, end, {self.task}))

    def a(self, e):
        return self._ok(self.s, e)

    def borc(self, e):
        return self._ok(self.h, e)

    def bdef(self, e):
        if self.kind != "push_object_into":
            return self._ok(self.hyb, e)
        obs, src_b, src_l, dst_b, dst_l = self.args
        fx = self.h["scene_info"]["fixed_objects"][dst_b]
        dst = (fx["uid"], fx["links"][dst_l])
        T = type(self.o)
        for ob in _objs(obs):
            if dst in _cs(self.h, ob):
                continue
            if T.push_object_into(ob, src_b, src_l, dst_b, dst_l, start_info=self.s, end_info=e):
                return True
        return False


class Probe:
    """Wraps env.step: computes the three judgements at every step of a target attempt; for rotate targets, accumulates yaw from the handoff."""
    def __init__(self, adapter):
        self.ad, self.active, self.J = adapter, False, None
        self.last, self.n, self.dis = None, 0, 0
        self.first, self.cur, self.track = {}, {}, None

    def attach(self, env):
        inner = env.step

        def step(action, _inner=inner):
            out = _inner(action)
            if self.track is not None:
                self._track(out[3])
            if self.active:
                self._see(out[3])
            return out
        env.step = step

    def start_track(self, obj, info):
        self.track = [obj, list(info["scene_info"]["movable_objects"][obj]["current_orn"]), 0.0]

    def _track(self, info):
        q1 = list(info["scene_info"]["movable_objects"][self.track[0]]["current_orn"])
        self.track[2] += rel_yaw(self.track[1], q1)
        self.track[1] = q1

    def yaw(self):
        return None if self.track is None else float(self.track[2])

    def begin(self, judge):
        self.J, self.last, self.n, self.dis = judge, None, 0, 0
        self.first, self.cur, self.active = dict(a=None, bdef=None, borc=None), {}, True

    def _see(self, info):
        self.last, self.n = info, self.n + 1
        self.cur = dict(a=self.J.a(info), bdef=self.J.bdef(info), borc=self.J.borc(info))
        for k, v in self.cur.items():
            if v and self.first[k] is None:
                self.first[k] = self.n

    def note(self, ok_a):
        if bool(ok_a) != bool(self.cur.get("a")):
            self.dis += 1

    def finish(self, att, ok, st, sha_a, sha_end):
        self.active = False
        J = self.J
        e = self.last if self.last is not None else snap(self.ad)
        return dict(attempt=att, success=bool(ok), steps=int(st), steps_seen=self.n, a_disagree=self.dis,
                    a_end=J.a(e), bdef_end=J.bdef(e), borc_end=J.borc(e),
                    a_first=self.first["a"], bdef_first=self.first["bdef"], borc_first=self.first["borc"],
                    sha_at_a_first=sha_a, sha_end=sha_end, start_poses=poses(J.s), end_poses=poses(e),
                    flags_start=flags(J.kind, J.args, J.s), yaw_cum=self.yaw())


def run_attempt(adapter, probe, task, max_steps, mode, h_info):
    """Same loop as rho.s04_replay.run_task; in mode B, success is declared when B_def first holds."""
    adapter.start_task(task)
    probe.begin(Judge(adapter.oracle, task, h_info, adapter.start_info))
    sha_a = None
    for steps in range(1, max_steps + 1):
        ok_a = adapter.step_task()
        probe.note(ok_a)
        if sha_a is None and probe.first["a"] is not None:
            sha_a = psha(adapter.env)
        if (ok_a if mode == "A" else probe.cur["bdef"]):
            return True, steps, sha_a
    return False, max_steps, sha_a


def run_chain(adapter, brain, cfg, seq, robot0, scene0, arm, chain_id, recorder, mode):
    """Copy of worker_cl2.run_arm (plus the fixed arm of worker_cl.py); differs only in the probe and the mode-B success criterion."""
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
    used, tasks_out, probes, success = 0, [], [], True
    lam, tau, delta, tau_r = cfg["lam"], cfg["tau_risk"], cfg["delta_benefit"], cfg["tau_retry"]
    for pos, task in enumerate(seq["tasks"]):
        rec = dict(pos=pos, task=task, target=(pos > 0 and task in brain.targets))
        pr = h_info = None
        if rec["target"]:
            h_info = snap(adapter)
            kind, args = task_spec(adapter.oracle, task)
            pr = dict(pos=pos, task=task, kind=kind, obj=(args[0] if args and isinstance(args[0], str) else None),
                      used_before=used, handoff=poses(h_info), flags_handoff=flags(kind, args, h_info),
                      repair_executed=False, attempts=[])
            if kind == "rotate_object":
                probe.start_track(args[0], h_info)
        if rec["target"] and arm in REPAIR:
            u, do = "none", False
            if arm == "inverse":
                u = INVERSE.get(task, "none"); do = u != "none"
            elif arm == "home_all":
                u, do = "home", True
            elif arm == "prior":
                pri = {c: brain.prior(task, c) for c in brain.options(task) if c != "none"}
                u = max(pri, key=pri.get); do = True
            elif arm == "fixed":      # worker_cl.py: observe and score first (not used for the decision), then take the fixed per-task repair
                E = brain.encode(adapter.observe())
                q = brain.score_arm(E, task, "benefit")
                reps = {k: v for k, v in q.items() if k != "none"}
                ustar = max(reps, key=reps.get)
                u = cfg["fixed_repair_per_task"].get(task, "none"); do = u != "none"
                rec.update(q_none=q["none"], ustar=ustar, q_ustar=reps[ustar])
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
                pr["repair_executed"] = True
        if rec["target"]:
            r_info = snap(adapter)
            pr.update(post_repair=poses(r_info), flags_post_repair=flags(kind, args, r_info), yaw_post_repair=probe.yaw())
        attempts = []
        for att in range(2):
            if used >= BUDGET:
                break
            seed_all(sseed("cl-attempt", chain_id, pos, att))
            if rec["target"]:
                ok, st, sha_a = run_attempt(adapter, probe, task, min(MAX_STEPS, BUDGET - used), mode, h_info)
                pr["attempts"].append(probe.finish(att, ok, st, sha_a, psha(adapter.env)))
            else:
                ok, st = run_task(adapter, task, min(MAX_STEPS, BUDGET - used))
            used += st
            attempts.append(dict(attempt=att, success=ok, steps=st))
            if ok:
                break
            if not rec["target"] or arm in NO_RETRY or used >= BUDGET:
                break
            if arm == "full":
                v = brain.verify(brain.encode(adapter.observe()), task)
                attempts[-1]["retry_score"] = v
                if v < tau_r:
                    break
        if rec["target"]:
            probe.track = None
            probes.append(pr)
        rec["attempts"] = attempts
        rec["success"] = bool(attempts and attempts[-1]["success"])
        tasks_out.append(rec)
        if not rec["success"]:
            success = False
            break
    return dict(arm=arm, mode=mode, success=success, completed=sum(t["success"] for t in tasks_out), steps=used,
                init_physics_sha=phys_hash, tasks=tasks_out, probes=probes)


def run_job(adapter, recorder, perturb, job, cfgs, brains, arms, modes):
    seq = adapter.sequences(job["index"], 1)[0]
    if list(seq["tasks"]) != list(job["tasks"]):
        raise RuntimeError(f"task chain does not match jobs: {seq['tasks']} vs {job['tasks']}")
    robot0, scene0, tries = perturb(adapter, seq, job["perturb_seed"], recorder)
    out = dict(job={k: v for k, v in job.items() if k != "orig"}, perturb_tries=tries,
               scene0=[float(x) for x in scene0], arms={}, config_sha256={})
    for arm in arms:
        key = job["config_key"][arm]
        brain, (cfg, csha) = brains[key], cfgs[key]
        if sorted(brain.targets) != sorted(job["targets"]):
            raise RuntimeError(f"{arm}: model target set {sorted(brain.targets)} ≠ plan {job['targets']}")
        out["config_sha256"][arm] = csha
        out["arms"][arm] = {m: run_chain(adapter, brain, cfg, seq, robot0, scene0, arm, job["orig_id"], recorder, m)
                            for m in modes}
    out["paired_initial_state_identical"] = len({r["init_physics_sha"] for v in out["arms"].values() for r in v.values()}) == 1
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
    ap.add_argument("--modes", default="A,B"); ap.add_argument("--arms", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.gpu not in (2, 3, 4, 5):
        raise SystemExit("only GPUs 2-5 may be used")
    run = Path(a.run).resolve()
    if run != ROOT.resolve():   # HANDOFF_RUN_ROOT may be a symlink: compare resolved paths
        raise SystemExit(f"--run must resolve to {ROOT.resolve()} (got {run})")
    modes = [m for m in a.modes.split(",") if m]
    if not modes or set(modes) - {"A", "B"}:
        raise SystemExit("--modes accepts only A and B")
    jobs = json.loads(Path(a.jobs).read_text())["jobs"]
    arm_filter = set(a.arms.split(",")) if a.arms else None
    suffix = "" if a.out == "results" else "_" + a.out
    dirs = [run / a.out, run / ("claims" + suffix), run / ("errors" + suffix)]
    for d in dirs:
        if d.resolve().parent != run:
            raise SystemExit(f"output directories must be inside {run}: {d}")
        d.mkdir(exist_ok=True)
    res_dir, claim_dir, err_dir = dirs
    paths = yaml.safe_load((RHO / "configs/paths.yaml").read_text())
    settings = yaml.safe_load((RHO / "configs/calvin.yaml").read_text())
    settings.update(max_steps=MAX_STEPS, sequence_catalog=CATALOG)
    adapter = CalvinAdapter(paths, settings); recorder = Recorder()
    cfgs, brains = {}, {}
    for key in sorted({k for j in jobs for k in j["config_key"].values()}):   # as in the original process: Brain is built before adapter.open
        txts = [Path(p).read_text() for p in key.split("+")]
        cfg = json.loads(txts[0])
        for t in txts[1:]:
            cfg.update(json.loads(t))          # same merge as worker_cl2 --addon
        cfgs[key] = (cfg, hashlib.sha256("\n".join(txts).encode()).hexdigest())
        brains[key] = Brain(cfg)
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
            arms = [x for x in job["arms"] if arm_filter is None or x in arm_filter]
            t0 = time.time()
            try:
                out = run_job(adapter, recorder, perturb, job, cfgs, brains, arms, modes)
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
            out.update(seconds=round(time.time() - t0, 1), gpu=a.gpu, worker="fc_worker.py", modes=modes,
                       max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
            try:
                out["max_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            except Exception:
                pass
            tmp = res_dir / f"{jid}.tmp"
            tmp.write_text(json.dumps(out)); os.replace(tmp, res_dir / f"{jid}.json")
            done += 1
            print(f"[gpu{a.gpu}] {jid} " + " ".join(
                f"{arm}:" + ",".join(f"{m}={int(r['success'])}/{r['completed']}/{r['steps']}" for m, r in v.items())
                for arm, v in out["arms"].items())
                + f" paired={out['paired_initial_state_identical']} {out['seconds']}s rss={out['max_rss_mb']}MB", flush=True)
        print("WORKER DONE", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
