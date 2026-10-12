# -*- coding: utf-8 -*-
"""B_def relabelling worker (relabel_bdef_20261011).

Job = one labelled branch (handoff × candidate u; with role=none_identity, an identity check of the none branch).
  1. Following the original label worker (multitask_vocab_20261002/worker.py, identical sha256 across the three directories; this process
     loads that module directly and calls its Runner.restore / load_boundary / hashes_from_npz), restore the handoff h (prefix replay,
     checking the physics vector to 1e-6 and the bytes of both camera images), and take h_info = env.get_info().
  2. For j = 0..7: restore the post-repair state s_u as in the original pipeline (prefix + repair_actions, same checks), seed_all(stable_seed("vocab-b", sample_id, j)).
     Mode A: the same loop as rho.s04_replay.run_task, stopping when the original criterion first holds (at most 360 steps); must match the stored success/steps exactly;
            B_def and B_orc are also evaluated at every step, recording the first step at which each of the three criteria holds and the physics hash when A and B_def first hold.
     Mode B: stops when B_def first holds (at most 360 steps). Both modes follow the same trajectory up to min(both termination steps), hence:
            · if during mode A B_def already holds before A terminates (including the same step), or A runs the full 360 steps, the mode-B result is fully observed and not re-run;
            · only when A succeeds at step k while B_def has not held by step k are s_u restored again and mode B re-run with the same seed;
            · for branches planned with verify_b=True, all 8 executions re-run mode B to check that "the B result inferred from A" agrees with the actual run.
     Every mode-B re-run checks that, within min(both termination steps), the first-hold steps of the three criteria and the corresponding physics hashes are identical.
  none_identity: the start is the handoff itself; only mode A is run, checking that the first-hold steps of B_def and B_orc equal that of A and that the poses of s and h are identical.
Probes only read simulator information; they do not change the physics state or consume random numbers. The B_def functions are copied verbatim from fc_worker.py and checked by AST at start-up (ignoring docstrings);
the process exits on any mismatch.
Usage: python -u -B relabel_worker.py --run <this directory> --jobs jobs.json --gpu <2..5> [--out results] [--limit N]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, ast, copy, hashlib, importlib.util, json, os, resource, sys, time, traceback
from pathlib import Path

ROOT = Path(f"{RUN_ROOT}/relabel_bdef_20261011")
LABEL_WORKER = Path(f"{RUN_ROOT}/multitask_vocab_20261002/worker.py")
FC_WORKER = Path(f"{RUN_ROOT}/fullchain_replay_20261010/fc_worker.py")
COPIED = ("snap", "psha", "poses", "rel_yaw", "task_spec", "_objs", "_cs", "flags", "hybrid_start", "Judge")
TOL = 1e-6


def _load_label_worker():
    spec = importlib.util.spec_from_file_location("label_worker_vocab", str(LABEL_WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LW = _load_label_worker()          # same os.environ / sys.path setup as the original label process
MAX_STEPS, REPEATS = LW.MAX_STEPS, LW.REPEATS


# ======== copied verbatim from fullchain_replay_20261010/fc_worker.py (AST-checked at start-up) ========
def snap(adapter):
    return copy.deepcopy(adapter.env.get_info())


def psha(env):
    import numpy as np
    from rho.s04_replay import physics_vector
    return hashlib.sha256(np.round(physics_vector(env), 10).tobytes()).hexdigest()


def poses(info):
    """Numeric fields of length 3/4 (current_pos, current_orn, etc.) and the contact count of every movable object."""
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
    """Descriptive flags related to predicate preconditions; not used in any decision."""
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
    """Reference state of B_def: a copy of s (at the start of this attempt) with the reference-related quantities replaced by their values at the handoff h."""
    if kind not in ("push_object", "lift_object", "rotate_object", "stack_objects"):
        return h
    y = copy.deepcopy(s)
    M, Mh = y["scene_info"]["movable_objects"], h["scene_info"]["movable_objects"]
    if kind in ("push_object", "lift_object"):
        M[args[0]]["current_pos"] = copy.deepcopy(Mh[args[0]]["current_pos"])
    elif kind == "rotate_object":
        for k in ("current_pos", "current_orn"):
            M[args[0]][k] = copy.deepcopy(Mh[args[0]][k])
    else:  # stack: start contacts = union of s and h ⇒ the start condition requires no contact with other blocks at both times
        for n in M:
            M[n]["contacts"] = list(M[n]["contacts"]) + list(Mh[n]["contacts"])
    return y


class Judge:
    """The three criteria for one attempt at one target subtask."""
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
# ======== end of copied code ========


def _strip_doc(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


def check_copied():
    def defs(path):
        tree = _strip_doc(ast.parse(Path(path).read_text(encoding="utf-8")))
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in COPIED}
    a, b = defs(FC_WORKER), defs(Path(__file__).resolve())
    return {n: bool(n in a and n in b and a[n] == b[n]) for n in COPIED}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jd(o):
    try:
        return o.item()
    except Exception:
        return str(o)


class Tap:
    """Wraps env.step and only records the info returned at each step; does not change the physics state or consume random numbers."""
    def __init__(self):
        self.last = None

    def attach(self, env):
        inner = env.step

        def step(action, _inner=inner):
            out = _inner(action)
            self.last = out[3]
            return out
        env.step = step


def target_pose(info, obj):
    if obj is None or info is None:
        return None
    d = info["scene_info"]["movable_objects"][obj]
    return dict(pos=[float(x) for x in d["current_pos"]], orn=[float(x) for x in d["current_orn"]])


def consistency(A, B):
    """Both modes follow the same trajectory within min(termination steps): first-hold steps and the corresponding physics hashes must be identical."""
    L = min(A["steps"], B["steps"])
    ok, firsts = True, {}
    for k in ("a", "bdef", "borc"):
        fa, fb = A[k + "_first"], B[k + "_first"]
        fa = fa if fa is not None and fa <= L else None
        fb = fb if fb is not None and fb <= L else None
        firsts[k] = [fa, fb]
        ok = ok and fa == fb
    for k in ("a", "bdef"):
        if firsts[k][0] is not None:
            ok = ok and A["sha_" + k] == B["sha_" + k]
    return dict(ok=bool(ok), L=L, firsts=firsts)


class Relabeler:
    def __init__(self, gpu):
        self.r = LW.Runner(ROOT.resolve(), gpu)
        self.tap = Tap()

    @property
    def ad(self):
        return self.r.adapter

    def open(self):
        self.r.open()

    def close(self):
        self.r.close()

    def restore(self, seq, actions, vec, imgs):
        err, same = self.r.restore(seq, actions, vec, imgs)     # restore of the original label process (new environment each time)
        self.tap.last = None
        self.tap.attach(self.ad.env)
        return float(err), bool(same)

    def execute(self, task, h_info, mode, obj):
        """Same loop as run_task; mode A terminates on the original criterion, mode B on B_def."""
        ad = self.ad
        ad.start_task(task)
        J = Judge(ad.oracle, task, h_info, ad.start_info)
        first = dict(a=None, bdef=None, borc=None)
        sha = dict(a=None, bdef=None)
        dis, cur, info = 0, {}, None
        for steps in range(1, MAX_STEPS + 1):
            ok_a = ad.step_task()
            info = self.tap.last
            cur = dict(a=J.a(info), bdef=J.bdef(info), borc=J.borc(info))
            if cur["a"] != bool(ok_a):
                dis += 1
            for k, v in cur.items():
                if v and first[k] is None:
                    first[k] = steps
            for k in ("a", "bdef"):
                if first[k] == steps:
                    sha[k] = psha(ad.env)
            if (ok_a if mode == "A" else cur["bdef"]):
                return self._pack(mode, True, steps, first, sha, dis, cur, info, obj)
        return self._pack(mode, False, MAX_STEPS, first, sha, dis, cur, info, obj)

    @staticmethod
    def _pack(mode, ok, steps, first, sha, dis, cur, info, obj):
        return dict(mode=mode, success=bool(ok), steps=int(steps), a_first=first["a"], bdef_first=first["bdef"],
                    borc_first=first["borc"], sha_a=sha["a"], sha_bdef=sha["bdef"], a_disagree=dis,
                    end=dict(cur), end_target=target_pose(info, obj))

    def job(self, job):
        import numpy as np
        from rho.s04_replay import seed_all, stable_seed
        t_restore, n_restore = 0.0, 0
        seq = self.ad.sequences(job["index"], 1)[0]
        if seq["sequence_id"] != job["sequence_id"]:
            raise RuntimeError(f"sequence_id mismatch: {seq['sequence_id']} vs {job['sequence_id']}")
        task, cand, role = job["task_b"], job["cand"], job["role"]
        prefix, vec0, img0 = self.r.load_boundary(job["npz"])
        out = dict(id=job["id"], role=role, dir=job["dir"], label_file=job["label_file"], label_job_id=job["job_id"],
                   sample_id=job["sample_id"], task_b=task, candidate=cand, verify_b=bool(job["verify_b"]),
                   prefix_len=len(prefix), repair_len=0, executions=[])
        t0 = time.time()
        err, same = self.restore(seq, prefix, vec0, img0)
        t_restore += time.time() - t0; n_restore += 1
        out["restore_handoff"] = dict(err=err, images_equal=same)
        if err > TOL or not same:
            out["invalid"] = f"handoff restore mismatch err={err} images_equal={same}"
            return out
        h_info = snap(self.ad)
        kind, args = task_spec(self.ad.oracle, task)
        obj = args[0] if args and isinstance(args[0], str) else None
        out.update(kind=kind, args=json.loads(json.dumps(list(args), default=str)), obj=obj,
                   handoff=dict(sha=psha(self.ad.env), poses=poses(h_info), flags=flags(kind, args, h_info)))
        if cand == "none":
            actions, vec, imgs = prefix, vec0, img0
        else:
            with np.load(job["screen"]) as z:
                ra = [a for a in z["repair_actions"]]
                vec = z["physics"].copy()
                imgs = LW.hashes_from_npz(z)
            actions = prefix + ra
            out["repair_len"] = len(ra)
        errs_a, errs_all = [], []
        for j in range(REPEATS):
            st = job["stored"][j]
            if int(st["repeat"]) != j:
                raise RuntimeError("unexpected repeat order in stored rollouts")
            seed = stable_seed("vocab-b", job["sample_id"], j)        # paired seed of the original label process
            t0 = time.time()
            err, same = self.restore(seq, actions, vec, imgs)
            t_restore += time.time() - t0; n_restore += 1
            errs_a.append(err); errs_all.append(err)
            if err > TOL or not same:
                out["invalid"] = f"post-repair restore mismatch at repeat {j}: err={err} images_equal={same}"
                break
            if j == 0:
                s_info = snap(self.ad)
                out["post_repair"] = dict(sha=psha(self.ad.env), poses=poses(s_info), flags=flags(kind, args, s_info),
                                          equals_handoff_poses=poses(s_info) == poses(h_info))
            seed_all(seed)
            A = self.execute(task, h_info, "A", obj)
            rec = dict(repeat=j, seed=seed, stored=st, A=A,
                       match_stored=bool(A["success"] == bool(st["success"]) and A["steps"] == int(st["steps"])))
            if A["bdef_first"] is not None:
                pred = (True, int(A["bdef_first"]))
            elif not A["success"]:
                pred = (False, MAX_STEPS)
            else:
                pred = None
            run_b = role != "none_identity" and (pred is None or job["verify_b"])
            if run_b:
                t0 = time.time()
                err, same = self.restore(seq, actions, vec, imgs)
                t_restore += time.time() - t0; n_restore += 1
                errs_all.append(err)
                if err > TOL or not same:
                    out["invalid"] = f"post-repair restore mismatch (mode B) at repeat {j}: err={err} images_equal={same}"
                    break
                seed_all(seed)
                B = self.execute(task, h_info, "B", obj)
                rec.update(B=B, consistency=consistency(A, B))
                lab = (B["success"], B["steps"])
                if pred is not None:
                    rec["pred_ok"] = bool(lab == pred)
                rec["b_source"] = "run_B" if pred is None else "observed_in_A+verified"
            elif pred is not None:
                lab, rec["b_source"] = pred, "observed_in_A"
            else:   # only possible for none_identity: A succeeds while B_def does not hold at the same step ⇒ identity check fails
                lab, rec["b_source"] = (A["success"], A["steps"]), "identity_violation"
            rec["label_B"] = dict(repeat=j, success=bool(lab[0]), steps=int(lab[1]))
            if role == "none_identity":
                rec["identity_ok"] = bool(A["bdef_first"] == A["a_first"] and A["borc_first"] == A["a_first"])
            out["executions"].append(rec)
        ex = out["executions"]
        out["seconds_restore"] = round(t_restore, 1)
        out["restore_max_error_all"] = max(errs_all) if errs_all else None
        if "invalid" not in out and len(ex) == REPEATS:
            rme = max(errs_a)
            out["arm_A"] = dict(candidate=cand, rollouts=[dict(repeat=e["repeat"], success=e["A"]["success"], steps=e["A"]["steps"])
                                                         for e in ex], k=sum(e["A"]["success"] for e in ex), n=REPEATS, restore_max_error=rme)
            out["arm_B"] = dict(candidate=cand, rollouts=[e["label_B"] for e in ex], k=sum(e["label_B"]["success"] for e in ex),
                                n=REPEATS, restore_max_error=rme)
            out["checks"] = dict(match_stored=all(e["match_stored"] for e in ex),
                                 a_disagree=sum(e["A"]["a_disagree"] + (e["B"]["a_disagree"] if "B" in e else 0) for e in ex),
                                 consistency=all(e["consistency"]["ok"] for e in ex if "consistency" in e),
                                 pred_ok=all(e.get("pred_ok", True) for e in ex),
                                 identity=all(e.get("identity_ok", True) for e in ex),
                                 n_b_runs=sum("B" in e for e in ex), n_restores=n_restore,
                                 steps_run=sum(e["A"]["steps"] + (e["B"]["steps"] if "B" in e else 0) for e in ex))
        return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.gpu not in (2, 3, 4, 5):
        raise SystemExit("only GPUs 2-5 are allowed")
    run = Path(a.run).resolve()
    if run != ROOT.resolve():   # HANDOFF_RUN_ROOT may be a symlink: compare resolved paths
        raise SystemExit(f"--run must resolve to {ROOT.resolve()} (got {run})")
    copied = check_copied()
    if not all(copied.values()):
        raise SystemExit(f"copied B_def code differs from fc_worker.py: {copied}")
    code = dict(label_worker=sha_file(LABEL_WORKER), fc_worker=sha_file(FC_WORKER), self=sha_file(Path(__file__).resolve()),
                copied_identical=True)
    jobs = json.loads(Path(a.jobs).read_text())["jobs"]
    suffix = "" if a.out == "results" else "_" + a.out
    dirs = [run / a.out, run / ("claims" + suffix), run / ("errors" + suffix)]
    for d in dirs:
        if d.resolve().parent != run:
            raise SystemExit(f"output directory must be under {run}: {d}")
        d.mkdir(exist_ok=True)
    res_dir, claim_dir, err_dir = dirs
    rl = Relabeler(a.gpu)
    done = 0
    try:
        rl.open()
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
            try:
                out = rl.job(job)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                ef = err_dir / f"{jid}.json"
                n = (json.loads(ef.read_text()).get("n", 0) if ef.exists() else 0) + 1
                ef.write_text(json.dumps(dict(n=n, gpu=a.gpu, time=time.time(), error=tb)))
                if n >= 2:   # second failure: write a result with error and do not reschedule
                    tmp = res_dir / f"{jid}.tmp"
                    tmp.write_text(json.dumps(dict(id=jid, error=tb, code=code)))
                    os.replace(tmp, res_dir / f"{jid}.json")
                else:
                    (claim_dir / f"{jid}.json").unlink(missing_ok=True)
                raise
            out.update(seconds=round(time.time() - t0, 1), gpu=a.gpu, worker="relabel_worker.py", code=code,
                       max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
            try:
                out["max_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            except Exception:
                pass
            tmp = res_dir / f"{jid}.tmp"
            tmp.write_text(json.dumps(out, default=_jd))
            os.replace(tmp, res_dir / f"{jid}.json")
            done += 1
            if "arm_A" in out:
                c = out["checks"]
                msg = (f"A={out['arm_A']['k']}/8 stored={job['stored_k']}/8 B={out['arm_B']['k']}/8 match={c['match_stored']} "
                       f"cons={c['consistency']} pred={c['pred_ok']} id={c['identity']} Bruns={c['n_b_runs']}")
            else:
                msg = "INVALID " + out.get("invalid", "")
            print(f"[gpu{a.gpu}] {jid} {job['role']} {job['task_b']} {job['cand']} {msg} {out['seconds']}s rss={out['max_rss_mb']}MB",
                  flush=True)
        print("WORKER DONE", flush=True)
    finally:
        rl.close()


if __name__ == "__main__":
    main()
