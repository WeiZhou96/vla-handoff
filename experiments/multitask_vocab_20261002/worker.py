# -*- coding: utf-8 -*-
"""Generic worker for the multi-task decomposition and instruction-vocabulary repair experiments.

Job types (the "type" field of each entry in the jobs file):
  j8     Run the preceding subtasks naturally up to the target handoff, save the handoff observation and action prefix, then execute the target task with 8 seeds.
  screen For one handoff, execute each candidate repair once (scripted action or CALVIN language instruction, at most 120 steps) and save the post-repair observation; the target task is not executed.
  label  For one handoff, replay the specified candidate repair (including none = no repair), check that the post-repair state matches the screening state exactly, then execute the target task with 8 paired seeds.
           Seeds depend only on (handoff, repetition index), not on the candidate, so all candidates are paired.
Every restore checks the physical state vector (tolerance 1e-6) and the bytes of both camera images; actions are replayed with their original dtype.
Reuses the CalvinAdapter and state replay of the rho project without modifying its source.
Usage: python -B worker.py --run <experiment dir> --jobs <jobs.json> --gpu <physical GPU index>
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")
sys.path.insert(0, os.path.dirname(RHO_ROOT))
RHO = Path(f"{RHO_ROOT}")
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
MAX_STEPS, REPEATS, LANG_MAX = 360, 8, 120


def key_of(s):
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def write_atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def claim(run, jid):
    p = run / "claims" / f"{jid}.json"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(dict(pid=os.getpid(), time=time.time()), f)
    return True


def hashes_from_npz(z):
    import numpy as np
    return {"rgb_static": hashlib.sha256(np.ascontiguousarray(z["image_static"]).tobytes()).hexdigest(),
            "rgb_gripper": hashlib.sha256(np.ascontiguousarray(z["image_wrist"]).tobytes()).hexdigest()}


def actions_to_array(actions):
    import numpy as np
    if not actions:
        return np.zeros((0, 7), np.float32)
    return np.asarray(actions, dtype=actions[0].dtype)


def scripted_repair(env, kind, sequence):
    """retreat: open the gripper and lift by 10 cm; home: retreat, then move the end-effector target pose back to the initial configuration. Same as repair_20261001."""
    import numpy as np
    rob, steps = env.robot, 0

    def act(dp=(0, 0, 0), do=(0, 0, 0), grip=1.0):
        nonlocal steps
        a = np.zeros(7, dtype=np.float64)
        a[:3], a[3:6], a[6] = np.clip(dp, -1, 1), np.clip(do, -1, 1), grip
        env.step(a); steps += 1

    for _ in range(5):
        act()
    goal_z = float(rob.target_pos[2]) + 0.10
    for _ in range(10):
        dz = goal_z - float(rob.target_pos[2])
        if dz < 1e-4:
            break
        act(dp=(0, 0, dz / (rob.max_rel_pos * rob.magic_scaling_factor_pos)))
    if kind == "home":
        goal_p = np.asarray(sequence["initial"]["robot"][:3], dtype=np.float64)
        goal_o = np.asarray(sequence["initial"]["robot"][3:6], dtype=np.float64)
        sp = rob.max_rel_pos * rob.magic_scaling_factor_pos
        so = rob.max_rel_orn * rob.magic_scaling_factor_orn
        for _ in range(100):
            dp = goal_p - np.asarray(rob.target_pos, dtype=np.float64)
            do = (goal_o - np.asarray(rob.target_orn, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi
            if np.max(np.abs(dp)) < 1e-3 and np.max(np.abs(do)) < 1e-2:
                break
            act(dp=dp / sp, do=do / so)
    for _ in range(10):
        act()
    return steps


class Runner:
    def __init__(self, run, gpu):
        import yaml
        from rho.envs.s02_calvin import CalvinAdapter
        from rho.s04_replay import Recorder
        self.run, self.gpu = run, gpu
        paths = yaml.safe_load((RHO / "configs/paths.yaml").read_text())
        settings = yaml.safe_load((RHO / "configs/calvin.yaml").read_text())
        settings.update(max_steps=MAX_STEPS, sequence_catalog=CATALOG)
        self.adapter = CalvinAdapter(paths, settings)
        self.recorder = Recorder()
        self.audits_left = 1

    def open(self):
        self.adapter.open(self.gpu, 0)

    def close(self):
        self.adapter.close()

    # ---------- basic operations ----------
    def restore(self, seq, actions, vec, imgs):
        from rho.s04_replay import restore
        return restore(self.adapter, seq, actions, vec, imgs, 0, self.recorder)

    def load_boundary(self, npz):
        import numpy as np
        with np.load(npz) as z:
            pa = z["prefix_actions"]
            prefix = [a.astype(np.float32) for a in pa]  # original execution used float32
            return prefix, z["physics"].copy(), hashes_from_npz(z)

    # ---------- job: j8 ----------
    def job_j8(self, job):
        import numpy as np
        from rho.s04_replay import fresh_env, image_hashes, physics_vector, run_task, seed_all, stable_seed
        seq = self.adapter.sequences(job["index"], 1)[0]
        if job.get("sequence_id"):
            assert seq["sequence_id"] == job["sequence_id"]
        tasks = seq["tasks"][: job["stop_after"]]
        assert tasks[-1] == job["target"]
        seed_all(stable_seed("mt-prefix", seq["sequence_id"]))
        fresh_env(self.adapter, 0, self.recorder)
        self.adapter.reset(seq)
        out = dict(job=job, reached=True, prefix=[])
        for t in tasks[:-1]:
            ok, steps = run_task(self.adapter, t, MAX_STEPS)
            out["prefix"].append(dict(task=t, success=ok, steps=steps))
            if not ok:
                out["reached"] = False
                return out
        prefix = [a.copy() for a in self.recorder.actions]
        obs = self.adapter.observe()
        vec, imgs = physics_vector(self.adapter.env), image_hashes(self.adapter.env)
        sid = f'{seq["sequence_id"]}:{len(tasks) - 1}'
        name = f"boundary_{key_of(sid)}.npz"
        np.savez_compressed(self.run / "boundaries" / name, state=obs["state"], scene_state=obs["scene_state"],
                            image_static=obs["images"]["static"], image_wrist=obs["images"]["wrist"],
                            physics=vec, prefix_actions=actions_to_array(prefix))
        reps, errs = [], []
        plan = [(j, stable_seed("mt-b", sid, j)) for j in range(REPEATS)]
        if self.audits_left > 0:
            plan.append(("audit", plan[0][1]))
        for j, s in plan:
            err, same = self.restore(seq, prefix, vec, imgs)
            errs.append(err)
            if err > 1e-6 or not same:
                out["restore_failed"] = dict(repeat=str(j), err=err, images_equal=same)
                return out
            seed_all(s)
            n0 = len(self.recorder.actions)
            ok, steps = run_task(self.adapter, job["target"], MAX_STEPS)
            dig = hashlib.sha256(actions_to_array(self.recorder.actions[n0:]).astype("float64").tobytes()).hexdigest()
            reps.append(dict(repeat=j, seed=s, success=ok, steps=steps, actions_sha256=dig))
        if self.audits_left > 0:
            self.audits_left -= 1
            out["determinism_audit"] = reps[-1]["actions_sha256"] == reps[0]["actions_sha256"] and reps[-1]["success"] == reps[0]["success"]
            reps = reps[:-1]
        out.update(sample_id=sid, parent_id=seq["parent_id"], task_a=tasks[-2], task_b=job["target"], observation=name,
                   repeats=reps, k=sum(r["success"] for r in reps), n=len(reps), restore_max_error=max(errs))
        return out

    # ---------- job: screen ----------
    def job_screen(self, job):
        import numpy as np
        from rho.s04_replay import image_hashes, physics_vector, run_task, seed_all, stable_seed
        seq = self.adapter.sequences(job["index"], 1)[0]
        assert seq["sequence_id"] == job["sequence_id"]
        prefix, vec0, img0 = self.load_boundary(job["npz"])
        out = dict(job_id=job["id"], sample_id=job["sample_id"], task_b=job["task_b"], candidates=[])
        for cand in job["candidates"]:
            rec = dict(candidate=cand)
            err, same = self.restore(seq, prefix, vec0, img0)
            if err > 1e-6 or not same:
                rec["invalid"] = f"boundary restore mismatch err={err} images_equal={same}"
                out["candidates"].append(rec); continue
            seed_all(stable_seed("vocab-screen", job["sample_id"], cand))
            n0 = len(self.recorder.actions)
            if cand in ("retreat", "home"):
                rec.update(steps=scripted_repair(self.adapter.env, cand, seq), task_success=None)
            else:
                ok, steps = run_task(self.adapter, cand, LANG_MAX)
                rec.update(steps=steps, task_success=ok)
            ra = self.recorder.actions[n0:]
            obs = self.adapter.observe()
            vec1, img1 = physics_vector(self.adapter.env), image_hashes(self.adapter.env)
            name = f"{key_of(job['sample_id'])}__{cand}.npz"
            np.savez_compressed(self.run / "screen" / name, state=obs["state"], scene_state=obs["scene_state"],
                                image_static=obs["images"]["static"], image_wrist=obs["images"]["wrist"],
                                physics=vec1, repair_actions=actions_to_array(ra))
            rec.update(npz=name, image_hashes=img1)
            out["candidates"].append(rec)
        return out

    # ---------- job: label ----------
    def job_label(self, job):
        import numpy as np
        from rho.s04_replay import run_task, seed_all, stable_seed
        seq = self.adapter.sequences(job["index"], 1)[0]
        assert seq["sequence_id"] == job["sequence_id"]
        prefix, vec0, img0 = self.load_boundary(job["npz"])
        out = dict(job_id=job["id"], sample_id=job["sample_id"], task_b=job["task_b"], arms=[])
        for cand in job["arms"]:
            rec = dict(candidate=cand)
            if cand == "none":
                actions, vec, imgs = prefix, vec0, img0
            else:
                with np.load(self.run / "screen" / job["screen_npz"][cand]) as z:
                    ra = [a for a in z["repair_actions"]]
                    vec = z["physics"].copy()
                    imgs = hashes_from_npz(z)
                actions = prefix + ra
            rollouts, errs = [], []
            for j in range(REPEATS):
                err, same = self.restore(seq, actions, vec, imgs)
                errs.append(err)
                if err > 1e-6 or not same:
                    rec["invalid"] = f"restore mismatch at repeat {j}: err={err} images_equal={same}"
                    break
                seed_all(stable_seed("vocab-b", job["sample_id"], j))   # paired seed: independent of the candidate
                ok, steps = run_task(self.adapter, job["task_b"], MAX_STEPS)
                rollouts.append(dict(repeat=j, success=ok, steps=steps))
            rec.update(rollouts=rollouts, k=sum(r["success"] for r in rollouts), n=len(rollouts),
                       restore_max_error=max(errs) if errs else None)
            out["arms"].append(rec)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    for d in ("claims", "results", "boundaries", "screen"):
        (run / d).mkdir(exist_ok=True)
    r = Runner(run, args.gpu)
    done = 0
    try:
        r.open()
        for job in jobs:
            if args.limit is not None and done >= args.limit:
                break
            jid = job["id"]
            if (run / "results" / f"{jid}.json").exists() or not claim(run, jid):
                continue
            t0 = time.time()
            out = getattr(r, "job_" + job["type"])(job)
            out["seconds"] = round(time.time() - t0, 1)
            out["gpu"] = args.gpu
            write_atomic(run / "results" / f"{jid}.json", out)
            done += 1
            if job["type"] == "j8":
                msg = f"j8 {job['target']} reached={out['reached']} k={out.get('k')}/{out.get('n')}"
            elif job["type"] == "screen":
                msg = f"screen {job['task_b']} {len(out['candidates'])} cands, invalid={sum('invalid' in c for c in out['candidates'])}"
            else:
                msg = f"label {job['task_b']} " + ", ".join(f"{a['candidate']}={a.get('k')}/{a.get('n')}" for a in out["arms"])
            print(f"[gpu{args.gpu}] {jid} {msg} {out['seconds']}s", flush=True)
        print("WORKER DONE", flush=True)
    finally:
        r.close()


if __name__ == "__main__":
    main()
