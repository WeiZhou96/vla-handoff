# -*- coding: utf-8 -*-
"""T1 inventory (read-only): scope of the B_def relabelling, file integrity, exact restore reproducibility and run-time estimate.

1. Enumerate every labelled branch (each arm of label_<id>.json) in the three labelling directories, classified as pr/other × repair/no repair;
   check that the job entry, handoff npz and post-repair npz exist and are readable; count branches and recorded steps and compare with the expected counts of 2026-10-11.
2. Statically decide whether a repair moved a block relevant to the reference quantity of the target predicate (scene_state of the handoff npz vs. the post-repair npz).
   B_def differs from the original criterion only in the reference quantity (push/lift: target block position; rotate: target block pose; push_into_drawer: whether a block is in the drawer;
   stack: contacts between blocks); when these quantities are unchanged along h→s_u, both criteria agree step by step (except at threshold edges, checked by sampling).
3. repair_20261001 and fix_20260928/runs/j8_two_task: rationale and static statistics for whether relabelling is needed.
4. A sample (default 10 repair branches) restores the handoff and post-repair states on the current software stack: physics vector (original tolerance 1e-6, also reports exact equality),
   bytes of both camera images, 15/24-dim observations; records restore time and fits "time = c0 + c1 × number of replayed actions".
5. Estimate run time for the default relabelling scope (plus a comparison estimate for "full relabelling of non-pr branches").
Read-only: no existing file is modified; only <this directory>/inventory/ is written.
Usage: python -u -B inventory.py [--gpu 2] [--sample 10] [--no-env] [--sec-per-step 0.05] [--procs 12]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, collections, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(f"{RUN_ROOT}/relabel_bdef_20261011")
CV = Path(f"{RUN_ROOT}")
LABEL_DIRS = {
    "multitask_vocab_20261002": ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"),
    "gating_vocab_20261003": ("jobs_label.json",),
    "newtasks_20261004": ("jobs_nt3_label.json",),
}
LABEL_WORKER = CV / "multitask_vocab_20261002" / "worker.py"
J8_TWO_TASK = CV / "fix_20260928" / "runs" / "j8_two_task"
REPAIR_HAND = CV / "repair_20261001"
PR_TASKS = frozenset({"push_red_block_right", "push_blue_block_right", "push_pink_block_right",
                      "rotate_pink_block_right", "rotate_red_block_left"})
EXPECT = {  # expected counts as of 2026-10-11 (number of branches, total steps over 8 executions)
    "multitask_vocab_20261002": {"pr_rep": (714, 1738150), "pr_none": (119, None), "other_rep": (372, 665924), "other_none": (62, None)},
    "gating_vocab_20261003": {"pr_rep": (540, 1165262), "pr_none": (90, None), "other_rep": (360, 466234), "other_none": (60, None)},
    "newtasks_20261004": {"pr_rep": (542, 942986), "pr_none": (81, None), "other_rep": (1089, 1701385), "other_none": (158, None)},
}
MAX_STEPS, REPEATS, TOL = 360, 8, 1e-6
BLOCK = {"red": slice(6, 12), "blue": slice(12, 18), "pink": slice(18, 24)}   # scene_state: xyz + euler per block
POS_TOL, ORN_TOL = 1e-4, 1e-3
SCOPE = dict(none_frac=0.10, unmoved_frac=0.10, verify_frac=0.10)
CLASSES = ("pr_rep", "pr_none", "other_rep", "other_none")


def rroot():
    return ROOT.resolve()


def _jd(o):
    try:
        return o.item()
    except Exception:
        return str(o)


def out_path(rel):
    """Output path inside this directory; paths outside it are refused (HANDOFF_RUN_ROOT may be a symlink, so resolved paths are compared)."""
    p = rroot() / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    rp = p.parent.resolve() / p.name
    if rroot() not in rp.parents:
        raise SystemExit(f"refusing to write outside this directory: {rp}")
    return rp


def write_json(rel, data):
    p = out_path(rel)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_jd))
    os.replace(tmp, p)
    return p


def key_of(s):
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def frac(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 2 ** 32


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resolve_npz(run, p):
    q = Path(p)
    if q.is_absolute():
        return q
    for c in (run / q, run / "boundaries" / q.name):
        if c.exists():
            return c
    return run / q


# ---------------- enumeration ----------------
def enumerate_branches():
    branches, problems, files = [], [], []
    for dname, jfs in LABEL_DIRS.items():
        run = (CV / dname).resolve()
        jobs = {}
        for jf in jfs:
            p = run / jf
            if not p.exists():
                problems.append(dict(dir=dname, what="job file missing", file=jf))
                continue
            for j in json.loads(p.read_text())["jobs"]:
                if j.get("type") != "label":
                    continue
                if j["id"] in jobs:
                    problems.append(dict(dir=dname, what="duplicate job id", id=j["id"]))
                jobs[j["id"]] = dict(j, _jobfile=jf)
        for f in sorted((run / "results").glob("label_*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception as e:
                problems.append(dict(dir=dname, what="unreadable result", file=f.name, error=repr(e)))
                continue
            job = jobs.get(d.get("job_id"))
            files.append(dict(dir=dname, file=f.name, job_id=d.get("job_id"), task_b=d.get("task_b"), has_job=job is not None,
                              n_arms=len(d.get("arms", [])), seconds=d.get("seconds"),
                              steps=sum(int(r["steps"]) for a in d.get("arms", []) for r in a.get("rollouts", []))))
            if job is None:
                problems.append(dict(dir=dname, what="no job entry for result", file=f.name))
                continue
            if job["sample_id"] != d["sample_id"] or job["task_b"] != d["task_b"]:
                problems.append(dict(dir=dname, what="job/result sample or task mismatch", file=f.name))
                continue
            for a in d["arms"]:
                c = a["candidate"]
                ro = a.get("rollouts", [])
                valid = ("invalid" not in a and a.get("n") == REPEATS and len(ro) == REPEATS
                         and [int(r["repeat"]) for r in ro] == list(range(REPEATS)))
                b = dict(dir=dname, run=str(run), label_file=f.name, job_id=d["job_id"], jobfile=job["_jobfile"],
                         sample_id=d["sample_id"], index=job["index"], sequence_id=job["sequence_id"],
                         parent_id=job.get("parent_id"), task_b=d["task_b"], cand=c, pr=d["task_b"] in PR_TASKS,
                         repair=c != "none", valid=valid, in_job_arms=c in job.get("arms", []),
                         stored=[dict(repeat=int(r["repeat"]), success=bool(r["success"]), steps=int(r["steps"])) for r in ro],
                         k=a.get("k"), n=a.get("n"), restore_max_error=a.get("restore_max_error"),
                         npz=str(resolve_npz(run, job["npz"])), screen=None)
                if c != "none":
                    sn = (job.get("screen_npz") or {}).get(c)
                    if sn is None:
                        problems.append(dict(dir=dname, what="no screen_npz for candidate", file=f.name, cand=c))
                        b["valid"] = False
                    else:
                        b["screen"] = str(run / "screen" / sn)
                b["cls"] = ("pr" if b["pr"] else "other") + ("_rep" if b["repair"] else "_none")
                branches.append(b)
    return branches, problems, files


# ---------------- static movement test ----------------
def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def target_blocks(task):
    for c in ("red", "blue", "pink"):
        if f"_{c}_" in f"_{task}_":
            return [c]
    return list(BLOCK)          # push_into_drawer, stack_block: any block


def movement(task, sc_h, sc_u):
    sc_h, sc_u = np.asarray(sc_h, float), np.asarray(sc_u, float)
    blocks, dpos, dorn = {}, 0.0, 0.0
    for c in target_blocks(task):
        a, b = sc_h[BLOCK[c]], sc_u[BLOCK[c]]
        dp = float(np.max(np.abs(b[:3] - a[:3])))
        do = float(np.max(np.abs(wrap(b[3:] - a[3:]))))
        blocks[c] = [dp, do]
        dpos, dorn = max(dpos, dp), max(dorn, do)
    d_doors = float(np.max(np.abs(sc_u[:2] - sc_h[:2])))
    moved = dpos > POS_TOL or dorn > ORN_TOL or (task == "push_into_drawer" and d_doors > POS_TOL)
    return dict(blocks=blocks, dpos=dpos, dorn=dorn, d_doors=d_doors, moved=bool(moved))


def read_npz(p, keys):
    with np.load(p) as z:
        miss = [k for k in keys if k not in z.files]
        if miss:
            raise KeyError(f"missing keys {miss}")
        return {k: z[k].copy() for k in keys}


def check_files(branches, verbose=True):
    bcache = {}
    t0 = time.time()
    for i, b in enumerate(branches):
        prob = []
        bp = b["npz"]
        if bp not in bcache:
            try:
                z = read_npz(bp, ("physics", "prefix_actions", "image_static", "image_wrist", "state", "scene_state"))
                ok = (z["physics"].ndim == 1 and bool(np.isfinite(z["physics"]).all()) and z["prefix_actions"].ndim == 2
                      and z["prefix_actions"].shape[1] == 7 and z["image_static"].dtype == np.uint8
                      and z["image_wrist"].dtype == np.uint8 and z["state"].shape == (15,) and z["scene_state"].shape == (24,))
                bcache[bp] = dict(ok=bool(ok), scene=z["scene_state"], plen=int(len(z["prefix_actions"])),
                                  nphys=int(z["physics"].size), error=None if ok else "bad content")
            except Exception as e:
                bcache[bp] = dict(ok=False, error=repr(e))
        bc = bcache[bp]
        b["prefix_len"] = bc.get("plen")
        if not bc["ok"]:
            prob.append("boundary: " + str(bc["error"]))
        if b["repair"] and b["screen"]:
            try:
                z = read_npz(b["screen"], ("state", "scene_state", "image_static", "image_wrist", "physics", "repair_actions"))
                ok = (z["state"].shape == (15,) and z["scene_state"].shape == (24,) and z["image_static"].shape == (200, 200, 3)
                      and z["image_wrist"].shape == (84, 84, 3) and z["image_static"].dtype == np.uint8
                      and z["physics"].ndim == 1 and bool(np.isfinite(z["physics"]).all())
                      and z["repair_actions"].ndim == 2 and z["repair_actions"].shape[1] == 7)
                if bc.get("nphys") is not None and z["physics"].size != bc["nphys"]:
                    ok = False
                b["repair_len"] = int(len(z["repair_actions"]))
                b["repair_dtype"] = str(z["repair_actions"].dtype)
                if not ok:
                    prob.append("screen: bad content")
                elif bc["ok"]:
                    b["move"] = movement(b["task_b"], bc["scene"], z["scene_state"])
            except Exception as e:
                prob.append("screen: " + repr(e))
        b["file_problem"] = prob or None
        if verbose and (i + 1) % 1000 == 0:
            print(f"  files checked {i + 1}/{len(branches)} ({time.time() - t0:.0f}s)", flush=True)
    return branches


# ---------------- label sources outside the three directories ----------------
def j8_check():
    n, per = 0, collections.Counter()
    for g in sorted(J8_TWO_TASK.glob("gpu*/labels.jsonl")):
        for line in g.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            per[r.get("task_b")] += 1
    return dict(records=n, per_task=dict(per), relabel=False,
                reason="no repair: the target starts at the handoff, start_info = handoff, hybrid_start substitutes identical values, so B_def == A at every step.")


def hand_check():
    out = dict(dir=str(REPAIR_HAND), exists=REPAIR_HAND.exists(), per_task={}, errors=[], relabel=False,
               reason=("targets are only push_into_drawer and lift_red_block_slider; 0 divergences on these tasks in the closed-loop audit; "
                       "B_def and A can differ only if the repair moved the relevant block; labelled by another worker without a "
                       "no-repair branch, so this worker cannot replay it as is. Static movement counts follow; moved branches are handled in a later stage."))
    if not REPAIR_HAND.exists():
        return out
    bmap = {}
    for g in sorted(J8_TWO_TASK.glob("gpu*/labels.jsonl")):
        for line in g.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("observation"):
                bmap[r["sample_id"]] = g.parent / "boundaries" / r["observation"]
    by_name = collections.defaultdict(list)
    for p in REPAIR_HAND.rglob("*.npz"):
        by_name[p.name].append(p)
    per = collections.defaultdict(lambda: dict(n=0, moved=0, unknown=0, max_dpos=0.0))
    for f in sorted((REPAIR_HAND / "results").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            out["errors"].append(f"{f.name}: {e!r}")
            continue
        sid, task = d.get("sample_id"), d.get("task_b")
        for r in d.get("repairs", []):
            if r.get("invalid") or r.get("n") != 8:
                continue
            e = per[task]
            e["n"] += 1
            try:
                cands = by_name.get(Path(r["repaired_npz"]).name, [])
                if len(cands) != 1:
                    raise FileNotFoundError(f"{r['repaired_npz']}: {len(cands)} matches")
                with np.load(bmap[sid]) as z:
                    sh = z["scene_state"].copy()
                with np.load(cands[0]) as z:
                    su = z["scene_state"].copy()
                m = movement(task, sh, su)
                e["moved"] += int(m["moved"])
                e["max_dpos"] = max(e["max_dpos"], m["dpos"])
            except Exception as ex:
                e["unknown"] += 1
                if len(out["errors"]) < 10:
                    out["errors"].append(repr(ex))
    out["per_task"] = {k: dict(v) for k, v in per.items()}
    return out


# ---------------- relabelling scope and jobs ----------------
def make_job(b, role, verify_b):
    u = f'{b["dir"]}|{b["label_file"]}|{b["cand"]}'
    steps = sum(r["steps"] for r in b["stored"])
    return dict(id="rb_" + key_of(u), type="relabel", role=role, verify_b=bool(verify_b),
                dir=b["dir"], run=b["run"], label_file=b["label_file"], job_id=b["job_id"], sample_id=b["sample_id"],
                index=b["index"], sequence_id=b["sequence_id"], parent_id=b["parent_id"], task_b=b["task_b"], cand=b["cand"],
                pr=b["pr"], npz=b["npz"], screen=b["screen"], stored=b["stored"], stored_k=int(b["k"]),
                stored_restore_max_error=b["restore_max_error"], move=b.get("move"),
                prefix_len=b.get("prefix_len") or 0, repair_len=b.get("repair_len") or 0,
                stored_steps=steps, est=steps + MAX_STEPS * int(b["k"] or 0))


def select_scope(branches, none_frac, unmoved_frac, verify_frac):
    jobs = []
    for b in branches:
        if not b["valid"] or b.get("file_problem"):
            continue
        u = f'{b["dir"]}|{b["label_file"]}|{b["cand"]}'
        moved = (b.get("move") or {}).get("moved", True)     # unknown is treated as "moved" (conservative)
        role = None
        if b["cls"] == "pr_rep":
            role = "pr_rep"
        elif b["cls"] == "pr_none":
            if frac(f'none|{b["dir"]}|{b["label_file"]}') < none_frac:
                role = "none_identity"
        elif b["cls"] == "other_rep":
            if moved:
                role = "other_moved"
            elif frac("unmoved|" + u) < unmoved_frac:
                role = "other_unmoved_sample"
        if role is None:
            continue
        jobs.append(make_job(b, role, role != "none_identity" and frac("verify|" + u) < verify_frac))
    return jobs


def estimate(jobs, sec_per_step, procs, c0, c1):
    """Three tiers: lower = only the verify sample re-runs B; central = every A success on a moved branch re-runs B for the full 360 steps; upper = every A success re-runs for the full 360 steps."""
    if not jobs:
        return dict(n_jobs=0)
    rep = [j for j in jobs if j["role"] != "none_identity"]
    a_steps = sum(j["stored_steps"] for j in jobs)
    avg = a_steps / (REPEATS * len(jobs))
    moved = lambda j: (j.get("move") or {}).get("moved", True)
    rest = lambda j, nb: (1 + REPEATS + nb) * (c0 + c1 * (j["prefix_len"] + j["repair_len"]))
    out = dict(n_jobs=len(jobs), executions=REPEATS * len(jobs), steps_A=a_steps,
               A_successes=sum(j["stored_k"] for j in rep), A_successes_moved=sum(j["stored_k"] for j in rep if moved(j)),
               verify_jobs=sum(j["verify_b"] for j in rep), sec_per_step=sec_per_step, restore_fit=[c0, c1], procs=procs)
    for name in ("lower", "central", "upper"):
        steps = a_steps
        rs = 0.0
        for j in jobs:
            nv = REPEATS if j.get("verify_b") else 0
            if name == "lower" or j["role"] == "none_identity":
                nb, sb = nv, nv * avg
            elif name == "central":
                nb = nv + (j["stored_k"] if moved(j) else 0)
                sb = nv * avg + (j["stored_k"] if moved(j) else 0) * MAX_STEPS
            else:
                nb, sb = nv + j["stored_k"], (nv + j["stored_k"]) * MAX_STEPS
            steps += sb
            rs += rest(j, nb)
        sec = steps * sec_per_step + rs
        out[name] = dict(steps=int(steps), restore_hours=round(rs / 3600, 1), process_hours=round(sec / 3600, 1),
                         wall_hours=round(sec / 3600 / procs, 1))
    return out


# ---------------- exact restore reproducibility (requires a GPU) ----------------
def load_label_worker():
    spec = importlib.util.spec_from_file_location("label_worker_vocab", str(LABEL_WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def determinism(branches, n, gpu):
    cand = [b for b in branches if b["repair"] and b["valid"] and not b.get("file_problem")]
    by = collections.defaultdict(list)
    for b in sorted(cand, key=lambda b: frac(f'inv|{b["dir"]}|{b["label_file"]}|{b["cand"]}')):
        by[b["dir"]].append(b)
    pick, i = [], 0
    while len(pick) < n and any(len(v) > i for v in by.values()):
        for d in sorted(by):
            if len(by[d]) > i and len(pick) < n:
                pick.append(by[d][i])
        i += 1
    lw = load_label_worker()
    from rho.s04_replay import physics_vector
    r = lw.Runner(rroot(), gpu)
    recs = []
    try:
        r.open()
        for b in pick:
            seq = r.adapter.sequences(b["index"], 1)[0]
            rec = dict(dir=b["dir"], label_file=b["label_file"], cand=b["cand"], task_b=b["task_b"],
                       seq_ok=seq["sequence_id"] == b["sequence_id"])
            prefix, vec0, img0 = r.load_boundary(b["npz"])
            t0 = time.time()
            e0, s0 = r.restore(seq, prefix, vec0, img0)
            rec.update(t_boundary=time.time() - t0, len_boundary=len(prefix), boundary_err=float(e0), boundary_images_equal=bool(s0),
                       boundary_exact=bool(np.array_equal(physics_vector(r.adapter.env), vec0)))
            with np.load(b["screen"]) as z:
                ra = [a for a in z["repair_actions"]]
                vec, imgs = z["physics"].copy(), lw.hashes_from_npz(z)
                st, sc = z["state"].copy(), z["scene_state"].copy()
            t0 = time.time()
            e1, s1 = r.restore(seq, prefix + ra, vec, imgs)
            rec.update(t_post=time.time() - t0, len_post=len(prefix) + len(ra), post_err=float(e1), post_images_equal=bool(s1),
                       post_exact=bool(np.array_equal(physics_vector(r.adapter.env), vec)))
            obs = r.adapter.observe()
            rec.update(state_equal=bool(np.array_equal(obs["state"], st)), scene_equal=bool(np.array_equal(obs["scene_state"], sc)))
            rec["pass"] = bool(rec["seq_ok"] and e0 <= TOL and s0 and e1 <= TOL and s1)
            recs.append(rec)
            print(f"  restore {b['task_b']} {b['cand']}: boundary err={e0:.2e} img={s0} exact={rec['boundary_exact']} | "
                  f"post err={e1:.2e} img={s1} exact={rec['post_exact']} obs={rec['state_equal']}/{rec['scene_equal']} "
                  f"t={rec['t_boundary']:.1f}/{rec['t_post']:.1f}s", flush=True)
    finally:
        r.close()
    return recs


def restore_fit(recs):
    x = [r[k] for r in recs for k in ("len_boundary", "len_post")]
    y = [r[k] for r in recs for k in ("t_boundary", "t_post")]
    if not x:
        return 2.0, 0.004, "assumed"
    if len(set(x)) >= 3:
        c1, c0 = np.polyfit(x, y, 1)
        if c1 > 0 and c0 >= 0:
            return float(c0), float(c1), "fit"
    return 0.0, float(sum(y) / max(1, sum(x))), "ratio"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--no-env", action="store_true")
    ap.add_argument("--sec-per-step", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.gpu not in (2, 3, 4, 5):
        raise SystemExit("only GPUs 2-5 are allowed")
    t_start = time.time()
    print("enumerating labelled branches...", flush=True)
    branches, problems, files = enumerate_branches()
    print(f"  {len(files)} label files, {len(branches)} branches; checking npz files...", flush=True)
    check_files(branches)
    rep = dict(created=time.strftime("%F %T"), root=str(rroot()), label_worker_sha256=sha_file(LABEL_WORKER),
               label_files=len(files), branches=len(branches), problems=problems[:200], n_problems=len(problems))
    # counts
    print(f"\n{'dir|class':42s} {'n':>5s} {'valid':>5s} {'filesOK':>7s} {'steps(all)':>11s} {'k_sum':>6s} {'expect':>18s} ok", flush=True)
    counts = {}
    for d in LABEL_DIRS:
        for cls in CLASSES:
            bs = [b for b in branches if b["dir"] == d and b["cls"] == cls]
            vs = [b for b in bs if b["valid"]]
            steps_all = sum(sum(r["steps"] for r in b["stored"]) for b in bs)
            en, es = EXPECT[d][cls]
            c = dict(n=len(bs), valid=len(vs), files_ok=sum(1 for b in vs if not b.get("file_problem")), steps_all=steps_all,
                     k_sum=sum(int(b["k"] or 0) for b in vs), expect_n=en, expect_steps=es,
                     matches_expect=bool(len(bs) == en and (es is None or steps_all == es)))
            counts[f"{d}|{cls}"] = c
            print(f"{d + '|' + cls:42s} {c['n']:5d} {c['valid']:5d} {c['files_ok']:7d} {steps_all:11d} {c['k_sum']:6d} "
                  f"{str((en, es)):>18s} {c['matches_expect']}", flush=True)
    rep["counts"] = counts
    tot = dict(pr_rep_branches=sum(counts[f"{d}|pr_rep"]["n"] for d in LABEL_DIRS),
               pr_rep_steps=sum(counts[f"{d}|pr_rep"]["steps_all"] for d in LABEL_DIRS))
    rep["totals"] = tot
    print(f"pr_rep total {tot['pr_rep_branches']} branches / {tot['pr_rep_steps']} steps (expected 1796 / 3846398)", flush=True)
    rep["file_problems"] = [dict(dir=b["dir"], file=b["label_file"], cand=b["cand"], problem=b["file_problem"])
                            for b in branches if b.get("file_problem")][:200]
    rep["n_file_problems"] = sum(1 for b in branches if b.get("file_problem"))
    rep["n_not_in_job_arms"] = sum(1 for b in branches if not b["in_job_arms"])
    print(f"file problems {rep['n_file_problems']}, job/result-level problems {len(problems)}, arms not in job['arms'] {rep['n_not_in_job_arms']}", flush=True)
    for p in problems[:10]:
        print("  ", p, flush=True)
    # all-inclusive rate of the original label processes
    rate = {}
    for d in LABEL_DIRS:
        fs = [f for f in files if f["dir"] == d and f.get("seconds")]
        if fs:
            rate[d] = sum(f["seconds"] for f in fs) / max(1, sum(f["steps"] for f in fs))
    rep["orig_label_sec_per_step_all_in"] = rate
    print("original label processes (including restore) s/step:", {k: round(v, 4) for k, v in rate.items()}, flush=True)
    # movement statistics
    mv = collections.defaultdict(lambda: dict(n=0, moved=0, unknown=0))
    for b in branches:
        if b["repair"] and b["valid"]:
            e = mv[f'{b["dir"]}|{b["task_b"]}']
            e["n"] += 1
            if b.get("move") is None:
                e["unknown"] += 1
            else:
                e["moved"] += int(b["move"]["moved"])
    rep["movement"] = dict(mv)
    print("\nbranches whose repair moved the reference block (dir|task: moved/n, unknown):", flush=True)
    for k in sorted(mv):
        print(f"  {k:55s} {mv[k]['moved']:5d}/{mv[k]['n']:5d}  unknown={mv[k]['unknown']}", flush=True)
    rep["j8_two_task"] = j8_check()
    rep["repair_20261001"] = hand_check()
    print("\nj8_two_task:", rep["j8_two_task"]["records"], "records,", rep["j8_two_task"]["reason"], flush=True)
    print("repair_20261001:", json.dumps(rep["repair_20261001"]["per_task"], ensure_ascii=False), flush=True)
    if rep["repair_20261001"]["errors"]:
        print("  (some could not be determined)", rep["repair_20261001"]["errors"][:3], flush=True)
    # exact restore reproducibility
    recs = []
    if not a.no_env and a.sample > 0:
        print(f"\non GPU {a.gpu}: checking restore on {a.sample} sampled repair branches...", flush=True)
        recs = determinism(branches, a.sample, a.gpu)
    c0, c1, how = restore_fit(recs)
    rep["determinism"] = dict(n=len(recs), all_pass=bool(recs) and all(r["pass"] for r in recs),
                              all_exact=bool(recs) and all(r["boundary_exact"] and r["post_exact"] for r in recs),
                              all_obs_equal=bool(recs) and all(r["state_equal"] and r["scene_equal"] for r in recs), records=recs)
    rep["restore_time_model"] = dict(c0=c0, c1=c1, how=how)
    # estimate
    jobs = select_scope(branches, **SCOPE)
    rep["scope_default"] = dict(params=SCOPE, roles=dict(collections.Counter(j["role"] for j in jobs)),
                                estimate=estimate(jobs, a.sec_per_step, a.procs, c0, c1))
    ids = {j["id"] for j in jobs}
    extra = [make_job(b, "other_unmoved_full", False) for b in branches
             if b["cls"] == "other_rep" and b["valid"] and not b.get("file_problem")
             and "rb_" + key_of(f'{b["dir"]}|{b["label_file"]}|{b["cand"]}') not in ids]
    rep["scope_all_other"] = dict(extra_jobs=len(extra), estimate=estimate(jobs + extra, a.sec_per_step, a.procs, c0, c1))
    print(f"\nrestore time model ({how}): t = {c0:.2f} + {c1:.5f} × number of replayed actions", flush=True)
    print("default scope:", rep["scope_default"]["roles"], flush=True)
    for name in ("lower", "central", "upper"):
        e = rep["scope_default"]["estimate"][name]
        print(f"  {name:7s} steps={e['steps']:>9d} restore={e['restore_hours']:6.1f}h process={e['process_hours']:6.1f}h "
              f"wall({a.procs} procs)={e['wall_hours']:5.1f}h", flush=True)
    e = rep["scope_all_other"]["estimate"]["central"]
    print(f"comparison: full relabelling of non-pr branches ({len(extra)} additional branches) central wall={e['wall_hours']}h", flush=True)
    rep["determinism_verdict"] = rep["determinism"]["all_pass"] if recs else None
    rep["seconds"] = round(time.time() - t_start, 1)
    write_json("inventory/inventory.json", rep)
    write_json("inventory/branches.json", branches)
    print(f"\nrestore check: {'all passed' if rep['determinism']['all_pass'] else ('not run' if not recs else 'some failed')}; "
          f"exactly equal={rep['determinism']['all_exact']}; observations equal={rep['determinism']['all_obs_equal']}", flush=True)
    print("wrote", out_path("inventory/inventory.json"), flush=True)


if __name__ == "__main__":
    main()
