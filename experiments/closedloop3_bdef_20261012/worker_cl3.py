# -*- coding: utf-8 -*-
"""Worker process of the round-3 stage-3 (B_def) closed-loop confirmation experiment (closedloop3_bdef_20261012).

One job = one chain of the locked jobs_cl3.json; the 7 arms each start in turn from the same perturbed initial state (worker_cl.perturb, seed job["perturb_seed"]);
each arm creates a new environment, resets, settles for 30 steps and records a hash of the initial physics state; the hashes must agree across arms (otherwise the whole chain is re-run once in this process, and if they still differ it is recorded as a pairing failure).
Each arm calls the locked fc_worker.run_chain(mode="B") directly (file sha256 checked before loading):
  every attempt at a target subtask ends and counts as success when B_def first holds (relative to the handoff; on retry the pattern condition is taken at the start of that attempt); retry and chain stopping are decided by B_def;
  criteria A and B_orc are recorded step by step (first step at which they hold, value at attempt end) and do not affect execution; non-target subtasks use the original criterion (no repair and no retry before them).
  Each target handoff records pose and physics hash at the handoff / after repair / at start and end of each attempt, the repair choice, q(none), u*, q(u*);
  this process additionally writes all q values of the 37 options at that handoff for gate arms to tasks[*]["q_all"] (only the return values of the scoring calls are recorded; no decision is changed).
Seeds: sseed("cl-repair", "cl3_<i>", pos, u), sseed("cl-attempt", "cl3_<i>", pos, att) (unchanged from fc_worker).
Arm mapping (prereg §5, locked_config.json["arm_spec"], checked item by item at start-up):
  none→none; retry→retry; prior→prior (argmax of the v4 prior table, excluding none); benefit_B→benefit (v4 M7, λ, δ_B, c̄);
  benefit_noretry_B→benefit_noretry; noloc_B→noloc (v4 M7n); benefit_A→benefit (v3_all M7, frozen round-2 λ/δ/c̄).
Start-up (before the first chain; if any step fails the worker refuses to run and writes a marker to refused/; processes started later exit as soon as they see the marker):
  1 sha256 of locked files matches lock_record.json; 2 load fc_worker, worker_cl2, worker_cl; 3 build Brain (before adapter.open);
  4 arm configuration check; 5 fixture_inputs.npz → fixture_expected.json (q within 1e-9, u*/repair identical; prior repair identical);
  6 files of loaded modules (fc_worker, worker_cl2, worker_cl, worker, features_patch, benefit_lib*, etc.) match the locked paths and sha;
  7 sha256 of worker_cl3.py and analyze_cl3.py in lock_addendum.json matches the files (not checked with --selftest).
Logs never print per-arm results. Writes only to this directory. Crashes: after 3 accumulated Python exceptions a result with error is written (excluded per prereg §10); otherwise the claim is released and the process exits.
Usage: python -u -B worker_cl3.py --run <this directory> --jobs <locked jobs_cl3.json> --gpu <2..5> [--out results|results_smoke] [--ids a,b] [--limit n]
       python -B worker_cl3.py --run <this directory> --jobs <...> --selftest      (no simulation, no files written)
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, hashlib, importlib, importlib.util, json, os, platform, resource, socket, sys, time, traceback
from pathlib import Path

sys.dont_write_bytecode = True
# Same as train_benefit_v4.py / lock_config3.py: deserialising the models requires benefit_lib_v2
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")

NEW = Path(f"{RUN_ROOT}/closedloop3_bdef_20261012")
LOCKDIR = Path(f"{RUN_ROOT}/benefit_model_bdef_20261011")
HERE = Path(__file__).resolve().parent
LOCKED_CONFIG_SHA256 = "7a0c297dfdd7a2869d3ebc1a4d94fdc6714589e238665d11ab08ee8bf32dfa2e"
PREREG_SHA256 = "4c1eaba310383dfba9505062c0a17c045b836259b605b13848e43f5001cf90d5"
FC_WORKER_SHA256 = "5e460ecd83682e54ba86dc173cc2f80168bd19557fe395a8479f1e38ad2ca695"
ARMS = ["none", "retry", "prior", "benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A"]
FC_ARM = dict(none="none", retry="retry", prior="prior", benefit_B="benefit", benefit_noretry_B="benefit_noretry",
              noloc_B="noloc", benefit_A="benefit")
GATE_ARMS = ("benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A")
OUTS = ("results", "results_smoke")
FIX_TOL = 1e-9
LOCAL = {"jobs_cl3": "jobs_cl3.json", "model_v4": "models/benefit_models_v4_all.joblib",
         "fixture_inputs": "analysis/fixture_inputs.npz", "fixture_expected": "analysis/fixture_expected.json",
         "round2_config_frozen_copy": "config_benefit_A_round2_frozen.json", "prereg_round3": "prereg_round3.md"}
PROV = {"fc_worker": "fc_worker_modeB", "worker_cl2": "worker_cl2", "worker_cl": "worker_cl_perturb",
        "worker": "worker_scripted_repair", "features_patch": "lib_features_patch", "object_locator": "lib_object_locator",
        "benefit_lib": "lib_benefit_lib", "benefit_lib_v2": "lib_benefit_lib_v2", "benefit_lib_v3": "lib_benefit_lib_v3",
        "evaluate_cv": "lib_evaluate_cv"}
REQUIRED_MODS = ("fc_worker", "worker_cl2", "worker_cl", "worker", "features_patch")


class Refuse(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Refuse(msg)


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def rp(p):
    return Path(p).resolve()


def W(p):
    q, root = rp(p), rp(NEW)
    if q != root and root not in q.parents:
        raise RuntimeError(f"refusing to write outside {root}: {q}")
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def write_json(p, obj):
    q = W(p)
    tmp = q.with_name(f"{q.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, q)


def refused_present():
    d = NEW / "refused"
    return d.is_dir() and any(d.iterdir())


# ------------------------------------------------------------------ locked files
def verify_lock():
    lc_p, lr_p = LOCKDIR / "locked_config.json", LOCKDIR / "lock_record.json"
    need(lc_p.is_file() and lr_p.is_file(), f"missing lock files {lc_p} / {lr_p}")
    s = sha(lc_p)
    need(s == LOCKED_CONFIG_SHA256, f"locked_config.json sha256 {s} ≠ {LOCKED_CONFIG_SHA256}")
    rec = json.loads(lr_p.read_text())
    need(rec.get("locked_config_sha256") == s, "locked_config_sha256 in lock_record.json does not match the file")
    need(rec.get("prereg_sha256") == PREREG_SHA256, "prereg_sha256 in lock_record.json does not match")
    pf = (LOCKDIR / "prereg_round3.sha256").read_text().split()
    need(bool(pf) and pf[0] == PREREG_SHA256, "prereg_round3.sha256 does not match")
    cfg = json.loads(lc_p.read_text())
    S = cfg["sha256"]
    need(set(S) == set(rec["files"]), "file lists of locked_config.json and lock_record.json differ")
    checked = {}
    for k, ent in sorted(S.items()):
        need(rec["files"][k] == ent, f"{k}: locked_config and lock_record entries differ")
        if ent["sha256"] is None:
            checked[k] = None
            continue
        p = Path(ent["path"])
        need(p.is_file(), f"{k}: missing file {p}")
        h = sha(p)
        need(h == ent["sha256"], f"{k}: sha256 {h} ≠ locked value {ent['sha256']} ({p})")
        checked[k] = h
    need(S["prereg_round3"]["sha256"] == PREREG_SHA256, "prereg sha does not match")
    need(cfg["criterion"]["name"] == "B_def", "criterion name is not B_def")
    need(cfg["criterion"]["fc_worker_sha256"] == FC_WORKER_SHA256 == S["fc_worker_modeB"]["sha256"], "fc_worker sha does not match")
    need(rp(cfg["criterion"]["fc_worker"]) == rp(S["fc_worker_modeB"]["path"]), "fc_worker path does not match")
    L = rp(LOCKDIR)
    for k, rel in LOCAL.items():
        need(rp(S[k]["path"]) == L / rel, f"{k}: path {S[k]['path']} is not in the lock directory {L}")
    need(rp(cfg["jobs_file"]) == L / "jobs_cl3.json", "jobs_file path does not match")
    need(rp(cfg["model_files"]["main"]) == rp(S["model_v4"]["path"]), "main model path does not match")
    need(rp(cfg["model_files"]["A"]) == rp(S["model_v3_all_benefit_A"]["path"]), "A model path does not match")
    need(rp(cfg["benefit_A"]["frozen_config"]) == rp(S["round2_config_frozen_copy"]["path"]), "benefit_A frozen config path does not match")
    need(S["round2_config_frozen_copy"]["sha256"] == S["round2_config_original"]["sha256"], "frozen copy differs from the original round-2 config")
    return cfg, rec, checked


def load_jobs(cfg):
    J = json.loads(Path(cfg["jobs_file"]).read_text())
    jobs = J["jobs"]
    need(len(jobs) == cfg["n_chains"] == J.get("n_total", len(jobs)), "number of jobs does not match the locked configuration")
    ids = [j["id"] for j in jobs]
    need(len(set(ids)) == len(ids), "duplicate job id")
    for j in jobs:
        need(j["id"] == f"cl3_{j['index']}", f"{j['id']}: id is not cl3_<index>")
        need(len(j["tasks"]) == 5 and all(k in j for k in ("base", "perturb_seed", "stratum")), f"{j['id']}: missing fields")
    return jobs


def verify_addendum():
    p = NEW / "lock_addendum.json"
    need(p.is_file(), "lock_addendum.json does not exist: run bash run.sh addendum first")
    A = json.loads(p.read_text())
    for f in ("worker_cl3.py", "analyze_cl3.py"):
        need(A.get("files", {}).get(f, {}).get("sha256") == sha(NEW / f), f"sha256 of {f} does not match lock_addendum.json")
    need(A.get("locked_config_sha256") == LOCKED_CONFIG_SHA256, "locked_config_sha256 in lock_addendum.json does not match")
    return dict(sha256=sha(p), time=A.get("time"), analyze_sha256=A["files"]["analyze_cl3.py"]["sha256"])


# ------------------------------------------------------------------ modules, Brain, configuration checks
def load_modules(cfg):
    need("fc_worker" not in sys.modules, "fc_worker has already been imported")
    path = rp(cfg["criterion"]["fc_worker"])
    spec = importlib.util.spec_from_file_location("fc_worker", str(path))
    fcw = importlib.util.module_from_spec(spec)
    sys.modules["fc_worker"] = fcw
    spec.loader.exec_module(fcw)          # its sys.path setup and import of worker_cl2 are the same as in rounds 1 and 2
    import worker_cl2
    need(fcw.Brain is worker_cl2.Brain, "Brain of fc_worker and worker_cl2 is not the same class")
    from worker_cl import perturb
    return fcw, perturb


def build_brains(cfg, fcw):
    fz = json.loads(Path(cfg["benefit_A"]["frozen_config"]).read_text())
    base = dict(tau_risk=None, tau_retry=None, targets=list(cfg["targets"]))
    cfg_B = dict(base, lam=cfg["lam"], delta_benefit=cfg["delta_benefit"], cbar=cfg["cbar"], cbar_global=cfg["cbar_global"],
                 model_file=cfg["model_files"]["main"],
                 arm_models={"benefit": ["main", "M7"], "benefit_noretry": ["main", "M7"], "noloc": ["main", "M7n"]})
    cfg_A = dict(base, lam=fz["lam"], delta_benefit=fz["delta_benefit"], cbar=fz["cbar"], cbar_global=fz["cbar_global"],
                 model_file=cfg["model_files"]["A"], arm_models={"benefit": ["main", "M7"]})
    bB, bA = fcw.Brain(cfg_B), fcw.Brain(cfg_A)
    arm_brain = {arm: ((bA, cfg_A) if arm == "benefit_A" else (bB, cfg_B)) for arm in ARMS}
    return arm_brain, cfg_B, cfg_A, fz


def check_spec(cfg, fcw, arm_brain, cfg_B, cfg_A, fz):
    need(list(cfg["arms"]) == ARMS, "arm list does not match the locked configuration")
    B = cfg["budget"]
    need((fcw.BUDGET, fcw.MAX_STEPS, fcw.LANG_MAX) == (B["BUDGET"], B["MAX_STEPS"], B["LANG_MAX"]), "budget constants do not match")
    need(rp(fcw.CATALOG) == rp(cfg["sha256"]["catalog"]["path"]), "sequence catalog path does not match")
    bB, bA = arm_brain["benefit_B"][0], arm_brain["benefit_A"][0]
    for b, nm in ((bB, "v4"), (bA, "v3_all")):
        need(sorted(b.targets) == sorted(cfg["targets"]) == sorted(b.tasks), f"{nm}: target set does not match the locked targets")
    need(list(bA.cands) == list(bB.cands), "v4 and v3_all candidates differ")
    need(fz["lam"] == cfg["benefit_A"]["lam"] and fz["delta_benefit"] == cfg["benefit_A"]["delta_benefit"], "benefit_A λ/δ do not match")
    need(fz["cbar"] == cfg["cbar"] and fz["cbar_global"] == cfg["cbar_global"], "frozen c̄ differs from the locked c̄")
    if fz.get("model_file"):
        need(rp(fz["model_file"]) == rp(cfg["model_files"]["A"]), "model_file in the frozen config differs from model_files['A']")
    need(cfg_B["lam"] == cfg["lam"] and cfg_B["delta_benefit"] == cfg["delta_benefit"], "λ/δ of the B arms do not match")
    out = {}
    for arm in ARMS:
        s, fa = cfg["arm_spec"][arm], FC_ARM[arm]
        brain, acfg = arm_brain[arm]
        need(bool(s["retry"]) == (fa not in fcw.NO_RETRY), f"{arm}: retry setting does not match fc_worker arm {fa}")
        if s["repair"] == "none":
            need(fa in ("none", "retry") and fa not in fcw.REPAIR, f"{arm}: wrong mapping for a no-repair arm")
            out[arm] = dict(fc_arm=fa, retry=bool(s["retry"]))
        elif s["repair"] == "argmax_prior_table_excluding_none":
            need(fa == "prior" and fa in fcw.REPAIR and brain is bB and s["model"][0] == "main", f"{arm}: wrong prior mapping")
            out[arm] = dict(fc_arm=fa, model=str(rp(brain.cfg["model_file"])), retry=bool(s["retry"]))
        elif s["repair"] == "benefit_gate":
            src, key = s["model"]
            need(fa in ("benefit", "benefit_noretry", "noloc") and fa in fcw.REPAIR, f"{arm}: wrong gate-arm mapping")
            need(list(brain.arm_models[fa]) == ["main", key], f"{arm}: scorer is not {key}")
            need(rp(brain.cfg["model_file"]) == rp(cfg["model_files"][src]), f"{arm}: model file is not {src}")
            need(list(cfg["arm_models"][arm]) == [src, key], f"{arm}: arm_models do not match")
            need(acfg["lam"] == s["lam"] and acfg["delta_benefit"] == s["delta"], f"{arm}: λ/δ do not match")
            ref = {"cbar": (cfg["cbar"], cfg["cbar_global"]), "round2_frozen": (fz["cbar"], fz["cbar_global"])}.get(s["cbar"])
            need(ref is not None and (acfg["cbar"], acfg["cbar_global"]) == ref, f"{arm}: c̄ source does not match")
            out[arm] = dict(fc_arm=fa, model=str(rp(brain.cfg["model_file"])), scorer=key, lam=acfg["lam"],
                            delta=acfg["delta_benefit"], cbar=s["cbar"], retry=bool(s["retry"]))
        else:
            need(False, f"{arm}: unknown repair rule {s['repair']}")
    return out


def gate(brain, acfg, task, q):
    """Same gate expression as in fc_worker.run_chain (used for the fixture self-test)."""
    reps = {k: v for k, v in q.items() if k != "none"}
    us = max(reps, key=reps.get)
    do = (reps[us] - q["none"] - acfg["lam"] * brain.cbar(task, us) / 360) > acfg["delta_benefit"]
    return us, reps[us], bool(do)


def prior_choice(brain, task):
    """Same as the prior arm in fc_worker.run_chain."""
    pri = {c: brain.prior(task, c) for c in brain.options(task) if c != "none"}
    return max(pri, key=pri.get)


def fixture_check(cfg, arm_brain):
    import numpy as np
    fx = np.load(cfg["sha256"]["fixture_inputs"]["path"])
    rows = json.loads(Path(cfg["sha256"]["fixture_expected"]["path"]).read_text())["rows"]
    need(len(rows) == len(fx["sid"]) > 0, "fixture row count does not match")
    maxd, nq, bad, prior_fx = 0.0, 0, [], {}
    for i, e in enumerate(rows):
        t, sid = str(fx["task"][i]), str(fx["sid"][i])
        if e["sid"] != sid or e["task"] != t:
            bad.append(f"row{i}: sid/task mismatch")
            continue
        E = dict(g16=fx["g16"][i:i + 1], g8=fx["g8"][i:i + 1], rob=fx["state"][i:i + 1])
        for arm in GATE_ARMS:
            ex = e["arms"]["benefit_B" if arm == "benefit_noretry_B" else arm]
            brain, acfg = arm_brain[arm]
            q = brain.score_arm(E, t, FC_ARM[arm])
            if set(q) != set(ex["q"]):
                bad.append(f"row{i} {arm}: option sets differ")
                continue
            d = max(abs(q[k] - ex["q"][k]) for k in q)
            maxd, nq = max(maxd, d), nq + len(q)
            us, qu, do = gate(brain, acfg, t, q)
            if (d > FIX_TOL or us != ex["ustar"] or abs(qu - ex["q_ustar"]) > FIX_TOL or abs(q["none"] - ex["q_none"]) > FIX_TOL
                    or (us if do else None) != ex["repair"]):
                bad.append(f"row{i} {arm}: max|Δq|={d:.3g} u*={us}/{ex['ustar']} repair={(us if do else None)}/{ex['repair']}")
        u = prior_choice(arm_brain["prior"][0], t)
        if u != e["arms"]["prior"]["repair"]:
            bad.append(f"row{i} prior: {u} ≠ {e['arms']['prior']['repair']}")
        if prior_fx.setdefault(t, u) != u:
            bad.append(f"row{i} prior: two different repairs for the same task")
    need(not bad, "fixture reproduction failed: " + "; ".join(bad[:10]))
    return dict(ok=True, n_rows=len(rows), n_q_compared=nq, max_abs_diff=maxd, tol=FIX_TOL,
                tasks=sorted({str(x) for x in fx["task"]}), prior_fixture=prior_fx)


def provenance(cfg):
    out = {}
    for mod, key in PROV.items():
        m = sys.modules.get(mod)
        if m is None or not getattr(m, "__file__", None):
            need(mod not in REQUIRED_MODS, f"module {mod} not loaded")
            out[mod] = None
            continue
        p, exp = rp(m.__file__), rp(cfg["sha256"][key]["path"])
        need(p == exp, f"module {mod} comes from {p}, not from the locked {exp}")
        h = sha(p)
        need(h == cfg["sha256"][key]["sha256"], f"module {mod} sha256 does not match")
        out[mod] = dict(path=str(p), sha256=h)
    return out


def wrap_scoring(brain):
    """Only records the return values of score_arm (all 37 q values); does not change them."""
    brain.qlog = []
    inner = brain.score_arm

    def score_arm(E, task, arm, _inner=inner, _b=brain):
        q = _inner(E, task, arm)
        _b.qlog.append(dict(task=task, fc_arm=arm, q=dict(q)))
        return q
    brain.score_arm = score_arm


def startup(a, selftest):
    cfg, rec, checked = verify_lock()
    need(rp(a.jobs) == rp(cfg["jobs_file"]), f"--jobs must be the locked {cfg['jobs_file']}")
    jobs = load_jobs(cfg)
    add = None if selftest else verify_addendum()
    fcw, perturb = load_modules(cfg)
    adapter = recorder = None
    if not selftest:                     # same order as fc_worker.main: construct adapter → Brain → adapter.open
        import yaml
        import torch  # noqa: F401
        from rho.envs.s02_calvin import CalvinAdapter
        from rho.s04_replay import Recorder
        paths = yaml.safe_load((fcw.RHO / "configs/paths.yaml").read_text())
        settings = yaml.safe_load((fcw.RHO / "configs/calvin.yaml").read_text())
        settings.update(max_steps=fcw.MAX_STEPS, sequence_catalog=fcw.CATALOG)
        adapter, recorder = CalvinAdapter(paths, settings), Recorder()
    arm_brain, cfg_B, cfg_A, fz = build_brains(cfg, fcw)
    spec = check_spec(cfg, fcw, arm_brain, cfg_B, cfg_A, fz)
    fix = fixture_check(cfg, arm_brain)
    importlib.import_module("features_patch")
    importlib.import_module("worker")    # fc_worker does `from worker import scripted_repair` inside run_chain; imported here first to check its provenance
    prov = provenance(cfg)
    bB = arm_brain["benefit_B"][0]
    prior_by_task = {t: prior_choice(bB, t) for t in sorted(cfg["targets"])}
    for t, u in fix["prior_fixture"].items():
        need(prior_by_task[t] == u, f"prior {t}: {prior_by_task[t]} ≠ fixture {u}")
    for b in {id(x[0]): x[0] for x in arm_brain.values()}.values():
        wrap_scoring(b)
    import numpy, sklearn, torch
    selfcheck = dict(time=time.strftime("%F %T"), lock_files_verified=sum(v is not None for v in checked.values()),
                     fixture=fix, spec=spec, provenance=prov, prior_by_task=prior_by_task,
                     versions=dict(python=sys.version.split()[0], numpy=numpy.__version__, sklearn=sklearn.__version__,
                                   torch=torch.__version__, platform=platform.platform()),
                     lock_record_versions=dict(sklearn=rec.get("sklearn"), numpy=rec.get("numpy")))
    return dict(cfg=cfg, jobs=jobs, fcw=fcw, perturb=perturb, adapter=adapter, recorder=recorder, arm_brain=arm_brain,
                addendum=add, selfcheck=selfcheck, self_sha=sha(Path(__file__).resolve()))


# ------------------------------------------------------------------ run
def attach_q(r, brain, arm):
    log, brain.qlog = brain.qlog, []
    recs = [t for t in r["tasks"] if t["target"]]
    if arm not in GATE_ARMS:
        if log:
            raise RuntimeError(f"{arm}: scoring call in a non-gate arm")
        return
    if len(log) != len(recs):
        raise RuntimeError(f"{arm}: {len(log)} scoring calls ≠ {len(recs)} target handoffs")
    for t, e in zip(recs, log):
        if e["task"] != t["task"] or e["fc_arm"] != FC_ARM[arm]:
            raise RuntimeError(f"{arm}: scoring records do not correspond to the handoffs")
        reps = {k: v for k, v in e["q"].items() if k != "none"}
        us = max(reps, key=reps.get)
        if us != t.get("ustar") or e["q"]["none"] != t.get("q_none") or reps[us] != t.get("q_ustar"):
            raise RuntimeError(f"{arm}: q_all is inconsistent with the run_chain record")
        t["q_all"] = e["q"]


def run_once(ctx, adapter, recorder, job, seq):
    robot0, scene0, tries = ctx["perturb"](adapter, seq, job["perturb_seed"], recorder)
    arms = {}
    for arm in ARMS:
        brain, acfg = ctx["arm_brain"][arm]
        brain.qlog = []
        r = ctx["fcw"].run_chain(adapter, brain, acfg, seq, robot0, scene0, FC_ARM[arm], job["id"], recorder, "B")
        r["fc_arm"] = r.pop("arm")
        r["arm"] = arm
        attach_q(r, brain, arm)
        arms[arm] = r
    hashes = {k: v["init_physics_sha"] for k, v in arms.items()}
    return dict(perturb_tries=int(tries), scene0=[float(x) for x in scene0], robot0=[float(x) for x in robot0], arms=arms,
                init_physics_sha=hashes, paired_initial_state_identical=len(set(hashes.values())) == 1)


def run_job(ctx, adapter, recorder, job):
    seq = adapter.sequences(job["index"], 1)[0]
    if list(seq["tasks"]) != list(job["tasks"]):
        raise RuntimeError(f"task chain inconsistent with jobs: {seq['tasks']} vs {job['tasks']}")
    out = run_once(ctx, adapter, recorder, job, seq)
    out["pairing_rerun"] = False
    if not out["paired_initial_state_identical"]:      # prereg §10: pairing check failed → re-run once with the same seed
        first = out["init_physics_sha"]
        out = run_once(ctx, adapter, recorder, job, seq)
        out.update(pairing_rerun=True, first_run_init_physics_sha=first)
    out["job"] = job
    return out


def meta(ctx, a):
    add = ctx["addendum"] or {}
    return dict(worker="worker_cl3.py", worker_sha256=ctx["self_sha"], analyze_sha256=add.get("analyze_sha256"),
                addendum_sha256=add.get("sha256"), locked_config_sha256=LOCKED_CONFIG_SHA256, fc_worker_sha256=FC_WORKER_SHA256,
                selfcheck=ctx["selfcheck"], gpu=a.gpu, out_dir=a.out, host=socket.gethostname(), pid=os.getpid(),
                finished=time.strftime("%F %T"))


def run_loop(a, ctx):
    import torch
    jobs = ctx["jobs"]
    if a.ids:
        want = {x for x in a.ids.split(",") if x}
        unknown = want - {j["id"] for j in jobs}
        if unknown:
            sys.exit(f"--ids contains ids outside the locked jobs: {sorted(unknown)}")
        jobs = [j for j in jobs if j["id"] in want]
    sfx = "" if a.out == "results" else "_" + a.out
    dirs = [W(NEW / n) for n in (a.out, "claims" + sfx, "errors" + sfx, "starts" + sfx)]
    for d in dirs:
        d.mkdir(exist_ok=True)
    res_dir, claim_dir, err_dir, start_dir = dirs
    adapter, recorder = ctx["adapter"], ctx["recorder"]
    done = 0
    try:
        adapter.open(a.gpu, 0)
        for job in jobs:
            if a.limit is not None and done >= a.limit:
                break
            if refused_present():
                print("marker present in refused/: no new jobs are claimed", flush=True)
                break
            jid = job["id"]
            if (res_dir / f"{jid}.json").exists():
                continue
            try:
                fd = os.open(claim_dir / f"{jid}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                json.dump(dict(pid=os.getpid(), time=time.time(), gpu=a.gpu, host=socket.gethostname()), f)
            sf = start_dir / f"{jid}.json"
            st = json.loads(sf.read_text()) if sf.exists() else dict(n_started=0, starts=[])
            st["n_started"] += 1
            st["starts"].append(dict(time=time.strftime("%F %T"), pid=os.getpid(), gpu=a.gpu))
            write_json(sf, st)
            t0 = time.time()
            try:
                out = run_job(ctx, adapter, recorder, job)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                ef = err_dir / f"{jid}.json"
                e = json.loads(ef.read_text()) if ef.exists() else dict(n=0, errors=[])
                e["n"] += 1
                e["errors"].append(dict(time=time.strftime("%F %T"), gpu=a.gpu, pid=os.getpid(), error=tb[-6000:]))
                write_json(ef, e)
                if e["n"] >= 3:      # prereg §10: 3 crashes with the same seed → excluded
                    write_json(res_dir / f"{jid}.json", dict(job=job, error=tb[-6000:], n_crashes=e["n"], arms={},
                                                             excluded_reason="crashed 3 times (prereg §10)", **meta(ctx, a)))
                else:
                    (claim_dir / f"{jid}.json").unlink(missing_ok=True)
                raise
            out.update(seconds=round(time.time() - t0, 1), **meta(ctx, a))
            out["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
            try:
                out["max_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            except Exception:
                pass
            write_json(res_dir / f"{jid}.json", out)
            done += 1
            print(f"[gpu{a.gpu}] {jid} paired={out['paired_initial_state_identical']} rerun={out['pairing_rerun']} "
                  f"{out['seconds']}s rss={out['max_rss_mb']}MB", flush=True)      # no per-arm results are printed
        print("WORKER DONE", flush=True)
    finally:
        adapter.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if HERE != rp(NEW):
        sys.exit(f"worker_cl3.py must be located in {rp(NEW)} (actual {HERE})")
    if rp(a.run) != rp(NEW):
        sys.exit(f"--run must resolve to {rp(NEW)}")
    if a.selftest:
        try:
            ctx = startup(a, selftest=True)
        except Refuse as e:
            print("SELFTEST FAILED:", e, flush=True)
            sys.exit(2)
        sc = ctx["selfcheck"]
        print(json.dumps(dict(lock_files_verified=sc["lock_files_verified"],
                              fixture={k: v for k, v in sc["fixture"].items() if k != "prior_fixture"},
                              spec=sc["spec"], provenance=sc["provenance"], prior_by_task=sc["prior_by_task"],
                              versions=sc["versions"]), indent=1, ensure_ascii=False))
        print("SELFTEST OK", flush=True)
        return
    if a.gpu not in (2, 3, 4, 5):
        sys.exit("only GPUs 2-5 may be used")
    if a.out not in OUTS:
        sys.exit(f"--out must be one of {OUTS}")
    if refused_present():
        sys.exit("marker present in refused/: not starting (remove the marker manually after the cause has been identified and recorded)")
    try:
        ctx = startup(a, selftest=False)
    except Refuse as e:
        write_json(NEW / "refused" / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json",
                   dict(time=time.strftime("%F %T"), pid=os.getpid(), gpu=a.gpu, out=a.out, reason=str(e)))
        print("REFUSED:", e, flush=True)
        sys.exit(2)
    print(f"[gpu{a.gpu}] self-check passed: {ctx['selfcheck']['lock_files_verified']} locked files; fixture {ctx['selfcheck']['fixture']['n_rows']} rows, "
          f"max|Δq|={ctx['selfcheck']['fixture']['max_abs_diff']:.3g}", flush=True)
    run_loop(a, ctx)


if __name__ == "__main__":
    main()
