# -*- coding: utf-8 -*-
"""Gate re-tuning and locking for the round-3 (stage 3, B_def) closed-loop confirmation experiment. Run before any stage-3 closed-loop data exist.

Re-tuning (--tune; under --lock it is recomputed and checked against tune_report)
  Objective, grid and tie-breaking are the same as in the round-1 lock_config.py:
    utility = p − λ·(repair steps + target-task steps)/360 (no repair: p0 − λ·target steps without repair/360), averaged over dev handoffs;
    benefit gate: q(u*) − q(none) − λ·c̄(task,u*)/360 > δ; u* = labelled candidate with the highest M7 score;
    δ grid [−inf] + (−0.4…0.6 step 0.025) + [inf]; the value with the highest mean utility is taken (ties: first in grid order).
  Inputs replaced by: B labels (p, p0, and target-task steps from the rollout step counts under B) and the 5-fold cross-fitted M7_B scores of train_benefit_v4.py.
  λ = 0.5 is not tuned: round 1 fixed it in advance as part of the utility definition; it is not a grid result.
  τ_risk and τ_retry are not set: the risk and full arms are not part of round 3.
  c̄ takes the round-2 locked values (screening step counts do not depend on the criterion) and is recomputed and checked as in lock_config2.py.
  Dev set: gate + hand + the "natural-distribution" groups of nt (group name contains gate/nat/rand/dist and not vocab/low); if none are found it falls back to round 1's gate + hand.
  Handoffs with missing no-repair step counts (hand) are imputed with the per-task mean, as in round 1; handoffs with missing repair step counts are excluded, as in round 1.
Lock (--lock --fc-worker PATH): check preconditions -> write config_benefit_A_round2_frozen.json (byte-identical copy of the round-2 config),
  analysis/fixture_expected.json (expected values for the closed-loop scoring self-test), analysis/label_files.sha256, locked_config.json (read-only),
  prereg_round3.sha256, lock_record.json. Refuses if any of locked_config.json / prereg_round3.sha256 / lock_record.json already exists.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import argparse, collections, glob, hashlib, json, os, platform, shutil, socket, stat, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NEWDIR = Path(f"{RUN_ROOT}/benefit_model_bdef_20261011").resolve()
if HERE != NEWDIR:
    sys.exit(f"must run from {NEWDIR}")
CV = Path(f"{RUN_ROOT}")
REL = CV / "relabel_bdef_20261011" / "labels_bdef"
ROWS = HERE / "analysis/v4_rows_all.json"
V4 = HERE / "models/benefit_models_v4_all.joblib"
V3 = CV / "benefit_model_20261003/models/benefit_models_v3_all.joblib"
R1CFG = CV / "closedloop_20261004/locked_config.json"
R2CFG = CV / "closedloop2_20261005/locked_config.json"
V2ROWS = CV / "benefit_model_20261003/analysis/v2_rows_with_gating.json"
OOF = Path(f"{RHO_ROOT}/outputs/experiments/offline_full_20260929_105330/evaluations/cv/oof_predictions.json")
CATALOG = Path(f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json")
JOBS, PLANREP = HERE / "jobs_cl3.json", HERE / "analysis/plan_report.json"
PREREG, TUNE = HERE / "prereg_round3.md", HERE / "analysis/tune_report.json"
LABCHK = HERE / "analysis/label_check.json"
VOCAB_YAML = f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml"
LAM = 0.5
DEL = [-np.inf] + list(np.round(np.arange(-0.4, 0.61, 0.025), 3)) + [np.inf]
NAT_KW, LOW_KW = ("gate", "nat", "rand", "dist"), ("vocab", "low")
ARMS = ["none", "retry", "prior", "benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A"]
BOOT_SEED, BOOT_N = 20261011, 5000

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--tune", action="store_true"); g.add_argument("--lock", action="store_true")
ap.add_argument("--fc-worker", default=None, help="fc_worker.py (implementation of the mode-B criterion); required for --lock")
a = ap.parse_args()


def W(p):
    rp = os.path.realpath(p)
    if not rp.startswith(str(HERE) + os.sep):
        raise RuntimeError(f"refusing to write outside {HERE}: {rp}")
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    return Path(rp)


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


oofj = json.loads(OOF.read_text())
fold_of = {s["parent_id"]: int(f) for s, f in zip(oofj["samples"], oofj["methods"]["tree | vision+robot15"]["folds"])}
fold = lambda p: fold_of.setdefault(p, int(hashlib.sha256(("mt-fold:" + p).encode()).hexdigest(), 16) % 5)


def boot(rs, fn, n=2000, seed=13):
    ps = sorted({r["parent"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["parent"]].append(r)
    rng = np.random.RandomState(seed)
    v = [fn([x for q in rng.choice(ps, len(ps)) for x in by[q]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def tune():
    rows = json.loads(ROWS.read_text())
    r1, r2 = json.loads(R1CFG.read_text()), json.loads(R2CFG.read_text())
    assert r1["lam"] == LAM and r2["lam"] == LAM, "round-1/2 λ differs from 0.5"
    # ---- c̄: keep the round-2 locked values; recompute and check as in lock_config2.py ----
    steps = collections.defaultdict(list)
    for d in ("multitask_vocab_20261002", "gating_vocab_20261003", "newtasks_20261004"):
        for f in glob.glob(f"{RUN_ROOT}/{d}/results/screen_*.json"):
            r = json.loads(open(f).read())
            for c in r["candidates"]:
                if "invalid" not in c and c.get("steps") is not None:
                    steps[(r["task_b"], c["candidate"])].append(c["steps"])
    cre = {f"{t}|{c}": float(np.mean(v)) for (t, c), v in steps.items()}
    cbar, cbar_global = r2["cbar"], r2["cbar_global"]
    cbar_check = dict(keys_equal=set(cre) == set(cbar),
                      max_abs_diff=float(max([abs(cre[k] - cbar[k]) for k in set(cre) & set(cbar)] or [0.0])))
    cb = lambda t, u: cbar.get(f"{t}|{u}", cbar_global.get(u, 100.0))
    # ---- impute no-repair step counts (as in round 1) ----
    bn = collections.defaultdict(list)
    for r in rows:
        if r.get("bsteps_none") is not None:
            bn[r["task"]].append(r["bsteps_none"])
    for r in rows:
        if r.get("bsteps_none") is None:
            r["bsteps_none"] = float(np.mean(bn[r["task"]])) if bn[r["task"]] else None
            r["bsteps_none_imputed"] = True
    complete = lambda r: (all(v is not None for v in r["rsteps"].values()) and all(v is not None for v in r["bsteps"].values())
                          and r["bsteps_none"] is not None)
    nt_groups = collections.Counter(r["group"] for r in rows if r["src"] == "nt")
    nt_nat = sorted(gq for gq in nt_groups if any(k in gq.lower() for k in NAT_KW) and not any(k in gq.lower() for k in LOW_KW))
    defs = {"gate+hand": (("gate", "hand"), []), "gate+hand+nt_natural": (("gate", "hand"), nt_nat),
            "gate+hand+nt_all": (("gate", "hand"), sorted(nt_groups))}
    primary = "gate+hand+nt_natural" if nt_nat else "gate+hand"

    def devset(name):
        srcs, ntg = defs[name]
        base = [r for r in rows if r["src"] in srcs or (r["src"] == "nt" and r["group"] in ntg)]
        return [r for r in base if complete(r)], len(base)

    def outcome(r, u, rep):
        return (r["p"][u], r["rsteps"][u] + r["bsteps"][u]) if rep else (r["p0"], r["bsteps_none"])

    def util(rs, rule, m="M7"):
        v = []
        for r in rs:
            sc = r["scores"][m]; u = max(sc, key=sc.get); q0 = r["none_scores"][m]
            s_, st = outcome(r, u, rule(r, u, sc[u], q0))
            v.append(s_ - LAM * st / 360)
        return float(np.mean(v))

    brule = lambda d: (lambda r, u, qu, q0: qu - q0 - LAM * cb(r["task"], u) / 360 > d)
    pick = lambda rs, m: max(DEL, key=lambda d: util(rs, brule(d), m))

    def per_row(rs, m, d):
        out = []
        for r in rs:
            sc = r["scores"][m]; u = max(sc, key=sc.get); q0 = r["none_scores"][m]
            rep = bool(brule(d)(r, u, sc[u], q0)); s_, st = outcome(r, u, rep)
            out.append(dict(parent=r["parent"], s=s_, steps=st, U=s_ - LAM * st / 360, rep=int(rep)))
        return out

    summ = lambda o: {k: float(np.mean([x[k] for x in o])) for k in ("s", "steps", "U", "rep")}
    rep = dict(created=time.strftime("%F %T"), lam=LAM, delta_grid=[float(x) for x in DEL], nt_groups=dict(nt_groups),
               nt_natural_groups=nt_nat, dev_rule=primary, cbar_check=cbar_check, dev_sets={})
    for name in defs:
        rs, n0 = devset(name)
        rep["dev_sets"][name] = dict(n=len(rs), n_before_completeness_filter=n0, tasks=dict(collections.Counter(r["task"] for r in rs)),
                                     delta=(pick(rs, "M7") if rs else None))
    dev, _ = devset(primary)
    delta = pick(dev, "M7")
    d_r1 = float(r2["delta_benefit"])
    rep["delta_benefit"] = float(delta)
    rep["delta_round1_carried_in_round2"] = d_r1
    rep["dev_utility_B"] = dict(
        never=util(dev, lambda *x: False), always=util(dev, lambda *x: True),
        benefit_B=summ(per_row(dev, "M7", delta)),
        round2_policy_crossfit_A=summ(per_row(dev, "M7_A", d_r1)),      # trained on A (cross-fitted) + round-1 δ, evaluated on B
        round2_policy_v3all_insample=summ(per_row(dev, "M7_v3all", d_r1)),
        M7_B_with_round1_delta=summ(per_row(dev, "M7", d_r1)))
    pb, pa = per_row(dev, "M7", delta), per_row(dev, "M7_A", d_r1)
    rep["dev_benefitB_minus_round2policyA_U"] = boot([dict(parent=x["parent"], d=x["U"] - y["U"]) for x, y in zip(pb, pa)],
                                                     lambda z: float(np.mean([w["d"] for w in z])))
    # cross-fitted δ (δ chosen on training folds, reported on the test fold; descriptive)
    cf = []
    for f in range(5):
        tr = [r for r in dev if fold(r["parent"]) != f]; te = [r for r in dev if fold(r["parent"]) == f]
        if tr and te:
            d = pick(tr, "M7")
            cf += [dict(x, fold=f, delta=float(d)) for x in per_row(te, "M7", d)]
    rep["crossfit_delta_test"] = dict(summ(cf), deltas=sorted({x["delta"] for x in cf}))
    # check repair step counts against the v2 rows (repair step counts do not depend on the criterion and should equal the values used in round 1)
    try:
        v2 = json.loads(V2ROWS.read_text())
        key = lambda r: (r.get("src"), r.get("task"), r.get("parent"), tuple(sorted(r.get("p", {}))))
        cnt = collections.Counter(key(r) for r in v2)
        m2 = {key(r): r for r in v2 if cnt[key(r)] == 1}
        n = eq = eqn = 0
        for r in rows:
            k = key(r)
            if r["src"] in ("gate", "hand") and k in m2:
                n += 1; b = m2[k].get("rsteps") or {}
                eq += all(r["rsteps"].get(c) is not None and b.get(c) is not None and abs(r["rsteps"][c] - b[c]) < 1e-6 for c in r["rsteps"])
                if not r.get("bsteps_none_imputed") and m2[k].get("bsteps_none") is not None:
                    eqn += abs(r["bsteps_none"] - m2[k]["bsteps_none"]) < 1e-6
        rep["rsteps_vs_v2_rows"] = dict(available=True, v2_rows=len(v2), matched=n, rsteps_equal=int(eq), bsteps_none_equal=int(eqn))
    except Exception as e:
        rep["rsteps_vs_v2_rows"] = dict(available=False, error=repr(e))
    return rep


def brain_scores(m, key, g16, g8, rob, task, vocab):
    """Same score construction as worker_cl2.Brain.score_arm."""
    opts = ["none"] + [v for v in vocab if v != task] + ["retreat", "home"]
    F = m["featurizer_nl"].transform(g8, rob) if key in ("M7n", "M5n") else m["featurizer"].transform(g16, g8, rob)
    parts = [np.repeat(F, len(opts), 0), np.array([[float(task == u) for u in m["tasks"]]] * len(opts)),
             np.array([[float(o == c) for c in m["cands"]] for o in opts])]
    if key in ("M7", "M7n", "M7a"):
        parts.append(np.array([[prior_of(m, task, o)] for o in opts]))
    return dict(zip(opts, map(float, m[key].predict(np.hstack(parts))))), opts


def prior_of(m, t, c):
    pt = m["prior_table"]
    if f"{t}|{c}" in pt:
        return pt[f"{t}|{c}"]
    return pt.get(f"{t}|none", 0.5) if c == "none" else m["task_mean"].get(t, 0.5)


if a.tune:
    rep = tune()
    W(TUNE).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "delta_grid"}, indent=1, ensure_ascii=False))
    sys.exit(0)

# ======================= --lock =======================
for p in (HERE / "locked_config.json", HERE / "prereg_round3.sha256", HERE / "lock_record.json"):
    if p.exists():
        sys.exit(f"{p.name} exists: refusing to overwrite a lock")
fail = []
res_dir = HERE / "results"
if res_dir.exists() and any(res_dir.iterdir()):
    fail.append("results/ is not empty: stage-3 data appear to exist")
if list(HERE.glob("**/cl3_*.json")):
    fail.append("files named cl3_*.json exist under this directory")
for p in (ROWS, V4, V3, R2CFG, JOBS, PLANREP, PREREG, TUNE, LABCHK, CATALOG, HERE / "analysis/fixture_inputs.npz"):
    if not p.exists():
        fail.append(f"missing {p}")
if not a.fc_worker or not Path(a.fc_worker).is_file():
    fail.append("--fc-worker missing or not a file")
if not fail:
    lc = json.loads(LABCHK.read_text())
    if not lc.get("ok"):
        fail.append("label_check.json ok=False")
    if not lc.get("mapping_json_found"):
        fail.append("mapping.json of the relabel not found")
    pr = json.loads(PLANREP.read_text())
    if not pr.get("ok"):
        fail.append("plan_report.json ok=False")
if fail:
    sys.exit("LOCK REFUSED:\n  " + "\n  ".join(fail))

rep = tune()
tune_saved = json.loads(TUNE.read_text())
if rep["delta_benefit"] != tune_saved["delta_benefit"] or rep["dev_rule"] != tune_saved["dev_rule"]:
    sys.exit(f"LOCK REFUSED: recomputed δ/dev rule {rep['delta_benefit']}/{rep['dev_rule']} differ from tune_report")
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")  # fix: same sys.path as train_benefit_v4.py
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")  # fix: same sys.path as train_benefit_v4.py
import joblib, sklearn, yaml
m4, m3 = joblib.load(V4), joblib.load(V3)
need = ["tasks", "cands", "featurizer", "featurizer_nl", "prior_table", "task_mean", "M7", "M7n", "M5v2", "M2v2"]
jobs = json.loads(JOBS.read_text())["jobs"]
chk = dict(missing_keys=[k for k in need if k not in m4], tasks_equal_v3=set(m4["tasks"]) == set(m3["tasks"]),
           cands_equal_v3=list(m4["cands"]) == list(m3["cands"]), targets_equal_plan=sorted(m4["tasks"]) == sorted(pr["targets"]),
           sklearn_model=m4.get("sklearn_version"), sklearn_now=sklearn.__version__, n_jobs=len(jobs), n_plan=pr["n_total"])
bad = (chk["missing_keys"] or not chk["tasks_equal_v3"] or not chk["cands_equal_v3"] or not chk["targets_equal_plan"]
       or chk["sklearn_model"] != chk["sklearn_now"] or chk["n_jobs"] != chk["n_plan"])
if bad:
    sys.exit("LOCK REFUSED: model/plan interface check failed: " + json.dumps(chk))
r2 = json.loads(R2CFG.read_text())
cbar, cbar_global = r2["cbar"], r2["cbar_global"]
delta = rep["delta_benefit"]

# ---- byte-identical frozen copy of the round-2 config (benefit_A) ----
frozen = W(HERE / "config_benefit_A_round2_frozen.json")
shutil.copyfile(R2CFG, frozen)
assert sha(frozen) == sha(R2CFG)

# ---- expected values for the closed-loop scoring self-test ----
vocab = sorted(yaml.safe_load(open(VOCAB_YAML)))
fx = np.load(HERE / "analysis/fixture_inputs.npz")
cbf = lambda cb_, cbg, t, u: cb_.get(f"{t}|{u}", cbg.get(u, 100.0))
fixture = []
for i in range(len(fx["sid"])):
    t = str(fx["task"][i]); g16, g8, rob = fx["g16"][i:i + 1], fx["g8"][i:i + 1], fx["state"][i:i + 1]
    rec = dict(sid=str(fx["sid"][i]), task=t, arms={})
    for arm, (mm, key, lam, dl, cb_, cbg) in {"benefit_B": (m4, "M7", LAM, delta, cbar, cbar_global),
                                              "noloc_B": (m4, "M7n", LAM, delta, cbar, cbar_global),
                                              "benefit_A": (m3, "M7", r2["lam"], r2["delta_benefit"], r2["cbar"], r2["cbar_global"])}.items():
        q, opts = brain_scores(mm, key, g16, g8, rob, t, vocab)
        reps = {k: v for k, v in q.items() if k != "none"}
        us = max(reps, key=reps.get)
        do = (reps[us] - q["none"] - lam * cbf(cb_, cbg, t, us) / 360) > dl
        rec["arms"][arm] = dict(q=q, ustar=us, q_ustar=reps[us], q_none=q["none"], repair=(us if do else None))
    opts = ["none"] + [v for v in vocab if v != t] + ["retreat", "home"]
    pri = {c: prior_of(m4, t, c) for c in opts if c != "none"}
    rec["arms"]["prior"] = dict(repair=max(pri, key=pri.get))
    fixture.append(rec)
fix_path = W(HERE / "analysis/fixture_expected.json")
fix_path.write_text(json.dumps(dict(note="expected Brain outputs on analysis/fixture_inputs.npz; 6c worker must reproduce "
                                         "q to 1e-9 and identical ustar/repair before its first job", rows=fixture), indent=1))

# ---- digest of the label file tree ----
lines = []
for d in sorted(REL.iterdir()):
    for f in sorted((d / "results").glob("label_*.json")):
        lines.append(f"{sha(f)}  {d.name}/results/{f.name}")
lab_list = W(HERE / "analysis/label_files.sha256")
lab_list.write_text("\n".join(lines) + "\n")
label_tree_digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()

fc = Path(a.fc_worker).resolve()
FILES = {
    "model_v4": V4, "model_v3_all_benefit_A": V3, "jobs_cl3": JOBS, "plan_report": PLANREP, "prereg_round3": PREREG,
    "round2_config_original": R2CFG, "round2_config_frozen_copy": frozen, "round1_config": R1CFG, "catalog": CATALOG,
    "label_check": LABCHK, "tune_report": TUNE, "v4_report": HERE / "analysis/v4_report_all.json", "v4_rows": ROWS,
    "v4_report_rpexclude": HERE / "analysis/v4_report_all_rpexclude.json", "cache_provenance": HERE / "analysis/cache_provenance.json",
    "fixture_inputs": HERE / "analysis/fixture_inputs.npz", "fixture_expected": fix_path, "label_files_list": lab_list,
    "code_train_benefit_v4": HERE / "train_benefit_v4.py", "code_lock_config3": HERE / "lock_config3.py",
    "code_plan_cl3": HERE / "plan_cl3.py", "code_run_sh": HERE / "run.sh",
    "lib_benefit_lib": CV / "benefit_model_20261003/benefit_lib.py", "lib_benefit_lib_v2": CV / "benefit_model_20261003/benefit_lib_v2.py",
    "lib_benefit_lib_v3": CV / "benefit_model_20261003/benefit_lib_v3.py", "lib_features_patch": CV / "fix_20260928/features_patch.py",
    "lib_object_locator": CV / "fix_20260928/object_locator.py", "lib_evaluate_cv": CV / "fix_20260928/evaluate_cv.py",
    "worker_cl2": CV / "closedloop_20261004/worker_cl2.py", "worker_cl_perturb": CV / "closedloop_20261004/worker_cl.py",
    "worker_scripted_repair": CV / "multitask_vocab_20261002/worker.py", "fc_worker_modeB": fc,
}
for i, mp in enumerate(lc["mapping_json_found"]):
    FILES[f"relabel_mapping_{i}"] = Path(mp)
REQUIRED = {"model_v4", "model_v3_all_benefit_A", "jobs_cl3", "plan_report", "prereg_round3", "round2_config_original", "catalog",
            "label_check", "tune_report", "code_train_benefit_v4", "code_lock_config3", "code_plan_cl3", "code_run_sh", "fc_worker_modeB",
            "relabel_mapping_0"}
shas = {}
for k, p in FILES.items():
    if Path(p).is_file():
        shas[k] = dict(path=str(p), sha256=sha(p))
    elif k in REQUIRED:
        sys.exit(f"LOCK REFUSED: required file missing: {p}")
    else:
        shas[k] = dict(path=str(p), sha256=None)
ret = lambda retry: retry
arm_spec = {
    "none": dict(repair="none", retry=False),
    "retry": dict(repair="none", retry=True),
    "prior": dict(repair="argmax_prior_table_excluding_none", model=["main", None], retry=True),
    "benefit_B": dict(repair="benefit_gate", model=["main", "M7"], lam=LAM, delta=delta, cbar="cbar", retry=True),
    "benefit_noretry_B": dict(repair="benefit_gate", model=["main", "M7"], lam=LAM, delta=delta, cbar="cbar", retry=False),
    "noloc_B": dict(repair="benefit_gate", model=["main", "M7n"], lam=LAM, delta=delta, cbar="cbar", retry=True),
    "benefit_A": dict(repair="benefit_gate", model=["A", "M7"], lam=r2["lam"], delta=r2["delta_benefit"], cbar="round2_frozen", retry=True),
}
cfg = dict(
    created=time.strftime("%F %T") + " (before any stage-3 closed-loop data)",
    criterion=dict(name="B_def", driver="mode B of fc_worker: target-subtask success and retry decisions judged relative to the "
                   "handoff state; for retries mode conditions taken at the start of the attempt; criterion A recorded only",
                   fc_worker=str(fc), fc_worker_sha256=shas["fc_worker_modeB"]["sha256"]),
    lam=LAM, delta_benefit=delta, tau_risk=None, tau_retry=None,
    model_files={"main": str(V4), "A": str(V3)}, model_file=str(V4), aux_model_file=str(V4),
    arm_models={"benefit_B": ["main", "M7"], "benefit_noretry_B": ["main", "M7"], "noloc_B": ["main", "M7n"], "benefit_A": ["A", "M7"]},
    arms=ARMS, arm_spec=arm_spec, targets=sorted(m4["tasks"]), cbar=cbar, cbar_global=cbar_global,
    benefit_A=dict(frozen_config=str(frozen), lam=r2["lam"], delta_benefit=r2["delta_benefit"], model_file=str(V3),
                   scorer="M7", cbar_source="round-2 locked config (identical to cbar above)"),
    budget=dict(BUDGET=2160, MAX_STEPS=360, LANG_MAX=120), jobs_file=str(JOBS), n_chains=len(jobs),
    analysis=dict(bootstrap_resamples=BOOT_N, bootstrap_seed=BOOT_SEED, cluster="job['base']"),
    dev_tuning={k: rep[k] for k in ("dev_rule", "nt_natural_groups", "dev_sets", "delta_benefit", "delta_round1_carried_in_round2",
                                      "dev_utility_B", "crossfit_delta_test", "cbar_check")},
    interface_check=chk, label_tree_digest=label_tree_digest, sha256=shas)
fd = os.open(HERE / "locked_config.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
with os.fdopen(fd, "w") as f:
    f.write(json.dumps(cfg, indent=1, ensure_ascii=False))
os.chmod(HERE / "locked_config.json", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
W(HERE / "prereg_round3.sha256").write_text(f"{shas['prereg_round3']['sha256']}  prereg_round3.md\n")
record = dict(time=time.strftime("%F %T"), host=socket.gethostname(), python=sys.version, platform=platform.platform(),
              sklearn=sklearn.__version__, numpy=np.__version__, locked_config_sha256=sha(HERE / "locked_config.json"),
              prereg_sha256=shas["prereg_round3"]["sha256"], label_tree_digest=label_tree_digest, files=shas)
W(HERE / "lock_record.json").write_text(json.dumps(record, indent=1))
print(json.dumps({k: cfg[k] for k in ("created", "lam", "delta_benefit", "arms", "n_chains", "label_tree_digest")}, indent=1, default=str))
print("LOCKED. locked_config.json sha256", record["locked_config_sha256"])
