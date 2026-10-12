# -*- coding: utf-8 -*-
"""Benefit model v4 (B_def labels): a copy of train_benefit_v3.py. Differences from v3 are limited to:
 (a) MT/GT/NT point to relabel_bdef_20261011/labels_bdef/<dir> (label_*.json are B_def labels; all other files are symlinks to the original dirs).
 (b) evaluate_cv.features_for is not called (it writes fix_20260928/cache when a cache is missing). features_for4 is used instead: new caches are
     always written to cache/ in this directory; if this directory has no cache but fix_20260928/cache has one with the same name,
     verify_old_cache first checks that the inputs are identical and uses it read-only only if the check passes; otherwise it is recomputed here.
     Checks: (1) the cache names cover all input file names; (2) each input (through its symlink) has the same realpath as the same-named
          file in the original dir, or else the same sha256; (3) 8 inputs drawn by a fixed hash are recomputed here with the same build_cache
          and compared key by key with the cache (state/scene_state relative diff <= 1e-6, image features <= 1e-2).
     Fixed-name caches (mt_screen_*, repaired_states_*) get the same check for their inputs that lie in the relabelled dirs;
     two_task_raw_all is unaffected by relabelling and is read only.
 (c) repair_20261001 (hand): --rp keep (default, deployed model) keeps it; --rp exclude is a sensitivity analysis only and saves no model.
 (d) Output models/benefit_models_v4_all.joblib with the same keys as v3 (plus criterion, label dirs, sklearn version).
 (e) All offline reports use B labels; in addition a "train on A, evaluate on B" reference:
       M7_A      same folds, same featurizer, same structure, training labels and prior taken from the original A labels (cross-fitted, paired with M7);
       M7_v3all  the previously deployed model benefit_models_v3_all.joblib scored as is (trained on all dev data, in-sample, optimistic, reference only).
 Also (model unchanged): label check analysis/label_check.json (lock requires ok), per-arm step counts (rsteps/bsteps, used by lock_config3 to tune delta),
 closed-loop scoring self-test inputs analysis/fixture_inputs.npz. Nothing is written outside this directory. The 5-task variant is no longer needed (--tag accepts only all).
Usage: CUDA_VISIBLE_DEVICES= python -B train_benefit_v4.py [--rp keep|exclude] [--no-loto] [--no-save]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import os, sys, json, hashlib, collections, argparse, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "32")
HERE = Path(__file__).resolve().parent
NEWDIR = Path(f"{RUN_ROOT}/benefit_model_bdef_20261011").resolve()
if HERE != NEWDIR:
    sys.exit(f"must run from {NEWDIR} (this file is in {HERE})")
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")
import numpy as np
import torch
torch.set_num_threads(32)
import joblib
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from benefit_lib_v2 import Featurizer2
from benefit_lib_v3 import FeaturizerNL
from features_patch import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="all", choices=["all"], help="only all (11 tasks) is supported; the 5-task variant is no longer needed")
ap.add_argument("--rp", default="keep", choices=["keep", "exclude"], help="repair_20261001 (hand) data: keep=deployed model; exclude=sensitivity analysis")
ap.add_argument("--no-loto", action="store_true")
ap.add_argument("--no-save", action="store_true")
args = ap.parse_args()
SUF = "" if args.rp == "keep" else "_rpexclude"
if (HERE / "locked_config.json").exists():
    sys.exit("locked_config.json exists: stage 2 is locked; retraining refused")


def W(p):
    """Every write path goes through here: writing is allowed only inside this directory."""
    rp = os.path.realpath(p)
    if not rp.startswith(str(HERE) + os.sep):
        raise RuntimeError(f"refusing to write outside {HERE}: {rp}")
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    return Path(rp)


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


CV = Path(f"{RUN_ROOT}")
FIX = CV / "fix_20260928"
REL = CV / "relabel_bdef_20261011" / "labels_bdef"
MT = REL / "multitask_vocab_20261002"
GT = REL / "gating_vocab_20261003"
NT = REL / "newtasks_20261004"
ORIG = {MT: CV / "multitask_vocab_20261002", GT: CV / "gating_vocab_20261003", NT: CV / "newtasks_20261004"}
RP = CV / "repair_20261001"
CACHE_OLD = FIX / "cache"
CACHE = HERE / "cache"
V3_MODEL = CV / "benefit_model_20261003/models/benefit_models_v3_all.joblib"
V3_REPORT = CV / "benefit_model_20261003/analysis/v3_report_all.json"
OOF = Path(f"{RHO_ROOT}/outputs/experiments/offline_full_20260929_105330/evaluations/cv/oof_predictions.json")
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))
CANDS = ["none"] + VOCAB + ["retreat", "home"]
ET = dict(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=32)
KEYS = ("static_grid16", "static_grid8", "state", "scene_state")
for d in (MT, GT, NT):
    if not (d / "results").is_dir():
        sys.exit(f"missing relabelled directory {d}/results")
USE_NT = (NT / "results").exists()
OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
# "k changed" column of section 3 of the stage-6a relabelling report (per task, total 245)
EXPECTED_K_CHANGE = {"lift_blue_block_slider": 0, "lift_pink_block_drawer": 4, "lift_pink_block_slider": 0, "lift_red_block_slider": 0,
                     "push_blue_block_right": 56, "push_into_drawer": 0, "push_pink_block_right": 66, "push_red_block_right": 46,
                     "rotate_pink_block_right": 31, "rotate_red_block_left": 42, "stack_block": 0}


# ---------------- label check (read-only) ----------------
def find_mapping():
    root, found = REL.parent, []
    for dp, dn, fn in os.walk(root, followlinks=False):
        if len(Path(dp).relative_to(root).parts) >= 3:
            dn[:] = []
        if "mapping.json" in fn:
            found.append(str(Path(dp) / "mapping.json"))
    return sorted(found)


def label_check():
    out = dict(dirs={}, problems=[], n_problems=0, k_change_by_task=collections.Counter(), criterion=collections.Counter(),
               label_source=collections.Counter())

    def prob(msg):
        out["n_problems"] += 1
        if len(out["problems"]) < 50:
            out["problems"].append(msg)
    for new, old in ORIG.items():
        rn = {f.name for f in (new / "results").glob("label_*.json")}
        ro = {f.name for f in (old / "results").glob("label_*.json")}
        dd = dict(label_new=len(rn), label_old=len(ro), same_file_set=rn == ro, arms_valid=0, k_changed=0, j8=0, j8_not_identical=0)
        if rn != ro:
            prob(f"{new.name}: label file sets differ ({len(rn - ro)} new-only, {len(ro - rn)} old-only)")
        for name in sorted(rn & ro):
            a = json.loads((new / "results" / name).read_text()); b = json.loads((old / "results" / name).read_text())
            if (a.get("sample_id"), a.get("task_b")) != (b.get("sample_id"), b.get("task_b")):
                prob(f"{new.name}/{name}: sample_id/task_b differ"); continue
            out["criterion"][str(a.get("criterion"))] += 1
            A = {x["candidate"]: x for x in a["arms"]}; B = {x["candidate"]: x for x in b["arms"]}
            if set(A) != set(B):
                prob(f"{new.name}/{name}: candidate sets differ"); continue
            for c in sorted(A):
                x, y = A[c], B[c]
                vx = "invalid" not in x and x.get("n") == 8; vy = "invalid" not in y and y.get("n") == 8
                if vx != vy:
                    prob(f"{new.name}/{name}/{c}: validity differs"); continue
                if not vx:
                    continue
                out["label_source"][str(x.get("label_source"))] += 1
                if len(x["rollouts"]) != x["n"] or x["k"] != sum(int(r["success"]) for r in x["rollouts"]):
                    prob(f"{new.name}/{name}/{c}: k != sum(success) or n != len(rollouts)")
                dd["arms_valid"] += 1
                if x["k"] != y["k"]:
                    dd["k_changed"] += 1; out["k_change_by_task"][a["task_b"]] += 1
        for f in (new / "results").glob("j8_*.json"):
            dd["j8"] += 1; g = old / "results" / f.name
            if not g.exists() or (os.path.realpath(f) != os.path.realpath(g) and sha256_file(f) != sha256_file(g)):
                dd["j8_not_identical"] += 1
        if dd["j8_not_identical"]:
            prob(f"{new.name}: {dd['j8_not_identical']} j8 files not identical to the original")
        out["dirs"][new.name] = dd
    kc = dict(out["k_change_by_task"])
    out["k_change_expected_6a"] = EXPECTED_K_CHANGE
    out["k_change_matches_6a_report"] = bool(all(kc.get(t, 0) == v for t, v in EXPECTED_K_CHANGE.items()) and set(kc) <= set(EXPECTED_K_CHANGE))
    out["mapping_json_found"] = find_mapping()
    for k in ("k_change_by_task", "criterion", "label_source"):
        out[k] = dict(out[k])
    return out


LC = label_check() if args.rp == "keep" else None
if LC is not None:
    print("label check: problems", LC["n_problems"], "| k changes", LC["k_change_by_task"], "| matches 6a:", LC["k_change_matches_6a_report"],
          "| criterion", LC["criterion"], flush=True)

# ---------------- natural handoffs (same as v3) ----------------
nat = []
for g in sorted((FIX / "runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            nat.append(dict(sid=r["sample_id"], npz=str(g.parent / "boundaries" / r["observation"]), k=r["k"], n=r["n"],
                            wins=[int(x["success"]) for x in r["repeats"]], task=r["task_b"], parent=r["parent_id"], src="dev2"))
for run, src in ((MT, "mt"), (NT, "nt")):
    if src == "nt" and not USE_NT:
        continue
    for f in sorted((run / "results").glob("j8_*.json")):
        d = json.loads(f.read_text())
        if d.get("reached") and d.get("n") == 8 and "restore_failed" not in d:
            nat.append(dict(sid=d["sample_id"], npz=str(run / "boundaries" / d["observation"]), k=d["k"], n=d["n"],
                            wins=[int(x["success"]) for x in d["repeats"]], task=d["task_b"], parent=d["parent_id"], src=src))
TASKS = sorted({r["task"] for r in nat})
nat_idx = {}
for i, r in enumerate(nat):
    nat_idx.setdefault(r["sid"], i)
fold_of = {}
oof = json.loads(OOF.read_text())
for s, fo in zip(oof["samples"], oof["methods"]["tree | vision+robot15"]["folds"]):
    fold_of[s["parent_id"]] = int(fo)
fold = lambda p: fold_of.setdefault(p, int(hashlib.sha256(("mt-fold:" + p).encode()).hexdigest(), 16) % 5)

# ---------------- repair labels (B labels; A labels and step counts are read as well) ----------------
reps, states = [], {}
STEP_SRC, HAND_STEP_KEY = collections.Counter(), collections.Counter()


def mean_steps(rollouts):
    v = [x.get("steps") for x in rollouts]
    return float(np.mean(v)) if v and all(s is not None for s in v) else None


def screen_index(run):
    """Measured step count of each candidate repair in the screening files (independent of the criterion)."""
    by_sid, by_stem = {}, {}
    for f in sorted((run / "results").glob("screen_*.json")):
        d = json.loads(f.read_text())
        st = {c["candidate"]: c["steps"] for c in d.get("candidates", []) if "invalid" not in c and c.get("steps") is not None}
        if d.get("sample_id") is not None:
            by_sid[d["sample_id"]] = st
        by_stem[f.stem[len("screen_"):]] = st
    return by_sid, by_stem


def add_vocab(run, job_files, src):
    orig = ORIG[run]
    scr_sid, scr_stem = screen_index(run)
    sel = {}
    for f in (run / "analysis").glob("*_selection.json"):
        sel.update(json.loads(f.read_text()))
    jobs = {}
    for jf in job_files:
        p = run / jf
        if p.exists():
            for j in json.loads(p.read_text())["jobs"]:
                jobs[j["sample_id"]] = j
    for f in sorted((run / "results").glob("label_*.json")):
        d = json.loads(f.read_text()); sid = d["sample_id"]
        if sid not in jobs or sid not in nat_idx or sid not in sel:
            continue
        job = jobs[sid]
        arms = {a["candidate"]: a for a in d["arms"] if "invalid" not in a and a.get("n") == 8}
        if "none" not in arms:
            continue
        dA = json.loads((orig / "results" / f.name).read_text())
        armsA = {a["candidate"]: a for a in dA["arms"] if "invalid" not in a and a.get("n") == 8}
        if set(armsA) != set(arms):
            raise RuntimeError(f"valid arm sets differ between B and A label files: {run.name}/{f.name}")
        if sid in scr_sid:
            rs = scr_sid[sid]; STEP_SRC[src + ":sample_id"] += 1
        elif str(job.get("id")) in scr_stem:
            rs = scr_stem[str(job["id"])]; STEP_SRC[src + ":job_id"] += 1
        elif sid in scr_stem:
            rs = scr_stem[sid]; STEP_SRC[src + ":filename"] += 1
        else:
            rs = {}; STEP_SRC[src + ":missing"] += 1
        key = sid + "#" + src
        st = dict(key=key, sid=sid, task=d["task_b"], parent=job["parent_id"], src=src, group=job.get("group", src),
                  k_none=arms["none"]["k"], wins_none=[int(x["success"]) for x in arms["none"]["rollouts"]],
                  wins_none_A=[int(x["success"]) for x in armsA["none"]["rollouts"]],
                  bsteps_none=mean_steps(arms["none"]["rollouts"]),
                  random=[c for c, _ in sel[sid]["random"]], cands=[])
        for c, a in arms.items():
            if c == "none":
                continue
            reps.append(dict(key=key, sid=sid, cand=c, k=a["k"], wins=[int(x["success"]) for x in a["rollouts"]],
                             kA=armsA[c]["k"], winsA=[int(x["success"]) for x in armsA[c]["rollouts"]],
                             rsteps=(float(rs[c]) if c in rs else None), bsteps=mean_steps(a["rollouts"]),
                             path=str(run / "screen" / job["screen_npz"][c]), file=job["screen_npz"][c], task=d["task_b"],
                             parent=job["parent_id"], src=src))
            st["cands"].append(c)
        states[key] = st


add_vocab(MT, ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"), "vocab")
add_vocab(GT, ("jobs_label.json",), "gate")
if USE_NT:
    add_vocab(NT, ("jobs_nt3_label.json",), "nt")
if args.rp == "keep":
    # hand: hand-designed repairs for push_into_drawer and lift_red_block_slider; labels kept as stored (A and B_def had zero disagreement on these two tasks in the audit and relabelling)
    rp_plan = {s["sample_id"]: s for s in json.loads((RP / "plan.json").read_text())["states"]}
    for f in sorted((RP / "results").glob("*.json")):
        d = json.loads(f.read_text()); sid = d["sample_id"]
        if sid not in nat_idx:
            continue
        key = sid + "#hand"
        st = dict(key=key, sid=sid, task=d["task_b"], parent=rp_plan[sid]["parent_id"], src="hand", group="hand_" + d["group"],
                  k_none=nat[nat_idx[sid]]["k"], wins_none=None, wins_none_A=None, bsteps_none=None, random=[], cands=[])
        for r in d["repairs"]:
            if r.get("invalid") or r.get("n") != 8:
                continue
            c = r["repair"].replace("lang:", "")
            rsk = next((k for k in ("repair_steps", "steps") if r.get(k) is not None), None)
            HAND_STEP_KEY[str(rsk)] += 1
            wins = [int(x["success"]) for x in r["rollouts"]]
            reps.append(dict(key=key, sid=sid, cand=c, k=r["k"], wins=wins, kA=r["k"], winsA=wins,
                             rsteps=(float(r[rsk]) if rsk else None), bsteps=mean_steps(r["rollouts"]),
                             path=None, file=r["repaired_npz"], task=d["task_b"], parent=st["parent"], src="hand"))
            st["cands"].append(c)
        states[key] = st
print("tag", args.tag, "rp", args.rp, "| natural", len(nat), collections.Counter(r["task"] for r in nat), flush=True)
print("repair arms", len(reps), "| states", collections.Counter(s["src"] for s in states.values()), flush=True)
print("repair-step sources", dict(STEP_SRC), "| hand step key", dict(HAND_STEP_KEY), flush=True)

if LC is not None:
    v3data = json.loads(V3_REPORT.read_text())["data"]
    mine = dict(natural=len(nat), repaired=len(reps), states=dict(collections.Counter(s["src"] for s in states.values())))
    LC["data_composition"] = dict(v4=mine, v3=v3data, equal=(v3data.get("natural") == mine["natural"] and
                                  v3data.get("repaired") == mine["repaired"] and v3data.get("states") == mine["states"]))
    LC["repair_step_sources"] = dict(STEP_SRC); LC["hand_step_key"] = dict(HAND_STEP_KEY)
    LC["ok"] = bool(LC["n_problems"] == 0 and LC["k_change_matches_6a_report"] and LC["data_composition"]["equal"])
    W(OUT / "label_check.json").write_text(json.dumps(LC, indent=1, ensure_ascii=False))
    print("LABEL CHECK OK" if LC["ok"] else "LABEL CHECK FAILED (lock will refuse)", flush=True)

# ---------------- features (caches written to this directory; old caches read-only after verification) ----------------
PROV = {}
SPOT = 8


def orig_path(p):
    s = str(p)
    for new, old in ORIG.items():
        if s.startswith(str(new) + os.sep):
            return Path(str(old) + s[len(str(new)):])
    return None


def verify_old_cache(cache_file, paths):
    info = dict(cache=str(cache_file), n_paths=len(paths))
    C = np.load(cache_file)
    cn = {n: i for i, n in enumerate(C["names"].tolist())}
    names = [Path(p).name for p in paths]
    info["missing_names"] = int(sum(n not in cn for n in names))
    real = collections.defaultdict(set)
    for p in paths:
        real[Path(p).name].add(os.path.realpath(p))
    info["ambiguous_basenames"] = int(sum(len(v) > 1 for v in real.values()))
    same_real = same_hash = differ = no_orig = 0
    for p in paths:
        op = orig_path(p)
        if op is None:
            no_orig += 1
        elif os.path.realpath(p) == os.path.realpath(op):
            same_real += 1
        elif op.exists() and sha256_file(p) == sha256_file(op):
            same_hash += 1
        else:
            differ += 1
    info.update(same_realpath=same_real, same_sha256=same_hash, differ=differ, not_in_relabel_dirs=no_orig)
    ok = info["missing_names"] == 0 and differ == 0
    if ok and paths:
        sample = sorted(paths, key=lambda p: hashlib.sha256(("v4-spot:" + Path(p).name).encode()).hexdigest())[:SPOT]
        spot = CACHE / f"spotcheck_{Path(cache_file).stem}.npz"
        if not spot.exists():
            build_cache(list(sample), W(spot), modes=("cls_crop", "grid8", "grid16"))
        S = np.load(spot); sn = {n: i for i, n in enumerate(S["names"].tolist())}
        keys = sorted((set(S.files) & set(C.files)) - {"names"})
        dev = {}
        for k in keys:
            Sk, Ck = S[k], C[k]
            a = np.stack([Sk[sn[Path(p).name]] for p in sample]).astype(np.float64)
            b = np.stack([Ck[cn[Path(p).name]] for p in sample]).astype(np.float64)
            dev[k] = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-8))
        info["spot_keys"], info["spot_rel_maxdiff"] = keys, dev
        ok = set(KEYS) <= set(keys) and all(v <= (1e-6 if k in ("state", "scene_state") else 1e-2) for k, v in dev.items())
    info["ok"] = bool(ok)
    return ok, info


def features_for4(paths, cache_name, prov, allow_old=True):
    names = [Path(x).name for x in paths]
    digest = hashlib.sha256("|".join(sorted(names)).encode()).hexdigest()[:12]
    fname = f"{Path(cache_name).stem}_{len(paths)}_{digest}.npz"
    new, old = CACHE / fname, CACHE_OLD / fname
    if new.exists():
        info = dict(source="new_dir_existing"); C = np.load(new)
    else:
        ok, info = verify_old_cache(old, paths) if (allow_old and old.exists()) else (False, dict(reason="no old cache or not allowed"))
        if ok:
            info["source"] = "old_verified_read_only"; C = np.load(old)
        else:
            build_cache(list(paths), W(new), modes=("cls_crop", "grid8", "grid16"))
            info["source"] = "built_in_new_dir"; C = np.load(new)
    order = {n: i for i, n in enumerate(C["names"].tolist())}
    idx = [order[n] for n in names]
    prov[fname] = info
    print(f"features {fname}: {info.get('source')}", flush=True)
    return {k: C[k][idx] for k in C.files if k != "names"}


NZ = features_for4([r["npz"] for r in nat], f"v3_natural_{args.tag}.npz", PROV)
NZ = {k: NZ[k] for k in KEYS}
R = {k: [None] * len(reps) for k in KEYS}
for fname in ("mt_screen_2170_eddcd0e44786.npz", "mt_screen_4165_aace2b415d9d.npz", "repaired_states_286_51a1f2c969bd.npz"):
    z = np.load(CACHE_OLD / fname); mp = {n: i for i, n in enumerate(z["names"])}
    want = [(j, mp[r["file"]]) for j, r in enumerate(reps) if r["src"] in ("vocab", "hand") and r["file"] in mp]
    if not want:
        PROV[fname] = dict(used=0); continue
    js_path = [j for j, _ in want if reps[j]["path"] is not None]
    ok, info = (verify_old_cache(CACHE_OLD / fname, [reps[j]["path"] for j in js_path]) if js_path
                else (True, dict(note="only hand states (not in relabel dirs)")))
    js, iz = zip(*want)
    for k in KEYS:
        arr = z[k][list(iz)]
        for j, a in zip(js, arr):
            R[k][j] = a
    if not ok:  # inputs under the relabelled dirs could not be confirmed identical: recompute these arms here (hand arms are not in the relabelled dirs and still read the original cache)
        G = features_for4([reps[j]["path"] for j in js_path], f"rebuilt_{Path(fname).stem}.npz", PROV, allow_old=False)
        for k in KEYS:
            for j, a in zip(js_path, G[k]):
                R[k][j] = a
    info.update(used=len(want), source=("old_fixed_name_read_only" if ok else "relabel-dir inputs rebuilt in new dir"))
    PROV[fname] = info
for src in ("gate", "nt"):
    js = [j for j, r in enumerate(reps) if r["src"] == src]
    if js:
        G = features_for4([reps[j]["path"] for j in js], "gate_repaired.npz" if src == "gate" else "nt_repaired.npz", PROV)
        for k in KEYS:
            for j, a in zip(js, G[k]):
                R[k][j] = a
assert all(x is not None for x in R["state"])
R = {k: np.stack(v) for k, v in R.items()}
RAW_FILE = CACHE_OLD / "two_task_raw_all_939_b59d93bfcc96.npz"   # raw handoffs (not affected by relabelling), read-only
_raw = np.load(RAW_FILE)
PROV[RAW_FILE.name] = dict(source="old_read_only (raw two-task boundaries, unaffected by relabelling)", size=RAW_FILE.stat().st_size)
RAW = {k: _raw[k] for k in ("names", "static_grid16", "static_grid8", "scene_state")}
raw_parent = {}
for f in sorted(Path(f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/raw").glob("gpu*/handoffs.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line); raw_parent[r["observation"]] = r["parent_id"]
rg = np.array([raw_parent[n] for n in RAW["names"]])
W(OUT / f"cache_provenance{SUF}.json").write_text(json.dumps(PROV, indent=1, default=str))

nf = np.array([fold(r["parent"]) for r in nat]); rf = np.array([fold(r["parent"]) for r in reps])
half = lambda p: int(hashlib.sha256(("v3-half:" + p).encode()).hexdigest(), 16) % 2 == 0
nh = np.array([half(r["parent"]) for r in nat]); rh = np.array([half(r["parent"]) for r in reps])
r_nat = np.array([nat_idx[r["sid"]] for r in reps])
ntask = [r["task"] for r in nat]; rtask = [r["task"] for r in reps]; rcand = [r["cand"] for r in reps]
none_rows = [(s["key"], nat_idx[s["sid"]], s["wins_none"], s["task"], s["parent"]) for s in states.values() if s["wins_none"] is not None]
X_WINS_B = [w for _, _, w, *_ in none_rows]
X_WINS_A = [states[k]["wins_none_A"] for k, *_ in none_rows]
xf = np.array([fold(p) for *_, p in none_rows], int); xh = np.array([half(p) for *_, p in none_rows], bool)
x_nat = np.array([i for _, i, *_ in none_rows], int); xtask = [t for *_, t, _ in none_rows]


def labels(j, crit="B"):
    """Success rate and weight over the first j rollouts (j=8 is the full label); crit=B gives B_def labels, A the original A labels (used only by the M7_A reference)."""
    wk = "wins" if crit == "B" else "winsA"
    xw = X_WINS_B if crit == "B" else X_WINS_A
    yN = np.array([np.mean(r["wins"][:j]) for r in nat]); wN = np.array([min(j, len(r["wins"])) for r in nat], float)
    yR = np.array([np.mean(r[wk][:j]) for r in reps]); wR = np.full(len(reps), float(j))
    yX = np.array([np.mean(w[:j]) for w in xw]) if none_rows else np.zeros(0)
    wX = np.full(len(none_rows), float(j))
    return yN, wN, yR, wR, yX, wX


LAB = {(j, "B"): labels(j, "B") for j in (2, 4, 8)}
LAB[(8, "A")] = labels(8, "A")
oh_t = lambda ts, tasks=TASKS: np.array([[float(t == u) for u in tasks] for t in ts]).reshape(len(ts), len(tasks))
oh_c = lambda cs: np.array([[float(c == u) for u in CANDS] for c in cs])


def prior_fn(mask_n, mask_x, mask_r, j=8, crit="B"):
    yN, _, yR, _, yX, _ = LAB[(j, crit)]
    agg = collections.defaultdict(list)
    for i in np.flatnonzero(mask_n):
        agg[(ntask[i], "none")].append(yN[i])
    for i in np.flatnonzero(mask_x):
        agg[(xtask[i], "none")].append(yX[i])
    for i in np.flatnonzero(mask_r):
        agg[(rtask[i], rcand[i])].append(yR[i])
    tmean = {}
    for t in TASKS:
        v = [x for (tt, c), vs in agg.items() if tt == t and c != "none" for x in vs]
        tmean[t] = float(np.mean(v)) if v else 0.5
    def prior(t, c):
        if agg.get((t, c)):
            return float(np.mean(agg[(t, c)]))
        if c == "none":
            return float(np.mean(agg[(t, "none")])) if agg.get((t, "none")) else 0.5
        return tmean[t]
    return prior, agg


def fit_rows(XN_, XR_, XX_, mn, mr, mx, j=8, crit="B"):
    yN, wN, yR, wR, yX, wX = LAB[(j, crit)]
    Xs, ys, ws = [XN_[mn], XR_[mr]], [yN[mn], yR[mr]], [wN[mn], wR[mr]]
    if XX_ is not None and mx.any():
        Xs.append(XX_[mx]); ys.append(yX[mx]); ws.append(wX[mx])
    return ExtraTreesRegressor(**ET).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))


def design(F_n, F_r, prior, use_prior=True):
    def X_(F_, tasks, cands):
        parts = [F_, oh_t(tasks), oh_c(cands)]
        if use_prior:
            parts.append(np.array([[prior(t, c)] for t, c in zip(tasks, cands)]))
        return np.hstack(parts)
    XN_ = X_(F_n, ntask, ["none"] * len(nat))
    XR_ = X_(F_n[r_nat], rtask, rcand)
    XX_ = X_(F_n[x_nat], xtask, ["none"] * len(x_nat)) if len(x_nat) else None
    return X_, XN_, XR_, XX_


SEL = ("M6", "M7", "M7n", "M7a", "M5v2", "M5n", "M7_j2", "M7_j4", "M7_half", "M5_j4", "M5_half", "M7_A", "M7_v3all")
S = {m: np.full(len(reps), np.nan) for m in SEL if m != "M6"}
S0 = {m: {} for m in SEL}
PRI = {}
for f in range(5):
    ntr, rtr, xtr = nf != f, rf != f, xf != f
    test_par = {r["parent"] for r, x in zip(nat, nf) if x == f} | {r["parent"] for r, x in zip(reps, rf) if x == f}
    raw_tr = ~np.isin(rg, list(test_par))
    G16 = np.concatenate([RAW["static_grid16"][raw_tr], NZ["static_grid16"][ntr], R["static_grid16"][rtr]]).astype(np.float32)
    G8 = np.concatenate([RAW["static_grid8"][raw_tr], NZ["static_grid8"][ntr], R["static_grid8"][rtr]])
    SC = np.concatenate([RAW["scene_state"][raw_tr], NZ["scene_state"][ntr], R["scene_state"][rtr]])
    fz = Featurizer2().fit(G16, G8, SC, seed=0); del G16
    fnl = FeaturizerNL().fit(G8, seed=0)
    F2N = fz.transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])
    FNN = fnl.transform(NZ["static_grid8"], NZ["state"])
    F1N = F2N[:, :32]
    te_keys = [s for s in states.values() if fold(s["parent"]) == f]
    variants = [  # name, features, use prior, label rollouts, training-mask modifier, training criterion
        ("M7", F2N, True, 8, None, "B"), ("M7n", FNN, True, 8, None, "B"), ("M7a", F1N, True, 8, None, "B"),
        ("M5v2", F2N, False, 8, None, "B"), ("M5n", FNN, False, 8, None, "B"), ("M7_j2", F2N, True, 2, None, "B"),
        ("M7_j4", F2N, True, 4, None, "B"), ("M7_half", F2N, True, 8, "half", "B"), ("M5_j4", F2N, False, 4, None, "B"),
        ("M5_half", F2N, False, 8, "half", "B"), ("M7_A", F2N, True, 8, None, "A")]
    for name, FN_, use_prior, j, mod, crit in variants:
        mn, mr, mx = ntr.copy(), rtr.copy(), xtr.copy()
        if mod == "half":
            mn &= nh; mr &= rh; mx &= xh
        prior, _ = prior_fn(mn, mx, mr, j, crit)
        X_, XN_, XR_, XX_ = design(FN_, None, prior, use_prior)
        m = fit_rows(XN_, XR_, XX_, mn, mr, mx, j, crit)
        S[name][rf == f] = m.predict(XR_[rf == f])
        for s in te_keys:
            i = nat_idx[s["sid"]]
            S0[name][s["key"]] = float(m.predict(X_(FN_[i:i + 1], [s["task"]], ["none"]))[0])
    prior, _ = prior_fn(ntr, xtr, rtr, 8, "B")
    for s in te_keys:
        PRI[s["key"]] = {c: prior(s["task"], c) for c in ["none"] + s["cands"]}
        S0["M6"][s["key"]] = PRI[s["key"]]["none"]
    print(f"fold {f}: natural {ntr.sum()} | repaired {rtr.sum()} | extra none {int(xtr.sum())}", flush=True)

# ---- reference: previously deployed model v3_all (A labels, trained on all dev data; in-sample) scored as is ----
v3 = joblib.load(V3_MODEL)
F3 = v3["featurizer"].transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])


def _pri3(t, c):
    pt = v3["prior_table"]
    if f"{t}|{c}" in pt:
        return pt[f"{t}|{c}"]
    return pt.get(f"{t}|none", 0.5) if c == "none" else v3["task_mean"].get(t, 0.5)


def X3(F_, ts, cs):
    return np.hstack([F_, np.array([[float(t == u) for u in v3["tasks"]] for t in ts]),
                      np.array([[float(c == u) for u in v3["cands"]] for c in cs]), np.array([[_pri3(t, c)] for t, c in zip(ts, cs)])])


S["M7_v3all"][:] = v3["M7"].predict(X3(F3[r_nat], rtask, rcand))
for s in states.values():
    i = nat_idx[s["sid"]]
    S0["M7_v3all"][s["key"]] = float(v3["M7"].predict(X3(F3[i:i + 1], [s["task"]], ["none"]))[0])

# ---------------- selection evaluation (B labels) ----------------
by_state = collections.defaultdict(list)
for j, r in enumerate(reps):
    by_state[r["key"]].append(j)


def state_eval(s):
    js = by_state[s["key"]]
    p = {reps[j]["cand"]: reps[j]["k"] / 8 for j in js}
    wins = {reps[j]["cand"]: np.array(reps[j]["wins"]) for j in js}
    p0 = s["k_none"] / 8
    o = dict(key=s["key"], sid=s["sid"], task=s["task"], parent=s["parent"], src=s["src"], group=s["group"], p0=p0, p=p,
             pA={reps[j]["cand"]: reps[j]["kA"] / 8 for j in js},
             rsteps={reps[j]["cand"]: reps[j]["rsteps"] for j in js}, bsteps={reps[j]["cand"]: reps[j]["bsteps"] for j in js},
             bsteps_none=s.get("bsteps_none"))
    sc_all = {m: {reps[j]["cand"]: float(S[m][j]) for j in js} for m in S}
    sc_all["M6"] = {c: PRI[s["key"]][c] for c in p}
    o["scores"], o["none_scores"] = sc_all, {m: S0[m].get(s["key"]) for m in SEL}
    for m, sc in sc_all.items():
        best = max(sc, key=sc.get)
        o["pick_" + m] = p[best]; o["choice_" + m] = best
        o["conc_" + m] = [float(np.sign(sc[a] - sc[b]) == np.sign(p[a] - p[b])) for a in p for b in p
                          if a < b and p[a] != p[b] and sc[a] != sc[b]]
    halves = []
    for a, b in ((slice(0, 4), slice(4, 8)), (slice(4, 8), slice(0, 4))):
        bst = max(wins, key=lambda c: wins[c][a].mean()); halves.append(wins[bst][b].mean())
    o["oracle_split"] = float(np.mean(halves))
    o["rand"] = float(np.mean([p[c] for c in s["random"] if c in p])) if s["random"] else float("nan")
    o["mean_labeled"] = float(np.mean(list(p.values())))
    return o


rows = [state_eval(s) for s in states.values() if by_state[s["key"]]]


def boot(rs, fn, n=2000, seed=9):
    ps = sorted({r["parent"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["parent"]].append(r)
    rng = np.random.RandomState(seed)
    v = [fn([x for q in rng.choice(ps, len(ps)) for x in by[q]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


OLD5 = {"lift_red_block_slider", "push_into_drawer", "push_red_block_right", "push_blue_block_right", "rotate_pink_block_right"}
INV_TASKS = {"push_red_block_right", "push_blue_block_right", "rotate_pink_block_right", "push_pink_block_right", "rotate_red_block_left"}
fam = lambda t: "old5" if t in OLD5 else ("new_inverse" if t in INV_TASKS else "new_noninverse")
report = dict(tag=args.tag, rp=args.rp, criterion="B_def",
              data=dict(natural=len(nat), repaired=len(reps), states=dict(collections.Counter(s["src"] for s in states.values()))),
              note=dict(M7_A="cross-fitted, same folds/featurizer, trained on A labels, evaluated on B labels",
                        M7_v3all="deployed v3_all model (A labels, all dev data): in-sample, optimistic reference"),
              selection={})
scopes = [("src", v) for v in sorted({r["src"] for r in rows})] + [("family", v) for v in ("old5", "new_inverse", "new_noninverse")] \
         + [("task", t) for t in sorted({r["task"] for r in rows})] + [("all", "ALL")]
print(f"\n{'scope':34s} {'n':>4s} {'p0':>6s} " + " ".join(f"{m:>8s}" for m in SEL) + f" {'rand':>6s} {'orac½':>6s}")
for kind, v in scopes:
    rs = [r for r in rows if kind == "all" or (kind == "src" and r["src"] == v) or (kind == "family" and fam(r["task"]) == v)
          or (kind == "task" and r["task"] == v)]
    if not rs:
        continue
    rec = dict(n=len(rs), p0=float(np.mean([r["p0"] for r in rs])), **{m: float(np.mean([r["pick_" + m] for r in rs])) for m in SEL},
               rand=float(np.nanmean([r["rand"] for r in rs])) if any(not np.isnan(r["rand"]) for r in rs) else None,
               oracle_split=float(np.mean([r["oracle_split"] for r in rs])),
               conc={m: (float(np.mean(sum((r["conc_" + m] for r in rs), []))) if any(r["conc_" + m] for r in rs) else None) for m in SEL})
    for m in SEL[1:]:
        rec[f"{m}_minus_M6"] = boot(rs, lambda x, m=m: float(np.mean([r["pick_" + m] - r["pick_M6"] for r in x])))
    rec["M7_minus_M7n"] = boot(rs, lambda x: float(np.mean([r["pick_M7"] - r["pick_M7n"] for r in x])))
    rec["M5v2_minus_M5n"] = boot(rs, lambda x: float(np.mean([r["pick_M5v2"] - r["pick_M5n"] for r in x])))
    rec["M5v2_minus_M7"] = boot(rs, lambda x: float(np.mean([r["pick_M5v2"] - r["pick_M7"] for r in x])))
    rec["M7_minus_M7_A"] = boot(rs, lambda x: float(np.mean([r["pick_M7"] - r["pick_M7_A"] for r in x])))
    rec["M7_minus_M7_v3all"] = boot(rs, lambda x: float(np.mean([r["pick_M7"] - r["pick_M7_v3all"] for r in x])))
    report["selection"][f"{kind}|{v}"] = rec
    print(f"{kind + ':' + v:34s} {len(rs):4d} {rec['p0']:6.3f} " + " ".join(f"{rec[m]:8.3f}" for m in SEL)
          + f" {(rec['rand'] if rec['rand'] is not None else float('nan')):6.3f} {rec['oracle_split']:6.3f}")
print("\ndifferences [95% CI] (B labels):")
for key, rec in report["selection"].items():
    print(f"  {key:34s} " + "  ".join(f"{m}-M6={rec[f'{m}_minus_M6'][0]:+.3f}[{rec[f'{m}_minus_M6'][1]:+.3f},{rec[f'{m}_minus_M6'][2]:+.3f}]"
                                      for m in ("M7", "M7n", "M5v2", "M7_A", "M7_v3all"))
          + f"  M7-M7_A={rec['M7_minus_M7_A'][0]:+.3f}[{rec['M7_minus_M7_A'][1]:+.3f},{rec['M7_minus_M7_A'][2]:+.3f}]")
W(OUT / f"v4_rows_{args.tag}{SUF}.json").write_text(json.dumps(rows, default=float))

# ---------------- final deployable model (all data, B labels) ----------------
G16 = np.concatenate([RAW["static_grid16"], NZ["static_grid16"], R["static_grid16"]]).astype(np.float32)
G8 = np.concatenate([RAW["static_grid8"], NZ["static_grid8"], R["static_grid8"]])
SC = np.concatenate([RAW["scene_state"], NZ["scene_state"], R["scene_state"]])
fz = Featurizer2().fit(G16, G8, SC, seed=0); del G16
fnl = FeaturizerNL().fit(G8, seed=0)
F2N = fz.transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])
FNN = fnl.transform(NZ["static_grid8"], NZ["state"])
F2R = fz.transform(R["static_grid16"].astype(np.float32), R["static_grid8"], R["state"])
report["featurizer_maxabs_diff_vs_v3_all"] = float(np.abs(F2N - F3).max()) if F2N.shape == F3.shape else None
alln, allr, allx = np.ones(len(nat), bool), np.ones(len(reps), bool), np.ones(len(none_rows), bool)
prior, agg = prior_fn(alln, allx, allr, 8, "B")
prior_table = {f"{t}|{c}": float(np.mean(v)) for (t, c), v in agg.items()}
task_mean = {}
for t in TASKS:
    v = [x for (tt, c), vs in agg.items() if tt == t and c != "none" for x in vs]
    task_mean[t] = float(np.mean(v)) if v else 0.5
_, XN7, XR7, XX7 = design(F2N, None, prior, True)
_, XNn, XRn, XXn = design(FNN, None, prior, True)
M7 = fit_rows(XN7, XR7, XX7, alln, allr, allx)
M7n = fit_rows(XNn, XRn, XXn, alln, allr, allx)
_, XN5, XR5, XX5 = design(F2N, None, prior, False)
_, XN5n, XR5n, XX5n = design(FNN, None, prior, False)
M5v2 = fit_rows(XN5, XR5, XX5, alln, allr, allx)
M5n = fit_rows(XN5n, XR5n, XX5n, alln, allr, allx)
yN, wN, yR, wR, yX, wX = LAB[(8, "B")]
XV = [np.hstack([F2N, oh_t(ntask)]), np.hstack([F2R, oh_t(rtask)])]
yV, wV = [yN, yR], [wN, wR]
if len(x_nat):
    XV.append(np.hstack([F2N[x_nat], oh_t(xtask)])); yV.append(yX); wV.append(wX)
M2v2 = ExtraTreesRegressor(**ET).fit(np.vstack(XV), np.concatenate(yV), sample_weight=np.concatenate(wV))
final = dict(featurizer=fz, featurizer_nl=fnl, tasks=TASKS, cands=CANDS, prior_table=prior_table, task_mean=task_mean,
             M7=M7, M7n=M7n, M5v2=M5v2, M5n=M5n, M2v2=M2v2,
             layout=dict(M7="[F2(o), onehot(task), onehot(cand), prior]", M7n="[FNL(o), onehot(task), onehot(cand), prior]",
                         M5v2="[F2(o), onehot(task), onehot(cand)]", M5n="[FNL(o), onehot(task), onehot(cand)]",
                         M2v2="[F2(o'), onehot(task)]"),
             data=report["data"], criterion="B_def", label_dirs={d.name: str(d) for d in (MT, GT, NT)},
             hand_data="repair_20261001 kept (labels as stored)", sklearn_version=sklearn.__version__, numpy_version=np.__version__,
             created=time.strftime("%F %T"))
if args.rp == "keep" and not args.no_save:
    joblib.dump(final, W(HERE / "models" / "benefit_models_v4_all.joblib"))
    print("saved", HERE / "models" / "benefit_models_v4_all.joblib", flush=True)
    # closed-loop scoring self-test inputs: 4 natural handoffs per task (fixed hash), used by lock to compute expected scores and by the closed-loop worker self-test
    fx = []
    for t in TASKS:
        ids = [i for i, r in enumerate(nat) if r["task"] == t]
        fx += sorted(ids, key=lambda i: hashlib.sha256(("v4-fixture:" + nat[i]["sid"]).encode()).hexdigest())[:4]
    np.savez_compressed(W(OUT / "fixture_inputs.npz"), g16=NZ["static_grid16"][fx].astype(np.float32),
                        g8=NZ["static_grid8"][fx].astype(np.float32), state=NZ["state"][fx].astype(np.float32),
                        task=np.array([nat[i]["task"] for i in fx]), sid=np.array([nat[i]["sid"] for i in fx]))

# ---------------- leave-one-task-out (B labels) ----------------
if not args.no_loto:
    report["loto"] = {}
    lab_tasks = sorted({s["task"] for s in states.values()})
    print("\n==== leave-one-task-out (task never seen in training) ====")
    for T in lab_tasks:
        others = [t for t in TASKS if t != T]
        mn = np.array([t != T for t in ntask]); mr = np.array([t != T for t in rtask]); mx = np.array([t != T for t in xtask], bool)
        _, agg_o = prior_fn(mn, mx, mr, 8, "B")
        cand_mean = collections.defaultdict(list)
        for (t, c), vs in agg_o.items():
            cand_mean[c].extend(vs)
        xprior = lambda c: float(np.mean(cand_mean[c])) if cand_mean.get(c) else 0.5

        def X_(F_, tasks, cands, up=True):
            parts = [F_, oh_t(tasks, others), oh_c(cands)]
            if up:
                parts.append(np.array([[xprior(c)] for c in cands]))
            return np.hstack(parts)
        XN_ = X_(F2N, ntask, ["none"] * len(nat)); XR_ = X_(F2N[r_nat], rtask, rcand)
        XX_ = X_(F2N[x_nat], xtask, ["none"] * len(x_nat)) if len(x_nat) else None
        m = fit_rows(XN_, XR_, XX_, mn, mr, mx)
        XNp = X_(F2N, ntask, ["none"] * len(nat), False); XRp = X_(F2N[r_nat], rtask, rcand, False)
        XXp = X_(F2N[x_nat], xtask, ["none"] * len(x_nat), False) if len(x_nat) else None
        mp = fit_rows(XNp, XRp, XXp, mn, mr, mx)
        rs = []
        for s in states.values():
            if s["task"] != T or not by_state[s["key"]]:
                continue
            js = by_state[s["key"]]
            p = {reps[j]["cand"]: reps[j]["k"] / 8 for j in js}
            sc = {reps[j]["cand"]: float(m.predict(XR_[j:j + 1])[0]) for j in js}
            scp = {reps[j]["cand"]: float(mp.predict(XRp[j:j + 1])[0]) for j in js}
            sp = {c: xprior(c) for c in p}
            rs.append(dict(parent=s["parent"], model=p[max(sc, key=sc.get)], model_np=p[max(scp, key=scp.get)], xprior=p[max(sp, key=sp.get)],
                           mean=float(np.mean(list(p.values()))), p0=s["k_none"] / 8))
        rec = dict(n=len(rs), **{k: float(np.mean([r[k] for r in rs])) for k in ("p0", "mean", "xprior", "model", "model_np")},
                   model_minus_xprior=boot(rs, lambda x: float(np.mean([r["model"] - r["xprior"] for r in x]))),
                   model_np_minus_xprior=boot(rs, lambda x: float(np.mean([r["model_np"] - r["xprior"] for r in x]))),
                   model_minus_mean=boot(rs, lambda x: float(np.mean([r["model"] - r["mean"] for r in x]))))
        report["loto"][T] = rec
        print(f"  {T:26s} n={rec['n']:3d} p0={rec['p0']:.3f} mean={rec['mean']:.3f} xprior={rec['xprior']:.3f} model={rec['model']:.3f} "
              f"model_np={rec['model_np']:.3f} model-xprior={rec['model_minus_xprior'][0]:+.3f}[{rec['model_minus_xprior'][1]:+.3f},{rec['model_minus_xprior'][2]:+.3f}] "
              f"model_np-xprior={rec['model_np_minus_xprior'][0]:+.3f}[{rec['model_np_minus_xprior'][1]:+.3f},{rec['model_np_minus_xprior'][2]:+.3f}]", flush=True)
report["provenance"] = PROV
W(OUT / f"v4_report_{args.tag}{SUF}.json").write_text(json.dumps(report, indent=1, default=float))
print("\nsaved", OUT / f"v4_report_{args.tag}{SUF}.json")
