# -*- coding: utf-8 -*-
"""Benefit model v3: adds data from the extended target tasks, ablations that remove privileged information and reduce data, and a leave-one-task-out test.

Data (used when present)
  Natural handoffs: dev two-task J8, multi-task J8 (multitask_vocab_20261002), extended-task J8 (newtasks_20261004)
  Repair labels: low-score vocabulary (vocab), natural distribution (gate), hand-designed repairs (hand), extended tasks (nt: low-score and natural-distribution groups)
Models (5-fold cross-fitting by initial configuration; the featurizer of each fold is fit only on the training-fold initial configurations)
  M6     task-level prior (mean success rate per (task, instruction) on the training folds), ignores the state
  M7     state features v2 (locator supervised with simulator object positions) + task + instruction + prior value  -- method used in closed loop
  M7n    locator-free features (PCA of DINOv2 grid + robot state, no simulator ground-truth supervision) + task + instruction + prior
  M5n    locator-free features + task + instruction, without the prior value
  M7a    state features v1 (32-d) + task + instruction + prior
  M5v2   state features v2 + task + instruction, without the prior value
  M7_j2 / M7_j4   training labels use only the first 2 / 4 rollouts of each arm (test labels still use 8)
  M7_half         uses only half of the training-fold initial configurations (fixed hash)
  M2v2   post-repair verifier (retry decision of the closed-loop full arm)
Leave-one-task-out (LOTO): an M7-type model is trained without any labels or natural handoffs of one task (task code set to zero, prior = instruction
  mean over the other tasks) and compared with the "instruction mean over other tasks" prior and with random candidates. The featurizer is fit on all
  data (it involves only object-position supervision, not success labels).
Usage: CUDA_VISIBLE_DEVICES= python -B train_benefit_v3.py [--tag 5task|all] [--no-loto]
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import os, sys, json, hashlib, collections, argparse
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "32")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{RUN_ROOT}/benefit_model_20261003")
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")
import numpy as np
import torch
torch.set_num_threads(32)
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from benefit_lib_v2 import Featurizer2
from benefit_lib_v3 import FeaturizerNL
import evaluate_cv as ev

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="all", help="5task: only data of the original 5 tasks; all: use all data present")
ap.add_argument("--no-loto", action="store_true")
ap.add_argument("--no-save", action="store_true")
args = ap.parse_args()

FIX = Path(f"{RUN_ROOT}/fix_20260928")
MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
RP = Path(f"{RUN_ROOT}/repair_20261001")
GT = Path(f"{RUN_ROOT}/gating_vocab_20261003")
NT = Path(f"{RUN_ROOT}/newtasks_20261004")
CACHE = FIX / "cache"
OOF = Path(f"{RHO_ROOT}/outputs/experiments/offline_full_20260929_105330/evaluations/cv/oof_predictions.json")
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))
CANDS = ["none"] + VOCAB + ["retreat", "home"]
ET = dict(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=32)
KEYS = ("static_grid16", "static_grid8", "state", "scene_state")
USE_NT = args.tag == "all" and (NT / "results").exists()
OUT = HERE / "analysis"; OUT.mkdir(exist_ok=True)

# ---------------- natural handoffs ----------------
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

# ---------------- repair labels ----------------
reps, states = [], {}


def add_vocab(run, job_files, src, group_of=None):
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
        key = sid + "#" + src
        st = dict(key=key, sid=sid, task=d["task_b"], parent=job["parent_id"], src=src, group=job.get("group", src),
                  k_none=arms["none"]["k"], wins_none=[int(x["success"]) for x in arms["none"]["rollouts"]],
                  random=[c for c, _ in sel[sid]["random"]], cands=[])
        for c, a in arms.items():
            if c == "none":
                continue
            reps.append(dict(key=key, sid=sid, cand=c, k=a["k"], wins=[int(x["success"]) for x in a["rollouts"]],
                             path=str(run / "screen" / job["screen_npz"][c]), file=job["screen_npz"][c], task=d["task_b"],
                             parent=job["parent_id"], src=src))
            st["cands"].append(c)
        states[key] = st


add_vocab(MT, ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"), "vocab")
add_vocab(GT, ("jobs_label.json",), "gate")
if USE_NT:
    add_vocab(NT, ("jobs_nt3_label.json",), "nt")
rp_plan = {s["sample_id"]: s for s in json.loads((RP / "plan.json").read_text())["states"]}
for f in sorted((RP / "results").glob("*.json")):
    d = json.loads(f.read_text()); sid = d["sample_id"]
    if sid not in nat_idx:
        continue
    key = sid + "#hand"
    st = dict(key=key, sid=sid, task=d["task_b"], parent=rp_plan[sid]["parent_id"], src="hand", group="hand_" + d["group"],
              k_none=nat[nat_idx[sid]]["k"], wins_none=None, random=[], cands=[])
    for r in d["repairs"]:
        if r.get("invalid") or r.get("n") != 8:
            continue
        c = r["repair"].replace("lang:", "")
        reps.append(dict(key=key, sid=sid, cand=c, k=r["k"], wins=[int(x["success"]) for x in r["rollouts"]],
                         path=None, file=r["repaired_npz"], task=d["task_b"], parent=st["parent"], src="hand"))
        st["cands"].append(c)
    states[key] = st
print("tag", args.tag, "| natural", len(nat), collections.Counter(r["task"] for r in nat), flush=True)
print("repair arms", len(reps), "| states", collections.Counter(s["src"] for s in states.values()), flush=True)

# ---------------- features ----------------
NZ = ev.features_for([r["npz"] for r in nat], f"v3_natural_{args.tag}.npz")
NZ = {k: NZ[k] for k in KEYS}
R = {k: [None] * len(reps) for k in KEYS}
for fname in ("mt_screen_2170_eddcd0e44786.npz", "mt_screen_4165_aace2b415d9d.npz", "repaired_states_286_51a1f2c969bd.npz"):
    z = np.load(CACHE / fname); mp = {n: i for i, n in enumerate(z["names"])}
    want = [(j, mp[r["file"]]) for j, r in enumerate(reps) if r["src"] in ("vocab", "hand") and r["file"] in mp]
    if want:
        js, iz = zip(*want)
        for k in KEYS:
            arr = z[k][list(iz)]
            for j, a in zip(js, arr):
                R[k][j] = a
for src in ("gate", "nt"):
    js = [j for j, r in enumerate(reps) if r["src"] == src]
    if js:
        G = ev.features_for([reps[j]["path"] for j in js], "gate_repaired.npz" if src == "gate" else "nt_repaired.npz")
        for k in KEYS:
            for j, a in zip(js, G[k]):
                R[k][j] = a
assert all(x is not None for x in R["state"])
R = {k: np.stack(v) for k, v in R.items()}
_raw = np.load(CACHE / "two_task_raw_all_939_b59d93bfcc96.npz")
RAW = {k: _raw[k] for k in ("names", "static_grid16", "static_grid8", "scene_state")}
raw_parent = {}
for f in sorted(Path(f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/raw").glob("gpu*/handoffs.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line); raw_parent[r["observation"]] = r["parent_id"]
rg = np.array([raw_parent[n] for n in RAW["names"]])

nf = np.array([fold(r["parent"]) for r in nat]); rf = np.array([fold(r["parent"]) for r in reps])
half = lambda p: int(hashlib.sha256(("v3-half:" + p).encode()).hexdigest(), 16) % 2 == 0
nh = np.array([half(r["parent"]) for r in nat]); rh = np.array([half(r["parent"]) for r in reps])
r_nat = np.array([nat_idx[r["sid"]] for r in reps])
ntask = [r["task"] for r in nat]; rtask = [r["task"] for r in reps]; rcand = [r["cand"] for r in reps]
none_rows = [(s["key"], nat_idx[s["sid"]], s["wins_none"], s["task"], s["parent"]) for s in states.values() if s["wins_none"] is not None]
xf = np.array([fold(p) for *_, p in none_rows], int); xh = np.array([half(p) for *_, p in none_rows], bool)
x_nat = np.array([i for _, i, *_ in none_rows], int); xtask = [t for *_, t, _ in none_rows]


def labels(j):
    """Success rate and weight over the first j rollouts (j=8 is the full label)."""
    yN = np.array([np.mean(r["wins"][:j]) for r in nat]); wN = np.array([min(j, len(r["wins"])) for r in nat], float)
    yR = np.array([np.mean(r["wins"][:j]) for r in reps]); wR = np.full(len(reps), float(j))
    yX = np.array([np.mean(w[:j]) for _, _, w, *_ in none_rows]) if none_rows else np.zeros(0)
    wX = np.full(len(none_rows), float(j))
    return yN, wN, yR, wR, yX, wX


LAB = {j: labels(j) for j in (2, 4, 8)}
oh_t = lambda ts, tasks=TASKS: np.array([[float(t == u) for u in tasks] for t in ts]).reshape(len(ts), len(tasks))
oh_c = lambda cs: np.array([[float(c == u) for u in CANDS] for c in cs])


def prior_fn(mask_n, mask_x, mask_r, j=8):
    yN, _, yR, _, yX, _ = LAB[j]
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


def fit_rows(XN_, XR_, XX_, mn, mr, mx, j=8):
    yN, wN, yR, wR, yX, wX = LAB[j]
    Xs, ys, ws = [XN_[mn], XR_[mr]], [yN[mn], yR[mr]], [wN[mn], wR[mr]]
    if XX_ is not None and mx.any():
        Xs.append(XX_[mx]); ys.append(yX[mx]); ws.append(wX[mx])
    return ExtraTreesRegressor(**ET).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))


def design(F_n, F_r, prior, use_prior=True):
    """Build [F, onehot(task), onehot(cand), prior] for three blocks: natural handoffs (none), repair arms (pre-repair features), paired none arms."""
    def X_(F_, tasks, cands):
        parts = [F_, oh_t(tasks), oh_c(cands)]
        if use_prior:
            parts.append(np.array([[prior(t, c)] for t, c in zip(tasks, cands)]))
        return np.hstack(parts)
    XN_ = X_(F_n, ntask, ["none"] * len(nat))
    XR_ = X_(F_n[r_nat], rtask, rcand)
    XX_ = X_(F_n[x_nat], xtask, ["none"] * len(x_nat)) if len(x_nat) else None
    return X_, XN_, XR_, XX_


SEL = ("M6", "M7", "M7n", "M7a", "M5v2", "M5n", "M7_j2", "M7_j4", "M7_half", "M5_j4", "M5_half")
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
    variants = [  # name, natural-handoff features, use prior, label rollouts, training-mask modifier
        ("M7", F2N, True, 8, None), ("M7n", FNN, True, 8, None), ("M7a", F1N, True, 8, None), ("M5v2", F2N, False, 8, None),
        ("M5n", FNN, False, 8, None), ("M7_j2", F2N, True, 2, None), ("M7_j4", F2N, True, 4, None), ("M7_half", F2N, True, 8, "half"),
        ("M5_j4", F2N, False, 4, None), ("M5_half", F2N, False, 8, "half")]
    for name, FN_, use_prior, j, mod in variants:
        mn, mr, mx = ntr.copy(), rtr.copy(), xtr.copy()
        if mod == "half":
            mn &= nh; mr &= rh; mx &= xh
        prior, _ = prior_fn(mn, mx, mr, j)
        X_, XN_, XR_, XX_ = design(FN_, None, prior, use_prior)
        m = fit_rows(XN_, XR_, XX_, mn, mr, mx, j)
        S[name][rf == f] = m.predict(XR_[rf == f])
        for s in te_keys:
            i = nat_idx[s["sid"]]
            S0[name][s["key"]] = float(m.predict(X_(FN_[i:i + 1], [s["task"]], ["none"]))[0])
    prior, _ = prior_fn(ntr, xtr, rtr, 8)
    for s in te_keys:
        PRI[s["key"]] = {c: prior(s["task"], c) for c in ["none"] + s["cands"]}
        S0["M6"][s["key"]] = PRI[s["key"]]["none"]
    print(f"fold {f}: natural {ntr.sum()} | repaired {rtr.sum()} | extra none {int(xtr.sum())}", flush=True)

# ---------------- selection evaluation ----------------
by_state = collections.defaultdict(list)
for j, r in enumerate(reps):
    by_state[r["key"]].append(j)


def state_eval(s):
    js = by_state[s["key"]]
    p = {reps[j]["cand"]: reps[j]["k"] / 8 for j in js}
    wins = {reps[j]["cand"]: np.array(reps[j]["wins"]) for j in js}
    p0 = s["k_none"] / 8
    o = dict(key=s["key"], task=s["task"], parent=s["parent"], src=s["src"], group=s["group"], p0=p0, p=p)
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
report = dict(tag=args.tag, data=dict(natural=len(nat), repaired=len(reps), states=dict(collections.Counter(s["src"] for s in states.values()))),
              selection={})
scopes = [("src", v) for v in sorted({r["src"] for r in rows})] + [("family", v) for v in ("old5", "new_inverse", "new_noninverse")] \
         + [("task", t) for t in sorted({r["task"] for r in rows})] + [("all", "ALL")]
print(f"\n{'scope':34s} {'n':>4s} {'p0':>6s} " + " ".join(f"{m:>7s}" for m in SEL) + f" {'rand':>6s} {'orac½':>6s}")
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
    report["selection"][f"{kind}|{v}"] = rec
    print(f"{kind + ':' + v:34s} {len(rs):4d} {rec['p0']:6.3f} " + " ".join(f"{rec[m]:7.3f}" for m in SEL)
          + f" {(rec['rand'] if rec['rand'] is not None else float('nan')):6.3f} {rec['oracle_split']:6.3f}")
print("\ndifferences vs M6 [95% CI]:")
for key, rec in report["selection"].items():
    print(f"  {key:34s} " + "  ".join(f"{m}={rec[f'{m}_minus_M6'][0]:+.3f}[{rec[f'{m}_minus_M6'][1]:+.3f},{rec[f'{m}_minus_M6'][2]:+.3f}]"
                                      for m in ("M7", "M7n", "M5v2", "M5n", "M5_j4", "M5_half")))
(OUT / f"v3_rows_{args.tag}.json").write_text(json.dumps(rows, default=float))

# ---------------- final deployable model (all data) ----------------
G16 = np.concatenate([RAW["static_grid16"], NZ["static_grid16"], R["static_grid16"]]).astype(np.float32)
G8 = np.concatenate([RAW["static_grid8"], NZ["static_grid8"], R["static_grid8"]])
SC = np.concatenate([RAW["scene_state"], NZ["scene_state"], R["scene_state"]])
fz = Featurizer2().fit(G16, G8, SC, seed=0); del G16
fnl = FeaturizerNL().fit(G8, seed=0)
F2N = fz.transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])
FNN = fnl.transform(NZ["static_grid8"], NZ["state"])
F2R = fz.transform(R["static_grid16"].astype(np.float32), R["static_grid8"], R["state"])
alln, allr, allx = np.ones(len(nat), bool), np.ones(len(reps), bool), np.ones(len(none_rows), bool)
prior, agg = prior_fn(alln, allx, allr, 8)
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
yN, wN, yR, wR, yX, wX = LAB[8]
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
             data=report["data"])
if not args.no_save:
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(final, HERE / "models" / f"benefit_models_v3_{args.tag}.joblib")
    print("saved", HERE / "models" / f"benefit_models_v3_{args.tag}.joblib", flush=True)

# ---------------- leave-one-task-out ----------------
if not args.no_loto:
    report["loto"] = {}
    lab_tasks = sorted({s["task"] for s in states.values()})
    print("\n==== leave-one-task-out (task never seen in training) ====")
    for T in lab_tasks:
        others = [t for t in TASKS if t != T]
        mn = np.array([t != T for t in ntask]); mr = np.array([t != T for t in rtask]); mx = np.array([t != T for t in xtask], bool)
        _, agg_o = prior_fn(mn, mx, mr, 8)
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
(OUT / f"v3_report_{args.tag}.json").write_text(json.dumps(report, indent=1, default=float))
print("\nsaved", OUT / f"v3_report_{args.tag}.json")
