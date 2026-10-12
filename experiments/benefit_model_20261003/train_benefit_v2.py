# -*- coding: utf-8 -*-
"""Benefit model v2: state features v2 (adds orientation, relative geometry, local appearance) + "task prior plus state correction", and gate evaluation.

Data (used when present)
  671 natural handoffs; 181 vocabulary repairs, low-score (vocab); 82 hand-designed repairs (hand);
  150 vocabulary repairs, natural distribution (gate, from gating_vocab_20261003, with paired "no repair" arms).
Models (5-fold cross-fitting by initial configuration; the featurizer of each fold uses only training-fold initial configurations)
  M2   v1 features, post-repair verifier              M2v2 v2 features, post-repair verifier
  M5   v1 features, pre-repair instruction model      M5v2 v2 features, pre-repair instruction model
  M6   task-level prior (ignores the state)
  M7   v2 pre-repair features + task/instruction + training-fold prior value (task prior plus state correction; decides before repairing)
  M7v  M7 plus post-repair v2 features (post-repair verifier variant)
Evaluation
  Selection: actual success rate of the one repair picked among labelled repairs, a variant that includes the "no repair" option, pairwise ranking
  concordance, and paired-difference intervals against M6.
  Gating (gate and hand handoffs, which all have a "no repair" outcome), with the same scorer:
    risk gate: repair only if the predicted no-repair success rate is below a threshold; benefit gate: repair only if the predicted repair benefit is above a threshold;
    thresholds are chosen only on training-fold handoffs (objective: mean actual success rate); never-repair, always-repair and a fixed per-task plan are reported as well.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
CALVIN_ROOT = os.environ.get("CALVIN_ROOT") or "calvin"
import os, sys, json, hashlib, collections
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "32")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{RUN_ROOT}/fix_20260928")
import numpy as np
import torch
torch.set_num_threads(32)
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from benefit_lib_v2 import Featurizer2

FIX = Path(f"{RUN_ROOT}/fix_20260928")
MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
RP = Path(f"{RUN_ROOT}/repair_20261001")
GT = Path(f"{RUN_ROOT}/gating_vocab_20261003")
CACHE = FIX / "cache"
OOF = Path(f"{RHO_ROOT}/outputs/experiments/offline_full_20260929_105330/evaluations/cv/oof_predictions.json")
VOCAB = sorted(__import__("yaml").safe_load(open(f"{CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml")))
CANDS = ["none"] + VOCAB + ["retreat", "home"]
ET = dict(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=32)
KEYS = ("static_grid16", "static_grid8", "state", "scene_state")


def npz_rows(fname, names):
    z = np.load(CACHE / fname); mp = {n: i for i, n in enumerate(z["names"])}
    idx = [mp[n] for n in names]
    return {k: z[k][idx] for k in KEYS}


# ---------------- data ----------------
nat = []
for g in sorted((FIX / "runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            nat.append(dict(sid=r["sample_id"], file=r["observation"], k=r["k"], n=r["n"], task=r["task_b"], parent=r["parent_id"]))
for f in sorted((MT / "results").glob("j8_*.json")):
    d = json.loads(f.read_text())
    if d.get("reached") and d.get("n") == 8 and "restore_failed" not in d:
        nat.append(dict(sid=d["sample_id"], file=d["observation"], k=d["k"], n=d["n"], task=d["task_b"], parent=d["parent_id"]))
TASKS = sorted({r["task"] for r in nat})
nat_idx = {}
for i, r in enumerate(nat):
    nat_idx.setdefault(r["sid"], i)
fold_of = {}
oof = json.loads(OOF.read_text())
for s, fo in zip(oof["samples"], oof["methods"]["tree | vision+robot15"]["folds"]):
    fold_of[s["parent_id"]] = int(fo)
fold = lambda p: fold_of.setdefault(p, int(hashlib.sha256(("mt-fold:" + p).encode()).hexdigest(), 16) % 5)

reps, states = [], {}


def add_vocab(run, job_files, src):
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
        st = dict(key=key, sid=sid, task=d["task_b"], parent=job["parent_id"], src=src, k_none=arms["none"]["k"],
                  wins_none=[int(x["success"]) for x in arms["none"]["rollouts"]], random=[c for c, _ in sel[sid]["random"]],
                  m0=sel[sid]["all_scores"], cands=[])
        rsteps = {}
        sp = run / "results" / ("screen_" + f.name[len("label_"):])
        if sp.exists():
            rsteps = {c["candidate"]: c.get("steps") for c in json.loads(sp.read_text())["candidates"] if "invalid" not in c}
        st["bsteps_none"] = float(np.mean([x["steps"] for x in arms["none"]["rollouts"]]))
        for c, a in arms.items():
            if c == "none":
                continue
            reps.append(dict(key=key, sid=sid, cand=c, k=a["k"], wins=[int(x["success"]) for x in a["rollouts"]],
                             file=job["screen_npz"][c], run=str(run), task=d["task_b"], parent=job["parent_id"], src=src,
                             k_base=arms["none"]["k"], rsteps=rsteps.get(c),
                             bsteps=float(np.mean([x["steps"] for x in a["rollouts"]]))))
            st["cands"].append(c)
        states[key] = st


add_vocab(MT, ("jobs_phase2_label_old.json", "jobs_phase3_label_new.json"), "vocab")
add_vocab(GT, ("jobs_label.json",), "gate")
rp_plan = {s["sample_id"]: s for s in json.loads((RP / "plan.json").read_text())["states"]}
for f in sorted((RP / "results").glob("*.json")):
    d = json.loads(f.read_text()); sid = d["sample_id"]
    if sid not in nat_idx:
        continue
    key = sid + "#hand"
    st = dict(key=key, sid=sid, task=d["task_b"], parent=rp_plan[sid]["parent_id"], src="hand", group=d["group"],
              k_none=nat[nat_idx[sid]]["k"], wins_none=None, random=[], m0={}, cands=[], bsteps_none=None)
    for r in d["repairs"]:
        if r.get("invalid") or r.get("n") != 8:
            continue
        c = r["repair"].replace("lang:", "")
        reps.append(dict(key=key, sid=sid, cand=c, k=r["k"], wins=[int(x["success"]) for x in r["rollouts"]],
                         file=r["repaired_npz"], run=str(RP), task=d["task_b"], parent=st["parent"], src="hand", k_base=st["k_none"],
                         rsteps=r.get("repair_steps"), bsteps=float(np.mean([x["steps"] for x in r["rollouts"]]))))
        st["cands"].append(c)
    states[key] = st
print("natural", len(nat), "| repair arms", len(reps), "| states", collections.Counter(s["src"] for s in states.values()), flush=True)

# ---------------- features ----------------
NZ = npz_rows("mt_natural_671_9e9e2921abe9.npz", [r["file"] for r in nat])
R = {k: [None] * len(reps) for k in KEYS}
for fname in ("mt_screen_2170_eddcd0e44786.npz", "mt_screen_4165_aace2b415d9d.npz", "repaired_states_286_51a1f2c969bd.npz"):
    z = np.load(CACHE / fname); mp = {n: i for i, n in enumerate(z["names"])}
    want = [(j, mp[r["file"]]) for j, r in enumerate(reps) if r["src"] in ("vocab", "hand") and r["file"] in mp]
    if want:
        js, iz = zip(*want)
        for k in KEYS:
            for j, a in zip(js, z[k][list(iz)]):
                R[k][j] = a
gate_j = [j for j, r in enumerate(reps) if r["src"] == "gate"]
if gate_j:
    import evaluate_cv as ev
    G = ev.features_for([str(Path(reps[j]["run"]) / "screen" / reps[j]["file"]) for j in gate_j], "gate_repaired.npz")
    for k in KEYS:
        for j, a in zip(gate_j, G[k]):
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
oh_t = lambda ts: np.array([[float(t == u) for u in TASKS] for t in ts])
oh_c = lambda cs: np.array([[float(c == u) for u in CANDS] for c in cs])
r_nat = np.array([nat_idx[r["sid"]] for r in reps])
nK = np.array([r["k"] for r in nat], float); nN = np.array([r["n"] for r in nat], float)
rK = np.array([r["k"] for r in reps], float)
ntask = [r["task"] for r in nat]; rtask = [r["task"] for r in reps]; rcand = [r["cand"] for r in reps]
# handoffs with a paired "no repair" arm: used as extra none training samples
none_rows = [(s["key"], nat_idx[s["sid"]], s["k_none"], s["task"], s["parent"]) for s in states.values() if s["wins_none"] is not None]
xf = np.array([fold(p) for *_, p in none_rows]) if none_rows else np.zeros(0, int)
x_nat = np.array([i for _, i, *_ in none_rows], int); xK = np.array([k for _, _, k, *_ in none_rows], float)
xtask = [t for *_, t, _ in none_rows]

MODELS = ("M2", "M2v2", "M5", "M5v2", "M7", "M7v")
S = {m: np.full(len(reps), np.nan) for m in MODELS}
S0 = {m: {} for m in MODELS}           # handoff key -> predicted no-repair success rate
PRIOR_SEL = {}                          # handoff key -> M6 prior score of each candidate (including none)
for f in range(5):
    ntr, rtr, xtr = nf != f, rf != f, xf != f
    test_par = {r["parent"] for r, x in zip(nat, nf) if x == f} | {r["parent"] for r, x in zip(reps, rf) if x == f}
    raw_tr = ~np.isin(rg, list(test_par))
    G16 = np.concatenate([RAW["static_grid16"][raw_tr], NZ["static_grid16"][ntr], R["static_grid16"][rtr]]).astype(np.float32)
    G8 = np.concatenate([RAW["static_grid8"][raw_tr], NZ["static_grid8"][ntr], R["static_grid8"][rtr]])
    SC = np.concatenate([RAW["scene_state"][raw_tr], NZ["scene_state"][ntr], R["scene_state"][rtr]])
    fz = Featurizer2().fit(G16, G8, SC, seed=0)
    F2N = fz.transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])
    F2R = fz.transform(R["static_grid16"].astype(np.float32), R["static_grid8"], R["state"])
    F1N, F1R = F2N[:, :32], F2R[:, :32]
    F2O, F1O = F2N[r_nat], F1N[r_nat]
    # training-fold prior: mean success rate per (task, instruction); none uses natural handoffs and paired none arms
    agg = collections.defaultdict(list)
    for i in np.flatnonzero(ntr):
        agg[(ntask[i], "none")].append(nK[i] / nN[i])
    for j in np.flatnonzero(xtr):
        agg[(xtask[j], "none")].append(xK[j] / 8)
    for j in np.flatnonzero(rtr):
        agg[(rtask[j], rcand[j])].append(rK[j] / 8)
    tmean = {t: np.mean([v for (tt, c), vs in agg.items() if tt == t and c != "none" for v in vs]) for t in TASKS}
    prior = lambda t, c: float(np.mean(agg[(t, c)])) if agg.get((t, c)) else (float(np.mean(agg[(t, "none")])) if c == "none" else tmean[t])
    pr_r = np.array([prior(t, c) for t, c in zip(rtask, rcand)]); pr_n = np.array([prior(t, "none") for t in ntask])

    def fit_pair(XN_, XR_, extra_none=None):
        Xs = [XN_[ntr], XR_[rtr]]; ys = [nK[ntr] / nN[ntr], rK[rtr] / 8]; ws = [nN[ntr], np.full(rtr.sum(), 8.0)]
        if extra_none is not None and xtr.any():
            Xs.append(extra_none[xtr]); ys.append(xK[xtr] / 8); ws.append(np.full(xtr.sum(), 8.0))
        return ExtraTreesRegressor(**ET).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))

    TN, TR = oh_t(ntask), oh_t(rtask)
    # post-repair verifier (none = the natural observation itself)
    for name, FN_, FR_ in (("M2", F1N, F1R), ("M2v2", F2N, F2R)):
        XN_, XR_ = np.hstack([FN_, TN]), np.hstack([FR_, TR])
        XX = np.hstack([FN_[x_nat], oh_t(xtask)]) if len(x_nat) else None
        m = fit_pair(XN_, XR_, XX)
        S[name][rf == f] = m.predict(XR_[rf == f])
        for key, i, _, t, p in none_rows + [(s["key"], nat_idx[s["sid"]], None, s["task"], s["parent"]) for s in states.values() if s["wins_none"] is None]:
            if fold(p) == f:
                S0[name][key] = float(m.predict(np.hstack([FN_[i:i + 1], oh_t([t])]))[0])
    # pre-repair instruction models and "prior plus state correction"
    for name, FO_, FN_, use_prior, post in (("M5", F1O, F1N, False, False), ("M5v2", F2O, F2N, False, False),
                                            ("M7", F2O, F2N, True, False), ("M7v", F2O, F2N, True, True)):
        def X_(F_pre, tasks, cands, pri, F_post=None):
            parts = [F_pre, oh_t(tasks), oh_c(cands)]
            if use_prior:
                parts.append(np.asarray(pri)[:, None])
            if post:
                parts.append(F_post if F_post is not None else F_pre)
            return np.hstack(parts)
        XN_ = X_(FN_, ntask, ["none"] * len(nat), pr_n)
        XR_ = X_(FO_, rtask, rcand, pr_r, F2R if post else None)
        XX = X_(FN_[x_nat], xtask, ["none"] * len(x_nat), [prior(t, "none") for t in xtask]) if len(x_nat) else None
        m = fit_pair(XN_, XR_, XX)
        S[name][rf == f] = m.predict(XR_[rf == f])
        for s in states.values():
            if fold(s["parent"]) == f:
                i = nat_idx[s["sid"]]
                S0[name][s["key"]] = float(m.predict(X_(FN_[i:i + 1], [s["task"]], ["none"], [prior(s["task"], "none")]))[0])
    for s in states.values():
        if fold(s["parent"]) == f:
            PRIOR_SEL[s["key"]] = {c: prior(s["task"], c) for c in ["none"] + s["cands"]}
    print(f"fold {f}: natural {ntr.sum()} | repaired {rtr.sum()} | extra none {int(xtr.sum())} | features {F2N.shape[1]}", flush=True)

# ---------------- selection evaluation ----------------
by_state = collections.defaultdict(list)
for j, r in enumerate(reps):
    by_state[r["key"]].append(j)


def state_eval(s):
    js = by_state[s["key"]]
    p = {reps[j]["cand"]: reps[j]["k"] / 8 for j in js}
    wins = {reps[j]["cand"]: np.array(reps[j]["wins"]) for j in js}
    p0 = s["k_none"] / 8
    o = dict(key=s["key"], task=s["task"], parent=s["parent"], src=s["src"], group=s.get("group", s["src"]), p0=p0, p=p,
             rsteps={reps[j]["cand"]: reps[j]["rsteps"] for j in js}, bsteps={reps[j]["cand"]: reps[j]["bsteps"] for j in js},
             bsteps_none=s.get("bsteps_none"))
    sc_all = {m: {reps[j]["cand"]: S[m][j] for j in js} for m in MODELS}
    sc_all["M6"] = {c: PRIOR_SEL[s["key"]][c] for c in p}
    if s["m0"]:
        sc_all["M0"] = {c: s["m0"][c] for c in p}
    s0 = {m: S0[m].get(s["key"]) for m in MODELS}
    s0["M6"] = PRIOR_SEL[s["key"]]["none"]
    o["scores"], o["none_scores"] = sc_all, s0
    for m, sc in sc_all.items():
        best = max(sc, key=sc.get)
        o["pick_" + m] = p[best]
        if m in s0 and s0[m] is not None:
            o["pick0_" + m] = p[best] if sc[best] > s0[m] else p0
        o["conc_" + m] = [float(np.sign(sc[a] - sc[b]) == np.sign(p[a] - p[b])) for a in p for b in p
                          if a < b and p[a] != p[b] and sc[a] != sc[b]]
    halves = []
    for a, b in ((slice(0, 4), slice(4, 8)), (slice(4, 8), slice(0, 4))):
        bst = max(wins, key=lambda c: wins[c][a].mean()); halves.append(wins[bst][b].mean())
    o["oracle_split"] = float(np.mean(halves))
    o["rand"] = float(np.mean([p[c] for c in s["random"] if c in p])) if s["random"] else np.nan
    return o


rows = [state_eval(s) for s in states.values() if by_state[s["key"]]]


def boot(rs, fn, n=2000, seed=9):
    ps = sorted({r["parent"] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r["parent"]].append(r)
    rng = np.random.RandomState(seed)
    v = [fn([x for q in rng.choice(ps, len(ps)) for x in by[q]]) for _ in range(n)]
    return float(fn(rs)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


report = dict(selection={}, gating={})
SEL_MODELS = ["M0", "M2", "M2v2", "M5", "M5v2", "M6", "M7", "M7v"]
for src in ("vocab", "gate", "hand"):
    rs_src = [r for r in rows if r["src"] == src]
    if not rs_src:
        continue
    print(f"\n==== selection among labeled repairs: {src} ====")
    print(f"{'task':26s} {'n':>3s} {'p0':>6s} " + " ".join(f"{m:>6s}" for m in SEL_MODELS) + f" {'rand':>6s} {'orac½':>6s}")
    for t in sorted({r["task"] for r in rs_src}) + ["ALL"]:
        rs = [r for r in rs_src if t == "ALL" or r["task"] == t]
        val = {m: float(np.mean([r["pick_" + m] for r in rs])) if all("pick_" + m in r for r in rs) else np.nan for m in SEL_MODELS}
        rec = dict(n=len(rs), p0=float(np.mean([r["p0"] for r in rs])), **val,
                   rand=float(np.nanmean([r["rand"] for r in rs])) if src != "hand" else None,
                   oracle_split=float(np.mean([r["oracle_split"] for r in rs])),
                   conc={m: float(np.mean(sum((r.get("conc_" + m, []) for r in rs), []))) if any(r.get("conc_" + m) for r in rs) else None for m in SEL_MODELS})
        for m in ("M2v2", "M5v2", "M7", "M7v"):
            rec[f"{m}_minus_M6"] = boot(rs, lambda x, m=m: float(np.mean([r["pick_" + m] - r["pick_M6"] for r in x])))
        report["selection"][f"{src}|{t}"] = rec
        print(f"{t:26s} {len(rs):3d} {rec['p0']:6.3f} " + " ".join("   n/a" if np.isnan(val[m]) else f"{val[m]:6.3f}" for m in SEL_MODELS)
              + f" {(rec['rand'] if rec['rand'] is not None else float('nan')):6.3f} {rec['oracle_split']:6.3f}")
    print("  differences vs M6 (task prior) [95% CI]:")
    for t in sorted({r["task"] for r in rs_src}) + ["ALL"]:
        rec = report["selection"][f"{src}|{t}"]
        print(f"    {t:26s} " + "  ".join(f"{m}-M6={rec[f'{m}_minus_M6'][0]:+.3f}[{rec[f'{m}_minus_M6'][1]:+.3f},{rec[f'{m}_minus_M6'][2]:+.3f}]"
                                         for m in ("M2v2", "M5v2", "M7", "M7v")))
    print("  pairwise concordance:", {m: (None if report['selection'][f'{src}|ALL']['conc'][m] is None else round(report['selection'][f'{src}|ALL']['conc'][m], 3)) for m in SEL_MODELS})

# ---------------- gate evaluation (requires "no repair" outcomes; gate and hand) ----------------
gate_rows = [r for r in rows if r["src"] in ("gate", "hand")]
if gate_rows:
    print("\n==== gating: when to repair (states with no-repair outcome; thresholds chosen on training folds) ====")
    folds_r = np.array([fold(r["parent"]) for r in gate_rows])
    TAU = [-np.inf] + list(np.linspace(0, 1, 21)) + [np.inf]
    DEL = [-np.inf] + list(np.round(np.arange(-0.3, 0.51, 0.05), 3)) + [np.inf]
    for m in ("M2v2", "M7", "M7v", "M6"):
        dec = {}
        for r in gate_rows:
            sc = r["scores"][m]; u = max(sc, key=sc.get)
            dec[r["key"]] = (u, sc[u], r["none_scores"][m])
        out = collections.defaultdict(list)
        for f in range(5):
            tr = [r for r, x in zip(gate_rows, folds_r) if x != f]; te = [r for r, x in zip(gate_rows, folds_r) if x == f]
            if not te:
                continue
            real = lambda r, rep: r["p"][dec[r["key"]][0]] if rep else r["p0"]
            best_tau = max(TAU, key=lambda tau: np.mean([real(r, dec[r["key"]][2] < tau) for r in tr]))
            best_del = max(DEL, key=lambda de: np.mean([real(r, dec[r["key"]][1] - dec[r["key"]][2] > de) for r in tr]))
            for r in te:
                u, qu, q0 = dec[r["key"]]
                out[r["key"]] = dict(never=r["p0"], always=r["p"][u], risk=real(r, q0 < best_tau), benefit=real(r, qu - q0 > best_del),
                                     repaired_risk=int(q0 < best_tau), repaired_benefit=int(qu - q0 > best_del))
        recs = []
        for r in gate_rows:
            if r["key"] in out:
                recs.append(dict(parent=r["parent"], src=r["src"], task=r["task"], **out[r["key"]]))
        for scope in ("gate", "hand", "ALL"):
            rs = [x for x in recs if scope == "ALL" or x["src"] == scope]
            if not rs:
                continue
            mean = {k: float(np.mean([x[k] for x in rs])) for k in ("never", "always", "risk", "benefit", "repaired_risk", "repaired_benefit")}
            d_br = boot(rs, lambda x: float(np.mean([y["benefit"] - y["risk"] for y in x])))
            d_bn = boot(rs, lambda x: float(np.mean([y["benefit"] - y["never"] for y in x])))
            d_ba = boot(rs, lambda x: float(np.mean([y["benefit"] - y["always"] for y in x])))
            report["gating"][f"{m}|{scope}"] = dict(n=len(rs), **mean, benefit_minus_risk=d_br, benefit_minus_never=d_bn, benefit_minus_always=d_ba)
            print(f"  {m:5s} {scope:5s} n={len(rs):3d} never={mean['never']:.3f} always={mean['always']:.3f} risk={mean['risk']:.3f}"
                  f"({100*mean['repaired_risk']:.0f}% repaired) benefit={mean['benefit']:.3f}({100*mean['repaired_benefit']:.0f}% repaired) "
                  f"| benefit-risk={d_br[0]:+.3f}[{d_br[1]:+.3f},{d_br[2]:+.3f}] benefit-never={d_bn[0]:+.3f}[{d_bn[1]:+.3f},{d_bn[2]:+.3f}] "
                  f"benefit-always={d_ba[0]:+.3f}[{d_ba[1]:+.3f},{d_ba[2]:+.3f}]")
else:
    print("\n(no gate data yet: gating evaluation skipped)")

(HERE / "analysis").mkdir(exist_ok=True)
tag = "with_gating" if any(r["src"] == "gate" for r in rows) else "pre_gating"
(HERE / "analysis" / f"v2_rows_{tag}.json").write_text(json.dumps(rows, default=float))

# ---------------- final deployable model (all dev data) ----------------
if os.environ.get("SAVE_FINAL", "1") == "1":
    G16 = np.concatenate([RAW["static_grid16"], NZ["static_grid16"], R["static_grid16"]]).astype(np.float32)
    G8 = np.concatenate([RAW["static_grid8"], NZ["static_grid8"], R["static_grid8"]])
    SC = np.concatenate([RAW["scene_state"], NZ["scene_state"], R["scene_state"]])
    fz = Featurizer2().fit(G16, G8, SC, seed=0)
    F2N = fz.transform(NZ["static_grid16"].astype(np.float32), NZ["static_grid8"], NZ["state"])
    F2R = fz.transform(R["static_grid16"].astype(np.float32), R["static_grid8"], R["state"])
    F2O = F2N[r_nat]
    agg = collections.defaultdict(list)
    for i in range(len(nat)):
        agg[(ntask[i], "none")].append(nK[i] / nN[i])
    for j in range(len(xK)):
        agg[(xtask[j], "none")].append(xK[j] / 8)
    for j in range(len(reps)):
        agg[(rtask[j], rcand[j])].append(rK[j] / 8)
    prior_table = {f"{t}|{c}": float(np.mean(v)) for (t, c), v in agg.items()}
    tmean = {t: float(np.mean([v for (tt, c), vs in agg.items() if tt == t and c != "none" for v in vs])) for t in TASKS}

    def prior(t, c):
        if f"{t}|{c}" in prior_table:
            return prior_table[f"{t}|{c}"]
        return prior_table[f"{t}|none"] if c == "none" else tmean[t]
    pr_r = np.array([prior(t, c) for t, c in zip(rtask, rcand)]); pr_n = np.array([prior(t, "none") for t in ntask])
    pr_x = np.array([prior(t, "none") for t in xtask])
    yN, wN, yR, wR = nK / nN, nN, rK / 8, np.full(len(reps), 8.0)

    def fit(XN_, XR_, XX_):
        Xs, ys, ws = [XN_, XR_], [yN, yR], [wN, wR]
        if len(xK):
            Xs.append(XX_); ys.append(xK / 8); ws.append(np.full(len(xK), 8.0))
        return ExtraTreesRegressor(**ET).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    TN, TR, TX = oh_t(ntask), oh_t(rtask), oh_t(xtask)
    CN, CR, CX = oh_c(["none"] * len(nat)), oh_c(rcand), oh_c(["none"] * len(xK))
    final = dict(
        featurizer=fz, tasks=TASKS, cands=CANDS, prior_table=prior_table, task_mean=tmean,
        M2v2=fit(np.hstack([F2N, TN]), np.hstack([F2R, TR]), np.hstack([F2N[x_nat], TX])),
        M5v2=fit(np.hstack([F2N, TN, CN]), np.hstack([F2O, TR, CR]), np.hstack([F2N[x_nat], TX, CX])),
        M7=fit(np.hstack([F2N, TN, CN, pr_n[:, None]]), np.hstack([F2O, TR, CR, pr_r[:, None]]),
               np.hstack([F2N[x_nat], TX, CX, pr_x[:, None]])),
        M7v=fit(np.hstack([F2N, TN, CN, pr_n[:, None], F2N]), np.hstack([F2O, TR, CR, pr_r[:, None], F2R]),
                np.hstack([F2N[x_nat], TX, CX, pr_x[:, None], F2N[x_nat]])),
        layout=dict(M2v2="[F(o'), onehot(task)]", M5v2="[F(o), onehot(task), onehot(cand)]",
                    M7="[F(o), onehot(task), onehot(cand), prior]", M7v="[F(o), onehot(task), onehot(cand), prior, F(o')]",
                    none="cand='none' and o'=o"),
        data=dict(natural=len(nat), repaired=len(reps), extra_none=len(xK)))
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(final, HERE / "models" / "benefit_models_v2.joblib")
    print("saved", HERE / "models" / "benefit_models_v2.joblib", flush=True)

(HERE / "analysis" / f"v2_report_{tag}.json").write_text(json.dumps(report, indent=1, default=float))
print("\nsaved", HERE / "analysis" / f"v2_report_{tag}.json")
