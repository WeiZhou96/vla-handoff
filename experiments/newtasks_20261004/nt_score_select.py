# -*- coding: utf-8 -*-
"""Extended tasks, phase-2 preparation (CPU): cross-fitted scoring of the screened post-repair states, selection of the label branches, and generation of label jobs.
Same algorithm as multitask_vocab_20261002/score_and_select.py (post-repair verifier: localisation features + ridge regression + shallow trees, 5-fold cross-fitting by initial configuration),
with the natural-handoff training data extended to 11 tasks.
Label branches = none + the 4 highest-scoring candidates + 2 of the remaining candidates chosen by a fixed hash + one pre-specified rule candidate (if not already included):
  tasks with an inverse skill use that inverse skill (INVERSE table), all other tasks use the scripted home. The rule candidate is used for offline comparison with the rule baseline and does not take part in model selection.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import os, sys, json, hashlib, argparse, collections
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "24")
HERE = Path(__file__).resolve().parent
FIX = Path(f"{RUN_ROOT}/fix_20260928")
MT = Path(f"{RUN_ROOT}/multitask_vocab_20261002")
sys.path.insert(0, str(FIX))
import numpy as np
import torch
torch.set_num_threads(24)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
import evaluate_cv as ev
from object_locator import fit_locator_normalized

INVERSE = {"push_red_block_right": "push_red_block_left", "push_blue_block_right": "push_blue_block_left",
           "rotate_pink_block_right": "rotate_pink_block_left", "push_pink_block_right": "push_pink_block_left",
           "rotate_red_block_left": "rotate_red_block_right"}
ap = argparse.ArgumentParser()
ap.add_argument("--screen-jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=4)
ap.add_argument("--random", type=int, default=2)
args = ap.parse_args()

OOF = json.loads(Path(f"{RHO_ROOT}/outputs/experiments/offline_full_20260929_105330/evaluations/cv/oof_predictions.json").read_text())
fold_of_parent = {}
for s, f in zip(OOF["samples"], OOF["methods"]["tree | vision+robot15"]["folds"]):
    fold_of_parent[s["parent_id"]] = int(f)
fold = lambda p: fold_of_parent.setdefault(p, int(hashlib.sha256(("mt-fold:" + p).encode()).hexdigest(), 16) % 5)
nat = []
for g in sorted((FIX / "runs/j8_two_task").glob("gpu*/labels.jsonl")):
    for line in g.read_text().splitlines():
        r = json.loads(line)
        if r.get("reached") and r.get("n") and "restore_failed" not in r:
            nat.append(dict(npz=str(g.parent / "boundaries" / r["observation"]), k=r["k"], n=r["n"], task=r["task_b"], parent=r["parent_id"]))
for run in (MT, HERE):
    for f in sorted((run / "results").glob("j8_*.json")):
        d = json.loads(f.read_text())
        if d.get("reached") and d.get("n") == 8 and "restore_failed" not in d:
            nat.append(dict(npz=str(run / "boundaries" / d["observation"]), k=d["k"], n=d["n"], task=d["task_b"], parent=d["parent_id"]))
tasks = sorted({r["task"] for r in nat})
print("natural boundaries", len(nat), collections.Counter(r["task"] for r in nat), flush=True)

jobs = {j["id"]: j for j in json.loads(Path(args.screen_jobs).read_text())["jobs"] if j["type"] == "screen"}
screens = []
for jid, job in jobs.items():
    p = HERE / "results" / f"{jid}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    for c in d["candidates"]:
        if "invalid" in c:
            continue
        screens.append(dict(job=jid, sample_id=job["sample_id"], task=job["task_b"], parent=job["parent_id"],
                            cand=c["candidate"], npz=str(HERE / "screen" / c["npz"]), steps=c["steps"]))
print("screened states", len(screens), "from jobs", len({s['job'] for s in screens}), flush=True)

NF = ev.features_for([r["npz"] for r in nat], "nt_all_natural.npz")
SF = ev.features_for([s["npz"] for s in screens], "nt_screen.npz")
RF, rg = ev.raw_boundaries()
nfold = np.array([fold(r["parent"]) for r in nat]); sfold = np.array([fold(s["parent"]) for s in screens])
K = np.array([r["k"] for r in nat], float); N = np.array([r["n"] for r in nat], float)
ntask = np.array([r["task"] for r in nat]); stask = np.array([s["task"] for s in screens])
onehot = lambda arr: np.array([[float(t == u) for u in tasks] for t in arr])


def feats(loc, ridges, G16, G8, rob):
    xyz, conf = loc(G16.astype(np.float32))
    art = np.column_stack([m.predict(sc.transform(G8)) for sc, m in ridges])
    dist = np.linalg.norm(xyz - rob[:, None, :3], axis=2)
    return np.hstack([xyz.reshape(len(rob), -1), conf, dist, art, rob])


score = np.full(len(screens), np.nan)
for f in range(5):
    tr = nfold != f
    te_s = sfold == f
    if not te_s.any():
        continue
    test_parents = {screens[i]["parent"] for i in np.flatnonzero(te_s)}
    raw_tr = ~np.isin(rg, list(test_parents))
    tr = tr & ~np.isin(np.array([r["parent"] for r in nat]), list(test_parents))
    Ptr = np.concatenate([RF["static_grid16"][raw_tr], NF["static_grid16"][tr]]).astype(np.float32)
    Str = np.concatenate([RF["scene_state"][raw_tr], NF["scene_state"][tr]])
    loc = fit_locator_normalized(Ptr, Str, epochs=600, seed=0)
    G8tr = np.concatenate([RF["static_grid8"][raw_tr], NF["static_grid8"][tr]])
    ridges = []
    for c in (0, 1):
        sc = StandardScaler().fit(G8tr)
        ridges.append((sc, RidgeCV(alphas=np.logspace(0, 5, 11)).fit(sc.transform(G8tr), Str[:, c])))
    Xn = feats(loc, ridges, NF["static_grid16"][tr], NF["static_grid8"][tr], NF["state"][tr])
    m = ExtraTreesRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0, n_jobs=24)
    m.fit(np.hstack([Xn, onehot(ntask[tr])]), K[tr] / N[tr], sample_weight=N[tr])
    Xs = feats(loc, ridges, SF["static_grid16"][te_s], SF["static_grid8"][te_s], SF["state"][te_s])
    score[te_s] = m.predict(np.hstack([Xs, onehot(stask[te_s])]))
    print(f"fold {f}: trained on {tr.sum()} natural, scored {te_s.sum()} screened", flush=True)
assert not np.isnan(score).any()

by = collections.defaultdict(list)
for s, sc in zip(screens, score):
    s["score"] = float(sc); by[s["sample_id"]].append(s)
label_jobs, sel_log = [], {}
for sid, rows in by.items():
    rows.sort(key=lambda r: -r["score"])
    top = rows[: args.top]
    rest = sorted(rows[args.top:], key=lambda r: hashlib.sha256(("mt-rand:" + sid + ":" + r["cand"]).encode()).hexdigest())
    rnd = rest[: args.random]
    job = jobs[rows[0]["job"]]
    rule = INVERSE.get(job["task_b"], "home")
    chosen = top + rnd
    forced = []
    if rule not in {r["cand"] for r in chosen}:
        forced = [r for r in rows if r["cand"] == rule]
    arms = ["none"] + [r["cand"] for r in chosen + forced]
    label_jobs.append(dict(type="label", id="label_" + hashlib.sha256(sid.encode()).hexdigest()[:24], sample_id=sid, group=job["group"],
                           index=job["index"], sequence_id=job["sequence_id"], task_b=job["task_b"], npz=job["npz"],
                           parent_id=job["parent_id"], arms=arms, rule_candidate=rule,
                           screen_npz={r["cand"]: Path(r["npz"]).name for r in chosen + forced}))
    sel_log[sid] = dict(task=job["task_b"], group=job["group"], top=[(r["cand"], r["score"]) for r in top],
                        random=[(r["cand"], r["score"]) for r in rnd], forced=[r["cand"] for r in forced],
                        all_scores={r["cand"]: r["score"] for r in rows})
# interleave tasks
by_t = collections.defaultdict(list)
for j in label_jobs:
    by_t[j["task_b"]].append(j)
inter = []
for i in range(max(map(len, by_t.values()))):
    for t in sorted(by_t):
        if i < len(by_t[t]):
            inter.append(by_t[t][i])
Path(args.out).write_text(json.dumps(dict(purpose="new-target labeling: none + top-4 cross-fitted post-repair score + 2 hash-random + rule candidate",
                                          jobs=inter), ensure_ascii=False, indent=1))
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis" / (Path(args.out).stem + "_selection.json")).write_text(json.dumps(sel_log, ensure_ascii=False, indent=1))
cnt = collections.Counter(c for v in sel_log.values() for c, _ in v["top"])
print("label jobs", len(label_jobs), "| forced rule arms", sum(bool(v["forced"]) for v in sel_log.values()),
      "| most-selected top candidates:", cnt.most_common(12))
