# -*- coding: utf-8 -*-
"""Fixes 3 and 4: evaluation protocol and low-capacity baselines, run on data with J repeated rollouts from the same handoff.

Evaluation (fix 3)
  * 5 folds grouped by initial configuration, repeated with 5 different splits; every sample is evaluated once per repeat.
  * Label definition: the 0/1 outcome of each rollout is one test label (comparable with the original single-rollout labels); scores are handoff-level predictions.
  * Reports AUROC, within-task AUROC (pairs only within the same fold and task, removing between-fold prior bias),
    the failure rate (risk) of admitted handoffs at fixed coverage 80%/90%, and initial-configuration cluster-bootstrap confidence intervals against the task prior.
  * Label ceiling: the mean of the other J-1 outcomes at the same handoff serves as a "state oracle" score.

Modelling (fix 4)
  * The task prior is estimated on the training folds only; other methods use logit(task prior) as a fixed offset and learn only the within-task residual.
  * Residual models are L2 logistic regression (binomial likelihood, using k/n soft labels directly) or shallow trees.
  * A control trained only on the first rollout (J=1) is also reported to quantify the effect of soft labels; the test label definition is unchanged.

Features
  robot15 robot state; cls original DINO CLS; vision positions/confidences/end-effector distances of the three blocks estimated by the static-camera locator
  + drawer and slider positions estimated by ridge regression on grid features (all fit on training folds only); scene simulator ground truth (privileged upper bound, reference only).
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import os, sys, json, glob, hashlib, collections
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "16")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
torch.set_num_threads(16)
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from features_patch import build_cache
from object_locator import fit_locator_normalized

RUN = HERE / "runs/j8_two_task"
RAW = Path(f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/raw")
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


# ---------- data ----------
def load_j8():
    recs = []
    for f in sorted(RUN.glob("gpu*/labels.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("reached") and "restore_failed" not in r and r.get("n"):
                r["npz"] = str(f.parent / "boundaries" / r["observation"])
                recs.append(r)
    return recs


def features_for(paths, cache_name):
    digest = hashlib.sha256("|".join(sorted(Path(x).name for x in paths)).encode()).hexdigest()[:12]
    cache = HERE / "cache" / f"{Path(cache_name).stem}_{len(paths)}_{digest}.npz"
    if not cache.exists():
        build_cache(paths, cache, modes=("cls_crop", "grid8", "grid16"))
    C = np.load(cache)
    order = {n: i for i, n in enumerate(C["names"])}
    idx = [order[Path(p).name] for p in paths]
    return {k: C[k][idx] for k in C.files if k != "names"}


def raw_boundaries():
    rows = {}
    for f in sorted(RAW.glob("gpu*/handoffs.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[r["observation"]] = (str(f.parent / r["observation"]), r["parent_id"])
    paths = [v[0] for v in rows.values()]
    F = features_for(paths, "two_task_raw_all.npz")
    return F, np.array([v[1] for v in rows.values()])


# ---------- models ----------
def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def offset_logreg(Xtr, k, n, off_tr, Xte, off_te, C=0.3):
    d = Xtr.shape[1]

    def f(w):
        z = off_tr + Xtr @ w[:-1] + w[-1]
        loss = np.sum(n * np.logaddexp(0, z) - k * z) + 0.5 / C * np.sum(w[:-1] ** 2)
        pr = 1 / (1 + np.exp(-z))
        g = n * pr - k
        return loss, np.concatenate([Xtr.T @ g + w[:-1] / C, [g.sum()]])
    w = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B").x
    return 1 / (1 + np.exp(-(off_te + Xte @ w[:-1] + w[-1])))


def rho_mlp(Xtr, k, n, off_tr, Xte, off_te, task_tr, task_te, seed, epochs=150):
    """The HandoffModel of the rho package (no image branch, state = input features, task embedding), modified to use a soft-label binomial loss, a task-prior offset and a fixed number of epochs."""
    sys.path.insert(0, os.path.dirname(RHO_ROOT))
    from rho.model import HandoffModel
    torch.manual_seed(seed)
    vocab = sorted(set(task_tr) | set(task_te))
    tid = lambda t: torch.tensor([vocab.index(x) for x in t], dtype=torch.long)
    model = HandoffModel(state_dim=Xtr.shape[1], num_tasks=len(vocab), num_views=0, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    X = torch.tensor(Xtr, dtype=torch.float32); O = torch.tensor(off_tr, dtype=torch.float32)
    Kt = torch.tensor(k, dtype=torch.float32); Nt = torch.tensor(n, dtype=torch.float32); T = tid(task_tr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        z = model(None, X, T) + O
        loss = (Nt * torch.nn.functional.softplus(z) - Kt * z).sum() / Nt.sum()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z = model(None, torch.tensor(Xte, dtype=torch.float32), tid(task_te)) + torch.tensor(off_te, dtype=torch.float32)
    return torch.sigmoid(z).numpy()


def tree_model(Xtr, k, n, Xte, seed):
    m = ExtraTreesRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=16)
    m.fit(Xtr, k / n, sample_weight=n)
    return m.predict(Xte)


# ---------- metrics ----------
def rollout_view(scores, K, N):
    """Expand handoff-level scores to the 0/1 label of each rollout."""
    y = np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for k, n in zip(K, N)])
    s = np.repeat(scores, N)
    return y, s


def within_fold_task_auc(scores, K, N, task, folds):
    tot = pairs = 0.0
    for f in np.unique(folds):
        for t in np.unique(task):
            m = (folds == f) & (task == t)
            pos = np.repeat(scores[m], K[m]); neg = np.repeat(scores[m], N[m] - K[m])
            if len(pos) and len(neg):
                tot += (pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()
                pairs += len(pos) * len(neg)
    return tot / pairs


def within_fold_auc(scores, K, N, folds):
    """Pairs all samples within a fold (regardless of task), weighted by the number of pairs; avoids cross-fold prior bias."""
    return within_fold_task_auc(scores, K, N, np.zeros(len(K)), folds)


def risk_at(scores, K, N, cov):
    order = np.argsort(-scores, kind="stable")
    cum = np.cumsum(N[order]); keep = order[: np.searchsorted(cum, cov * N.sum()) + 1]
    return 1 - K[keep].sum() / N[keep].sum()


def main():
    recs = load_j8()
    K = np.array([r["k"] for r in recs]); N = np.array([r["n"] for r in recs])
    task = np.array([r["task_b"] for r in recs]); g = np.array([r["parent_id"] for r in recs])
    paths = [r["npz"] for r in recs]
    F = features_for(paths, "j8_boundaries.npz")
    rob, scn = F["state"], F["scene_state"]
    cls = np.hstack([F["static_cls_crop"], F["wrist_cls_crop"]])
    RF, rg = raw_boundaries()
    onehot = (task == "push_into_drawer").astype(float)[:, None]
    report = {"boundaries": int(len(K)), "rollouts": int(N.sum()), "parents": int(len(set(g))),
              "success_rate": float(K.sum() / N.sum())}

    # ---- label ceiling (scene D, 360 steps) ----
    p = K / N
    pbar = K.sum() / N.sum(); tot = pbar * (1 - pbar)
    within_state = np.sum(K * (N - K) / (N - 1)) / N.sum()
    tm = {t: K[task == t].sum() / N[task == t].sum() for t in set(task)}
    between_task = np.sum(N * (np.array([tm[t] for t in task]) - pbar) ** 2) / N.sum()
    report["variance_share"] = dict(task=between_task / tot, state_within_task=(tot - within_state - between_task) / tot,
                                    rollout_noise=within_state / tot)
    ys, ss = [], []
    for i in range(len(K)):
        for j in range(N[i]):
            yj = 1 if j < K[i] else 0
            ys.append(yj); ss.append((K[i] - yj) / (N[i] - 1))
    ys, ss = np.array(ys), np.array(ss)
    tt = np.repeat(task, N)
    wa = []
    for t in set(task):
        m = tt == t; pos, neg = ss[m & (ys == 1)], ss[m & (ys == 0)]
        wa.append(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum(), len(pos) * len(neg)))
    report["oracle"] = dict(auroc=roc_auc_score(ys, ss), within_task_auroc=sum(a for a, _ in wa) / sum(b for _, b in wa),
                            mixed_state_fraction=float(np.mean((p > 0) & (p < 1))),
                            all_fail_fraction=float(np.mean(p == 0)))
    print(json.dumps(report, indent=1), flush=True)

    # ---- grouped cross-validation ----
    ug = np.unique(g)
    methods = collections.defaultdict(list)
    oof_first = {}
    for rep in range(5):
        rng = np.random.RandomState(rep); perm = rng.permutation(len(ug))
        fold_of = {ug[perm[i]]: i % 5 for i in range(len(ug))}
        folds = np.array([fold_of[x] for x in g])
        S = collections.defaultdict(lambda: np.zeros(len(K)))
        for f in range(5):
            tr, te = folds != f, folds == f
            pri = {t: K[tr & (task == t)].sum() / N[tr & (task == t)].sum() for t in set(task)}
            off_tr = logit(np.array([pri[t] for t in task[tr]])); off_te = logit(np.array([pri[t] for t in task[te]]))
            S["task prior"][te] = [pri[t] for t in task[te]]
            # visual locator features: the locator uses only raw handoffs of training-fold initial configurations and training-fold handoffs of this dataset
            raw_tr = ~np.isin(rg, g[te])
            Ptr = np.concatenate([RF["static_grid16"][raw_tr], F["static_grid16"][tr]]).astype(np.float32)
            Str = np.concatenate([RF["scene_state"][raw_tr], scn[tr]])
            loc = fit_locator_normalized(Ptr, Str, epochs=600, seed=rep)
            xyz, conf = loc(F["static_grid16"].astype(np.float32))
            G8tr = np.concatenate([RF["static_grid8"][raw_tr], F["static_grid8"][tr]])
            sc8 = StandardScaler().fit(G8tr)
            art = np.column_stack([RidgeCV(alphas=np.logspace(0, 5, 11)).fit(sc8.transform(G8tr), Str[:, c]).predict(sc8.transform(F["static_grid8"])) for c in (0, 1)])
            dist = np.linalg.norm(xyz - rob[:, None, :3], axis=2)
            vision = np.hstack([xyz.reshape(len(K), -1), conf, dist, art])
            sets = {"robot15": rob, "vision+robot15": np.hstack([vision, rob]),
                    "scene(privileged)+robot15": np.hstack([scn, rob])}
            for name, X in sets.items():
                s = StandardScaler().fit(X[tr]); Xtr, Xte = s.transform(X[tr]), s.transform(X[te])
                S[f"prior-offset logistic | {name}"][te] = offset_logreg(Xtr, K[tr], N[tr], off_tr, Xte, off_te)
                if name == "vision+robot15":
                    S[f"rho-MLP fixed | {name}"][te] = rho_mlp(Xtr, K[tr], N[tr], off_tr, Xte, off_te, task[tr], task[te], rep)
                S[f"tree | {name}"][te] = tree_model(np.hstack([X[tr], onehot[tr]]), K[tr], N[tr], np.hstack([X[te], onehot[te]]), rep)
            # original CLS features: PCA followed by offset logistic regression
            from sklearn.decomposition import PCA
            sc = StandardScaler().fit(cls[tr]); pca = PCA(32, random_state=0).fit(sc.transform(cls[tr]))
            Ztr, Zte = pca.transform(sc.transform(cls[tr])), pca.transform(sc.transform(cls[te]))
            S["prior-offset logistic | cls(his)+robot15"][te] = offset_logreg(
                np.hstack([Ztr, StandardScaler().fit(rob[tr]).transform(rob[tr])]), K[tr], N[tr], off_tr,
                np.hstack([Zte, StandardScaler().fit(rob[tr]).transform(rob[te])]), off_te)
            # J=1 control: training uses only the first rollout outcome of each handoff
            first = np.array([int(r["repeats"][0]["success"]) for r in recs])
            X = np.hstack([vision, rob]); s = StandardScaler().fit(X[tr])
            pri1 = {t: first[tr & (task == t)].mean() for t in set(task)}
            S["J=1 training | prior-offset logistic | vision+robot15"][te] = offset_logreg(
                s.transform(X[tr]), first[tr].astype(float), np.ones(tr.sum()),
                logit(np.array([pri1[t] for t in task[tr]])), s.transform(X[te]), logit(np.array([pri1[t] for t in task[te]])))
            S["J=1 training | tree | vision+robot15"][te] = tree_model(
                np.hstack([X[tr], onehot[tr]]), first[tr].astype(float), np.ones(tr.sum()), np.hstack([X[te], onehot[te]]), rep)
        for name, sc_ in S.items():
            y, s = rollout_view(sc_, K, N)
            methods[name].append(dict(auroc=within_fold_auc(sc_, K, N, folds), auroc_pooled=roc_auc_score(y, s), within=within_fold_task_auc(sc_, K, N, task, folds),
                                      risk80=risk_at(sc_, K, N, 0.8), risk90=risk_at(sc_, K, N, 0.9)))
            if rep == 0:
                oof_first[name] = (sc_.copy(), folds.copy())
        print(f"repeat {rep} done", flush=True)

    # ---- cluster bootstrap against the task prior (out-of-fold scores of split 0, resampled by initial configuration) ----
    base_s, folds0 = oof_first["task prior"]
    boot = {}
    rng = np.random.RandomState(20260928)
    groups = {x: np.flatnonzero(g == x) for x in ug}
    draws = [np.concatenate([groups[x] for x in rng.choice(ug, len(ug))]) for _ in range(1000)]
    for name, (s_, f_) in oof_first.items():
        if name == "task prior":
            continue
        diffs = []
        for idx in draws:
            try:
                diffs.append(within_fold_task_auc(s_[idx], K[idx], N[idx], task[idx], f_[idx]) - 0.5)
            except ZeroDivisionError:
                pass
        boot[name] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    table = {}
    print(f"\n{'method':56s} {'AUROC':>12s} {'within-task':>12s} {'risk@80%':>9s} {'risk@90%':>9s}  within-task gain vs prior 95% CI")
    for name, rs in methods.items():
        a = np.array([[r["auroc"], r["within"], r["risk80"], r["risk90"]] for r in rs])
        table[name] = dict(auroc=a[:, 0].mean(), auroc_sd=a[:, 0].std(), within=a[:, 1].mean(), within_sd=a[:, 1].std(),
                           risk80=a[:, 2].mean(), risk90=a[:, 3].mean(), within_gain_ci=boot.get(name))
        ci = boot.get(name)
        print(f"{name:56s} {a[:,0].mean():.3f}±{a[:,0].std():.3f} {a[:,1].mean():.3f}±{a[:,1].std():.3f} "
              f"{a[:,2].mean():9.3f} {a[:,3].mean():9.3f}  {'' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'}")
    report["base_failure_rate"] = float(1 - K.sum() / N.sum())
    report["cv"] = table
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("saved", OUT / "cv_report.json")


if __name__ == "__main__":
    main()
