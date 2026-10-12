# -*- coding: utf-8 -*-
"""Recompute every number used in the paper from per-chain and per-handoff records.

Inputs : data/derived/*.csv (extracted read-only from the experiment results), data/raw/libero_j8/*.jsonl
Output : data/derived/stats.json  (single source for tables, figures and numbers quoted in the text)
Checks : recomputed summaries are compared against the analysis reports produced on the server (data/raw/**/analysis/*.json);
         any mismatch beyond 1e-9 stops the script.
Statistics: chain-level paired differences with 95% percentile intervals from a cluster bootstrap over base initial
states (5000 resamples, seed 17), identical to the original analysis scripts; per-task handoff comparisons use 2000
resamples (seed 3 for round 1 by-task, as in analyze_cl.py; seed 17 elsewhere).
"""
import csv, json, glob, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DER, RAW = ROOT / "data/derived", ROOT / "data/raw"
S = {}


def rcsv(name):
    with open(DER / name, newline="") as f:
        return list(csv.DictReader(f))


def boot(rows, fn, key="base", n=5000, seed=17):
    groups = sorted({r[key] for r in rows}); by = collections.defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    rng = np.random.RandomState(seed)
    vals = [fn([x for g in rng.choice(groups, len(groups)) for x in by[g]]) for _ in range(n)]
    return [float(fn(rows)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def chains(name, arms, scene=None):
    rows = []
    for r in rcsv(name):
        if scene and r["scene"] != scene:
            continue
        d = dict(base=r["base"], chain=r["chain"], stratum=r["stratum"], same=int(r["same_initial_state"]))
        for a in arms:
            d[a] = dict(success=int(r[f"{a}_success"]), completed=int(r[f"{a}_completed"]), steps=int(r[f"{a}_steps"]),
                        repairs=int(r[f"{a}_repairs"]), retries=int(r[f"{a}_retries"]))
            d[a]["U"] = d[a]["completed"] / 5 - 0.5 * d[a]["steps"] / 1800
        rows.append(d)
    return rows


def summarize(rows, arms):
    return {a: {k: float(np.mean([r[a][k] for r in rows])) for k in ("success", "completed", "steps", "U", "repairs", "retries")} for a in arms}


def paired(rows, A, B, metrics=("success", "steps", "U")):
    return {m: boot(rows, lambda rs, m=m: float(np.mean([x[A][m] - x[B][m] for x in rs]))) for m in metrics}


def check(a, b, what):
    if abs(a - b) > 1e-9:
        raise SystemExit(f"MISMATCH {what}: {a} vs {b}")


# ---------------- Round 1 ----------------
A1 = ["none", "retry", "fixed", "risk", "benefit", "full", "inverse", "prior", "benefit_noretry", "home_all", "noloc", "benefit_m5"]
r1 = chains("round1_per_chain.csv", A1)
rep1 = json.loads((RAW / "closedloop_20261004/analysis/cl_report.json").read_text())
rep1x = json.loads((RAW / "closedloop_20261004/analysis/cl_extra2_report.json").read_text())
sum1 = summarize(r1, A1)
for a in rep1["summary"]:
    for k in ("success", "steps", "U"):
        check(sum1[a][k], rep1["summary"][a][k], f"r1 {a} {k}")
for a in rep1x["summary"]:
    check(sum1[a]["success"], rep1x["summary"][a]["success"], f"r1x {a}")
S["round1"] = dict(chains=len(r1), bases=len({r["base"] for r in r1}), same_initial_state=sum(r["same"] for r in r1), summary=sum1, paired={})
for A, B in [("benefit", "none"), ("benefit", "retry"), ("benefit", "fixed"), ("benefit", "prior"), ("benefit", "inverse"), ("benefit", "risk"),
             ("full", "benefit"), ("retry", "none"), ("benefit_noretry", "none"), ("benefit_noretry", "retry"), ("benefit", "home_all"),
             ("noloc", "benefit"), ("noloc", "prior"), ("benefit_m5", "benefit"), ("home_all", "retry"), ("benefit", "benefit_noretry")]:
    S["round1"]["paired"][f"{A}-{B}"] = paired(r1, A, B)
for a in A1:
    if a != "retry" and f"{a}-retry" not in S["round1"]["paired"]:
        S["round1"]["paired"][f"{a}-retry"] = paired(r1, a, "retry")
for k, v in rep1["paired"].items():
    if k in S["round1"]["paired"]:
        for m in ("success", "steps", "U"):
            for i in range(3):
                check(S["round1"]["paired"][k][m][i], v[m][i], f"r1 paired {k} {m}")

# first target handoff (state identical across arms before it)
def first_target(name, arms, seed_by_task=17, nb=2000):
    ft = collections.defaultdict(dict)
    for r in rcsv(name):
        ft[r["chain"]][r["arm"]] = dict(task=r["task"], success=int(r["success"]), repair=r["repair"], base=r["base"])
    rows = [dict(base=v[arms[0]]["base"], task=v[arms[0]]["task"], **{a: v[a] for a in arms}) for v in ft.values() if all(a in v for a in arms)]
    out = dict(n=len(rows), all={a: float(np.mean([r[a]["success"] for r in rows])) for a in arms}, by_task={})
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]
        rec = dict(n=len(rs), **{a: float(np.mean([r[a]["success"] for r in rs])) for a in arms})
        rec["choices"] = {a: collections.Counter(r[a]["repair"] for r in rs).most_common(3) for a in arms}
        for B in ("none", "inverse", "prior", "retry"):
            if B in arms:
                rec[f"benefit-{B}"] = boot(rs, lambda x, B=B: float(np.mean([y["benefit"]["success"] - y[B]["success"] for y in x])), n=nb, seed=seed_by_task)
        out["by_task"][t] = rec
    return rows, out


r1ft_rows, S["round1"]["first_target"] = first_target("round1_first_target.csv", A1, seed_by_task=3, nb=3000)

# ---------------- Round 2 ----------------
A2 = ["none", "retry", "inverse", "prior", "home_all", "benefit", "benefit_noretry", "noloc", "benefit_alt"]
r2 = chains("round2_per_chain.csv", A2)
rep2 = json.loads((RAW / "closedloop2_20261005/analysis/cl2_report.json").read_text())
sum2 = summarize(r2, A2)
for a in A2:
    for k in ("success", "steps", "U"):
        check(sum2[a][k], rep2["summary"][a][k], f"r2 {a} {k}")
S["round2"] = dict(chains=len(r2), bases=len({r["base"] for r in r2}), same_initial_state=sum(r["same"] for r in r2), summary=sum2, paired={})
for A, B in [("benefit", "prior"), ("benefit", "inverse"), ("benefit", "retry"), ("benefit", "none"), ("benefit", "home_all"), ("noloc", "benefit"),
             ("benefit_noretry", "none"), ("benefit_noretry", "retry"), ("prior", "retry"), ("inverse", "retry"), ("home_all", "retry"),
             ("retry", "none"), ("benefit", "benefit_alt"), ("benefit_alt", "prior")]:
    S["round2"]["paired"][f"{A}-{B}"] = paired(r2, A, B)
    for m in ("success", "steps", "U"):
        for i in range(3):
            check(S["round2"]["paired"][f"{A}-{B}"][m][i], rep2["paired"][f"{A}-{B}"][m][i], f"r2 paired {A}-{B} {m}")
OLD5 = {"lift_red_block_slider", "push_into_drawer", "push_red_block_right", "push_blue_block_right", "rotate_pink_block_right"}
NEW_INV = {"push_pink_block_right", "rotate_red_block_left"}
fam = lambda t: "old5" if t in OLD5 else ("new_inverse" if t in NEW_INV else "new_noninverse")
r2ft_rows, S["round2"]["first_target"] = first_target("round2_first_target.csv", A2)
S["round2"]["first_target"]["by_family"] = {}
for g in ("old5", "new_inverse", "new_noninverse"):
    rs = [r for r in r2ft_rows if fam(r["task"]) == g]
    rec = dict(n=len(rs), **{a: float(np.mean([r[a]["success"] for r in rs])) for a in A2})
    for B in ("prior", "inverse", "none"):
        rec[f"benefit-{B}"] = boot(rs, lambda x, B=B: float(np.mean([y["benefit"]["success"] - y[B]["success"] for y in x])), n=2000)
    S["round2"]["first_target"]["by_family"][g] = rec
    check(rec["benefit-prior"][0], rep2["first_target"]["family"][g]["benefit-prior"][0], f"r2 family {g}")
    check(rec["benefit-prior"][1], rep2["first_target"]["family"][g]["benefit-prior"][1], f"r2 family lo {g}")
S["round2"]["by_stratum"] = {s: dict(n=v["n"], **{a: v[a] for a in A2}) for s, v in rep2["by_stratum"].items()}

# ---------------- Cross-scene ----------------
AS = ["none", "retry", "inverse", "prior", "benefit", "benefit_m5"]
S["scenes"] = {}
for tag in ("scA", "scB", "scC"):
    rs = chains("scene_per_chain.csv", AS, scene=tag)
    rep = json.loads((RAW / f"crossscene_20261005/analysis/{tag}_report.json").read_text())
    sm = summarize(rs, AS)
    for a in AS:
        check(sm[a]["success"], rep["summary"][a]["success"], f"{tag} {a}")
    rec = dict(chains=len(rs), bases=len({r["base"] for r in rs}), same_initial_state=sum(r["same"] for r in rs), summary=sm, paired={})
    for A, B in [("benefit", "retry"), ("benefit", "none"), ("benefit", "inverse"), ("benefit", "prior"), ("benefit_m5", "benefit")]:
        rec["paired"][f"{A}-{B}"] = paired(rs, A, B)
        check(rec["paired"][f"{A}-{B}"]["success"][1], rep["paired_diff"][f"{A}-{B}"]["success"][1], f"{tag} {A}-{B}")
    S["scenes"][tag] = rec

# ---------------- Decomposition ----------------
def decompose(rs):
    K = np.array([r["k"] for r in rs], float); N = np.array([r["n"] for r in rs], float)
    T = np.array([r["task"] for r in rs])
    pbar = K.sum() / N.sum(); tot = pbar * (1 - pbar)
    within = np.sum(K * (N - K) / (N - 1)) / N.sum()
    tm = {t: K[T == t].sum() / N[T == t].sum() for t in set(T)}
    between = np.sum(N * (np.array([tm[t] for t in T]) - pbar) ** 2) / N.sum()
    conc = pairs = 0.0
    for t in set(T):
        pos, neg = [], []
        for r in (x for x in rs if x["task"] == t):
            w = r["wins"]
            for j, y in enumerate(w):
                (pos if y else neg).append((sum(w) - y) / (len(w) - 1))
        pos, neg = np.array(pos), np.array(neg)
        if len(pos) and len(neg):
            conc += (pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum(); pairs += len(pos) * len(neg)
    p = K / N
    return dict(n=len(rs), success=float(pbar), task_share=float(between / tot), state_share=float((tot - within - between) / tot),
                noise_share=float(within / tot), oracle_auroc=float(conc / pairs) if pairs else None, mixed=float(np.mean((p > 0) & (p < 1))),
                all_fail=float(np.mean(p == 0)), all_success=float(np.mean(p == 1)),
                k_hist=[int(x) for x in np.bincount(K.astype(int), minlength=9)])


H = [dict(task=r["task"], k=int(r["k"]), n=int(r["n"]), wins=[int(c) for c in r["wins"]], parent=r["parent_id"], source=r["source"]) for r in rcsv("handoffs_j8.csv")]
dref = json.loads((RAW / "newtasks_20261004/analysis/decomposition_11tasks.json").read_text())
S["decomposition"] = dict(pooled=decompose(H), by_task={})
check(S["decomposition"]["pooled"]["state_share"], dref["pooled_all"]["state_share"], "decomp pooled")
for t in sorted({h["task"] for h in H}):
    rs = [h for h in H if h["task"] == t]
    d = decompose(rs)
    check(d["state_share"], dref["by_task"][t]["state_share"], f"decomp {t}")
    rng = np.random.RandomState(5); v = []
    for _ in range(1000):
        s = [rs[i] for i in rng.randint(0, len(rs), len(rs))]
        v.append(decompose(s)["state_share"])
    d["state_share_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    d["family"] = fam(t)
    S["decomposition"]["by_task"][t] = d
# LIBERO
rows = [json.loads(l) for f in sorted(glob.glob(str(RAW / "libero_j8/w*.jsonl"))) for l in open(f)]
by = collections.defaultdict(list)
for r in rows:
    by[r["hid"]].append(r)
L = []
for hid, v in by.items():
    v = sorted(v, key=lambda r: r["j"])
    L.append(dict(task=f"B{v[0]['b']}", k=sum(int(r["success"]) for r in v), n=len(v), wins=[int(r["success"]) for r in v], a=v[0]["a"], parent=str(v[0]["init"])))
lib = decompose(L)
lsum = json.loads((RAW / "libero_feasibility_20261004/runs/gate_j8/summary.json").read_text())
check(lib["state_share"], lsum["decomposition"]["pooled"]["state_share"], "libero state share")
S["libero"] = dict(decomposition=lib, by_task={t: decompose([x for x in L if x["task"] == t]) for t in sorted({x["task"] for x in L})},
                   gate=lsum["gate"], b_from_reset=lsum.get("B_from_reset"), a_phase=lsum.get("A_phase"))

# ---------------- offline (reports only) ----------------
for tag in ("5task", "all"):
    S[f"offline_{tag}"] = json.loads((RAW / f"benefit_model_20261003/analysis/v3_report_{tag}.json").read_text())
S["vocab_newtasks"] = json.loads((RAW / "newtasks_20261004/analysis/vocab_repair_newtasks.json").read_text())
S["vocab_old"] = json.loads((RAW / "multitask_vocab_20261002/analysis/vocab_repair_report.json").read_text())
S["locked_round1"] = {k: v for k, v in json.loads((RAW / "closedloop_20261004/locked_config.json").read_text()).items() if k not in ("cbar", "cbar_global")}
S["locked_round2"] = {k: v for k, v in json.loads((RAW / "closedloop2_20261005/locked_config.json").read_text()).items() if k not in ("cbar", "cbar_global")}
(DER / "stats.json").write_text(json.dumps(S, indent=1, default=float))
print("all checks passed; wrote", DER / "stats.json")
print("R1 benefit-retry", S["round1"]["paired"]["benefit-retry"]["success"])
print("R2 noloc-benefit", S["round2"]["paired"]["noloc-benefit"]["success"])
print("decomp pooled", {k: S["decomposition"]["pooled"][k] for k in ("n", "state_share", "noise_share", "task_share")})
print("libero", {k: lib[k] for k in ("n", "success", "state_share", "noise_share", "task_share", "oracle_auroc", "all_fail", "k_hist")})
