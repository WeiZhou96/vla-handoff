# -*- coding: utf-8 -*-
"""Closed-loop numbers of the paper from the full-chain dual-criterion replay.

Inputs (read-only copies from the compute server)
  data/raw/fullchain_replay_20261010/analysis/summary.json   written by experiments/fullchain_replay_20261010/analyze.py
  data/raw/fullchain_replay_20261010/analysis/records.jsonl  one line per (round, chain, arm)
  data/derived/stats.json   original analysis (compute_stats.py): cross-checks against the published numbers and the
                            decomposition quantities for the across-task correlations
Outputs
  data/derived/fullchain_stats.json     every closed-loop number used by tables, figures and text (ASCII; KEYS below)
  data/derived/fullchain_per_chain.csv  main sample P, one row per (round, chain, arm)

Estimates
  A    replay under the original start-relative CALVIN criterion (bit-identical to the published runs)
  (i)  post hoc rescoring of the A trajectories with the handoff-referenced criterion B_def ((io): B_orc)
  (ii) counterfactual re-execution in which B_def ends attempts and triggers retries and chain stops (main estimate)
Paired differences X - Y: cluster bootstrap over base initial configurations, 2000 resamples, seed 17, a fresh
generator per cell and the same resamples for every metric of a cell; this re-implements analyze.py exactly.

Checks (any failure prints CHECK FAILED and exits with code 2 before anything is written)
  1 replay gate: no errors or missing jobs; mode-A match rate 1; every instrument check 1; every divergence check 1;
    the no-intervention arms never diverge; cross-run determinism 1
  2 every arm summary, paired difference, hypothesis and first-handoff entry of summary.json is recomputed from
    records.jsonl: point estimates to 1e-12, interval bounds to 1e-9 (relative to max(1, |value|))
  3 the replayed A numbers equal the published numbers in stats.json (arm means; paired and hypothesis point estimates)
  4 in chains without a push/rotate target, (ii) equals A exactly (L(ii) = 0 with a zero-width interval)

KEYS of fullchain_stats.json
  meta                            inputs (sha256), sample sizes, replay checks, bootstrap settings
  dv["r1|benefit"]                n, ok (divergence-check pass rate), identical (share of chains that never diverged)
  arms["r1|benefit"]              n, match (mode-A reproduction rate), S_O, S_A, S_i, S_io, S_ii (chain success),
                                  C_A, C_ii (completed subtasks), N_A, N_ii (steps/chain), U_A, U_ii (utility),
                                  rep_A, rep_ii (repairs/chain), A0B1_stop, identical
  paired["r1|all|benefit-retry"]  stratum: all | pr (chains with a push/rotate target) | nonpr;
                                  n, bases, and [point, lo95, hi95] for S_O, S_A, S_i, S_io, S_ii, C_A, C_ii, N_A, N_ii,
                                  U_A, U_ii, L_i, L_ii (success and utility as fractions, steps per chain);
                                  cls_A, cls_ii in {"+", "0", "-"} from the intervals of S_A and S_ii
  H["H1"] ... H["H5b"]            pre-registered round-2 comparisons in the same layout (H3: f_A, f_Bi, f_Bii);
                                  H_noinv_tasks: the task set used for H3
  first["r1|task|arm"]            n, A, Bi, Bii (success at the first target handoff), cells, choices (top repairs)
  first_overall["r1|arm"]         n, A, Bi, Bii pooled over target tasks
  first_paired["r1|group|X-Y"]    group: a task, pr_pooled, other_pooled or all; n, bases, f_A, f_Bi, f_Bii
  audit[task]                     kind, n, A_ne_Bdef, A_ne_Borc, Bdef_ne_Borc, repaired, flag_handoff, flag_start,
                                  flag_yaw180 (rotations), flags_raw
  audit_pooled["pr" | "other"]    sums over the two task groups
  across                          Spearman correlations across the eleven target tasks between decomposition
                                  quantities and first-handoff gains over no intervention, under A and (ii)
"""
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "fullchain_replay_20261010" / "analysis"
DER = ROOT / "data" / "derived"

NBOOT, SEED = 2000, 17                 # analyze.py
NPERM, NB_ACROSS = 20000, 5000         # across-task correlations (this script)
METRICS = ("S_O", "S_A", "S_i", "S_io", "S_ii", "C_A", "C_ii", "N_A", "N_ii", "U_A", "U_ii")
KEYS = METRICS + ("L_i", "L_ii")
FKEYS = ("f_A", "f_Bi", "f_Bii")
I_A, I_I, I_II = METRICS.index("S_A"), METRICS.index("S_i"), METRICS.index("S_ii")
ARMKEYS = ("S_O", "S_A", "S_i", "S_io", "S_ii", "C_A", "C_ii", "N_A", "N_ii", "U_A", "U_ii", "rep_A", "rep_ii")
PR_TASKS = frozenset(("push_red_block_right", "push_blue_block_right", "push_pink_block_right",
                      "rotate_pink_block_right", "rotate_red_block_left"))
# Keys written by experiments/fullchain_replay_20261010/analyze.py (Chinese labels, used verbatim; see the glossary in
# README.md): all chains / chains with a push or rotation target / chains without one; push/rotate pooled / other tasks
# pooled / all.
STRATUM = {"全部链": "all", "含 push/rotate 目标的链": "pr", "不含 push/rotate 目标的链": "nonpr"}
SFILT = {"all": lambda r: True, "pr": lambda r: bool(r["pr_chain"]), "nonpr": lambda r: not r["pr_chain"]}
FGROUP = {"push/rotate 合计": "pr_pooled", "其他任务合计": "other_pooled", "全部": "all"}
NEW_NOINV = ("lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer", "stack_block")
NOINV_CANDIDATES = (NEW_NOINV, NEW_NOINV + ("lift_red_block_slider", "push_into_drawer"))
R1_TASKS = ("push_red_block_right", "push_blue_block_right", "rotate_pink_block_right", "lift_red_block_slider",
            "push_into_drawer")
R2_TASKS = ("push_pink_block_right", "rotate_red_block_left", "lift_blue_block_slider", "lift_pink_block_slider",
            "lift_pink_block_drawer", "stack_block")
DECOMP_COLS = ("state_share", "mixed", "all_fail", "success")
PUBLISHED_ACROSS_A = {"retry": (-0.65, 0.83, -0.20, 0.13), "prior": (0.28, 0.28, 0.79, -0.73),
                      "benefit": (0.19, 0.43, 0.83, -0.83)}   # Table across of the submitted manuscript
TOL_PT, TOL_CI, TOL_PUB = 1e-12, 1e-9, 1e-9
FAILED, WARNED = [], []


# ---------------------------------------------------------------- helpers
def fail(msg):
    FAILED.append(msg)
    print("CHECK FAILED:", msg)


def warn(msg):
    WARNED.append(msg)
    print("WARN:", msg)


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel_close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def cls(v):
    return "+" if v[1] > 0 else ("-" if v[2] < 0 else "0")


def tolist(c, keys):
    if c is None:
        return None
    out = dict(n=c["n"], bases=c["bases"])
    for k in keys:
        out[k] = [c[k]["pt"], c[k]["lo"], c[k]["hi"]]
    return out


def clean(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def task_kind(t):
    if t == "push_into_drawer":
        return "push_into"
    if t == "stack_block":
        return "stack"
    return t.split("_")[0]          # push, rotate, lift


# ---------------------------------------------------------------- statistics (identical to analyze.py)
def boot(data, keys):
    if not data:
        return None
    bases = sorted({b for b, _ in data})
    bi = {b: i for i, b in enumerate(bases)}
    Ssum, N = np.zeros((len(bases), len(keys))), np.zeros(len(bases))
    for b, d in data:
        Ssum[bi[b]] += d
        N[bi[b]] += 1
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(bases), size=(NBOOT, len(bases)))
    est = Ssum[draw].sum(1) / N[draw].sum(1)[:, None]
    pt = Ssum.sum(0) / N.sum()
    lo, hi = np.percentile(est, 2.5, axis=0), np.percentile(est, 97.5, axis=0)
    out = dict(n=int(N.sum()), bases=len(bases))
    out.update({k: dict(pt=float(pt[i]), lo=float(lo[i]), hi=float(hi[i])) for i, k in enumerate(keys)})
    return out


def paired(rows, X, Y, filt):
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in (X, Y) and filt(r):
            by[(r["round"], r["orig_id"])][r["arm"]] = r
    data = []
    for v in by.values():
        if X in v and Y in v and all(v[X].get(k) is not None and v[Y].get(k) is not None for k in METRICS):
            d = np.array([v[X][k] - v[Y][k] for k in METRICS], float)
            data.append((v[X]["base"], np.concatenate([d, [d[I_I] - d[I_A], d[I_II] - d[I_A]]])))
    return boot(data, KEYS)


def paired_first(rows, X, Y, filt):
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in (X, Y) and filt(r) and r.get("f_reached") and r.get("f_Bii") is not None:
            by[(r["round"], r["orig_id"])][r["arm"]] = r
    data = [(v[X]["base"], np.array([v[X][k] - v[Y][k] for k in FKEYS], float)) for v in by.values() if X in v and Y in v]
    return boot(data, FKEYS)


def cmp_cell(mine, ref, keys, label):
    if mine is None or ref is None:
        if (mine is None) != (ref is None):
            fail(f"{label}: recomputed empty={mine is None}, summary empty={ref is None}")
        return
    if mine["n"] != ref["n"] or mine["bases"] != ref["bases"]:
        fail(f"{label}: n/bases {mine['n']}/{mine['bases']} vs summary {ref['n']}/{ref['bases']}")
        return
    for k in keys:
        if k not in ref:
            continue
        for part, tol in (("pt", TOL_PT), ("lo", TOL_CI), ("hi", TOL_CI)):
            if not rel_close(mine[k][part], ref[k][part], tol):
                fail(f"{label} {k}.{part}: recomputed {mine[k][part]!r} vs summary {ref[k][part]!r}")
                return


# ---------------------------------------------------------------- across-task correlations
def _pearson_rows(X, Y):
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((Xc * Xc).sum(axis=1) * (Yc * Yc).sum(axis=1))
    num = (Xc * Yc).sum(axis=1)
    out = np.full(len(num), np.nan)
    ok = den > 1e-15
    out[ok] = num[ok] / den[ok]
    return out


def spearman_rows(X, Y):
    return _pearson_rows(rankdata(X, axis=1), rankdata(Y, axis=1))


def _pct(v):
    v = v[np.isfinite(v)]
    if len(v) < 0.5 * NB_ACROSS:
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def across(first_paired, ST):
    D = ST["decomposition"]["by_task"]
    tasks = [(1, t) for t in R1_TASKS] + [(2, t) for t in R2_TASKS]
    n = len(tasks)
    x = {c: np.array([float(D[t][c]) for _, t in tasks]) for c in DECOMP_COLS}
    res = dict(tasks=[f"r{r}|{t}" for r, t in tasks], columns=list(DECOMP_COLS), gains={}, cells={},
               method=dict(rho="Spearman over the eleven target tasks (round 1: original five, round 2: six added)",
                           p_perm=f"two-sided permutation p, {NPERM} permutations of the gains, seed {SEED}",
                           ci_tasks=f"95% percentile interval, {NB_ACROSS} resamples of tasks with replacement, seed {SEED}",
                           ci_noise=(f"95% percentile interval, {NB_ACROSS} draws of each task gain from a normal distribution "
                                     "with SE = (hi - lo) / 3.92 of its cluster-bootstrap interval (approximation, this script)"),
                           bold="p_perm < 0.05 and both intervals exclude zero"))
    any_bold_ii = False
    for arm in ("retry", "prior", "benefit"):
        for crit in ("f_A", "f_Bii"):
            g = np.array([first_paired[f"r{r}|{t}|{arm}-none"][crit] for r, t in tasks], float)
            pt, se = g[:, 0], (g[:, 2] - g[:, 1]) / (2 * 1.959963984540054)
            res["gains"][f"{arm}|{crit}"] = g.tolist()
            for ci_, c in enumerate(DECOMP_COLS):
                xc = x[c]
                rho = float(spearman_rows(xc[None, :], pt[None, :])[0])
                p = None
                if math.isfinite(rho):
                    rng = np.random.default_rng(SEED)
                    perm = np.argsort(rng.random((NPERM, n)), axis=1)
                    rp = spearman_rows(np.tile(xc, (NPERM, 1)), pt[perm])
                    p = float((np.sum(np.abs(rp) >= abs(rho) - 1e-12) + 1) / (NPERM + 1))
                rng = np.random.default_rng(SEED)
                idx = rng.integers(0, n, size=(NB_ACROSS, n))
                ci_t = _pct(spearman_rows(xc[idx], pt[idx]))
                rng = np.random.default_rng(SEED)
                G = pt[None, :] + se[None, :] * rng.standard_normal((NB_ACROSS, n))
                ci_n = _pct(spearman_rows(np.tile(xc, (NB_ACROSS, 1)), G))
                excl = lambda ci: ci is not None and (ci[0] > 0 or ci[1] < 0)
                bold = bool(p is not None and p < 0.05 and excl(ci_t) and excl(ci_n))
                res["cells"][f"{arm}|{crit}|{c}"] = dict(rho=rho if math.isfinite(rho) else None, p_perm=p,
                                                        ci_tasks=ci_t, ci_noise=ci_n, bold=bold)
                if crit == "f_Bii" and bold:
                    any_bold_ii = True
                if crit == "f_A":
                    pub = PUBLISHED_ACROSS_A[arm][ci_]
                    if not math.isfinite(rho) or abs(round(rho, 2) - pub) > 0.0051:
                        warn(f"across A {arm}/{c}: recomputed rho {rho:.3f} vs published {pub:+.2f}")
    res["any_bold_ii"] = any_bold_ii
    return res


# ---------------------------------------------------------------- checks
def gate(SUM, ROWS):
    if SUM.get("errors"):
        fail(f"replay errors: {len(SUM['errors'])}")
    if SUM.get("missing"):
        fail(f"replay missing jobs: {len(SUM['missing'])}")
    if SUM.get("match_rate") != 1.0:
        fail(f"mode-A match rate {SUM.get('match_rate')}")
    for k, v in SUM.get("instrument_checks", {}).items():
        if v is not None and v != 1.0:
            fail(f"instrument check {k} = {v}")
    for g, v in SUM.get("dv", {}).items():
        if v.get("ok") != 1.0:
            fail(f"divergence check {g} = {v.get('ok')}")
        if g.split("·")[-1] == "none" and v.get("identical") != 1.0:
            fail(f"no-intervention arm {g} diverged: identical = {v.get('identical')}")
    if "cross_run_identical" in SUM and SUM["cross_run_identical"] not in (None, 1.0):
        fail(f"cross-run determinism {SUM['cross_run_identical']}")
    if len(ROWS) != SUM.get("n_rows"):
        fail(f"records.jsonl has {len(ROWS)} rows, summary.json says {SUM.get('n_rows')}")


def group_filter(rd, grp):
    if grp == "pr_pooled":
        return lambda r: r["round"] == rd and r["first_task"] in PR_TASKS
    if grp == "other_pooled":
        return lambda r: r["round"] == rd and r["first_task"] not in PR_TASKS
    if grp == "all":
        return lambda r: r["round"] == rd
    return lambda r: r["round"] == rd and r["first_task"] == grp


def parse_flag(s):
    m = re.search(r"(\d+)\s*/\s*(\d+)", s or "")
    if m:
        return int(m.group(1)), int(m.group(2)), None
    m = re.search(r"(\d+)\s*$", s or "")
    return None, None, (int(m.group(1)) if m else None)


def check_published(out, ST):
    for rd in (1, 2):
        R = ST[f"round{rd}"]
        for arm, v in R["summary"].items():
            a = out["arms"].get(f"r{rd}|{arm}")
            if a is None:
                fail(f"published arm r{rd} {arm} missing in the replay")
                continue
            for kp, kn in (("success", "S_A"), ("completed", "C_A"), ("steps", "N_A"), ("U", "U_A")):
                if not rel_close(v[kp], a[kn], TOL_PUB):
                    fail(f"published r{rd} {arm} {kp} {v[kp]!r} vs replay {kn} {a[kn]!r}")
            if not rel_close(v["repairs"], a["rep_A"], 1e-9):
                warn(f"r{rd} {arm}: published repairs/chain {v['repairs']:.4f} vs replay rep_A {a['rep_A']:.4f}")
        for key, v in R["paired"].items():
            c = out["paired"].get(f"r{rd}|all|{key}")
            if c is None:
                continue
            for kp, kn in (("success", "S_A"), ("steps", "N_A"), ("U", "U_A")):
                if not rel_close(v[kp][0], c[kn][0], TOL_PUB):
                    fail(f"published r{rd} paired {key} {kp} {v[kp][0]!r} vs replay {kn} {c[kn][0]!r}")
        ft = R.get("first_target", {}).get("by_task", {})
        for t, rec in ft.items():
            for arm in R["summary"]:
                f = out["first"].get(f"r{rd}|{t}|{arm}")
                if f is not None and arm in rec and not rel_close(rec[arm], f["A"], 1e-9):
                    warn(f"first handoff r{rd} {t} {arm}: published {rec[arm]:.4f} vs replay A {f['A']:.4f}")
    R2 = ST["round2"]["paired"]
    for h, key in (("H1", "benefit-prior"), ("H2", "benefit-inverse"), ("H4", "noloc-benefit"),
                   ("H5a", "benefit_noretry-none"), ("H5b", "benefit_noretry-retry")):
        if not rel_close(R2[key]["success"][0], out["H"][h]["S_A"][0], TOL_PUB):
            fail(f"published {h} {R2[key]['success'][0]!r} vs replay {out['H'][h]['S_A'][0]!r}")
    fam = ST["round2"]["first_target"]["by_family"]["new_noninverse"]
    if not rel_close(fam["benefit-prior"][0], out["H"]["H3"]["f_A"][0], TOL_PUB):
        fail(f"published H3 {fam['benefit-prior'][0]!r} vs replay {out['H']['H3']['f_A'][0]!r}")


# ---------------------------------------------------------------- main
def main():
    SUM = load_json(SRC / "summary.json")
    ROWS = [json.loads(l) for l in (SRC / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ST = load_json(DER / "stats.json")
    gate(SUM, ROWS)
    P = [r for r in ROWS if r.get("match") and r.get("dv_ok", True)]
    out = dict(meta={}, dv={}, arms={}, paired={}, H={}, first={}, first_overall={}, first_paired={},
               audit={}, audit_pooled={}, across=None)

    # divergence and reproduction per group
    for g, v in SUM["dv"].items():
        rtag, arm = g.split("·")
        out["dv"][f"{rtag}|{arm}"] = dict(n=v["n"], ok=v["ok"], identical=v["identical"])

    # arm summaries
    for key, ref in SUM["arms"].items():
        rtag, arm = key.split("|")
        rd = int(rtag[1:])
        rs = [r for r in P if r["round"] == rd and r["arm"] == arm]
        mine = {k: mean(rs, k) for k in ARMKEYS}
        mine["A0B1_stop"] = sum(r.get("A0B1_stop", 0) for r in rs)
        mine["n"] = len(rs)
        for k in ARMKEYS:
            if not rel_close(mine[k], ref.get(k), TOL_PT):
                fail(f"arm {key} {k}: recomputed {mine[k]!r} vs summary {ref.get(k)!r}")
        if mine["n"] != ref["n"] or mine["A0B1_stop"] != ref["A0B1_stop"]:
            fail(f"arm {key} n/A0B1_stop differ")
        allr = [r for r in ROWS if r["round"] == rd and r["arm"] == arm and "match" in r]
        mine["match"] = float(np.mean([bool(r["match"]) for r in allr])) if allr else None
        mine["sha_eq"] = float(np.mean([bool(r.get("sha_eq")) for r in allr])) if allr else None
        dv = out["dv"].get(f"r{rd}|{arm}")
        mine["identical"] = dv["identical"] if dv else None
        out["arms"][f"r{rd}|{arm}"] = mine

    # paired chain-level differences
    for key, ref in SUM["paired"].items():
        sname, rtag, scn, comp = key.split("|")
        if sname != "P":
            continue
        if scn not in STRATUM:
            fail(f"unknown stratum in {key}")
            continue
        rd, st = int(rtag[1:]), STRATUM[scn]
        X, Y = comp.split("-")
        mine = paired(P, X, Y, lambda r, rd=rd, f=SFILT[st]: r["round"] == rd and f(r))
        cmp_cell(mine, ref, KEYS, f"paired {rtag}|{st}|{comp}")
        c = tolist(mine, KEYS)
        if c:
            c["cls_A"], c["cls_ii"] = cls(c["S_A"]), cls(c["S_ii"])
        out["paired"][f"r{rd}|{st}|{X}-{Y}"] = c
    for k, c in out["paired"].items():
        if c and k.split("|")[1] == "nonpr" and any(abs(x) > 1e-12 for x in c["L_ii"]):
            fail(f"{k}: (ii) differs from A in chains without push/rotate targets: L_ii = {c['L_ii']}")

    # pre-registered hypotheses
    r2 = lambda r: r["round"] == 2
    for h, (X, Y) in {"H1": ("benefit", "prior"), "H2": ("benefit", "inverse"), "H4": ("noloc", "benefit"),
                      "H5a": ("benefit_noretry", "none"), "H5b": ("benefit_noretry", "retry")}.items():
        mine = paired(P, X, Y, r2)
        cmp_cell(mine, SUM["H"][h], KEYS, f"hypothesis {h}")
        out["H"][h] = tolist(mine, KEYS)
    ref3, chosen = SUM["H"]["H3"], None
    for cand in NOINV_CANDIDATES:
        mine = paired_first(P, "benefit", "prior", lambda r, c=cand: r2(r) and r["first_task"] in c)
        if mine and mine["n"] == ref3["n"] and mine["bases"] == ref3["bases"] and all(
                rel_close(mine[k][p], ref3[k][p], TOL_CI) for k in FKEYS for p in ("pt", "lo", "hi")):
            chosen = (cand, mine)
            break
    if chosen is None:
        fail("H3 could not be reproduced with either candidate set of no-inverse tasks")
    else:
        out["H"]["H3"] = tolist(chosen[1], FKEYS)
        out["H_noinv_tasks"] = list(chosen[0])

    # first target handoff
    for key, ref in SUM["first"].items():
        rtag, task, arm = key.split("|")
        rd = int(rtag[1:])
        rs = [r for r in P if r["round"] == rd and r["arm"] == arm and r["first_task"] == task and r.get("f_reached")]
        cells = dict(Counter(f"A{r['f_A']}B{r['f_Bi']}" for r in rs))
        mine = dict(n=len(rs), A=mean(rs, "f_A"), Bi=mean(rs, "f_Bi"), Bii=mean(rs, "f_Bii"), cells=cells)
        if mine["n"] != ref["n"] or cells != ref["cells"] or not all(
                rel_close(mine[k], ref[k], TOL_PT) for k in ("A", "Bi", "Bii")):
            fail(f"first {key}: recomputed {mine} vs summary {ref}")
        ch = Counter("none" if r.get("f_rep") in (None, "") else str(r["f_rep"]) for r in rs)
        mine["choices"] = [[k, v] for k, v in ch.most_common(5)]
        out["first"][f"r{rd}|{task}|{arm}"] = mine
    for rd in (1, 2):
        for arm in sorted({r["arm"] for r in P if r["round"] == rd}):
            rs = [r for r in P if r["round"] == rd and r["arm"] == arm and r.get("f_reached") and r.get("f_Bii") is not None]
            out["first_overall"][f"r{rd}|{arm}"] = dict(n=len(rs), A=mean(rs, "f_A"), Bi=mean(rs, "f_Bi"), Bii=mean(rs, "f_Bii"))
            pub = ST[f"round{rd}"].get("first_target", {}).get("all", {}).get(arm)
            if pub is not None and not rel_close(pub, out["first_overall"][f"r{rd}|{arm}"]["A"], 1e-9):
                warn(f"first overall r{rd} {arm}: published {pub:.4f} vs replay A {out['first_overall'][f'r{rd}|{arm}']['A']:.4f}")
    for key, ref in SUM["first_paired"].items():
        rtag, grp, comp = key.split("|")
        rd = int(rtag[1:])
        X, Y = comp.split("-")
        g = FGROUP.get(grp, grp)
        mine = paired_first(P, X, Y, group_filter(rd, g))
        cmp_cell(mine, ref, FKEYS, f"first_paired {key}")
        out["first_paired"][f"r{rd}|{g}|{X}-{Y}"] = tolist(mine, FKEYS)

    # predicate audit (handoff-level; copied from summary.json, flags parsed)
    pooled = {"pr": Counter(), "other": Counter()}
    for task, a in SUM["audit"].items():
        h, s, y = parse_flag(a.get("flags", ""))
        rec = dict(kind=task_kind(task), n=a["n"], A_ne_Bdef=a["A_ne_Bdef"], A_ne_Borc=a["A_ne_Borc"],
                   Bdef_ne_Borc=a["Bdef_ne_Borc"], repaired=a["repaired"], flag_handoff=h, flag_start=s,
                   flag_yaw180=y, flags_raw=a.get("flags", ""))
        out["audit"][task] = rec
        grp = "pr" if task in PR_TASKS else "other"
        for k in ("n", "A_ne_Bdef", "A_ne_Borc", "Bdef_ne_Borc", "repaired"):
            pooled[grp][k] += rec[k]
    out["audit_pooled"] = {g: dict(c) for g, c in pooled.items()}

    # cross-check with the published analysis and across-task correlations
    if not FAILED:
        check_published(out, ST)
        out["across"] = across(out["first_paired"], ST)

    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed; nothing written.")
        sys.exit(2)

    out["meta"] = dict(
        inputs={str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                for p in (SRC / "summary.json", SRC / "records.jsonl", DER / "stats.json")},
        n_rows=len(ROWS), n_P=len(P), match_rate=SUM["match_rate"], errors=len(SUM["errors"]),
        missing=len(SUM["missing"]), instrument_checks=SUM["instrument_checks"],
        cross_run_identical=SUM.get("cross_run_identical"),
        bootstrap=dict(resamples=NBOOT, seed=SEED, cluster="base initial configuration", interval="95% percentile"),
        recomputed=dict(arms=len(out["arms"]), paired=sum(v is not None for v in out["paired"].values()),
                        first=len(out["first"]), first_paired=sum(v is not None for v in out["first_paired"].values()),
                        hypotheses=len(out["H"])),
        warnings=WARNED)
    (DER / "fullchain_stats.json").write_text(json.dumps(clean(out), indent=1, ensure_ascii=True), encoding="utf-8")

    cols = ("round", "chain", "base", "arm", "tier", "pr_chain", "first_task", "first_pos", "L", "S_O", "S_A", "C_A", "N_A",
            "U_A", "rep_A", "S_i", "S_io", "A0B1_stop", "S_ii", "C_ii", "N_ii", "U_ii", "rep_ii", "f_reached", "f_A",
            "f_Bi", "f_Bii", "f_rep", "dv_status")

    def cv(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return int(v)
        return v

    with open(DER / "fullchain_per_chain.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in P:
            w.writerow([cv(r.get("orig_id") if c == "chain" else r.get(c)) for c in cols])

    print("all checks passed; wrote", DER / "fullchain_stats.json", "and", DER / "fullchain_per_chain.csv")
    print("recomputed:", out["meta"]["recomputed"], "| warnings:", len(WARNED))
    for k in ("r1|all|benefit-retry", "r1|all|benefit-none", "r1|all|benefit-prior", "r1|nonpr|benefit-retry",
              "r2|all|benefit-prior", "r2|all|benefit_noretry-retry"):
        c = out["paired"][k]
        print(f"{k:32s} A {100*c['S_A'][0]:+5.1f}  (ii) {100*c['S_ii'][0]:+5.1f} [{100*c['S_ii'][1]:+.1f}, {100*c['S_ii'][2]:+.1f}]")
    print("first overall r1 none/benefit (ii):", out["first_overall"]["r1|none"]["Bii"], out["first_overall"]["r1|benefit"]["Bii"])
    print("audit pooled:", out["audit_pooled"])
    print("across: any bold cell under (ii) =", out["across"]["any_bold_ii"])


if __name__ == "__main__":
    main()
