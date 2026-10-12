# -*- coding: utf-8 -*-
"""Round-3 statistics for the paper: pre-registered closed-loop confirmation under B_def, relabelling under
B_def, offline v4 model, delta tuning and plan.

Inputs (paths relative to the repository root)
  data/raw/closedloop3_bdef_20261012/analysis/summary.json, chains.jsonl     analyze_cl3.py output, as run
  data/raw/relabel_bdef_20261011/analysis/summary.json                       relabel report
  data/raw/benefit_model_bdef_20261011/{v4_report_all.json, tune_report.json, plan_report.json, label_check.json}
      (searched in that folder, then in its analysis/ subfolder; v4_report_all_rpexclude.json is read if present)
Output
  data/derived/round3_stats.json

All paired differences, percentile intervals, one-sided p-values and Holm decisions are recomputed from chains.jsonl with the
pre-registered bootstrap (prereg_round3.md section 7; analyze_cl3.py): clusters = base configurations (sorted), 5,000 resamples
drawn as numpy RandomState(20261011).randint(0, K, (5000, K)), one index set shared by all comparisons, every chain weighted
equally, two-sided 95% percentile intervals, p = (1 + #{d* <= threshold + 1e-12}) / 5001, Holm step-down over H2-H6 at
one-sided alpha = 0.025. The script exits with code 2 and lists every discrepancy if a point estimate or interval bound
differs from summary.json by more than 1e-9, or a p-value, Holm rank/threshold or decision differs.
"""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CL3 = ROOT / "data" / "raw" / "closedloop3_bdef_20261012" / "analysis"
RLB = ROOT / "data" / "raw" / "relabel_bdef_20261011" / "analysis"
BMD = ROOT / "data" / "raw" / "benefit_model_bdef_20261011"
OUT = ROOT / "data" / "derived" / "round3_stats.json"

ARMS = ["none", "retry", "prior", "benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A"]
HYP = [("H1", "benefit_B", "retry", 0.0), ("H2", "benefit_B", "prior", 0.0), ("H3", "benefit_B", "none", 0.0),
       ("H4", "benefit_B", "benefit_A", 0.0), ("H5", "benefit_noretry_B", "none", 0.0), ("H6", "noloc_B", "benefit_B", -0.05)]
FAMILY = ["H2", "H3", "H4", "H5", "H6"]
MK = ("S", "C", "N", "R", "T")
NB, SEED, ALPHA, EPS = 5000, 20261011, 0.025, 1e-12
TOL, PTOL = 1e-9, 1e-12
BLOCKING = ("worker_sha", "fixture_ok", "job_match", "arms_complete", "mode_B", "probe_set", "chain_book", "repair_rule",
            "repair_exec", "nontarget_plain", "retry_rule", "probe_align", "bdef_stop", "steps_seen", "a_recorded_consistent",
            "rotate_def_eq_orc", "prior_table_consistent")
# Time of the pre-registration lock, as reported with the lock record; it is not contained in the inputs read here
# and must be checked against prereg/lock_record.json.
PREREG_LOCK_TIME = "2026-10-11 21:00:43"
# Keys written by analyze_cl3.py (Chinese labels, used verbatim; see the glossary in README.md): chains with a push or
# rotation target (positions 1-4) / without one / base configuration not seen before / stratum prefix / all / (no repair)
SUB_PR = "含 push/rotate 目标（位置 1–4）"
SUB_NONPR = "不含 push/rotate 目标"
SUB_UNSEEN = "base 此前未出现过"
STRATUM_PREFIX = "层 "
FH_ALL = "全部"
NOREP = "(不修复)"

ERRORS = []


def die(msg):
    print("compute_round3: FAILED", file=sys.stderr)
    print(msg, file=sys.stderr)
    sys.exit(2)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        die(f"missing input: {p}")


def locate(name, required=True):
    for d in (BMD, BMD / "analysis"):
        if (d / name).is_file():
            return d / name
    if required:
        die(f"missing input: {name} (looked in {BMD} and {BMD / 'analysis'})")
    return None


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


# ------------------------------------------------------------------ bootstrap, identical to analyze_cl3.py
class Boot:
    def __init__(self, bases):
        self.bases = sorted(set(bases))
        self.ix, self.K = {b: i for i, b in enumerate(self.bases)}, len(self.bases)
        draw = np.random.RandomState(SEED).randint(0, self.K, size=(NB, self.K))
        self.Cm = np.zeros((NB, self.K))
        np.add.at(self.Cm, (np.repeat(np.arange(NB), self.K), draw.ravel()), 1)

    def __call__(self, bases, vals):
        if not bases:
            return None
        vals = np.asarray(vals, float).reshape(len(bases), -1)
        S, N = np.zeros((self.K, vals.shape[1])), np.zeros(self.K)
        for b, v in zip(bases, vals):
            S[self.ix[b]] += v
            N[self.ix[b]] += 1
        num, den = self.Cm @ S, self.Cm @ N
        with np.errstate(invalid="ignore", divide="ignore"):
            est = num / den[:, None]
        return dict(n=int(N.sum()), n_bases=int((N > 0).sum()), pt=S.sum(0) / N.sum(), est=est, n_empty=int((den == 0).sum()))


def summ(st, j=0, thr=0.0):
    if st is None:
        return None
    e = st["est"][:, j]
    e = e[np.isfinite(e)]
    if e.size == 0:
        return None
    lo, hi = np.percentile(e, [2.5, 97.5])
    return dict(pt=float(st["pt"][j]), lo=float(lo), hi=float(hi), p=(1 + int(np.sum(e <= thr + EPS))) / (NB + 1),
                thr=thr, n=st["n"], n_bases=st["n_bases"], n_empty=st["n_empty"])


def comp(boot, rs, X, Y, keys=MK, thr=0.0):
    if not rs:
        return {k: None for k in keys}
    st = boot([r["base"] for r in rs], [[r[k][X] - r[k][Y] for k in keys] for r in rs])
    return {k: summ(st, i, thr if i == 0 else 0.0) for i, k in enumerate(keys)}


def holm(H):
    go = True
    for k, h in enumerate(sorted(FAMILY, key=lambda x: (H[x]["p"], FAMILY.index(x))), 1):
        thr = ALPHA / (len(FAMILY) - k + 1)
        rej = go and H[h]["p"] <= thr
        H[h].update(holm_rank=k, holm_threshold=thr, holm_reject=bool(rej))
        go = rej


# ------------------------------------------------------------------ comparison with summary.json
def cmp_summ(name, mine, ref):
    if mine is None or ref is None:
        if not (mine is None and ref is None):
            ERRORS.append(f"{name}: recomputed is {'None' if mine is None else 'a value'}, summary.json is "
                          f"{'None' if ref is None else 'a value'}")
        return
    for k in ("pt", "lo", "hi"):
        if abs(float(mine[k]) - float(ref[k])) > TOL:
            ERRORS.append(f"{name}.{k}: recomputed {mine[k]!r}, summary.json {ref[k]!r}")
    if abs(float(mine["p"]) - float(ref["p"])) > PTOL:
        ERRORS.append(f"{name}.p: recomputed {mine['p']!r}, summary.json {ref['p']!r}")
    for k in ("n", "n_bases"):
        if int(mine[k]) != int(ref[k]):
            ERRORS.append(f"{name}.{k}: recomputed {mine[k]}, summary.json {ref[k]}")
    if float(mine["thr"]) != float(ref["thr"]):
        ERRORS.append(f"{name}.thr: recomputed {mine['thr']}, summary.json {ref['thr']}")


def cmp_num(name, a, b, tol=TOL):
    if a is None or b is None or abs(float(a) - float(b)) > tol:
        ERRORS.append(f"{name}: recomputed {a!r}, reference {b!r}")


def tri(s):
    return None if s is None else [float(s["pt"]), float(s["lo"]), float(s["hi"])]


def ts(x, what):
    if not isinstance(x, str):
        die(f"{what}: time stamp missing or not a string ({x!r})")
    s = x.strip().replace("T", " ")[:19]
    try:
        return s, datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        die(f"{what}: cannot parse time stamp {x!r}")


# ------------------------------------------------------------------ main
def main():
    p_sum, p_chains, p_rl = CL3 / "summary.json", CL3 / "chains.jsonl", RLB / "summary.json"
    p_v4, p_tu = locate("v4_report_all.json"), locate("tune_report.json")
    p_pl, p_lc = locate("plan_report.json"), locate("label_check.json")
    p_rpx = locate("v4_report_all_rpexclude.json", required=False)
    S = jload(p_sum)
    if not p_chains.is_file():
        die(f"missing input: {p_chains}")
    rows = [json.loads(x) for x in p_chains.read_text(encoding="utf-8").splitlines() if x.strip()]

    # ---------------- conduct: sample, exclusions, integrity
    excl = S["excluded"]
    if not isinstance(excl, list):
        die("summary.json: 'excluded' is not a list")
    if len(rows) != S["n_analysed"]:
        ERRORS.append(f"chains.jsonl has {len(rows)} chains, summary.json n_analysed = {S['n_analysed']}")
    if S["n_analysed"] + len(excl) != S["n_planned"]:
        ERRORS.append(f"n_analysed + excluded != n_planned ({S['n_analysed']} + {len(excl)} vs {S['n_planned']})")
    if S["n_results"] != S["n_planned"]:
        ERRORS.append(f"n_results {S['n_results']} != n_planned {S['n_planned']}")
    if len({r["id"] for r in rows}) != len(rows):
        ERRORS.append("duplicate chain ids in chains.jsonl")
    for r in rows:
        for k in MK:
            if set(r[k]) != set(ARMS):
                ERRORS.append(f"chain {r['id']}: field {k} does not contain exactly the seven arms")
                break
    if S.get("integrity_ok") is not True:
        ERRORS.append("summary.json integrity_ok is not true")
    bad = [k for k in BLOCKING if S["integrity"].get(k) != 1.0]
    if bad:
        ERRORS.append(f"blocking integrity checks not equal to 1.0: {bad}")
    if S["meta"].get("addendum_before_data") is not True:
        ERRORS.append("summary.json meta.addendum_before_data is not true")

    # ---------------- arm means
    arms = {a: {k: float(np.mean([r[k][a] for r in rows])) for k in MK} for a in ARMS}
    for a in ARMS:
        for k in MK:
            cmp_num(f"arms.{a}.{k}", arms[a][k], S["arms"][a][k])

    # ---------------- hypotheses with Holm
    boot = Boot([r["base"] for r in rows])
    if boot.K != S["n_bases"]:
        ERRORS.append(f"number of base configurations: recomputed {boot.K}, summary.json {S['n_bases']}")
    H = {}
    for h, X, Y, thr in HYP:
        cs = comp(boot, rows, X, Y, MK, thr)
        H[h] = dict(X=X, Y=Y, **cs["S"], completed=cs["C"], steps=cs["N"])
    holm(H)
    H["H1"]["supported"] = bool(H["H1"]["lo"] > 0)
    for h, X, Y, thr in HYP:
        ref = S["hypotheses"][h]
        if ref.get("X") != X or ref.get("Y") != Y:
            ERRORS.append(f"{h}: comparison in summary.json is {ref.get('X')}-{ref.get('Y')}, expected {X}-{Y}")
        cmp_summ(f"{h}.S", H[h], ref)
        cmp_summ(f"{h}.completed", H[h]["completed"], ref.get("completed"))
        cmp_summ(f"{h}.steps", H[h]["steps"], ref.get("steps"))
    for h in FAMILY:
        ref = S["hypotheses"][h]
        if H[h]["holm_rank"] != ref.get("holm_rank"):
            ERRORS.append(f"{h}.holm_rank: recomputed {H[h]['holm_rank']}, summary.json {ref.get('holm_rank')}")
        if H[h]["holm_reject"] != ref.get("holm_reject"):
            ERRORS.append(f"{h}.holm_reject: recomputed {H[h]['holm_reject']}, summary.json {ref.get('holm_reject')}")
        cmp_num(f"{h}.holm_threshold", H[h]["holm_threshold"], ref.get("holm_threshold"), 1e-15)
    if H["H1"]["supported"] != S["hypotheses"]["H1"].get("supported"):
        ERRORS.append(f"H1.supported: recomputed {H['H1']['supported']}, summary.json {S['hypotheses']['H1'].get('supported')}")

    # ---------------- all 21 pairwise comparisons
    PW = {}
    for i in range(len(ARMS)):
        for k in range(i + 1, len(ARMS)):
            X, Y = ARMS[k], ARMS[i]
            cs = comp(boot, rows, X, Y, ("S", "C", "N"))
            key = f"{X}-{Y}"
            PW[key] = cs
            if key not in S["pairwise"]:
                ERRORS.append(f"pairwise {key} missing in summary.json")
                continue
            for m in ("S", "C", "N"):
                cmp_summ(f"pairwise.{key}.{m}", cs[m], S["pairwise"][key][m])

    # ---------------- every arm against retry (table column), cross-checked against the pairwise results
    VR = {}
    for a in ARMS:
        if a == "retry":
            continue
        cs = comp(boot, rows, a, "retry", ("S", "N"))
        VR[a] = {"S": tri(cs["S"]), "N": tri(cs["N"])}
        key = f"{a}-retry"
        for m in ("S", "N"):
            mine = cs[m]
            if key in PW:
                ref = PW[key][m]
                for q in ("pt", "lo", "hi"):
                    cmp_num(f"vs_retry.{a}.{m}.{q}", mine[q], ref[q])
            else:
                ref = PW[f"retry-{a}"][m]
                cmp_num(f"vs_retry.{a}.{m}.pt", mine["pt"], -ref["pt"])
                cmp_num(f"vs_retry.{a}.{m}.lo", mine["lo"], -ref["hi"])
                cmp_num(f"vs_retry.{a}.{m}.hi", mine["hi"], -ref["lo"])

    # ---------------- worst-case sensitivity for H1 (section 10) and harm check (section 8)
    wc = [(r["base"], r["S"]["benefit_B"] - r["S"]["retry"]) for r in rows] + [(e["base"], -1.0) for e in excl]
    bw = Boot([b for b, _ in wc])
    WC = summ(bw([b for b, _ in wc], [[d] for _, d in wc]))
    cmp_summ("H1_worst_case", WC, S["H1_worst_case"])
    harm = {Y: bool(H[h]["hi"] < 0) for h, Y in (("H1", "retry"), ("H3", "none"), ("H2", "prior"))}
    for Y, v in harm.items():
        if v != S["harm"].get(Y):
            ERRORS.append(f"harm.{Y}: recomputed {v}, summary.json {S['harm'].get(Y)}")

    # ---------------- subgroups (descriptive)
    strata = sorted({r["stratum"] for r in rows})
    subs = [(SUB_PR, "pr_target", lambda r: r["pr"]), (SUB_NONPR, "no_pr_target", lambda r: not r["pr"]),
            (SUB_UNSEEN, "unseen_base", lambda r: r["unseen"])] + \
           [(STRATUM_PREFIX + s, f"stratum|{s}", (lambda r, s=s: r["stratum"] == s)) for s in strata]
    SG = {}
    for name, key, f in subs:
        rs = [r for r in rows if f(r)]
        ref = S["subgroups"].get(name)
        if ref is None:
            ERRORS.append(f"subgroup '{name}' missing in summary.json")
            continue
        if len(rs) != ref["n"]:
            ERRORS.append(f"subgroup {key}: n recomputed {len(rs)}, summary.json {ref['n']}")
        rates = {a: (float(np.mean([r["S"][a] for r in rs])) if rs else None) for a in ARMS}
        comps = {}
        for _, X, Y, _ in HYP:
            c = comp(boot, rs, X, Y, ("S",))["S"]
            comps[f"{X}-{Y}"] = tri(c)
            cmp_summ(f"subgroup.{key}.{X}-{Y}", c, ref["comps"].get(f"{X}-{Y}"))
        for a in ARMS:
            if rates[a] is not None:
                cmp_num(f"subgroup.{key}.rate.{a}", rates[a], ref["arms"][a])
        SG[key] = dict(n=len(rs), n_bases=len({r["base"] for r in rs}), arms=rates, comps=comps)

    # ---------------- first target handoff shared by all arms (descriptive)
    fr = [r for r in rows if r["shared"]]
    FHr = S["first_handoff"]
    if len(fr) != FHr["n_chains"]:
        ERRORS.append(f"first_handoff.n_chains: recomputed {len(fr)}, summary.json {FHr['n_chains']}")
    FH = dict(n_chains=len(fr), n_not_shared=len(rows) - len(fr), comps={}, rates=FHr["rates"])
    for _, X, Y, _ in HYP:
        c = comp(boot, fr, X, Y, ("F",))["F"]
        FH["comps"][f"{X}-{Y}"] = tri(c)
        cmp_summ(f"first_handoff.{X}-{Y}", c, FHr["comps"][FH_ALL].get(f"{X}-{Y}"))

    # ---------------- strata as planned
    per_stratum = {s: sum(r["stratum"] == s for r in rows) for s in strata}
    if per_stratum != {k: int(v) for k, v in S["per_stratum_planned"].items()}:
        ERRORS.append(f"chains per stratum differ from summary.json per_stratum_planned: {per_stratum}")

    # ---------------- relabelling
    RL = jload(p_rl)
    if RL.get("complete_and_verified") is not True:
        ERRORS.append("relabel summary: complete_and_verified is not true")
    if not (RL["jobs"] == RL["results"] == RL["good"]) or RL["missing"] != 0 or RL["failed"]:
        ERRORS.append("relabel summary: jobs/results/good/missing/failed inconsistent")
    m_all = RL["match"]["ALL|ALL"]
    if not (m_all["exec"] == m_all["exec_match"] == m_all["success_match"] == m_all["steps_match"]) \
            or not (m_all["arms"] == m_all["k_match"]):
        ERRORS.append("relabel summary: mode-A executions do not all reproduce the stored labels")
    per_task = {t: dict(n=v["n"], moved=v["moved"], changed=v["changed"], yA=v["yA"], yB=v["yB"], diff_ci=v["diff_ci"])
                for t, v in RL["per_task"].items()}
    lemma = RL["lemma"]
    n_changed = sum(v["changed"] for v in per_task.values())
    n_changed_lemma = lemma["pr_rep|moved"]["branches_changed"] + lemma["other_moved|moved"]["branches_changed"]
    if n_changed != n_changed_lemma:
        ERRORS.append(f"relabel: changed branches per task ({n_changed}) != lemma moved-branch changes ({n_changed_lemma})")
    unmoved_changed = lemma["pr_rep|unmoved"]["branches_changed"] + lemma["other_unmoved_sample|unmoved"]["branches_changed"]
    need = ("n", "gainA", "gainB", "gain_diff_ci", "a_best_still_best", "unique_a_best_still_best", "regret",
            "none_best_A", "none_best_B")
    rank = {}
    for t, v in RL["rankings_pr"].items():
        miss = [k for k in need if k not in v]
        if miss:
            ERRORS.append(f"relabel rankings_pr[{t}] lacks {miss}")
            continue
        rank[t] = {k: v[k] for k in need}
        rank[t]["all_fields"] = v
    if "ALL_PR" not in rank:
        ERRORS.append("relabel rankings_pr has no ALL_PR entry")
    pr_tasks = [t for t in per_task if t.startswith("push_") and t != "push_into_drawer" or t.startswith("rotate_")]
    relabel = dict(
        created=RL["created"], branches=RL["jobs"], verified=RL["good"], complete_and_verified=RL["complete_and_verified"],
        mode_A_executions=m_all["exec"], mode_A_executions_reproduced=m_all["exec_match"],
        divergence_checks=RL["divergence_checks"], smoke_vs_full=RL["smoke_vs_full"],
        branches_label_changed=n_changed, tasks_with_label_change={t: v["changed"] for t, v in per_task.items() if v["changed"]},
        unmoved_branches_checked=lemma["pr_rep|unmoved"]["branches"] + lemma["other_unmoved_sample|unmoved"]["branches"],
        unmoved_branches_changed=unmoved_changed, lemma=lemma,
        pr_mean_label_change_range=[min(per_task[t]["yB"] - per_task[t]["yA"] for t in pr_tasks),
                                    max(per_task[t]["yB"] - per_task[t]["yA"] for t in pr_tasks)],
        per_task=per_task, rankings_pr=rank, label_sources=RL["label_sources"],
        escalated_tasks=len(RL["escalate_tasks"]) if isinstance(RL["escalate_tasks"], list) else RL["escalate_tasks"],
        pr_unmoved_changed=len(RL["pr_unmoved_changed"]) if isinstance(RL["pr_unmoved_changed"], list) else RL["pr_unmoved_changed"],
        process_hours=RL["cost"]["seconds"] / 3600.0, steps_run=RL["cost"]["steps_run"], b_runs=RL["cost"]["b_runs"])

    # ---------------- offline v4 report (B labels)
    V = jload(p_v4)
    if V.get("criterion") != "B_def":
        ERRORS.append(f"v4_report_all.json criterion is {V.get('criterion')!r}, expected 'B_def'")
    sel = V["selection"]["all|ALL"]
    keys = ("n", "p0", "rand", "M7", "M7n", "M7_A", "M7_v3all", "oracle_split")
    diffs = ("M7_minus_M7_A", "M7_minus_M7n", "M7_minus_M7_v3all")
    offline = {k: sel[k] for k in keys}
    offline.update({k: sel[k] for k in diffs})
    offline.update(data=V["data"], note=V["note"], featurizer_maxabs_diff_vs_v3_all=V["featurizer_maxabs_diff_vs_v3_all"],
                   provenance_ok=all(v.get("ok", True) for v in V["provenance"].values() if isinstance(v, dict)))
    offline_rpx = None
    if p_rpx is not None:
        Vx = jload(p_rpx)
        sx = Vx.get("selection", {}).get("all|ALL", {})
        offline_rpx = {k: sx[k] for k in keys + diffs if k in sx}
        offline_rpx["rp"] = Vx.get("rp")

    # ---------------- delta tuning
    T = jload(p_tu)
    dev_sets = {k: dict(n=v["n"], delta=v["delta"]) for k, v in T["dev_sets"].items()}
    if T["delta_benefit"] != 0.0 or any(v["delta"] != 0.0 for v in dev_sets.values()):
        ERRORS.append(f"tune_report: delta_B is not 0.0 on every development set ({T['delta_benefit']}, {dev_sets})")
    for k, v in T["dev_utility_B"].items():
        a = v["U"] if isinstance(v, dict) else v
        r = S["dev_utility_B"].get(k)
        b = (r["U"] if isinstance(r, dict) else r) if r is not None else None
        cmp_num(f"dev_utility_B.{k} (tune_report vs locked config in summary.json)", a, b, 1e-12)
    tune = dict(created=T["created"], lam=T["lam"], delta_B=T["delta_benefit"], delta_rounds12=T["delta_round1_carried_in_round2"],
                dev_rule=T["dev_rule"], dev_sets=dev_sets, dev_utility_B=T["dev_utility_B"],
                dev_benefitB_minus_round2policyA_U=T["dev_benefitB_minus_round2policyA_U"],
                crossfit_delta_test=T["crossfit_delta_test"], cbar_check=T["cbar_check"])

    # ---------------- plan and sample size
    P = jload(p_pl)
    ss = P["sample_size"]
    z = NormalDist()
    n_req = math.ceil((z.inv_cdf(1 - ss["alpha_one_sided"]) + z.inv_cdf(ss["power"])) ** 2 * ss["sd"] ** 2 * ss["deff"]
                      / ss["effect"] ** 2)
    if n_req != ss["n_required"]:
        ERRORS.append(f"sample size: recomputed n_required {n_req}, plan_report {ss['n_required']}")
    if P["n_total"] != S["n_planned"] or P["bases"] != S["n_bases"]:
        ERRORS.append(f"plan_report n_total/bases ({P['n_total']}/{P['bases']}) differ from summary.json "
                      f"({S['n_planned']}/{S['n_bases']})")
    if {k: int(v) for k, v in P["strata"].items()} != per_stratum:
        ERRORS.append("plan_report strata differ from the chains per stratum")
    plan = dict(n_total=P["n_total"], per_task=P["per_task"], bases=P["bases"], max_chains_per_base=P["max_chains_per_base"],
                catalog=P["catalog"], previous_rounds=P["previous_rounds"], shortfall=P["shortfall"], filled=P["filled"],
                strata=P["strata"], bases_per_stratum=P["bases_per_stratum"], selection_order=P["selection_order"])
    power = dict(alpha_one_sided=ss["alpha_one_sided"], power=ss["power"], effect=ss["effect"], sd=ss["sd"], deff=ss["deff"],
                 n_required=ss["n_required"], n_inflated=ss["n_inflated"], n_total=ss["n_total"],
                 n_analysed_assumed=ss["n_analysed_assumed"], mde_at_n_analysed=ss["mde_at_n_analysed"],
                 replay_round1_gate_minus_retry_pp=2.6)

    # ---------------- label check (recorded only)
    LC = jload(p_lc)
    label_check = dict(sha256=sha256(p_lc), top_level_keys=sorted(LC) if isinstance(LC, dict) else None)

    # ---------------- timeline
    order = [("relabel_complete", RL["created"], "relabel summary.json"), ("delta_tuned", T["created"], "tune_report.json"),
             ("prereg_locked", PREREG_LOCK_TIME, "lock_record.json (constant in this script)"),
             ("code_hashed", S["meta"].get("addendum_time"), "lock_addendum.json"),
             ("first_job", S["meta"].get("first_start"), "start records"), ("analysis", S["time"], "analysis summary.json")]
    timeline, prev = {}, None
    for key, raw, src in order:
        s, d = ts(raw, key)
        if prev is not None and not d > prev[1]:
            ERRORS.append(f"timeline: {key} ({s}) is not after {prev[0]}")
        timeline[key] = s
        prev = (key, d)
    timeline["sources"] = {k: src for k, _, src in order}

    if ERRORS:
        die(f"{len(ERRORS)} discrepancies:\n  " + "\n  ".join(ERRORS))

    # ---------------- output
    hyp = {}
    for h, X, Y, thr in HYP:
        x = H[h]
        hyp[h] = dict(X=X, Y=Y, threshold=thr, S=tri(x), p=x["p"], C=tri(x["completed"]), N=tri(x["steps"]),
                      n=x["n"], n_bases=x["n_bases"],
                      holm_rank=x.get("holm_rank"), holm_threshold=x.get("holm_threshold"),
                      decision=(("supported" if x["supported"] else "not supported") if h == "H1"
                                else ("rejected" if x["holm_reject"] else "not rejected")))
    rc = {k: {("(no repair)" if u == NOREP else u): n for u, n in v.items()} for k, v in S["repair_choices"].items()}
    out = dict(
        created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        inputs={rel(p): sha256(p) for p in (p_sum, p_chains, p_rl, p_v4, p_tu, p_pl, p_lc) + ((p_rpx,) if p_rpx else ())},
        recomputed_and_matched_summary=True,
        bootstrap=dict(resamples=NB, seed=SEED, clusters="base configuration", ci="two-sided 95% percentile",
                       p="(1 + #{d* <= threshold}) / 5001", holm_family=FAMILY, alpha_one_sided=ALPHA),
        code_sha256=dict(addendum=S["addendum_sha256"], analyze=S["analyze_sha256"], worker=S["worker_sha256"]),
        n_planned=S["n_planned"], n_results=S["n_results"], n_analysed=S["n_analysed"], n_bases=boot.K,
        n_excluded=len(excl), excluded_frac=S["excluded_frac"], pairing=S["pairing"],
        integrity=S["integrity"], integrity_ok=S["integrity_ok"], n_blocking_checks=len(BLOCKING),
        cross_run=S["cross_run"], stratum_fill=S["stratum_fill"], per_stratum=per_stratum,
        timeline=timeline,
        arms=arms, vs_retry=VR, hyp=hyp, h1_worst_case=tri(WC), harm=harm,
        pairwise={k: {m: tri(v[m]) for m in ("S", "C", "N")} for k, v in PW.items()},
        subgroups=SG, first_handoff=FH, repair_choices=rc, disagreement=S["disagreement"],
        power=power, plan=plan, relabel=relabel, offline_v4=offline, offline_v4_rpexclude=offline_rpx, tune=tune,
        label_check=label_check)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"round3_stats.json written: {OUT}")
    print(f"recomputed from {len(rows)} chains / {boot.K} bases; all estimates, intervals, p-values and decisions match summary.json")
    for h in hyp:
        v = hyp[h]
        print(f"  {h} {v['X']}-{v['Y']}: {100 * v['S'][0]:+.1f} [{100 * v['S'][1]:+.1f}, {100 * v['S'][2]:+.1f}] pp, "
              f"p={v['p']:.4f}, {v['decision']}")


if __name__ == "__main__":
    main()
