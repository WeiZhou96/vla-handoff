# -*- coding: utf-8 -*-
"""Pre-registered analysis of the round-3 stage-3 (B_def) closed-loop confirmation experiment (prereg_round3.md §6–§11).

Checked before any result file is opened: lock_addendum.json exists and the sha256 values it records for worker_cl3.py and
analyze_cl3.py match the files; the sha256 values of locked_config.json and jobs_cl3.json match the locked values.
  --analyze       final analysis: runs only when all 627 chains have results (no interim analysis); writes analysis/report.md, summary.json, chains.jsonl
  --status        prints only completion counts and integrity checks (pairing, errors, criterion and bookkeeping consistency, pooled over arms); shows no per-arm results
  --smoke-gate    smoke gate: checks the smoke chains listed in lock_addendum.json (results_smoke/); dry-runs the analysis in memory (output discarded); writes smoke_gate.json
  --require-gate  before the full run: smoke_gate.json passed and matches the current addendum; sha of run.sh and of the scheduler copy match the addendum
  --pick-smoke    picks smoke chains from the locked jobs by a fixed rule (reads no results); prints only comma-separated ids
Statistics: cluster bootstrap over job["base"] (numbered after sorting); one draw set RandomState(20261011).randint(0,K,(5000,K)) shared by all comparisons;
  estimate per resample = sum of paired differences over all chains in the drawn clusters / number of chains (chains weighted equally); percentile 95% CI;
  one-sided p = (1 + #{d̄* ≤ threshold}) / 5001 (threshold 0; −0.05 for H6; a 1e-12 tolerance is added in the comparison against floating-point rounding); Holm over H2–H6 (one-sided α=0.025).
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
import argparse, contextlib, hashlib, io, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
NEW = Path(f"{RUN_ROOT}/closedloop3_bdef_20261012")
LOCKDIR = Path(f"{RUN_ROOT}/benefit_model_bdef_20261011")
LOCKED_CONFIG_SHA256 = "7a0c297dfdd7a2869d3ebc1a4d94fdc6714589e238665d11ab08ee8bf32dfa2e"
ARMS = ["none", "retry", "prior", "benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A"]
FC_ARM = dict(none="none", retry="retry", prior="prior", benefit_B="benefit", benefit_noretry_B="benefit_noretry",
              noloc_B="noloc", benefit_A="benefit")
REPAIR_ARMS = ("prior", "benefit_B", "benefit_noretry_B", "noloc_B", "benefit_A")
HYP = [("H1", "benefit_B", "retry", 0.0), ("H2", "benefit_B", "prior", 0.0), ("H3", "benefit_B", "none", 0.0),
       ("H4", "benefit_B", "benefit_A", 0.0), ("H5", "benefit_noretry_B", "none", 0.0), ("H6", "noloc_B", "benefit_B", -0.05)]
NB, SEED, ALPHA, EPS = 5000, 20261011, 0.025, 1e-12
MK = ("S", "C", "N", "R", "T")       # chain success, completed subtasks, steps, repairs, retries
BLOCKING = ("worker_sha", "fixture_ok", "job_match", "arms_complete", "mode_B", "probe_set", "chain_book", "repair_rule",
            "repair_exec", "nontarget_plain", "retry_rule", "probe_align", "bdef_stop", "steps_seen", "a_recorded_consistent",
            "rotate_def_eq_orc", "prior_table_consistent")
MUST_COVER = ("bdef_stop", "repair_rule", "retry_rule")
SMOKE_RULE = (("push_into_drawer",), ("stack_block",), ("rotate_pink_block_right", "rotate_red_block_left"),
              ("push_red_block_right", "push_blue_block_right", "push_pink_block_right"))


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def W(p):
    q, root = Path(p).resolve(), NEW.resolve()
    if root not in q.parents:
        raise RuntimeError(f"refusing to write outside {root}: {q}")
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def nonempty(d):
    return d.is_dir() and any(d.iterdir())


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


# ------------------------------------------------------------------ lock and addendum
def load_cfg():
    p = LOCKDIR / "locked_config.json"
    if sha(p) != LOCKED_CONFIG_SHA256:
        sys.exit("refused: locked_config.json sha256 does not match the locked value")
    cfg = json.loads(p.read_text())
    jp = Path(cfg["jobs_file"])
    if sha(jp) != cfg["sha256"]["jobs_cl3"]["sha256"]:
        sys.exit("refused: jobs_cl3.json sha256 does not match the locked value")
    jobs = json.loads(jp.read_text())["jobs"]
    if len(jobs) != cfg["n_chains"]:
        sys.exit("refused: number of jobs does not match the locked configuration")
    return cfg, jobs


def check_addendum():
    p = NEW / "lock_addendum.json"
    if not p.is_file():
        sys.exit("refused: lock_addendum.json does not exist; no result file is opened")
    A = json.loads(p.read_text())
    F = A.get("files", {})
    for f in ("worker_cl3.py", "analyze_cl3.py"):
        if F.get(f, {}).get("sha256") != sha(NEW / f):
            sys.exit(f"refused: sha256 of {f} does not match lock_addendum.json; no result file is opened")
    if A.get("locked_config_sha256") != LOCKED_CONFIG_SHA256:
        sys.exit("refused: locked_config_sha256 in lock_addendum.json does not match")
    warn = [f for f in ("run.sh", "supervise_multi.py")
            if not (NEW / f).is_file() or F.get(f, {}).get("sha256") != sha(NEW / f)]
    return A, sha(p), warn


def load(jobs, out):
    d, R = NEW / out, {}
    for j in jobs:
        p = d / f"{j['id']}.json"
        if p.is_file():
            R[j["id"]] = json.loads(p.read_text())
    return R


# ------------------------------------------------------------------ per-chain integrity checks (no between-arm comparison)
def check_arm(arm, A, job, cfg, pbt, c):
    ok = lambda k, v: c[k].append(bool(v))
    spec, targets, B = cfg["arm_spec"][arm], set(cfg["targets"]), cfg["budget"]["BUDGET"]
    ok("mode_B", A.get("mode") == "B" and A.get("fc_arm") == FC_ARM[arm] and A.get("arm") == arm)
    T, P = A["tasks"], {p["pos"]: p for p in A.get("probes", [])}
    ok("probe_set", set(P) == {t["pos"] for t in T if t["target"]})
    used, book = 0, 1 <= len(T) <= 5
    for i, t in enumerate(T):
        book = book and t["pos"] == i and t["task"] == job["tasks"][i] and t["target"] == (i > 0 and t["task"] in targets) \
            and t["success"] == bool(t["attempts"] and t["attempts"][-1]["success"]) and (t["success"] or i == len(T) - 1)
        pr = P.get(i)
        if t["target"] and pr is not None:
            if spec["repair"] == "none":
                rule = t.get("repair") is None and "q_none" not in t
            elif spec["repair"] == "argmax_prior_table_excluding_none":
                rule = pbt.get(t["task"]) is not None and t.get("repair") == pbt[t["task"]]
            else:
                u, qa = t.get("ustar"), t.get("q_all") or {}
                reps = {k: v for k, v in qa.items() if k != "none"}
                if u is None or not reps:
                    rule = False
                else:     # same expression as in fc_worker.run_chain; the frozen c̄ of benefit_A equals the locked c̄ (checked at worker start-up)
                    cb = cfg["cbar"].get(f"{t['task']}|{u}", cfg["cbar_global"].get(u, 100.0))
                    do = (t["q_ustar"] - t["q_none"] - spec["lam"] * cb / 360) > spec["delta"]
                    rule = (t.get("repair") == (u if do else None) and max(reps, key=reps.get) == u
                            and qa.get("none") == t["q_none"] and reps[u] == t["q_ustar"])
            ok("repair_rule", rule)
            ok("repair_exec", (t.get("repair") is not None and used < B) == bool(pr["repair_executed"]) == ("repair_steps" in t))
        elif not t["target"]:
            ok("nontarget_plain", "repair" not in t and "q_none" not in t)
        used += t.get("repair_steps") or 0
        at, u0 = t["attempts"], used
        if u0 >= B:
            exp = 0
        elif not at:
            exp = 1
        else:
            exp = 1 if (not t["target"] or not spec["retry"] or at[0]["success"] or u0 + at[0]["steps"] >= B) else 2
        ok("retry_rule", len(at) == exp and all(x["attempt"] == n for n, x in enumerate(at)))
        used += sum(x["steps"] for x in at)
        if t["target"] and pr is not None:
            pa = pr["attempts"]
            ok("probe_align", len(pa) == len(at) and all(x["success"] == y["success"] and x["steps"] == y["steps"] for x, y in zip(pa, at)))
            for x in pa:
                ok("bdef_stop", x["success"] == (x["bdef_first"] is not None) and bool(x["bdef_end"]) == bool(x["success"])
                   and (not x["success"] or x["bdef_first"] == x["steps"]))
                ok("steps_seen", x["steps_seen"] == x["steps"])
                ok("a_recorded_consistent", x["a_disagree"] == 0)
                if pr["kind"] == "rotate_object":
                    ok("rotate_def_eq_orc", x["bdef_first"] == x["borc_first"] and x["bdef_end"] == x["borc_end"])
            if not pr["repair_executed"] and pa:       # descriptive: without repair, the three criteria should agree on the first attempt
                x = pa[0]
                ok("att0_norepair_identity", x["a_first"] == x["bdef_first"] == x["borc_first"] and x["a_end"] == x["bdef_end"] == x["borc_end"])
    ok("chain_book", book and bool(A["success"]) == (len(T) == 5 and all(t["success"] for t in T))
       and A["completed"] == sum(bool(t["success"]) for t in T) and A["steps"] == used <= B)


def chain_checks(r, job, cfg, wsha, c, ex, pbt0):
    ok = lambda k, v: c[k].append(bool(v))
    ok("no_error", not r.get("error"))
    ok("worker_sha", r.get("worker_sha256") == wsha)
    if r.get("error"):
        return None, False
    sc = r.get("selfcheck") or {}
    fx = sc.get("fixture") or {}
    ok("fixture_ok", fx.get("ok") is True and fx.get("max_abs_diff") is not None and fx["max_abs_diff"] <= 1e-9)
    jj = r.get("job") or {}
    ok("job_match", jj.get("id") == job["id"] and jj.get("tasks") == job["tasks"] and jj.get("perturb_seed") == job["perturb_seed"])
    arms = r.get("arms") or {}
    ok("arms_complete", set(arms) == set(ARMS))
    paired = r.get("paired_initial_state_identical") is True and len({a.get("init_physics_sha") for a in arms.values()}) == 1
    ok("paired", paired)
    ex["pairing_rerun"] += bool(r.get("pairing_rerun"))
    ex["pairing_failed"] += not paired
    pbt = sc.get("prior_by_task") or {}
    pbt0.setdefault("v", pbt)
    ok("prior_table_consistent", pbt == pbt0["v"] and set(pbt) == set(cfg["targets"]))
    for arm in ARMS:
        if arm in arms:
            try:
                check_arm(arm, arms[arm], job, cfg, pbt, c)
            except (KeyError, TypeError, IndexError):
                ok("chain_book", False)
    targets = set(cfg["targets"])
    fp = next((i for i in range(1, 5) if job["tasks"][i] in targets), None)
    ok("first_pos_plan", fp == job.get("first_target_pos"))
    shared = False
    if fp is not None and set(arms) == set(ARMS):
        hs = []
        for a in arms.values():
            p = next((p for p in a.get("probes", []) if p["pos"] == fp), None)
            hs.append(None if p is None else json.dumps([p["handoff"], p["used_before"]], sort_keys=True))
        shared = None not in hs and len(set(hs)) == 1
    c["first_handoff_shared"].append(shared)
    return fp, shared


def chain_row(j, r, fp, shared):
    row = dict(id=j["id"], base=j["base"], stratum=j["stratum"], pr=bool(j.get("push_rotate_target")),
               unseen=not j.get("base_seen_before", True), fill=bool(j.get("fill")), first_pos=fp,
               first_task=(j["tasks"][fp] if fp is not None else None), shared=shared, **{k: {} for k in MK + ("F",)})
    for arm in ARMS:
        A = r["arms"][arm]
        row["S"][arm], row["C"][arm], row["N"][arm] = int(bool(A["success"])), int(A["completed"]), int(A["steps"])
        row["R"][arm] = sum(bool(p["repair_executed"]) for p in A["probes"])
        row["T"][arm] = sum(len(t["attempts"]) >= 2 for t in A["tasks"] if t["target"])
        if shared:
            row["F"][arm] = int(bool(A["tasks"][fp]["success"]))
    return row


# ------------------------------------------------------------------ statistics
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
    fam = ["H2", "H3", "H4", "H5", "H6"]
    go = True
    for k, h in enumerate(sorted(fam, key=lambda x: (H[x]["p"], fam.index(x))), 1):
        thr = ALPHA / (len(fam) - k + 1)
        rej = go and H[h]["p"] <= thr
        H[h].update(holm_rank=k, holm_threshold=thr, holm_reject=bool(rej))
        go = rej


def pp(s):
    return "—" if not s else f"{100 * s['pt']:+.1f} [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}]"


def nn(s, d=1):
    return "—" if not s else f"{s['pt']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def tab(head, body):
    return (["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
            + ["| " + " | ".join(str(x) for x in b) + " |" for b in body])


def canon(r):
    return json.dumps(dict(arms=r.get("arms"), tries=r.get("perturb_tries"), scene0=r.get("scene0")), sort_keys=True)


# ------------------------------------------------------------------ analysis pipeline
def compute(cfg, jobs, R, A, asha, Rs=None, meta=None):
    wsha = A["files"]["worker_cl3.py"]["sha256"]
    c, ex, pbt0 = defaultdict(list), Counter(), {}
    rows, excl = [], []
    for j in jobs:
        r = R.get(j["id"])
        if r is None:
            continue
        fp, shared = chain_checks(r, j, cfg, wsha, c, ex, pbt0)
        reason = ("crashed 3 times with the same seed (§10)" if r.get("error") else
                  "pairing check still failed after one re-run (§10)" if not r.get("paired_initial_state_identical") else
                  "arms incomplete (implementation defect, handled per §10/§12)" if set(r.get("arms", {})) != set(ARMS) else None)
        if reason:
            excl.append(dict(id=j["id"], base=j["base"], stratum=j["stratum"], reason=reason))
        else:
            rows.append(chain_row(j, r, fp, shared))
    ck = {k: float(np.mean(v)) for k, v in sorted(c.items()) if v}
    integ = all(ck.get(k, 1.0) == 1.0 for k in BLOCKING)
    n_all, n_an = len(jobs), len(rows)
    SUM = dict(time=time.strftime("%F %T"), addendum_sha256=asha, analyze_sha256=sha(Path(__file__).resolve()), worker_sha256=wsha,
               n_planned=n_all, n_results=len(R), n_analysed=n_an, excluded=excl, excluded_frac=len(excl) / max(n_all, 1),
               integrity=ck, integrity_ok=integ, pairing=dict(ex), meta=meta or {})
    if not rows:
        return SUM, ["# No chains to analyse"], rows
    boot = Boot([r["base"] for r in rows])
    SUM["n_bases"] = boot.K
    L = ["# Round-3 closed-loop confirmation experiment (criterion B_def): pre-registered analysis report", "",
         f"- Generated {SUM['time']}; lock_addendum.json sha256 `{asha}`; analyze_cl3.py `{SUM['analyze_sha256']}`; worker_cl3.py `{wsha}`",
         f"- Planned {n_all} chains; with results {len(R)}; **analysed {n_an} ({boot.K} bases)**; excluded {len(excl)} ({100 * SUM['excluded_frac']:.1f}%)",
         "- All outcomes are whole-chain success (all 5 subtasks succeed within the budget); success, retry and chain stopping of every attempt are decided by B_def; criterion A is only recorded.",
         "- Smoke chains are not part of the analysed sample (prereg §10 requires all 627 chains to run with the full design); the full run re-ran these chains unchanged, and bitwise identity of the two runs is reported in Section 12.", ""]
    if SUM["excluded_frac"] > 0.05:
        L.append("> **Note: the exclusion fraction exceeds 5% (prereg §10 requires this to be reported prominently).**\n")
    if not integ:
        L.append("> **Not all integrity checks passed (see Section 12): treated as an implementation error under §10; the results below cannot be taken as final.**\n")
    # 1 per arm
    SUM["arms"] = {a: {k: float(np.mean([r[k][a] for r in rows])) for k in MK} for a in ARMS}
    L += ["## 1. Whole-chain results per arm (§11)", ""] + tab(
        ["Arm", "n chains", "Chain success rate", "Completed subtasks/chain", "Steps/chain", "Repairs/chain", "Retries/chain"],
        [[a, n_an, f3(v["S"]), f3(v["C"]), f"{v['N']:.0f}", f3(v["R"]), f3(v["T"])] for a, v in SUM["arms"].items()]) + [""]
    # 2 hypotheses
    H = {}
    for h, X, Y, thr in HYP:
        cs = comp(boot, rows, X, Y, MK, thr)
        H[h] = dict(X=X, Y=Y, **cs["S"], completed=cs["C"], steps=cs["N"])
    holm(H)
    H["H1"]["supported"] = bool(H["H1"]["lo"] > 0)
    SUM["hypotheses"] = H
    body = []
    for h, X, Y, thr in HYP:
        v = H[h]
        dec = ("supported (95% CI lower bound > 0)" if v["supported"] else "not supported") if h == "H1" else \
            (f"H0 rejected (Holm rank {v['holm_rank']}, threshold {v['holm_threshold']:.4f})" if v["holm_reject"]
             else f"not rejected (Holm rank {v['holm_rank']}, threshold {v['holm_threshold']:.4f})")
        body.append([h, f"{X}−{Y}", "−5 pp (non-inferiority)" if h == "H6" else "0", pp(v), f"{v['p']:.4f}", dec, nn(v["steps"], 0)])
    L += ["## 2. Pre-registered hypotheses (paired difference in chain success, percentage points; cluster bootstrap, 5000 resamples, seed 20261011)", ""] + tab(
        ["Hypothesis", "Comparison", "Threshold", "Estimate [95% CI]", "One-sided p", "Decision", "Δ steps/chain"], body) + [
        "", "H1 is tested alone at one-sided α=0.025 (decision by 95% CI lower bound > 0); H2–H6 form one family, tested with Holm step-down at family-wise one-sided α=0.025, independently of the H1 result.", ""]
    # 3 decisions and conclusions
    h1 = H["H1"]
    L += ["## 3. Decision table and conclusions permitted in the paper (§8)", ""]
    if h1["supported"]:
        L.append(f"- **H1 holds**: the gate trained on handoff-referenced labels improves whole-chain success over retry under B_def; estimate {pp(h1)} percentage points.")
    else:
        L.append(f"- **H1 does not hold**: the paper states that the method shows no whole-chain improvement over retry under B_def; estimate {pp(h1)} percentage points; "
                 "conclusions about the method are limited to secondary hypotheses rejected under Holm; no subgroup is promoted to a conclusion.")
    rej = [h for h in ("H2", "H3", "H4", "H5", "H6") if H[h]["holm_reject"]]
    L.append(f"- Secondary hypotheses rejected under Holm: {', '.join(rej) if rej else 'none'}.")
    if not H["H4"]["holm_reject"]:
        L.append("- H4 not rejected: no claim that relabelling improved closed-loop performance; offline B-label results are reported as offline results only.")
    if not H["H6"]["holm_reject"]:
        L.append("- H6 not rejected: no claim that the locator-free model is equivalent to the original model.")
    L += ["- Stated regardless of outcome: a true gain of +2.6 pp (the replay estimate for the earlier gate) is likely undetectable at this sample size (§4).", ""]
    # 4 harm
    harm = {Y: bool(H[h]["hi"] < 0) for h, Y in (("H1", "retry"), ("H3", "none"), ("H2", "prior"))}
    SUM["harm"] = harm
    L += ["## 4. Harm check (§8)", ""] + [f"- benefit_B−{Y}: 95% CI {pp(H[h])} → "
                                       + ("**entire interval below 0: the paper reports the method as harmful in this comparison**" if harm[Y] else "interval not entirely below 0")
                                       for h, Y in (("H1", "retry"), ("H3", "none"), ("H2", "prior"))] + [""]
    # 5 worst case
    wc = [(r["base"], r["S"]["benefit_B"] - r["S"]["retry"]) for r in rows] + [(e["base"], -1.0) for e in excl]
    bw = Boot([b for b, _ in wc])
    SUM["H1_worst_case"] = summ(bw([b for b, _ in wc], [[d] for _, d in wc]))
    L += ["## 5. H1 worst-case sensitivity (§10: excluded chains counted as benefit_B failure and retry success; resampled over the bases of all chains, same seed)", "",
          f"benefit_B−retry: {pp(SUM['H1_worst_case'])} (n={len(wc)}, excluded {len(excl)})", ""]
    # 6 21 pairs
    PW = {}
    body = []
    for i in range(len(ARMS)):
        for k in range(i + 1, len(ARMS)):
            X, Y = ARMS[k], ARMS[i]
            cs = comp(boot, rows, X, Y, ("S", "C", "N"))
            PW[f"{X}-{Y}"] = cs
            body.append([f"{X}−{Y}", pp(cs["S"]), nn(cs["C"], 3), nn(cs["N"], 0)])
    SUM["pairwise"] = PW
    L += ["## 6. All 21 pairwise paired differences (descriptive, unadjusted 95% CI)", ""] + tab(["Comparison", "Δ chain success (pp)", "Δ completed subtasks", "Δ steps"], body) + [""]
    # 7 subgroups
    strata = sorted({r["stratum"] for r in rows})
    subs = [("含 push/rotate 目标（位置 1–4）", lambda r: r["pr"]), ("不含 push/rotate 目标", lambda r: not r["pr"]),
            ("base 此前未出现过", lambda r: r["unseen"])] + [(f"层 {s}", (lambda r, s=s: r["stratum"] == s)) for s in strata]
    SG = {}
    for name, f in subs:
        rs = [r for r in rows if f(r)]
        SG[name] = dict(n=len(rs), n_bases=len({r["base"] for r in rs}),
                        arms={a: (float(np.mean([r["S"][a] for r in rs])) if rs else None) for a in ARMS},
                        comps={f"{X}-{Y}": comp(boot, rs, X, Y, ("S",))["S"] for _, X, Y, _ in HYP})
    SUM["subgroups"] = SG
    L += ["## 7. Subgroups (descriptive, no confirmatory conclusions)", "", "### 7a. Chain success rate per arm", ""] + tab(
        ["Subgroup", "n", "n base"] + ARMS, [[k, v["n"], v["n_bases"]] + [f3(v["arms"][a]) for a in ARMS] for k, v in SG.items()]) + [
        "", "### 7b. Paired differences for the hypothesis comparisons (pp)", ""] + tab(
        ["Subgroup", "n"] + [f"{X}−{Y}" for _, X, Y, _ in HYP],
        [[k, v["n"]] + [pp(v["comps"][f"{X}-{Y}"]) for _, X, Y, _ in HYP] for k, v in SG.items()]) + [""]
    # 8 first target handoff
    fr = [r for r in rows if r["shared"]]
    ft = sorted({r["first_task"] for r in fr})
    FH = dict(n_chains=len(fr), n_not_shared=n_an - len(fr), rates={}, comps={})
    for t in ft:
        rs = [r for r in fr if r["first_task"] == t]
        FH["rates"][t] = dict(n=len(rs), **{a: float(np.mean([r["F"][a] for r in rs])) for a in ARMS})
    for name, f in [("全部", lambda r: True)] + [(t, (lambda r, t=t: r["first_task"] == t)) for t in ft]:
        rs = [r for r in fr if f(r)]
        FH["comps"][name] = dict(n=len(rs), **{f"{X}-{Y}": comp(boot, rs, X, Y, ("F",))["F"] for _, X, Y, _ in HYP})
    SUM["first_handoff"] = FH
    L += [f"## 8. First target handoff (chains in which all arms reach this handoff with identical pose and steps used: {len(fr)}/{n_an})", "", "### 8a. Success rate (including retry, B_def)", ""] + tab(
        ["Task", "n"] + ARMS, [[t, v["n"]] + [f3(v[a]) for a in ARMS] for t, v in FH["rates"].items()]) + [
        "", "### 8b. Paired differences (pp)", ""] + tab(
        ["Group", "n"] + [f"{X}−{Y}" for _, X, Y, _ in HYP],
        [[k, v["n"]] + [pp(v[f"{X}-{Y}"]) for _, X, Y, _ in HYP] for k, v in FH["comps"].items()]) + [""]
    # 9 repair choices, 10 criterion disagreement
    rep, dis = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        for arm in ARMS:
            Aa = R[row["id"]]["arms"][arm]
            P = {p["pos"]: p for p in Aa["probes"]}
            for t in Aa["tasks"]:
                if not t["target"]:
                    continue
                if arm in REPAIR_ARMS:
                    rep[(t["task"], arm)][t.get("repair") or "(不修复)"] += 1
                d = dis[(t["task"], arm)]
                d["handoffs"] += 1
                d["repaired"] += bool(P[t["pos"]]["repair_executed"])
                for x in P[t["pos"]]["attempts"]:
                    d["att"] += 1
                    d["A_seen_B_fail"] += x["a_first"] is not None and not x["success"]
                    d["B_succ_A_unseen"] += bool(x["success"]) and x["a_first"] is None
                    d["end_A_ne_Bdef"] += bool(x["a_end"]) != bool(x["bdef_end"])
                    d["end_Borc_ne_Bdef"] += bool(x["borc_end"]) != bool(x["bdef_end"])
    SUM["repair_choices"] = {f"{t}|{a}": dict(v) for (t, a), v in sorted(rep.items())}
    SUM["disagreement"] = {f"{t}|{a}": dict(v) for (t, a), v in sorted(dis.items())}
    L += ["## 9. Repair choices per task (all target handoffs; after the first one the arms are in different states; descriptive)", ""] + tab(
        ["Task", "Arm", "Handoffs", "Repair choice (count)"],
        [[t, a, sum(v.values()), ", ".join(f"{u}×{n}" for u, n in v.most_common(6))] for (t, a), v in sorted(rep.items())]) + [
        "", "## 10. Disagreement counts between criterion A and B_def (per target attempt; A is recorded only and does not drive execution)", "",
        "A seen / B_def not met: A held at some step within the attempt, but the attempt ended as a B_def failure; B_def met / A not seen: A had not yet held when B_def was met.", ""] + tab(
        ["Task", "Arm", "Handoffs", "Repairs executed", "Attempts", "A seen/B failed", "B met/A not seen", "A≠B_def at end", "B_orc≠B_def at end"],
        [[t, a, v["handoffs"], v["repaired"], v["att"], v["A_seen_B_fail"], v["B_succ_A_unseen"], v["end_A_ne_Bdef"], v["end_Borc_ne_Bdef"]]
         for (t, a), v in sorted(dis.items())]) + [""]
    # 11 offline and development set
    du = cfg["dev_tuning"]["dev_utility_B"]
    offl = {k: dict(path=cfg["sha256"][k]["path"], sha256_ok=sha(cfg["sha256"][k]["path"]) == cfg["sha256"][k]["sha256"])
            for k in ("v4_report", "v4_report_rpexclude", "v4_rows", "tune_report", "label_check", "cache_provenance")}
    fill = Counter(j["stratum"] for j in jobs if j.get("fill"))
    SUM.update(offline_files=offl, dev_utility_B=du, stratum_fill=dict(fill), per_stratum_planned=dict(Counter(j["stratum"] for j in jobs)))
    L += ["## 11. Offline B-label tables and development-set utility (§11, locked files cited unchanged)", ""] + [
        f"- `{v['path']}`: sha256 {'matches the locked value' if v['sha256_ok'] else '**does not match the locked value**'}" for v in offl.values()] + [
        "- The offline tables (including M7_A cross-fitting, the M7_v3all in-sample reference and the `--rp exclude` sensitivity) are taken unchanged from the v4_report file listed above.", ""] + tab(
        ["Dev-set policy", "Success rate", "Steps", "Utility U", "Repair fraction"],
        [[k, f3(v.get("s")), f"{v['steps']:.1f}", f3(v.get("U")), f3(v.get("rep"))] if isinstance(v, dict) else [k, "—", "—", f3(v), "—"]
         for k, v in du.items()]) + [
        "", f"δ_B = {cfg['delta_benefit']} (dev-set rule {cfg['dev_tuning']['dev_rule']}); cross-fitted δ test: {cfg['dev_tuning']['crossfit_delta_test']}",
        f"Stratum fill (chains with fill=True in jobs, per stratum): {dict(fill) if fill else 'none'}", ""]
    # 12 integrity
    cross = None
    if Rs:
        same = [canon(Rs[i]) == canon(R[i]) for i in Rs if i in R and not Rs[i].get("error") and not R[i].get("error")]
        cross = dict(n=len(same), identical=sum(same))
    SUM["cross_run"] = cross
    L += ["## 12. Integrity checks, exclusion list and cross-run consistency", "",
          f"Blocking checks (expected 1.000): { {k: round(ck[k], 4) for k in BLOCKING if k in ck} }",
          f"Descriptive checks: { {k: round(v, 4) for k, v in ck.items() if k not in BLOCKING} }",
          f"Pairing: {ex['pairing_rerun']} chains re-run, {ex['pairing_failed']} still inconsistent after re-run",
          f"Smoke chains bitwise identical to the full run: {'—' if not cross else f'{cross[chr(105) + chr(100) + chr(101) + chr(110) + chr(116) + chr(105) + chr(99) + chr(97) + chr(108)]}/{cross[chr(110)]}'}",
          f"Addendum written at {SUM['meta'].get('addendum_time')}; first stage-3 job started at {SUM['meta'].get('first_start')}; "
          f"addendum precedes data: {SUM['meta'].get('addendum_before_data')}", "", "Excluded chains:", ""]
    L += tab(["id", "base", "Stratum", "Reason"], [[e["id"], e["base"], e["stratum"], e["reason"]] for e in excl]) if excl else ["None"]
    return SUM, L, rows


# ------------------------------------------------------------------ modes
def status(cfg, jobs, out, quiet=False):
    A, asha, warn = check_addendum()
    if out == "results_smoke":
        ids = set(A.get("smoke_ids") or [])
        jobs = [j for j in jobs if j["id"] in ids]
    sfx = "" if out == "results" else "_" + out
    R = load(jobs, out)
    c, ex, pbt0 = defaultdict(list), Counter(), {}
    for j in jobs:
        if j["id"] in R:
            chain_checks(R[j["id"]], j, cfg, A["files"]["worker_cl3.py"]["sha256"], c, ex, pbt0)
    claims, live = 0, set()
    for j in jobs:
        p = NEW / ("claims" + sfx) / f"{j['id']}.json"
        if j["id"] not in R and p.is_file():
            claims += 1
            try:
                pid = json.loads(p.read_text()).get("pid")
            except Exception:
                pid = None
            if pid_alive(pid):
                live.add(pid)
    errs = [json.loads(p.read_text()) for p in (NEW / ("errors" + sfx)).glob("*.json")] if (NEW / ("errors" + sfx)).is_dir() else []
    starts = [json.loads(p.read_text()) for p in (NEW / ("starts" + sfx)).glob("*.json")] if (NEW / ("starts" + sfx)).is_dir() else []
    n_err_res = sum(bool(r.get("error")) for r in R.values())
    secs = [r["seconds"] for r in R.values() if r.get("seconds")]
    if not secs and out == "results":
        secs = [r["seconds"] for r in load(jobs, "results_smoke").values() if r.get("seconds")]
    rem = len(jobs) - len(R)
    ck = {k: (round(float(np.mean(v)), 4), len(v)) for k, v in sorted(c.items()) if v}
    lines = [f"[{out}] done {len(R)}/{len(jobs)} (results with error: {n_err_res}); in progress {claims} (live worker processes {len(live)}); not started {rem - claims}",
             f"Crash records: {len(errs)} chains, {sum(e.get('n', 0) for e in errs)} crashes in total; chains restarted (process killed/rescheduled) {sum(s.get('n_started', 0) > 1 for s in starts)}",
             f"refused markers: {len(list((NEW / 'refused').glob('*'))) if (NEW / 'refused').is_dir() else 0}",
             f"Pairing: initial-state hash identical {sum(c['paired'])}/{len(c['paired'])}; pairing re-runs {ex['pairing_rerun']}; still inconsistent after re-run {ex['pairing_failed']}",
             f"Checks (pooled over arms; value, number of entries; expected 1.0): {ck}"]
    if warn:
        lines.append(f"Warning: sha256 of {warn} does not match lock_addendum.json or the file is missing")
    if secs and rem:
        lines.append(f"Estimated remaining: {rem} chains × {np.mean(secs):.0f} s ÷ {max(len(live), 1)} processes ≈ {rem * np.mean(secs) / max(len(live), 1) / 3600:.1f} h")
    if not quiet:
        print("\n".join(lines), flush=True)
    return A, asha, warn, c, ex, R, jobs


def smoke_gate(cfg, jobs):
    A, asha, warn, c, ex, R, sj = status(cfg, jobs, "results_smoke")
    ids = A.get("smoke_ids") or []
    chk = dict(smoke_ids_valid=3 <= len(sj) == len(ids) <= 4, all_results_present=bool(sj) and all(j["id"] in R for j in sj),
               no_error_results=not any(r.get("error") for r in R.values()), no_crash_records=not nonempty(NEW / "errors_results_smoke"),
               no_refused=not nonempty(NEW / "refused"), no_pairing_rerun=ex["pairing_rerun"] == 0, addendum_files_match=not warn)
    for k in BLOCKING + ("no_error", "paired"):
        chk[f"check:{k}"] = all(c.get(k, [])) and (k not in MUST_COVER + ("no_error", "paired") or bool(c.get(k)))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            compute(cfg, sj, R, A, asha)
        chk["analysis_dry_run"] = True
    except Exception as e:      # record only the exception type, not its message
        chk["analysis_dry_run"] = False
        chk["analysis_dry_run_error_type"] = type(e).__name__
    ok = all(v for k, v in chk.items() if not k.endswith("_type"))
    secs = [r["seconds"] for r in R.values() if r.get("seconds")]
    g = dict(time=time.strftime("%F %T"), **{"pass": ok}, checks=chk, smoke_ids=ids, addendum_sha256=asha,
             worker_sha256=A["files"]["worker_cl3.py"]["sha256"], analyze_sha256=A["files"]["analyze_cl3.py"]["sha256"],
             seconds_per_chain=(float(np.mean(secs)) if secs else None), uncovered=[k for k in BLOCKING if not c.get(k)])
    W(NEW / "smoke_gate.json").write_text(json.dumps(g, indent=1, ensure_ascii=False))
    print("\nSmoke gate checks:")
    for k, v in chk.items():
        print(f"  {k}: {v}")
    print(f"  Uncovered checks (no entries): {g['uncovered']}")
    print(f"Time per chain {g['seconds_per_chain']} s" if secs else "")
    print("SMOKE GATE PASS" if ok else "SMOKE GATE FAIL", flush=True)
    sys.exit(0 if ok else 3)


def require_gate():
    A, asha, warn = check_addendum()
    p = NEW / "smoke_gate.json"
    g = json.loads(p.read_text()) if p.is_file() else {}
    ok = g.get("pass") is True and g.get("addendum_sha256") == asha and not warn \
        and g.get("worker_sha256") == A["files"]["worker_cl3.py"]["sha256"]
    print("GATE OK" if ok else f"GATE NOT OK (smoke_gate.json missing / not passed / inconsistent with the addendum; file warnings {warn})", flush=True)
    sys.exit(0 if ok else 3)


def pick_smoke(jobs):
    ids = []
    for grp in SMOKE_RULE:
        cand = sorted((j for j in jobs if j["stratum"] in grp),
                      key=lambda j: (not (j.get("first_target_pos") == 1 and j["tasks"][1] == j["stratum"]), j["index"]))
        if cand:
            ids.append(cand[0]["id"])
    print(",".join(ids))


def analyze(cfg, jobs):
    A, asha, warn = check_addendum()
    R = load(jobs, "results")
    miss = [j["id"] for j in jobs if j["id"] not in R]
    if miss:
        sys.exit(f"refused: {len(miss)} chains have no result (prereg §10: analysis only after all runs complete, no interim analysis)")
    bad = [i for i, r in R.items() if r.get("worker_sha256") != A["files"]["worker_cl3.py"]["sha256"]]
    if bad:
        sys.exit(f"refused: {len(bad)} results were not produced by the worker_cl3.py registered in the addendum (e.g. {bad[:3]})")
    sids = set(A.get("smoke_ids") or [])
    Rs = load([j for j in jobs if j["id"] in sids], "results_smoke")
    st = []
    for d in ("starts", "starts_results_smoke"):
        for p in (NEW / d).glob("*.json") if (NEW / d).is_dir() else []:
            st += [s["time"] for s in json.loads(p.read_text()).get("starts", [])]
    first = min(st) if st else None
    meta = dict(addendum_time=A.get("time"), first_start=first, addendum_before_data=bool(first and A.get("time") and A["time"] < first),
                file_warnings=warn)
    SUM, L, rows = compute(cfg, jobs, R, A, asha, Rs, meta)
    out = NEW / "analysis"
    W(out / "report.md").write_text("\n".join(L) + "\n")
    W(out / "summary.json").write_text(json.dumps(SUM, indent=1, ensure_ascii=False))
    with open(W(out / "chains.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("analyze", "status", "smoke-gate", "require-gate", "pick-smoke"):
        g.add_argument("--" + m, action="store_true")
    ap.add_argument("--out", default="results", choices=("results", "results_smoke"))
    a = ap.parse_args()
    if Path(__file__).resolve().parent != NEW.resolve():
        sys.exit(f"analyze_cl3.py must be located in {NEW.resolve()}")
    cfg, jobs = load_cfg()
    if a.pick_smoke:
        pick_smoke(jobs)
    elif a.status:
        status(cfg, jobs, a.out)
    elif a.smoke_gate:
        smoke_gate(cfg, jobs)
    elif a.require_gate:
        require_gate()
    else:
        analyze(cfg, jobs)


if __name__ == "__main__":
    main()
