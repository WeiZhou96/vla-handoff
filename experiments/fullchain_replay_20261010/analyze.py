# -*- coding: utf-8 -*-
"""Full-chain dual-criterion replay analysis.
Three full-chain success measures:
  A   : mode-A replay (driven by the original criterion) = the original experiment, checked bit for bit;
  (i) : post-hoc rescoring along the mode-A trajectory -- a target subtask is judged by B_def at the end of its last attempt (B_orc as sensitivity); A0B1 always counts as a failure;
  (ii): mode-B counterfactual rerun -- B_def drives attempt termination, retries and chain stopping (headline number).
P = chains whose mode-A replay matches the original result bit for bit and whose mode-B divergence check passes (main sample); S = all replayed chains.
Paired difference X − Y: cluster bootstrap over bases (2000 resamples, seed 17; the seed is reset for every cell, and all metrics within a cell share the same resamples).
Usage: python -B analyze.py [--jobs jobs.json] [--res results] [--out analysis] [--max-tier 3] [--cross results_smoke] [--gate]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan_jobs import NOINV, PR, summarize  # noqa: E402

NBOOT, SEED, LAM, H = 2000, 17, 0.5, 360
FIELDS = ("pos", "task", "target", "repair", "repair_steps", "repair_task_success", "attempts", "success")
METRICS = ("S_O", "S_A", "S_i", "S_io", "S_ii", "C_A", "C_ii", "N_A", "N_ii", "U_A", "U_ii")
KEYS = METRICS + ("L_i", "L_ii")
FKEYS = ("f_A", "f_Bi", "f_Bii")


# ---------------------------------------------------------------- loading and per-chain metrics
def load(jobs, res):
    d = HERE / res
    runs, errors, missing = {}, [], []
    for j in jobs:
        p = d / f"{j['id']}.json"
        if not p.exists():
            missing.append(j["id"]); continue
        r = json.loads(p.read_text())
        if r.get("error"):
            errors.append(dict(id=j["id"], error=r["error"][-800:])); continue
        for arm, mv in r["arms"].items():
            runs[(j["round"], j["orig_id"], arm)] = dict(job=j, A=mv.get("A"), B=mv.get("B"), scene0=r["scene0"],
                                                         tries=r["perturb_tries"], gpu=r.get("gpu"))
    return runs, errors, missing


def compare(o, s):
    d = []
    if o["init_physics_sha"] != s["init_physics_sha"]:
        d.append("init_sha")
    d += [k for k in ("success", "completed", "steps") if o[k] != s[k]]
    if len(o["tasks"]) != len(s["tasks"]):
        d.append("n_tasks")
    for x, y in zip(o["tasks"], s["tasks"]):
        d += [f"t{x['pos']}.{k}" for k in FIELDS if x.get(k) != y.get(k)]
    return d


def predict_dv(A, B):
    """Whether mode B agrees with mode A up to the first divergence point (predicted from the steps at which the criteria first hold in mode A)."""
    if A["init_physics_sha"] != B["init_physics_sha"]:
        return "?", False, "init_sha"
    sa, sb = summarize(A)["tasks"], summarize(B)["tasks"]
    pa = {p["pos"]: p for p in A["probes"]}; pb = {p["pos"]: p for p in B["probes"]}
    for i, ta in enumerate(sa):
        if i >= len(sb):
            return "?", False, f"B 缺 pos{ta['pos']}"
        tb, pos = sb[i], ta["pos"]
        if not ta["target"]:
            if ta != tb:
                return "?", False, f"非目标 pos{pos} 不一致"
            continue
        if any(ta.get(k) != tb.get(k) for k in ("task", "repair", "repair_steps", "repair_task_success")):
            return "?", False, f"pos{pos} 修复不一致"
        xa_l, xb_l = pa[pos]["attempts"], pb[pos]["attempts"]
        for n, xa in enumerate(xa_l):
            if n >= len(xb_l):
                return "?", False, f"pos{pos} att{n} B 缺失"
            xb, ka, kb = xb_l[n], xa["a_first"], xa["bdef_first"]
            if ka is not None:
                if kb is not None and kb <= ka:
                    if kb < ka:
                        return "diverged", bool(xb["success"] and xb["steps"] == kb), f"pos{pos} att{n}: B 先于 A 成立"
                    if not (xb["success"] and xb["steps"] == ka and xb["sha_end"] == xa["sha_end"]):
                        return "?", False, f"pos{pos} att{n}: 同步成立但 B 记录不同"
                    continue
                return "diverged", bool(xb["a_first"] == ka and xb["sha_at_a_first"] == xa["sha_end"]), f"pos{pos} att{n}: A 成立而 B 未成立"
            if kb is not None:
                return "diverged", bool(xb["success"] and xb["steps"] == kb), f"pos{pos} att{n}: A 失败而 B 成立"
            if xb["success"] or xb["steps"] != xa["steps"] or xb["sha_end"] != xa["sha_end"]:
                return "?", False, f"pos{pos} att{n}: 双方失败但 B 记录不同"
        if len(xb_l) != len(xa_l) or ta["success"] != tb["success"]:
            return "?", False, f"pos{pos} 尝试次数/结果不同"
    ok = len(sa) == len(sb) and A["success"] == B["success"] and A["steps"] == B["steps"]
    return "identical", bool(ok), "" if ok else "整链不同"


def rescore(run, key, L):
    P = {p["pos"]: p for p in run["probes"]}
    C, a0b1 = 0, 0
    for t in run["tasks"]:
        if t["target"]:
            at = P[t["pos"]]["attempts"]
            b = bool(at and at[-1][key])
            if not t["success"]:
                a0b1 = int(b); break
            if not b:
                break
        elif not t["success"]:
            break
        C += 1
    return int(C == L), C, a0b1


def util(C, N, L):
    return C / L - LAM * N / (L * H)


def n_rep(run):
    return sum(1 for t in run["tasks"] if t.get("repair") and t.get("repair_steps") is not None)


def first(run, j, mode):
    fp = j["first_pos"]
    t = next((t for t in run["tasks"] if t["pos"] == fp), None) if fp is not None else None
    if mode == "A":
        if t is None:
            return dict(f_reached=False)
        at = next(p for p in run["probes"] if p["pos"] == fp)["attempts"]
        last = at[-1] if at else None
        return dict(f_reached=True, f_A=int(t["success"]), f_Bi=int(bool(last and last["bdef_end"])),
                    f_Bio=int(bool(last and last["borc_end"])), f_rep=t.get("repair"))
    return dict(f_reached_B=t is not None, f_Bii=(int(t["success"]) if t is not None else None))


def _yaw(q0, q1):
    from scipy.spatial.transform import Rotation as R
    return float((R.from_quat(q1) * R.from_quat(q0).inv()).as_euler("xyz", degrees=True)[2])


def handoffs(run, row, mode):
    out = []
    tmap = {t["pos"]: t for t in run["tasks"]}
    for pr in run["probes"]:
        t, at = tmap[pr["pos"]], pr["attempts"]
        last = at[-1] if at else None
        h = dict(round=row["round"], orig_id=row["orig_id"], base=row["base"], arm=row["arm"], mode=mode, task=pr["task"],
                 kind=pr["kind"], pos=pr["pos"], is_first=pr["pos"] == row["first_pos"], repair=t.get("repair"),
                 repair_executed=pr["repair_executed"], success=bool(t["success"]), n_att=len(at),
                 A_end=bool(last and last["a_end"]), Bdef=bool(last and last["bdef_end"]), Borc=bool(last and last["borc_end"]),
                 flags_h=pr.get("flags_handoff", {}), flags_s=(last or {}).get("flags_start", {}),
                 yaw_rep=pr.get("yaw_post_repair"), yaw_end=(last or {}).get("yaw_cum"))
        ob = pr.get("obj")
        if mode == "A" and ob and ob in pr["handoff"] and "current_pos" in pr["handoff"][ob]:
            ph, pr_, pe = pr["handoff"][ob], pr["post_repair"][ob], (last["end_poses"][ob] if last else pr["post_repair"][ob])
            if pr["kind"] == "push_object":
                h["d_rep"] = 100 * (pr_["current_pos"][0] - ph["current_pos"][0])
                h["d_end"] = 100 * (pe["current_pos"][0] - ph["current_pos"][0])
            elif pr["kind"] == "rotate_object" and "current_orn" in ph:
                h["d_rep"] = _yaw(ph["current_orn"], pr_["current_orn"])
                h["d_end"] = _yaw(ph["current_orn"], pe["current_orn"])
        out.append(h)
    return out


def checks_A(run, chk):
    for pr in run["probes"]:
        for x in pr["attempts"]:
            chk["a_recomputed"].append(x["a_end"] == x["success"] and x["a_disagree"] == 0
                                       and (x["a_first"] == x["steps"] if x["success"] else x["a_first"] is None))
            chk["steps_seen"].append(x["steps_seen"] == x["steps"])
            if pr["kind"] == "rotate_object":
                chk["rotate_def_eq_orc"].append(x["bdef_first"] == x["borc_first"] and x["bdef_end"] == x["borc_end"])
        if not pr["repair_executed"] and pr["attempts"]:
            x = pr["attempts"][0]
            chk["att0_norepair_identity"].append(x["a_first"] == x["bdef_first"] == x["borc_first"]
                                                 and x["a_end"] == x["bdef_end"] == x["borc_end"])


def checks_B(run, chk):
    for pr in run["probes"]:
        for x in pr["attempts"]:
            chk["b_stop"].append(x["success"] == (x["bdef_first"] is not None) and x["bdef_end"] == x["success"]
                                 and (x["bdef_first"] == x["steps"] if x["success"] else True))
            chk["steps_seen"].append(x["steps_seen"] == x["steps"])
            chk["a_disagree_B"].append(x["a_disagree"] == 0)
            if pr["kind"] == "rotate_object":
                chk["rotate_def_eq_orc"].append(x["bdef_first"] == x["borc_first"] and x["bdef_end"] == x["borc_end"])


def build(jobs, res):
    runs, errors, missing = load(jobs, res)
    rows, hand, chk, scenes = [], [], defaultdict(list), defaultdict(set)
    for (rnd, oid, arm), R in runs.items():
        j, A, B = R["job"], R.get("A"), R.get("B")
        o, src, L = j["orig"][arm], j["orig"][arm]["source"], len(j["tasks"])
        scenes[(rnd, oid)].add(tuple(np.round(R["scene0"], 9)))
        row = dict(round=rnd, orig_id=oid, base=j["base"], arm=arm, tier=j["tier"], pr_chain=j["pr_chain"],
                   first_task=j["first_task"], first_pos=j["first_pos"], L=L, S_O=int(o["success"]),
                   worker=src["worker"], orig_dir=src["dir"], gpu=R["gpu"], same_gpu=src.get("gpu") == R["gpu"],
                   scene0_eq=bool(src.get("scene0") is not None and np.allclose(src["scene0"], R["scene0"], atol=1e-9, rtol=0)),
                   tries_eq=src.get("perturb_tries") == R["tries"])
        if A:
            d = compare(o, summarize(A))
            row.update(match=not d, diffs=d, sha_eq=o["init_physics_sha"] == A["init_physics_sha"],
                       S_A=int(A["success"]), C_A=int(A["completed"]), N_A=int(A["steps"]), rep_A=n_rep(A))
            row["S_i"], row["C_i"], row["A0B1_stop"] = rescore(A, "bdef_end", L)
            row["S_io"], row["C_io"], _ = rescore(A, "borc_end", L)
            row["U_A"] = util(row["C_A"], row["N_A"], L)
            row.update(first(A, j, "A")); checks_A(A, chk); hand += handoffs(A, row, "A")
        if B:
            row.update(S_ii=int(B["success"]), C_ii=int(B["completed"]), N_ii=int(B["steps"]), rep_ii=n_rep(B))
            row["U_ii"] = util(row["C_ii"], row["N_ii"], L)
            row.update(first(B, j, "B")); checks_B(B, chk); hand += handoffs(B, row, "B")
        if A and B:
            s, ok, info = predict_dv(A, B)
            row.update(dv_status=s, dv_ok=ok, dv_info=info)
            chk["dv_ok"].append(ok)
            if arm == "none":
                chk["none_identical"].append(s == "identical" and ok)
        rows.append(row)
    for v in scenes.values():
        chk["scene0_across_tiers"].append(len(v) == 1)
    return rows, hand, chk, errors, missing, runs


# ---------------------------------------------------------------- statistics
def boot(data, keys):
    if not data:
        return None
    bases = sorted({b for b, _ in data}); bi = {b: i for i, b in enumerate(bases)}
    Ssum, N = np.zeros((len(bases), len(keys))), np.zeros(len(bases))
    for b, d in data:
        Ssum[bi[b]] += d; N[bi[b]] += 1
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
    iA, ii, iI = METRICS.index("S_A"), METRICS.index("S_ii"), METRICS.index("S_i")
    for v in by.values():
        if X in v and Y in v and all(v[X].get(k) is not None and v[Y].get(k) is not None for k in METRICS):
            d = np.array([v[X][k] - v[Y][k] for k in METRICS], float)
            data.append((v[X]["base"], np.concatenate([d, [d[iI] - d[iA], d[ii] - d[iA]]])))
    return boot(data, KEYS)


def paired_first(rows, X, Y, filt):
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] in (X, Y) and filt(r) and r.get("f_reached") and r.get("f_Bii") is not None:
            by[(r["round"], r["orig_id"])][r["arm"]] = r
    data = [(v[X]["base"], np.array([v[X][k] - v[Y][k] for k in FKEYS], float)) for v in by.values() if X in v and Y in v]
    return boot(data, FKEYS)


def fpp(c, k):
    if not c or k not in c:
        return "—"
    v = c[k]
    return f"{100 * v['pt']:+.1f} [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]"


def fst(c, k):
    if not c or k not in c:
        return "—"
    v = c[k]
    return f"{v['pt']:+.0f} [{v['lo']:+.0f}, {v['hi']:+.0f}]"


def cls(c, k):
    if not c or k not in c:
        return "—"
    v = c[k]
    return "+" if v["lo"] > 0 else ("−" if v["hi"] < 0 else "0")


def mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f0(x):
    return "—" if x is None else f"{x:.0f}"


def med(xs):
    xs = [x for x in xs if x is not None]
    return "—" if not xs else f"{float(np.median(xs)):+.1f}"


def comps_for(rnd, arms):
    spec = [("benefit", "retry"), ("benefit", "none"), ("benefit", "inverse"), ("benefit", "prior"), ("retry", "none"),
            ("inverse", "none"), ("prior", "none"), ("inverse", "retry"), ("benefit_noretry", "none"),
            ("benefit_noretry", "retry"), ("benefit", "benefit_noretry"), ("noloc", "benefit"), ("benefit", "home_all")]
    spec += {1: [("benefit", "risk"), ("full", "benefit"), ("benefit", "fixed"), ("benefit_m5", "benefit")],
             2: [("benefit_alt", "benefit"), ("benefit_alt", "prior")]}[rnd]
    spec += [(x, "retry") for x in sorted(arms) if x != "retry"]
    out = []
    for c in spec:
        if c[0] in arms and c[1] in arms and c not in out:
            out.append(c)
    return out


FCOMPS = (("benefit", "none"), ("benefit", "retry"), ("benefit", "inverse"), ("benefit", "prior"),
          ("retry", "none"), ("inverse", "none"), ("prior", "none"))
STRATA = (("全部链", lambda r: True), ("含 push/rotate 目标的链", lambda r: r["pr_chain"]),
          ("不含 push/rotate 目标的链", lambda r: not r["pr_chain"]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.json"); ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="analysis"); ap.add_argument("--max-tier", type=int, default=9)
    ap.add_argument("--cross", default=None); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--min-match", type=float, default=1.0)
    a = ap.parse_args()
    jobs = [j for j in json.loads((HERE / a.jobs).read_text())["jobs"] if j["tier"] <= a.max_tier]
    out = HERE / a.out; out.mkdir(exist_ok=True)
    rows, hand, chk, errors, missing, runs = build(jobs, a.res)
    with open(out / "records.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "mismatches.jsonl", "w") as f:
        for r in rows:
            if not r.get("match", True) or not r.get("dv_ok", True):
                f.write(json.dumps({k: r.get(k) for k in ("round", "orig_id", "arm", "worker", "orig_dir", "diffs", "dv_status",
                                                          "dv_info", "same_gpu", "scene0_eq", "tries_eq")}, ensure_ascii=False) + "\n")
    SUM = dict(n_rows=len(rows), errors=errors, missing=missing)
    Lr = [f"# Full-chain dual-criterion replay analysis ({a.jobs}, results directory {a.res}, tier ≤ {a.max_tier})\n"]
    tot, done = Counter(j["tier"] for j in jobs), Counter(j["tier"] for j in jobs if (HERE / a.res / f"{j['id']}.json").exists())
    Lr.append("Progress: " + ", ".join(f"tier {t} {done[t]}/{tot[t]}" for t in sorted(tot)) + f"; errors {len(errors)}; missing {len(missing)}\n")
    # 1 reproduction
    RA = [r for r in rows if "match" in r]
    m = sum(r["match"] for r in RA)
    SUM["match_rate"] = m / len(RA) if RA else None
    Lr.append(f"## 1. Mode-A reproduction check (attempt outcomes and steps of every subtask, repairs, full-chain steps, initial physics hash)\n\nOverall {m}/{len(RA)} = {f3(SUM['match_rate'])}\n")
    Lr.append("| Group | n | match rate | sha equal | scene0 equal | perturbation tries equal |\n|---|---:|---:|---:|---:|---:|")
    for name, kf in (("round×arm", lambda r: f"r{r['round']}·{r['arm']}"), ("original worker", lambda r: f"r{r['round']}·{r['worker']}·{r['orig_dir']}"),
                     ("same GPU", lambda r: f"same_gpu={r['same_gpu']}")):
        g = defaultdict(list)
        for r in RA:
            g[kf(r)].append(r)
        for k in sorted(g):
            rs = g[k]
            Lr.append(f"| {k} | {len(rs)} | {f3(mean(rs, 'match'))} | {f3(mean(rs, 'sha_eq'))} | {f3(mean(rs, 'scene0_eq'))} | {f3(mean(rs, 'tries_eq'))} |")
    SUM["mismatch_fields"] = dict(Counter(d.split(".")[-1] for r in RA for d in r["diffs"]))
    Lr.append(f"\nMismatched field counts: {SUM['mismatch_fields']}\n")
    # 2 divergence check and self-checks
    RB = [r for r in rows if "dv_status" in r]
    SUM["dv"] = {f"r{rd}·{arm}": dict(n=len(rs), ok=mean(rs, "dv_ok"), identical=float(np.mean([x["dv_status"] == "identical" for x in rs])))
                 for (rd, arm), rs in sorted({(r["round"], r["arm"]): [x for x in RB if x["round"] == r["round"] and x["arm"] == r["arm"]] for r in RB}.items())}
    ck = {k: (float(np.mean(v)) if v else None) for k, v in sorted(chk.items())}
    inst_ok = all(v is None or v == 1.0 for v in ck.values())
    SUM["instrument_checks"] = ck
    Lr.append("## 2. Mode-B divergence check and instrument self-checks (all expected to be 1.000)\n")
    Lr.append(f"{ck}{'' if inst_ok else '  **INSTRUMENT_CHECK_FAILED**'}\n")
    Lr.append("| round·arm | n | divergence check passed | fraction identical to mode A (no divergence) |\n|---|---:|---:|---:|")
    for k, v in SUM["dv"].items():
        Lr.append(f"| {k} | {v['n']} | {f3(v['ok'])} | {f3(v['identical'])} |")
    P = [r for r in rows if r.get("match") and r.get("dv_ok", True)]
    samples = [("P", P)] + ([("S", rows)] if SUM["match_rate"] is not None and SUM["match_rate"] < 1.0 else [])
    # 3 full-chain results per arm
    Lr.append("\n## 3. Full-chain results per arm (P). orig = original file; A = replay; (i) = rescored along the original trajectory with B_def / B_orc; (ii) = counterfactual rerun driven by B_def\n")
    Lr.append("| round | arm | n chains | orig S | A | (i) B_def | (i) B_orc | **(ii)** | completed A/(ii) | steps A/(ii) | utility A/(ii) | repairs/chain A/(ii) | stopped at A0B1 in (i) |\n"
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    arms_by_round = defaultdict(set)
    for r in rows:
        arms_by_round[r["round"]].add(r["arm"])
    SUM["arms"] = {}
    for rd in sorted(arms_by_round):
        for arm in sorted(arms_by_round[rd]):
            rs = [r for r in P if r["round"] == rd and r["arm"] == arm]
            if not rs:
                continue
            s = {k: mean(rs, k) for k in ("S_O", "S_A", "S_i", "S_io", "S_ii", "C_A", "C_ii", "N_A", "N_ii", "U_A", "U_ii", "rep_A", "rep_ii")}
            s["A0B1_stop"] = sum(r.get("A0B1_stop", 0) for r in rs); s["n"] = len(rs)
            SUM["arms"][f"r{rd}|{arm}"] = s
            Lr.append(f"| r{rd} | {arm} | {len(rs)} | {f3(s['S_O'])} | {f3(s['S_A'])} | {f3(s['S_i'])} | {f3(s['S_io'])} | **{f3(s['S_ii'])}** | "
                      f"{f3(s['C_A'])}/{f3(s['C_ii'])} | {f0(s['N_A'])}/{f0(s['N_ii'])} | {f3(s['U_A'])}/{f3(s['U_ii'])} | "
                      f"{f3(s['rep_A'])}/{f3(s['rep_ii'])} | {s['A0B1_stop']} |")
    # 4 paired differences
    SUM["paired"] = {}
    for sname, samp in samples:
        Lr.append(f"\n## 4{'a' if sname == 'P' else 'b'}. Full-chain paired differences ({sname}): ΔS in percentage points; L(ii)=ΔS(ii)−ΔS(A); class A→(ii): + interval>0, 0 interval contains 0, − interval<0\n")
        for rd in sorted(arms_by_round):
            for stn, stf in STRATA:
                Lr.append(f"\n### Round {rd} · {stn}\n")
                Lr.append("| comparison | n chains | n base | ΔS original criterion A | ΔS (i) | **ΔS (ii)** | L(ii) | class A→(ii) | Δsteps (ii) | Δutility (ii) (pp) |\n|---|---:|---:|---|---|---|---|---|---|---|")
                for X, Y in comps_for(rd, arms_by_round[rd]):
                    c = paired(samp, X, Y, lambda r, rd=rd, stf=stf: r["round"] == rd and stf(r))
                    SUM["paired"][f"{sname}|r{rd}|{stn}|{X}-{Y}"] = c
                    if c:
                        Lr.append(f"| {X}−{Y} | {c['n']} | {c['bases']} | {fpp(c, 'S_A')} | {fpp(c, 'S_i')} | **{fpp(c, 'S_ii')}** | {fpp(c, 'L_ii')} | "
                                  f"{cls(c, 'S_A')}→{cls(c, 'S_ii')} | {fst(c, 'N_ii')} | {fpp(c, 'U_ii')} |")
    # 5 pre-registered hypotheses
    Lr.append("\n## 5. Round-2 pre-registered hypotheses (A = replayed original criterion; (i) and (ii) are post-hoc rescorings under the handoff criterion, not pre-registered results)\n")
    Lr.append("| hypothesis | comparison | criterion | A | (i) | (ii) | verdict A / (ii) |\n|---|---|---|---|---|---|---|")
    r2 = lambda r: r["round"] == 2
    H = [("H1", "benefit−prior full chain", "lower bound>0", paired(P, "benefit", "prior", r2), ("S_A", "S_i", "S_ii"), lambda v: v["lo"] > 0),
         ("H2", "benefit−inverse full chain", "lower bound>0", paired(P, "benefit", "inverse", r2), ("S_A", "S_i", "S_ii"), lambda v: v["lo"] > 0),
         ("H3", "benefit−prior first handoff, new tasks without an inverse skill", "lower bound>0",
          paired_first(P, "benefit", "prior", lambda r: r2(r) and r["first_task"] in NOINV), ("f_A", "f_Bi", "f_Bii"), lambda v: v["lo"] > 0),
         ("H4", "noloc−benefit full chain", "lower bound>−5", paired(P, "noloc", "benefit", r2), ("S_A", "S_i", "S_ii"), lambda v: v["lo"] > -0.05),
         ("H5a", "benefit_noretry−none full chain", "lower bound>0", paired(P, "benefit_noretry", "none", r2), ("S_A", "S_i", "S_ii"), lambda v: v["lo"] > 0),
         ("H5b", "benefit_noretry−retry full chain", "point estimate≥0 (lower bound listed separately)", paired(P, "benefit_noretry", "retry", r2), ("S_A", "S_i", "S_ii"), lambda v: v["pt"] >= 0)]
    SUM["H"] = {}
    for name, desc, crit, c, ks, rule in H:
        SUM["H"][name] = c
        ver = "—" if not c else " / ".join("holds" if rule(c[k]) else "does not hold" for k in (ks[0], ks[2]))
        Lr.append(f"| {name} | {desc} | {crit} | {fpp(c, ks[0])} | {fpp(c, ks[1])} | {fpp(c, ks[2])} | {ver} |")
    # 6 first handoff
    Lr.append("\n## 6. First target handoff (all arms in the same state up to this point; P)\n\n| round | target task | arm | n | A | (i) B_def | (ii) | A1B0(i) | A0B1(i) |\n|---|---|---|---:|---:|---:|---:|---:|---:|")
    SUM["first"] = {}
    tasks_by_round = defaultdict(set)
    for r in P:
        if r.get("f_reached"):
            tasks_by_round[r["round"]].add(r["first_task"])
    for rd in sorted(tasks_by_round):
        for t in sorted(tasks_by_round[rd]):
            for arm in sorted(arms_by_round[rd]):
                rs = [r for r in P if r["round"] == rd and r["arm"] == arm and r["first_task"] == t and r.get("f_reached")]
                if not rs:
                    continue
                cells = Counter(f"A{r['f_A']}B{r['f_Bi']}" for r in rs)
                SUM["first"][f"r{rd}|{t}|{arm}"] = dict(n=len(rs), A=mean(rs, "f_A"), Bi=mean(rs, "f_Bi"), Bii=mean(rs, "f_Bii"), cells=dict(cells))
                Lr.append(f"| r{rd} | {t} | {arm} | {len(rs)} | {f3(mean(rs, 'f_A'))} | {f3(mean(rs, 'f_Bi'))} | {f3(mean(rs, 'f_Bii'))} | "
                          f"{cells.get('A1B0', 0)} | {cells.get('A0B1', 0)} |")
    Lr.append("\n### Paired differences at the first handoff (percentage points)\n\n| round | group | comparison | n | n base | A | (i) | **(ii)** | class A→(ii) |\n|---|---|---|---:|---:|---|---|---|---|")
    SUM["first_paired"] = {}
    for rd in sorted(tasks_by_round):
        groups = [(t, (lambda r, t=t: r["first_task"] == t)) for t in sorted(tasks_by_round[rd])]
        groups += [("push/rotate 合计", lambda r: r["first_task"] in PR), ("其他任务合计", lambda r: r["first_task"] not in PR),
                   ("全部", lambda r: True)]
        for gn, gf in groups:
            for X, Y in FCOMPS:
                if X not in arms_by_round[rd] or Y not in arms_by_round[rd]:
                    continue
                c = paired_first(P, X, Y, lambda r, rd=rd, gf=gf: r["round"] == rd and gf(r))
                SUM["first_paired"][f"r{rd}|{gn}|{X}-{Y}"] = c
                if c:
                    Lr.append(f"| r{rd} | {gn} | {X}−{Y} | {c['n']} | {c['bases']} | {fpp(c, 'f_A')} | {fpp(c, 'f_Bi')} | **{fpp(c, 'f_Bii')}** | "
                              f"{cls(c, 'f_A')}→{cls(c, 'f_Bii')} |")
    # 7 all target handoffs and predicate audit
    Pk = {(r["round"], r["orig_id"], r["arm"]) for r in P}
    HP = [h for h in hand if (h["round"], h["orig_id"], h["arm"]) in Pk]
    Lr.append("\n## 7. All target handoffs (descriptive; after the first one the arms are in different states) and predicate audit (P)\n\n"
              "| round | task | arm | mode A n | A | B_def | B_orc | mode B n | (ii) success |\n|---|---|---|---:|---:|---:|---:|---:|---:|")
    for rd in sorted(arms_by_round):
        for t in sorted({h["task"] for h in HP if h["round"] == rd}):
            for arm in sorted(arms_by_round[rd]):
                ha = [h for h in HP if h["round"] == rd and h["task"] == t and h["arm"] == arm and h["mode"] == "A"]
                hb = [h for h in HP if h["round"] == rd and h["task"] == t and h["arm"] == arm and h["mode"] == "B"]
                if ha or hb:
                    Lr.append(f"| r{rd} | {t} | {arm} | {len(ha)} | {f3(mean(ha, 'success'))} | {f3(mean(ha, 'Bdef'))} | {f3(mean(ha, 'Borc'))} | "
                              f"{len(hb)} | {f3(mean(hb, 'success'))} |")
    Lr.append("\n### Predicate audit (mode A, all arms pooled): criterion disagreement counts and precondition flags\n\n| task | n | A≠B_def | A≠B_orc | B_def≠B_orc | repair executed | precondition flags (handoff / start of last attempt) |\n|---|---:|---:|---:|---:|---:|---|")
    SUM["audit"] = {}
    for t in sorted({h["task"] for h in HP}):
        hs = [h for h in HP if h["task"] == t and h["mode"] == "A"]
        k = hs[0]["kind"] if hs else ""
        if k == "push_object":
            fl = f"非机器人接触为空：{sum(h['flags_h'].get('nonrobot_contacts') == 0 for h in hs)} / {sum(h['flags_s'].get('nonrobot_contacts') == 0 for h in hs)}"
        elif k == "lift_object":
            fl = f"不在源表面：{sum(h['flags_h'].get('on_source_surface') is False for h in hs)} / {sum(h['flags_s'].get('on_source_surface') is False for h in hs)}"
        elif k == "push_object_into":
            fl = f"有积木已在抽屉：{sum(any(h['flags_h'].get('in_drawer', {}).values()) for h in hs)} / {sum(any(h['flags_s'].get('in_drawer', {}).values()) for h in hs)}"
        elif k == "stack_objects":
            fl = f"已有积木相互接触：{sum(any(h['flags_h'].get('touching_block', {}).values()) for h in hs)} / {sum(any(h['flags_s'].get('touching_block', {}).values()) for h in hs)}"
        else:
            fl = f"累计偏航 |yaw|≥180°：{sum(abs(h['yaw_end'] or 0) >= 180 for h in hs)}"
        au = dict(n=len(hs), A_ne_Bdef=sum(h["success"] != h["Bdef"] for h in hs), A_ne_Borc=sum(h["success"] != h["Borc"] for h in hs),
                  Bdef_ne_Borc=sum(h["Bdef"] != h["Borc"] for h in hs), repaired=sum(h["repair_executed"] for h in hs), flags=fl)
        SUM["audit"][t] = au
        Lr.append(f"| {t} | {au['n']} | {au['A_ne_Bdef']} | {au['A_ne_Borc']} | {au['Bdef_ne_Borc']} | {au['repaired']} | {fl} |")
    # 8 mechanism
    Lr.append("\n## 8. Net displacement of push/rotate targets (mode A, all target handoffs; dx in cm for push, dyaw in degrees for rotate, relative to the handoff)\n\n"
              "| task | arm | n | rep p50 | end p50 | A1B0 n | A1B0 rep p50 | A1B0 end p50 |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for t in PR:
        for arm in sorted({h["arm"] for h in HP if h["task"] == t}):
            hs = [h for h in HP if h["task"] == t and h["arm"] == arm and h["mode"] == "A"]
            c = [h for h in hs if h["success"] and not h["Bdef"]]
            if hs:
                Lr.append(f"| {t} | {arm} | {len(hs)} | {med([h.get('d_rep') for h in hs])} | {med([h.get('d_end') for h in hs])} | "
                          f"{len(c)} | {med([h.get('d_rep') for h in c])} | {med([h.get('d_end') for h in c])} |")
    # 9 cross-run determinism
    if a.cross and (HERE / a.cross).is_dir():
        other, _, _ = load(jobs, a.cross)
        same = [summarize(runs[k][m]) == summarize(other[k][m]) for k in runs if k in other for m in ("A", "B")
                if runs[k].get(m) and other[k].get(m)]
        SUM["cross_run_identical"] = float(np.mean(same)) if same else None
        Lr.append(f"\n## 9. Cross-run determinism (fraction of records overlapping between {a.res} and {a.cross} that are bit-identical): {f3(SUM['cross_run_identical'])} (n={len(same)})\n")
    Lr.append("\n## Notes\n\n- When A matches the original results bit for bit, column A equals the originally reported numbers (2000 bootstrap resamples here, so intervals differ slightly from the original 5000-resample intervals).\n"
              "- (i) only answers how many of the chains that succeed under the original criterion remain successful under the handoff criterion along the original trajectory; attempts terminated early by the original criterion and A0B1 chain stops both bias (i) downward for repair arms, so it is not a deployment estimate.\n"
              "- (ii) estimates each arm's full-chain performance when the handoff criterion is used as the success detector; everything before the divergence point is checked bit for bit (Section 2); everything after it is a new execution.\n"
              "- Criterion B_def is defined in the fc_worker.py docstring; B_orc (same oracle, start := handoff) is a sensitivity variant.\n")
    (out / "summary.json").write_text(json.dumps(SUM, indent=1, ensure_ascii=False))
    (out / "report.md").write_text("\n".join(Lr) + "\n")
    print("\n".join(Lr))
    if a.gate:
        fail = []
        if errors or missing:
            fail.append(f"errors={len(errors)} missing={len(missing)}")
        if not RA or SUM["match_rate"] is None or SUM["match_rate"] < a.min_match:
            fail.append(f"match_rate={SUM['match_rate']} < {a.min_match}")
        if not inst_ok:
            fail.append("instrument checks")
        print(("GATE FAIL: " + "; ".join(fail)) if fail else f"GATE OK: matched {m}/{len(RA)}")
        sys.exit(3 if fail else 0)


if __name__ == "__main__":
    main()
