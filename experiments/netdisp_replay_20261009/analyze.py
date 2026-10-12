# -*- coding: utf-8 -*-
"""Net-displacement replay analysis.
Inputs: jobs*.json (including summaries of the original results); replay results under results/ and results_retry/.
Outputs (--out directory): records.jsonl (one line per (chain, arm)), mismatches.jsonl, summary.json, report.md.
P = records that reproduce and reach the target (main analysis); S = all replayed records that reach the target (rerun sensitivity); O = full range of original results ((a) only).
Criteria: A = original experimental criterion (each attempt referenced to its own start; final outcome); B = at the end of the target subtask, relative to the handoff (before repair);
C = relative to the post-repair state; Bany = B satisfied at any step of an attempt (truncated by A's early termination; reference only).
Paired difference X − Y: cluster bootstrap over bases (2000 resamples, seed 17, seed reset for every cell; Da, Db and L=Db−Da use the same resamples).
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan_jobs import summarize  # noqa: E402

PUSH = ("push_red_block_right", "push_blue_block_right", "push_pink_block_right")
ROT = ("rotate_pink_block_right", "rotate_red_block_left")
GROUPS = [(t, (t,)) for t in PUSH + ROT] + [("push_all", PUSH), ("rotate_all", ROT), ("all5", PUSH + ROT)]
ARMS = ("none", "inverse", "prior", "benefit", "retry")
COMPS = (("inverse", "none"), ("benefit", "none"), ("prior", "none"),
         ("retry", "none"), ("benefit", "retry"), ("inverse", "retry"), ("benefit", "inverse"))
NBOOT, SEED = 2000, 17
EDGES = {"dx": [-15, -10, -5, -2, 2, 5, 10, 15], "dyaw": [-120, -90, -60, -30, 30, 60, 90, 120]}


def compare(o, s):
    diffs = []
    if o["prefix"] != s["prefix"]:
        diffs.append("prefix")
    ot, st = o["target"], s["target"]
    if (ot is None) != (st is None):
        diffs.append("reached")
    elif ot is not None:
        diffs += [k for k in ("repair", "repair_steps", "attempts", "success") if ot.get(k) != st.get(k)]
    return diffs


def _vec(e, key, n):
    v = e.get(key)
    if v is not None and len(v) == n:
        return v
    return next((x for k, x in e.items() if isinstance(x, list) and len(x) == n), None)


def rel_yaw(q0, q1):
    from scipy.spatial.transform import Rotation as R
    return float((R.from_quat(q1) * R.from_quat(q0).inv()).as_euler("xyz", degrees=True)[2])


def block_key(task, p):
    color = task.split("_")[1]
    if f"block_{color}" in p:
        return f"block_{color}"
    c = [n for n in p if color in n]
    return c[0] if c else None


def delta(p0, p1, k):
    if k is None or k not in p0 or k not in p1:
        return None
    pa, pb = _vec(p0[k], "current_pos", 3), _vec(p1[k], "current_pos", 3)
    if pa is None or pb is None:
        return None
    d = [100.0 * (y - x) for x, y in zip(pa, pb)]
    qa, qb = _vec(p0[k], "current_orn", 4), _vec(p1[k], "current_orn", 4)
    return dict(dx=d[0], dy=d[1], dz=d[2], dyaw=rel_yaw(qa, qb) if qa and qb else None)


def outcomes(pr, tgt, task):
    atts = pr["attempts"]
    last = atts[-1] if atts else None
    h, r = pr["handoff"], pr["post_repair"]
    e = last["end_poses"] if last else r
    k = block_key(task, h)
    others = []
    for n in h:
        if n != k:
            dd = delta(h, e, n)
            if dd:
                others.append(float(np.hypot(dd["dx"], dd["dy"])))
    return dict(A=bool(tgt["success"]), A0=bool(atts and atts[0]["success"]), B=bool(last and last["b_end"]),
                C=bool(last and last["c_end"]), Bany=any(x["b_any"] for x in atts), n_attempts=len(atts),
                repair=tgt.get("repair"), repair_executed=bool(pr.get("repair_executed")),
                chk_a=all(x["a_end"] == x["success"] for x in atts),
                chk_c0=(not atts) or atts[0]["c_end"] == atts[0]["a_end"],
                chk_steps=all(x["steps_seen"] == x["steps"] for x in atts),
                block=k, d_rep=delta(h, r, k), d_end=delta(h, e, k), d_att=delta(r, e, k),
                other_max_cm=max(others) if others else None)


def build(jobs):
    res, errors = {}, []
    for sub in ("results", "results_retry"):
        d = HERE / sub
        if not d.is_dir():
            continue
        for j in jobs:
            p = d / f"{j['id']}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            if r.get("error"):
                errors.append(dict(id=j["id"], sub=sub, error=r["error"][-800:])); continue
            for arm, v in r["arms"].items():
                res.setdefault(j["id"], {})[arm] = (v, dict(sub=sub, scene0=r["scene0"], tries=r["perturb_tries"], gpu=r.get("gpu")))
    records, missing = [], []
    retry_ran = (HERE / "results_retry").is_dir()
    for j in jobs:
        got = res.get(j["id"], {})
        for arm in j["arms"]:
            if arm == "retry" and not retry_ran:
                continue
            if arm not in got:
                missing.append(f"{j['id']}:{arm}"); continue
            v, m = got[arm]
            o, src = j["orig"][arm], j["orig"][arm]["source"]
            s = summarize(v, j["first_pos"])
            diffs = compare(o, s)
            rec = dict(job_id=j["id"], round=j["round"], orig_id=j["orig_id"], base=j["base"], task=j["first_task"],
                       first_pos=j["first_pos"], arm=arm, match=not diffs, diffs=diffs,
                       sha_eq=o["init_physics_sha"] == v["init_physics_sha"],
                       scene0_eq=bool(src.get("scene0") is not None and np.allclose(src["scene0"], m["scene0"], atol=1e-9, rtol=0)),
                       tries_eq=src.get("perturb_tries") == m["tries"], orig_dir=src.get("dir"),
                       orig_gpu=src.get("gpu"), gpu=m["gpu"], same_gpu=src.get("gpu") == m["gpu"],
                       orig_reached=o["target"] is not None, reached=s["target"] is not None,
                       orig_success=(o["target"] or {}).get("success"), orig_repair=(o["target"] or {}).get("repair"),
                       replay_repair=(s["target"] or {}).get("repair"),
                       orig_steps_to_end=o["steps_to_end"], steps_to_end=s["steps_to_end"])
            if o["target"] and s["target"]:
                rec["ustar_eq"] = o["target"].get("ustar") == s["target"].get("ustar")
            if s["target"] and v.get("probe"):
                rec.update(outcomes(v["probe"], s["target"], j["first_task"]))
            records.append(rec)
    return records, errors, missing


def rate(rs, key):
    return float(np.mean([bool(r[key]) for r in rs])) if rs else None


def cluster_boot(rows):
    """rows: (base, da, db); returns the chain-level mean paired differences and 95% percentile intervals from a cluster bootstrap over bases."""
    if not rows:
        return None
    bases = sorted({r[0] for r in rows}); bi = {b: i for i, b in enumerate(bases)}
    Ssum, N = np.zeros((len(bases), 2)), np.zeros(len(bases))
    for b, da, db in rows:
        Ssum[bi[b]] += (da, db); N[bi[b]] += 1
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(bases), size=(NBOOT, len(bases)))
    est = Ssum[draw].sum(1) / N[draw].sum(1)[:, None]
    pt = Ssum.sum(0) / N.sum()
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(n=int(N.sum()), bases=len(bases), Da=float(pt[0]), Da_ci=ci(est[:, 0]), Db=float(pt[1]),
                Db_ci=ci(est[:, 1]), L=float(pt[1] - pt[0]), L_ci=ci(est[:, 1] - est[:, 0]),
                ratio=float(pt[1] / pt[0]) if pt[0] != 0 else None)


def paired(recs, X, Y, tasks, bkey="B"):
    by = defaultdict(dict)
    for r in recs:
        if r["task"] in tasks and r["arm"] in (X, Y):
            by[r["job_id"]][r["arm"]] = r
    rows = [(v[X]["base"], int(v[X]["A"]) - int(v[Y]["A"]), int(v[X][bkey]) - int(v[Y][bkey]))
            for v in by.values() if X in v and Y in v]
    return cluster_boot(rows)


def classify(r):
    if r is None:
        return "no data"
    if r["Da_ci"][0] <= 0:
        return "gain not significant under the original criterion"
    if r["Db_ci"][0] <= 0:
        return "not significant under the absolute criterion"
    if r["L_ci"][1] < 0:
        return "partly retained"
    if r["L_ci"][0] > 0:
        return "larger under the absolute criterion"
    return "largely retained"


def quant(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    p = np.percentile(xs, [5, 25, 50, 75, 95])
    return dict(n=len(xs), mean=float(np.mean(xs)), p5=float(p[0]), p25=float(p[1]), p50=float(p[2]), p75=float(p[3]), p95=float(p[4]))


def hist(xs, edges):
    xs = [x for x in xs if x is not None]
    idx = np.searchsorted(edges, xs, side="right")
    c = Counter(int(i) for i in idx)
    labels = [f"<{edges[0]}"] + [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)] + [f">={edges[-1]}"]
    return {labels[i]: c.get(i, 0) for i in range(len(labels))}


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fci(r, k):
    return "—" if r is None else f"{r[k]:+.3f} [{r[k + '_ci'][0]:+.3f}, {r[k + '_ci'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.json"); ap.add_argument("--out", default="analysis")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    jobs = json.loads((HERE / a.jobs).read_text())["jobs"]
    out = HERE / a.out; out.mkdir(exist_ok=True)
    records, errors, missing = build(jobs)
    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out / "mismatches.jsonl", "w") as f:
        for r in records:
            if not r["match"]:
                f.write(json.dumps({k: r[k] for k in ("job_id", "arm", "task", "diffs", "sha_eq", "scene0_eq", "tries_eq",
                                                      "same_gpu", "orig_dir", "orig_success", "orig_repair", "replay_repair",
                                                      "orig_steps_to_end", "steps_to_end")}) + "\n")
    L, SUM = [], dict(n_records=len(records), errors=errors, missing=missing)
    L.append(f"# Net-displacement replay analysis ({a.jobs})\n")
    # 1 reproduction
    m = sum(r["match"] for r in records)
    SUM["match_rate"] = m / len(records) if records else None
    L.append(f"## 1. Reproduction check\n\nOverall match {m}/{len(records)} = {f3(SUM['match_rate'])}; errors {len(errors)}; missing {len(missing)}\n")
    L.append("| group | n | match rate | sha equal | scene0 equal | target reached (replay) |\n|---|---:|---:|---:|---:|---:|")
    rep = {}
    keyfuns = [("round×arm", lambda r: f"r{r['round']}·{r['arm']}"), ("orig_dir", lambda r: f"r{r['round']}·{r['orig_dir']}"),
               ("same_gpu", lambda r: f"same_gpu={r['same_gpu']}"), ("task", lambda r: r["task"])]
    for name, kf in keyfuns:
        g = defaultdict(list)
        for r in records:
            g[kf(r)].append(r)
        for k in sorted(g):
            rs = g[k]
            rep[f"{name}:{k}"] = dict(n=len(rs), match=rate(rs, "match"))
            L.append(f"| {k} | {len(rs)} | {f3(rate(rs, 'match'))} | {f3(rate(rs, 'sha_eq'))} | {f3(rate(rs, 'scene0_eq'))} | {f3(rate(rs, 'reached'))} |")
    SUM["repro"] = rep
    SUM["mismatch_fields"] = dict(Counter(d for r in records for d in r["diffs"]))
    L.append(f"\nMismatched field counts: {SUM['mismatch_fields']}\n")
    # 2 instrument self-checks
    RA = [r for r in records if r["reached"] and "A" in r]
    chk = dict(a_recomputed=rate(RA, "chk_a"), c0_equals_a=rate(RA, "chk_c0"), steps_seen=rate(RA, "chk_steps"),
               none_A_equals_B=(float(np.mean([r["A"] == r["B"] for r in RA if r["arm"] == "none"]))
                                if any(r["arm"] == "none" for r in RA) else None))
    SUM["instrument_checks"] = chk
    inst_ok = all(v is None or v == 1.0 for v in chk.values())
    L.append(f"## 2. Instrument self-checks (all expected to be 1.000)\n\n{chk}{'' if inst_ok else '  **INSTRUMENT_CHECK_FAILED**'}\n")
    P = [r for r in RA if r["match"]]; S = RA
    O = [dict(job_id=j["id"], base=j["base"], task=j["first_task"], arm=arm, A=o["target"]["success"], B=o["target"]["success"])
         for j in jobs for arm, o in j["orig"].items() if o["target"] is not None]
    # 3 success rates
    L.append("## 3. Success rate per task × arm (P: matched and target reached)\n\n| task | arm | n | A | B | Bany | C | A1B1 | A1B0 | A0B1 | A0B0 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    succ = {}
    for samp_name, samp in (("P", P), ("S", S)):
        for t in PUSH + ROT:
            for arm in ARMS:
                rs = [r for r in samp if r["task"] == t and r["arm"] == arm]
                if not rs:
                    continue
                cells = Counter(f"A{int(r['A'])}B{int(r['B'])}" for r in rs)
                succ[f"{samp_name}|{t}|{arm}"] = dict(n=len(rs), A=rate(rs, "A"), B=rate(rs, "B"), Bany=rate(rs, "Bany"), C=rate(rs, "C"), cells=dict(cells))
                if samp_name == "P":
                    L.append(f"| {t} | {arm} | {len(rs)} | {f3(rate(rs, 'A'))} | {f3(rate(rs, 'B'))} | {f3(rate(rs, 'Bany'))} | {f3(rate(rs, 'C'))} | "
                             + " | ".join(str(cells.get(c, 0)) for c in ("A1B1", "A1B0", "A0B1", "A0B0")) + " |")
    SUM["success"] = succ
    # 4 paired differences
    pairs = {}
    for samp_name, samp in (("P", P), ("S", S), ("O", O)):
        for gname, tasks in GROUPS:
            for X, Y in COMPS:
                r = paired(samp, X, Y, tasks)
                if r:
                    r["class"] = classify(r) if samp_name != "O" else None
                    pairs[f"{samp_name}|{gname}|{X}-{Y}"] = r
    SUM["paired"] = pairs
    for samp_name, title in (("P", "main analysis P"), ("S", "rerun sensitivity S (regardless of match)")):
        L.append(f"\n## 4{'a' if samp_name == 'P' else 'b'}. Paired differences ({title}): Da = original criterion, Db = handoff criterion, L = Db−Da; brackets give 95% cluster-bootstrap intervals over bases\n")
        L.append("| group | comparison | n chains | n base | Da | Db | L | Db/Da | class | O (full-range original results) Da |\n|---|---|---:|---:|---|---|---|---:|---|---|")
        for gname, _ in GROUPS:
            for X, Y in COMPS:
                r = pairs.get(f"{samp_name}|{gname}|{X}-{Y}")
                if r is None:
                    continue
                o = pairs.get(f"O|{gname}|{X}-{Y}")
                L.append(f"| {gname} | {X}−{Y} | {r['n']} | {r['bases']} | {fci(r, 'Da')} | {fci(r, 'Db')} | {fci(r, 'L')} | "
                         f"{f3(r['ratio'])} | {r['class']} | {fci(o, 'Da')} |")
    # 5 net displacement
    L.append("\n## 5. Net displacement of the relevant block relative to the handoff (P; dx in cm for push, dyaw in degrees for rotate; rep = after repair, end = end of the target subtask, att = end relative to post-repair)\n")
    L.append("| task | arm | n | rep p25/p50/p75 | end p5/p25/p50/p75/p95 | att p50 | max translation of other blocks p50/p95 cm |\n|---|---|---:|---|---|---:|---|")
    disp = {}
    for t in PUSH + ROT:
        c = "dx" if t.startswith("push") else "dyaw"
        for arm in ARMS:
            rs = [r for r in P if r["task"] == t and r["arm"] == arm]
            if not rs:
                continue
            get = lambda r, k: (r[k] or {}).get(c)
            qr, qe, qa = quant([get(r, "d_rep") for r in rs]), quant([get(r, "d_end") for r in rs]), quant([get(r, "d_att") for r in rs])
            qo = quant([r["other_max_cm"] for r in rs])
            cellq = {}
            for cell in ("A1B1", "A1B0", "A0B1", "A0B0"):
                cs = [r for r in rs if f"A{int(r['A'])}B{int(r['B'])}" == cell]
                cellq[cell] = dict(n=len(cs), rep=quant([get(r, "d_rep") for r in cs]), end=quant([get(r, "d_end") for r in cs]))
            disp[f"{t}|{arm}"] = dict(coord=c, rep=qr, end=qe, att=qa, other=qo, by_cell=cellq,
                                      hist_end=hist([get(r, "d_end") for r in rs], EDGES[c]),
                                      hist_rep=hist([get(r, "d_rep") for r in rs], EDGES[c]),
                                      dy_end=quant([(r["d_end"] or {}).get("dy") for r in rs]))
            g3 = lambda q, ks: "—" if q is None else "/".join(f"{q[k]:+.1f}" for k in ks)
            L.append(f"| {t} | {arm} | {len(rs)} | {g3(qr, ('p25', 'p50', 'p75'))} | {g3(qe, ('p5', 'p25', 'p50', 'p75', 'p95'))} | "
                     f"{g3(qa, ('p50',))} | {'—' if qo is None else f'{qo[chr(112)+chr(53)+chr(48)]:.1f}/{qo[chr(112)+chr(57)+chr(53)]:.1f}'} |")
    L.append("\n### Median net displacement by (A,B) cell (P)\n\n| task | arm | cell | n | rep p50 | end p50 |\n|---|---|---|---:|---:|---:|")
    for k, v in disp.items():
        for cell, cv in v["by_cell"].items():
            if cv["n"]:
                L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {cell} | {cv['n']} | {f3(cv['rep']['p50']) if cv['rep'] else '—'} | {f3(cv['end']['p50']) if cv['end'] else '—'} |")
    L.append("\n### Histogram of end net displacement (P)\n")
    for k, v in disp.items():
        L.append(f"- {k} [{v['coord']}]: {v['hist_end']}")
    SUM["displacement"] = disp
    (out / "summary.json").write_text(json.dumps(SUM, indent=1))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.gate:
        if errors or missing:
            print(f"GATE FAIL: errors={len(errors)} missing={len(missing)}"); sys.exit(4)
        if not records or m == 0:
            print("GATE FAIL: 0 matched records"); sys.exit(3)
        if not inst_ok:
            print("GATE FAIL: instrument checks"); sys.exit(5)
        print(f"GATE OK: matched {m}/{len(records)}")


if __name__ == "__main__":
    main()
