# -*- coding: utf-8 -*-
"""Read-only export of per-handoff displacements of push and rotation targets from the full-chain replay.

Run on the compute server inside $HANDOFF_RUN_ROOT/fullchain_replay_20261010/:
    python -B server_export_displacement.py            # jobs.json, results/, tiers <= 3
Reads jobs.json and results/*.json through analyze.py (imported, unchanged) and writes ONLY
analysis/handoff_displacement.json. Definitions as in analyze.handoffs (report section 8): push targets dx of the target
block along x in cm; rotation targets dyaw in degrees (z of the xyz Euler angles of q_end * q_handoff^-1); both relative
to the handoff; d_rep after the repair (= handoff without repair), d_end at the end of the last attempt.
Both modes are exported: A (original criterion, bit-exact replay) and B (B_def-driven re-execution, estimate (ii)).
Only chains of the main sample P (mode A reproduced and divergence check passed) are exported.
Check: every mode-A value equals analyze.handoffs to 1e-9; otherwise nothing is written (exit code 2).
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402

KINDS = {"push_object": "push", "rotate_object": "rotate"}
COLS = ("round", "orig_id", "base", "arm", "mode", "task", "kind", "pos", "is_first", "repair", "repair_executed", "n_att",
        "success", "A_end", "Bdef", "Borc", "d_rep", "d_end", "yaw_cum")


def disp(pr, last):
    ob, kind = pr.get("obj"), pr.get("kind")
    ph = (pr.get("handoff") or {}).get(ob) if ob else None
    prr = (pr.get("post_repair") or {}).get(ob) if ob else None
    if not ph or prr is None:
        return None, None
    pe = ((last.get("end_poses") or {}).get(ob)) if last else prr
    if pe is None:
        return None, None
    if kind == "push_object" and "current_pos" in ph:
        return 100 * (prr["current_pos"][0] - ph["current_pos"][0]), 100 * (pe["current_pos"][0] - ph["current_pos"][0])
    if kind == "rotate_object" and "current_orn" in ph:
        return analyze._yaw(ph["current_orn"], prr["current_orn"]), analyze._yaw(ph["current_orn"], pe["current_orn"])
    return None, None


def same(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-9)


def r4(x):
    return None if x is None else round(float(x), 4)


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.median(v)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.json"); ap.add_argument("--res", default="results")
    ap.add_argument("--max-tier", type=int, default=3)
    a = ap.parse_args()
    out_path = HERE / "analysis" / "handoff_displacement.json"
    if not out_path.parent.is_dir():
        sys.exit(f"{out_path.parent} does not exist; run from the replay directory")
    jobs = [j for j in json.loads((HERE / a.jobs).read_text())["jobs"] if j["tier"] <= a.max_tier]
    rows, hand, chk, errors, missing, runs = analyze.build(jobs, a.res)
    if errors or missing:
        print(f"errors={len(errors)} missing={len(missing)}; nothing written"); sys.exit(2)
    P = {(r["round"], r["orig_id"], r["arm"]) for r in rows if r.get("match") and r.get("dv_ok", True)}
    ref = {(h["round"], h["orig_id"], h["arm"], h["pos"]): h for h in hand if h["mode"] == "A"}
    out, checked, bad = [], 0, []
    for (rnd, oid, arm), R in sorted(runs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        if (rnd, oid, arm) not in P:
            continue
        j = R["job"]
        for mode in ("A", "B"):
            run = R.get(mode)
            if not run:
                continue
            tmap = {t["pos"]: t for t in run["tasks"]}
            for pr in run["probes"]:
                if pr.get("kind") not in KINDS:
                    continue
                t, at = tmap[pr["pos"]], pr["attempts"]
                last = at[-1] if at else None
                d_rep, d_end = disp(pr, last)
                rec = [rnd, oid, j["base"], arm, mode, pr["task"], KINDS[pr["kind"]], pr["pos"], pr["pos"] == j["first_pos"],
                       t.get("repair"), bool(pr["repair_executed"]), len(at), bool(t["success"]),
                       bool(last and last["a_end"]), bool(last and last["bdef_end"]), bool(last and last["borc_end"]),
                       r4(d_rep), r4(d_end), r4((last or {}).get("yaw_cum"))]
                if mode == "A":
                    checked += 1
                    h = ref.get((rnd, oid, arm, pr["pos"]))
                    if h is None or not same(h.get("d_rep"), d_rep) or not same(h.get("d_end"), d_end) \
                            or h["success"] != rec[12] or h["Bdef"] != rec[14] or h["Borc"] != rec[15]:
                        bad.append((rnd, oid, arm, pr["pos"]))
                out.append(rec)
    if bad:
        print(f"CHECK FAILED: {len(bad)} of {checked} mode-A handoffs differ from analyze.handoffs, e.g. {bad[:5]}; nothing written")
        sys.exit(2)
    g = defaultdict(list)
    for r in out:
        if r[4] == "A":
            g[(r[5], r[3])].append(r)
    sec8 = {}
    for (task, arm), hs in sorted(g.items()):
        c = [r for r in hs if r[12] and not r[14]]
        sec8[f"{task}|{arm}"] = dict(n=len(hs), rep_p50=med([r[16] for r in hs]), end_p50=med([r[17] for r in hs]),
                                     A1B0_n=len(c), A1B0_rep_p50=med([r[16] for r in c]), A1B0_end_p50=med([r[17] for r in c]))
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script=Path(__file__).name, max_tier=a.max_tier,
                jobs_sha256=hashlib.sha256((HERE / a.jobs).read_bytes()).hexdigest(), n_jobs=len(jobs), n_chain_arms_P=len(P),
                n_rows=len(out), checked_mode_A=checked, units=dict(push="cm along x", rotate="deg yaw"),
                section8_medians=sec8)
    out_path.write_text(json.dumps(dict(meta=meta, columns=COLS, rows=out), ensure_ascii=True), encoding="utf-8")
    print(f"checked {checked} mode-A handoffs against analyze.handoffs: all equal; wrote {len(out)} rows to {out_path}")
    for k in ("push_red_block_right|benefit", "push_blue_block_right|benefit", "rotate_pink_block_right|benefit"):
        print(k, sec8.get(k))


if __name__ == "__main__":
    main()
