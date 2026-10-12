# -*- coding: utf-8 -*-
"""Figures fig_paired_cost and fig_paired_stopping: the benefit gate (ours) against one comparator on the same chains,
within one round, under the handoff-referenced criterion, estimate (ii). Comparators: retry once, task-prior repair,
inverse-skill rule. The same quantities under the original criterion A are written to the derived JSON (not plotted).

cost      Chains are split by the observed pair of final outcomes (both fail, comparator only, ours only, both succeed).
          Column width = share of chains; dot = one chain at its step difference (ours - comparator); shaded bar = class
          mean, so the signed shaded area is the class contribution to the mean step difference. Corner text: paired
          differences in chain success and steps with 95% cluster-bootstrap intervals over base configurations
          (2000 resamples, seed 17) from fullchain_stats.json.
stopping  Cell (x, y) counts chains on which the comparator completed x consecutive subtasks and ours completed y.
Data: data/derived/fullchain_per_chain.csv, data/derived/fullchain_stats.json.
Outputs: figures/fig_paired_{cost,stopping}.*, figures/fig_paired_derived.json. Classes describe observed executions,
not causal recovery types.
"""
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from figcommon import ROOT, FIG, TEXT_W, OURS, INK, MUTED, setup, save

setup()
DER = ROOT / "data/derived"
F = json.loads((DER / "fullchain_stats.json").read_text(encoding="utf-8"))
COMPS = [("retry", "Retry once"), ("prior", "Task-prior repair"), ("inverse", "Inverse-skill rule")]
CLASSES = ["both_fail", "cmp_only", "ours_only", "both_succeed"]
CLASS_LABEL = {"both_fail": "Both fail", "cmp_only": "Comparator only succeeds", "ours_only": "Ours only succeeds",
               "both_succeed": "Both succeed"}
CLASS_DOT = {"both_fail": "#9AA3AB", "cmp_only": "#0072B2", "ours_only": OURS, "both_succeed": "#3E4C59"}
CLASS_BAR = {"both_fail": "#DDE1E5", "cmp_only": "#B9D3E8", "ours_only": "#F3C3A3", "both_succeed": "#C3CAD1"}
CRIT = {"ii": "_ii", "A": "_A"}
FS = 6.6

ROWS = defaultdict(lambda: defaultdict(dict))
with open(DER / "fullchain_per_chain.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ROWS[int(r["round"])][r["chain"]][r["arm"]] = r
for ri in (1, 2):    # same records as the stored statistics
    for arm in ["benefit"] + [a for a, _ in COMPS]:
        a = F["arms"][f"r{ri}|{arm}"]
        rs = [v[arm] for v in ROWS[ri].values() if arm in v]
        assert len(rs) == a["n"], (ri, arm)
        for m in ("S_A", "S_ii", "N_A", "N_ii", "C_A", "C_ii"):
            assert abs(np.mean([float(r[m]) for r in rs]) - a[m]) < 1e-10, (ri, arm, m)


def jitter(chain):
    """Deterministic horizontal position in [0, 1) from the chain id, so the figure is reproducible."""
    return int(hashlib.sha256(("opus-rev|" + chain).encode()).hexdigest()[:8], 16) / 16 ** 8


def minus(s):
    return s.replace("-", "\u2212")


def ci_text(v, scale, fmt):
    f2 = fmt.replace("+", "")
    return minus(f"{scale * v[0]:{fmt}} [{scale * v[1]:{f2}}, {scale * v[2]:{f2}}]")


def panel(ri, arm, crit):
    sfx = CRIT[crit]
    chains = sorted(c for c, v in ROWS[ri].items() if "benefit" in v and arm in v)
    col = lambda a, m: np.array([int(float(ROWS[ri][c][a][m + sfx])) for c in chains])
    bs, bn, bc = col("benefit", "S"), col("benefit", "N"), col("benefit", "C")
    cs, cn, cc = col(arm, "S"), col(arm, "N"), col(arm, "C")
    d, n = bn - cn, len(chains)
    masks = {"both_fail": (bs == 0) & (cs == 0), "cmp_only": (bs == 0) & (cs == 1),
             "ours_only": (bs == 1) & (cs == 0), "both_succeed": (bs == 1) & (cs == 1)}
    M = np.zeros((6, 6), int)
    np.add.at(M, (bc, cc), 1)
    fu, ea = bc > cc, bc < cc
    P = F["paired"][f"r{ri}|all|benefit-{arm}"]
    rec = {"criterion": crit, "chains": n, "bases": len({ROWS[ri][c]["benefit"]["base"] for c in chains}),
           "classes": {k: {"n": int(m.sum()), "mean_step_diff": float(d[m].mean()) if m.any() else 0.0,
                           "median_step_diff": float(np.median(d[m])) if m.any() else 0.0,
                           "share_identical_steps": float(np.mean(d[m] == 0)) if m.any() else 0.0,
                           "contribution_steps_per_chain": float(d[m].sum() / n)} for k, m in masks.items()},
           "mean_step_diff": float(d.mean()), "success_diff": float(bs.mean() - cs.mean()),
           "paired_ci": {"S": P["S" + sfx], "N": P["N" + sfx]},
           "matrix_ours_rows_comparator_cols": M.tolist(),
           "ours_further": int(fu.sum()), "ours_further_and_completed_all": int((fu & (bc == 5)).sum()),
           "comparator_further": int(ea.sum()), "comparator_further_and_completed_all": int((ea & (cc == 5)).sum()),
           "same_stopping_point": int((bc == cc).sum()), "mean_completed_diff": float((bc - cc).mean())}
    assert P["n"] == n, (ri, arm, crit)
    assert abs(sum(v["contribution_steps_per_chain"] for v in rec["classes"].values()) - P["N" + sfx][0]) < 1e-9
    assert abs(rec["success_diff"] - P["S" + sfx][0]) < 1e-12
    return rec, d, masks, chains


derived = {"definitions": {
    "criterion": "ii: handoff-referenced criterion B_def drives attempts, retries and chain stops (plotted); "
                 "A: original start-relative criterion (published runs, reproduced by the replay)",
    "step_difference": "steps(ours) - steps(comparator) on the same chain; steps include unsuccessful chains",
    "class_contribution": "sum of step differences in the class / all chains in the round; contributions add up to the mean step difference",
    "matrix": "cell [y][x] = chains with comparator completed == x and ours completed == y (consecutive subtasks, 0-5)"},
    "panels": {}}
PLOT = {}
for ri in (1, 2):
    for arm, _ in COMPS:
        for crit in CRIT:
            rec, d, masks, chains = panel(ri, arm, crit)
            derived["panels"][f"round{ri}|{arm}|{crit}"] = rec
            if crit == "ii":
                PLOT[(ri, arm)] = (rec, d, masks, chains)

YMAX = max(np.abs(p[1]).max() for p in PLOT.values())
YLIM = 100 * np.ceil(YMAX / 100) + 60
YPAD = 420     # room below the lowest chain for the interval text
YT = [v for v in range(-3000, 3001, 500) if -YLIM - YPAD <= v <= YLIM]


def grid(fig, top, bottom, left=0.085, right=0.985, wspace=0.07, hspace=0.36):
    return fig.add_gridspec(2, 3, left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)


# ------------------------------------------------------------------ paired outcome-cost mosaic
fig = plt.figure(figsize=(TEXT_W, 4.2))
gs = grid(fig, top=0.885, bottom=0.095)
tag = iter("abcdef")
for ri in (1, 2):
    for ci, (arm, cname) in enumerate(COMPS):
        p, d, masks, chains = PLOT[(ri, arm)]
        ax = fig.add_subplot(gs[ri - 1, ci])
        n, x0 = p["chains"], 0.0
        ax.axhline(0, color="#8C949B", lw=0.6, zorder=2)
        for k in CLASSES:
            m = masks[k]; w = 100 * m.sum() / n; mean = p["classes"][k]["mean_step_diff"]
            ax.add_patch(Rectangle((x0, 0), w, mean, facecolor=CLASS_BAR[k], edgecolor="none", zorder=1))
            ax.plot([x0, x0 + w], [mean, mean], color=CLASS_DOT[k], lw=1.1, solid_capstyle="butt", zorder=4)
            xs = [x0 + w * (0.14 + 0.72 * jitter(c)) for c, mm in zip(chains, m) if mm]
            ax.scatter(xs, d[m], s=2.2, color=CLASS_DOT[k], lw=0, alpha=0.75, zorder=3)
            if x0 > 0:
                ax.axvline(x0, color="white", lw=1.0, zorder=2.5)
            lx, ha = {"cmp_only": (x0 + w - 0.9, "right"), "ours_only": (x0 + 0.9, "left")}.get(k, (x0 + w / 2, "center"))
            hot = k in ("cmp_only", "ours_only")
            ax.text(lx, YLIM - 60, f"{m.sum()}", ha=ha, va="top", fontsize=FS, color=CLASS_DOT[k] if hot else MUTED,
                    weight="bold" if hot else "normal")
            x0 += w
        for label, v, scale, fmt, suffix, ypos in (("\u0394 success ", p["paired_ci"]["S"], 100, "+.1f", " pp", 0.108),
                                                   ("\u0394 steps ", p["paired_ci"]["N"], 1, "+.0f", "", 0.035)):
            ax.text(0.985, ypos, label + ci_text(v, scale, fmt) + suffix, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color=INK if v[1] > 0 or v[2] < 0 else MUTED, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_xlim(0, 100); ax.set_ylim(-YLIM - YPAD, YLIM)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(YT); ax.set_yticklabels([minus(f"{v:+d}") if v else "0" for v in YT])
        if ci:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("Steps, ours \u2212 comparator")
        if ri == 2:
            ax.set_xlabel("Cumulative share of chains (%)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"({next(tag)})  Round {ri} vs {cname.lower()}", loc="left", fontsize=7.5, pad=4)
lx = 0.085
for k in CLASSES:
    fig.add_artist(Rectangle((lx, 0.952), 0.012, 0.02, transform=fig.transFigure, facecolor=CLASS_BAR[k],
                             edgecolor=CLASS_DOT[k], lw=0.8))
    fig.text(lx + 0.017, 0.962, CLASS_LABEL[k], fontsize=6.9, color=INK, va="center")
    lx += 0.03 + 0.0062 * len(CLASS_LABEL[k])
fig.text(0.985, 0.962, "handoff criterion (ii);  dot = one chain;  bar = class mean", fontsize=FS, color=MUTED,
         ha="right", va="center")
print(save(fig, "fig_paired_cost"))


# ------------------------------------------------------------------ paired stopping-point matrix
def blend(c, a):
    r, g, b = to_rgb(c)
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


mats = {k: np.array(v[0]["matrix_ours_rows_comparator_cols"]) for k, v in PLOT.items()}
vmax = max((M - np.diag(np.diag(M))).max() for M in mats.values())
dmax = max(np.diag(M).max() for M in mats.values())
fig = plt.figure(figsize=(TEXT_W, 4.45))
gs = grid(fig, top=0.955, bottom=0.075, left=0.07, right=0.90, wspace=0.52)
tag = iter("abcdef")
for ri in (1, 2):
    for ci, (arm, cname) in enumerate(COMPS):
        p, M = PLOT[(ri, arm)][0], mats[(ri, arm)]
        ax = fig.add_subplot(gs[ri - 1, ci])
        for yy in range(6):
            for xx in range(6):
                v = M[yy, xx]
                if yy == xx:
                    face, txt = blend("#5B6670", 0.10 + 0.5 * v / max(dmax, 1)), INK
                elif v == 0:
                    face, txt = "white", None
                else:
                    a = 0.18 + 0.82 * (v / vmax) ** 0.7
                    face, txt = blend(OURS if yy > xx else "#0072B2", a), ("white" if a > 0.6 else INK)
                ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=0.8))
                if v:
                    ax.text(xx, yy, str(v), ha="center", va="center", fontsize=6.5, color=txt,
                            weight="bold" if (yy == 5 and xx < 5) else "normal")
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5); ax.set_aspect("equal")
        ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        if ci == 0:
            ax.set_ylabel("Ours: subtasks completed")
        ax.set_xlabel(f"{cname}: subtasks completed", labelpad=2)
        ax.set_title(f"({next(tag)})  Round {ri} vs {cname.lower()}", loc="left", fontsize=7.5, pad=4)
        ax.text(1.04, 0.98, f"Ours further\n{p['ours_further']} chains\n{p['ours_further_and_completed_all']} to the end",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS, color=OURS, linespacing=1.2)
        ax.text(1.04, 0.02, f"Comparator\nfurther\n{p['comparator_further']} chains\n"
                            f"{p['comparator_further_and_completed_all']} to the end",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=FS, color="#0072B2", linespacing=1.2)
print(save(fig, "fig_paired_stopping"))

# ------------------------------------------------------------------ derived data and provenance
derived["y_limit_steps"] = float(YLIM)
derived["sources_sha256"] = {f: hashlib.sha256((DER / f).read_bytes()).hexdigest()
                             for f in ("fullchain_stats.json", "fullchain_per_chain.csv")}
FIG.mkdir(exist_ok=True)
(FIG / "fig_paired_derived.json").write_text(json.dumps(derived, indent=2), encoding="utf8")
for k, p in derived["panels"].items():
    print(k, {c: (v["n"], round(v["contribution_steps_per_chain"], 1)) for c, v in p["classes"].items()},
          "further", p["ours_further"], p["ours_further_and_completed_all"],
          "| cmp further", p["comparator_further"], p["comparator_further_and_completed_all"])
