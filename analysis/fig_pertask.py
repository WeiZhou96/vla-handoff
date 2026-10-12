# -*- coding: utf-8 -*-
"""Figure (rules, a): success at the first target handoff of each chain, per target task.

No arm may intervene before the first target handoff, so all arms face the identical state there.
Rows: the five round-1 tasks (round-1 chains), then the six tasks added in round 2 (round-2 chains).
Left panel:
  - no intervention (identical under both criteria);
  - inverse-skill rule (ii);
  - task-prior repair (ii);
  - ours under the original criterion A (hollow) and under the handoff-referenced criterion (ii) (filled; arrow A -> (ii)).
Right panel: ours minus no intervention, under A (light) and under (ii) (vermillion). Each shows a point estimate and a
95% cluster-bootstrap interval over base configurations (2000 resamples). A filled marker means the interval excludes
0; a hollow marker means it includes 0 (keyed in the legend).
The three-row legend spans the content block, from the left edge of the task labels to the right edge of the right axes.
Data: data/derived/fullchain_stats.json (first, first_paired).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figcommon import ROOT, COL_W, OURS, GREY, LIGHT, MUTED, C_RULE, C_PRIOR, TASK_NAME, setup, save, style_ax

setup()
F = json.loads((ROOT / "data/derived/fullchain_stats.json").read_text(encoding="utf-8"))
FT, FP = F["first"], F["first_paired"]
C_A = "#E8A57C"     # light tint of the ours colour: original criterion A
FS = 6.6
INV = {"push_red_block_right", "push_blue_block_right", "push_pink_block_right", "rotate_pink_block_right", "rotate_red_block_left"}
old = ["push_red_block_right", "push_blue_block_right", "rotate_pink_block_right", "lift_red_block_slider", "push_into_drawer"]
new = ["push_pink_block_right", "rotate_red_block_left", "lift_blue_block_slider", "lift_pink_block_slider",
       "lift_pink_block_drawer", "stack_block"]
rows = [(1, t) for t in old] + [(2, t) for t in new]

H = 3.75                          # inches
AX_B, AX_H = 0.414, 2.66          # axes bottom and height in inches (unchanged from the previous version)
BOTF, TOPF = AX_B / H, (AX_B + AX_H) / H
fig = plt.figure(figsize=(COL_W, H))
gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.15], wspace=0.08, left=0.385, right=0.975, top=TOPF, bottom=BOTF)
ax, dx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
ys, labels, lows, his, xmin = [], [], [], [], 1.0
y = 0.0
for i, (rd, t) in enumerate(rows):
    if i == len(old):
        y -= 0.9
    ys.append(y)
    g = {a: FT[f"r{rd}|{t}|{a}"] for a in ("none", "inverse", "prior", "benefit")}
    assert abs(g["none"]["A"] - g["none"]["Bii"]) < 1e-12 and len({v["n"] for v in g.values()}) == 1, (rd, t)
    fp = FP[f"r{rd}|{t}|benefit-none"]
    assert fp["n"] == g["benefit"]["n"], (rd, t)
    pts = [g["none"]["Bii"], g["inverse"]["Bii"], g["prior"]["Bii"], g["benefit"]["A"], g["benefit"]["Bii"]]
    xmin = min(xmin, *pts)
    ax.plot([min(pts), max(pts)], [y, y], color="#D0D0D0", lw=1.2, zorder=1)
    ax.scatter([g["none"]["Bii"]], [y], s=16, marker="o", facecolor="white", edgecolor=GREY, lw=0.9, zorder=3)
    ax.scatter([g["inverse"]["Bii"]], [y + 0.26], s=22, marker="P", color=C_RULE, edgecolor="white", lw=0.3, zorder=3)
    ax.scatter([g["prior"]["Bii"]], [y - 0.26], s=18, marker="X", color=C_PRIOR, edgecolor="white", lw=0.3, zorder=3)
    oa, oi = g["benefit"]["A"], g["benefit"]["Bii"]
    if abs(oa - oi) > 1e-9:
        ax.scatter([oa], [y], s=20, marker="o", facecolor="white", edgecolor=OURS, lw=0.8, zorder=4)
        ax.annotate("", xy=(oi, y), xytext=(oa, y), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.6, shrinkA=2.5, shrinkB=2.5, mutation_scale=5))
    ax.scatter([oi], [y], s=22, marker="o", color=OURS, edgecolor="white", lw=0.5, zorder=5)
    for v, yy, col in ((fp["f_A"], y + 0.17, C_A), (fp["f_Bii"], y - 0.17, OURS)):
        e, lo, hi = (100 * x for x in v)
        lows.append(lo); his.append(hi)
        dx.plot([lo, hi], [yy, yy], color=col, lw=0.9, zorder=2)
        dx.scatter([e], [yy], s=11, color=col if (lo > 0 or hi < 0) else "white", edgecolor=col, lw=0.8, zorder=3)
    labels.append(f"{TASK_NAME[t]}{' *' if t in INV else ''}  ({g['benefit']['n']})")
    y -= 1
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=FS)
ax.tick_params(axis="y", length=0)
x0 = np.floor(10 * (xmin - 0.04)) / 10
ax.set_xlim(x0, 1.02); ax.set_xticks([v for v in np.arange(0.2, 1.01, 0.2) if v >= x0 - 1e-9])
ax.set_xlabel("Success at first target handoff")
lo_, hi_ = np.floor(min(lows) / 10) * 10 - 3, np.ceil(max(his) / 10) * 10 + 3
dx.axvline(0, color="#333333", lw=0.6)
dx.tick_params(axis="y", left=False, labelleft=False)
dx.set_xlabel("Ours $-$ none (pp)")
dx.set_xlim(lo_, hi_); dx.set_xticks([v for v in range(-50, 101, 25) if lo_ <= v <= hi_])
sep = (ys[len(old) - 1] + ys[len(old)]) / 2
for a in (ax, dx):
    style_ax(a); a.spines["left"].set_visible(False); a.set_ylim(min(ys) - 0.7, 0.9)
    a.axhline(sep, color=LIGHT, lw=0.6, ls=(0, (2, 2)))
ax.text(x0, 0.55, "round-1 chains", fontsize=FS, color=MUTED, ha="left", va="bottom")
ax.text(x0, ys[len(old)] + 0.55, "round-2 chains, added tasks", fontsize=FS, color=MUTED, ha="left", va="bottom")

# legend aligned with the content block: from the left edge of the task labels to the right edge of the right axes
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
L = min(t.get_window_extent(rnd).x0 for t in ax.get_yticklabels() if t.get_text()) / fig.bbox.width
R = dx.get_position().x1
yb = TOPF + 0.05 / H
yh = (H - 0.03) / H - yb
fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="no intervention"),
                    Line2D([], [], marker="P", ls="", color=C_RULE, label="inverse rule (ii)"),
                    Line2D([], [], marker="X", ls="", color=C_PRIOR, label="task prior (ii)"),
                    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=OURS, label="ours, A"),
                    Line2D([], [], marker="o", ls="", color=OURS, label="ours (ii)"),
                    Line2D([], [], color=C_A, lw=0.9, marker="o", markersize=3, label="ours $-$ none, A"),
                    Line2D([], [], color=OURS, lw=0.9, marker="o", markersize=3, label="ours $-$ none, (ii)"),
                    Line2D([], [], color=GREY, lw=0.9, marker="o", markersize=3, markerfacecolor=GREY,
                           markeredgecolor=GREY, label="95% CI excludes 0"),
                    Line2D([], [], color=GREY, lw=0.9, marker="o", markersize=3, markerfacecolor="white",
                           markeredgecolor=GREY, label="95% CI includes 0")],
           loc="lower left", bbox_to_anchor=(L, yb, R - L, yh), mode="expand", borderaxespad=0.0, ncol=3,
           frameon=False, fontsize=FS, handlelength=1.6, handletextpad=0.3, columnspacing=0.6, markerscale=0.8)
print(f"legend block: x = {L:.3f} .. {R:.3f} (figure fraction)")
print(save(fig, "fig_pertask"))
