# -*- coding: utf-8 -*-
"""Figure: how much of handoff failure is decided by the handoff state.
(a) per-task split of single-rollout outcome variance into state (between handoffs) and rollout noise (within handoff),
    with a 95% bootstrap interval for the state share; pooled rows add the between-task share.
(b) distribution of k (successes out of J = 8 repeated executions from the identical handoff state) per task.
Data: data/derived/stats.json (computed by compute_stats.py from 1,256 CALVIN handoffs and 120 LIBERO-Goal handoffs)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from figcommon import *  # noqa

setup()
D = STATS["decomposition"]
tasks = sorted(D["by_task"], key=lambda t: -D["by_task"][t]["state_share"])
rows = [("task", t) for t in tasks] + [("pool", "CALVIN"), ("pool", "LIBERO")]
INV = {"push_red_block_right", "push_blue_block_right", "push_pink_block_right", "rotate_pink_block_right", "rotate_red_block_left"}
labels = [TASK_NAME[t] + (" *" if t in INV else "") for t in tasks] + ["CALVIN, 11 tasks pooled", "LIBERO-Goal, pooled"]

fig = plt.figure(figsize=(TEXT_W, 3.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.06, left=0.165, right=0.985, top=0.9, bottom=0.14)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharey=ax)
y = np.arange(len(rows))[::-1].astype(float)
y[-2:] -= 0.0
gap = 0.55
y = np.array([len(rows) - 1 - i + (gap if i < len(tasks) else 0) for i in range(len(rows))], float)
for yi, (kind, key) in zip(y, rows):
    if kind == "task":
        d = D["by_task"][key]; task_s = 0.0
    elif key == "CALVIN":
        d = D["pooled"]; task_s = d["task_share"]
    else:
        d = STATS["libero"]["decomposition"]; task_s = d["task_share"]
    st, no = d["state_share"], d["noise_share"]
    ax.barh(yi, task_s, color=C_TASK, height=0.68, edgecolor="white", lw=0.4)
    ax.barh(yi, st, left=task_s, color=C_STATE, height=0.68, edgecolor="white", lw=0.4)
    ax.barh(yi, no, left=task_s + st, color=C_NOISE, height=0.68, edgecolor="white", lw=0.4)
    if kind == "task":
        lo, hi = d["state_share_ci"]
        ax.plot([lo, hi], [yi, yi], color="#0B2E4F", lw=0.8, solid_capstyle="butt", zorder=4)
        ax.plot([lo, lo], [yi - 0.16, yi + 0.16], color="#0B2E4F", lw=0.8, zorder=4)
        ax.plot([hi, hi], [yi - 0.16, yi + 0.16], color="#0B2E4F", lw=0.8, zorder=4)
    ax.text(task_s + 0.012, yi, f"{100*st:.0f}%", ha="left", va="center", color="white", fontsize=6.6, fontweight="bold")
    ax.text(1.015, yi, f"{d['success']:.2f}", ha="left", va="center", fontsize=6.6, color=INK)
    ax.text(1.115, yi, f"{d['n']}", ha="left", va="center", fontsize=6.6, color=MUTED)
ax.text(1.015, y[0] + 0.85, r"$\bar p$", ha="left", va="center", fontsize=6.8, color=INK)
ax.text(1.115, y[0] + 0.85, "$n$", ha="left", va="center", fontsize=6.8, color=MUTED)
ax.set_xlim(0, 1); ax.set_ylim(-0.7, y[0] + 1.25)
ax.set_yticks(y); ax.set_yticklabels(labels)
for lab, (kind, key) in zip(ax.get_yticklabels(), rows):
    if kind == "pool":
        lab.set_fontweight("bold")
ax.axhline((y[len(tasks) - 1] + y[len(tasks)]) / 2, color=LIGHT, lw=0.6, ls=(0, (2, 2)))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
ax.set_xlabel("Share of outcome variance")
style_ax(ax)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_STATE, label="handoff state"), Patch(color=C_NOISE, label="rollout noise"),
                   Patch(color=C_TASK, label="between tasks (pooled)")], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
          frameon=False, handlelength=1.1, columnspacing=1.0, borderaxespad=0.2)
ax.set_xlim(0, 1.22)
ax.spines["bottom"].set_bounds(0, 1)
panel_tag(ax, "(a)", dx=-0.235, dy=1.11)

# (b) k distribution
cmap = LinearSegmentedColormap.from_list("blues_soft", ["#FFFFFF", "#DCE9F5", "#8DB8DA", "#2F6EA6", "#0B2E4F"])
M = []
for kind, key in rows:
    if kind == "task":
        h = np.array(D["by_task"][key]["k_hist"], float)
    elif key == "CALVIN":
        h = np.array(D["pooled"]["k_hist"], float)
    else:
        h = np.array(STATS["libero"]["decomposition"]["k_hist"], float)
    M.append(h / h.sum())
M = np.array(M)
for yi, row in zip(y, M):
    for k, v in enumerate(row):
        bx.add_patch(plt.Rectangle((k - 0.5, yi - 0.4), 1, 0.8, color=cmap(min(v / 0.6, 1.0)), lw=0))
        if v >= 0.095:
            bx.text(k, yi, f"{100*v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.33 else INK)
bx.set_xlim(-0.6, 8.6)
bx.set_xticks(range(9)); bx.set_xlabel(r"Successes $k$ in $J=8$ executions from the same state")
bx.tick_params(axis="y", left=False, labelleft=False)
for s in ("left", "right", "top"):
    bx.spines[s].set_visible(False)
bx.spines["bottom"].set_color("#333333"); bx.spines["bottom"].set_linewidth(0.6)
bx.axhline((y[len(tasks) - 1] + y[len(tasks)]) / 2, color=LIGHT, lw=0.6, ls=(0, (2, 2)))
bx.text(0, y[0] + 0.85, "all fail", ha="center", va="center", fontsize=6.6, color=MUTED)
bx.text(8, y[0] + 0.85, "all succeed", ha="center", va="center", fontsize=6.6, color=MUTED)
bx.text(4, y[0] + 0.85, "mixed outcomes", ha="center", va="center", fontsize=6.6, color=MUTED)
bx.annotate("", xy=(1.0, y[0] + 0.85), xytext=(2.75, y[0] + 0.85), arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6))
bx.annotate("", xy=(7.0, y[0] + 0.85), xytext=(5.25, y[0] + 0.85), arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6))
panel_tag(bx, "(b)", dx=-0.02, dy=1.11)
print(save(fig, "fig_decomposition"))
