# -*- coding: utf-8 -*-
"""Shared settings for the paper figures: style, widths, display names, colours, data loading."""
import json
from pathlib import Path

from plotstyle import use_style, save_figure, ROLE, OKABE_ITO, style_ax, panel_tag  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
STATS = json.loads((ROOT / "data/derived/stats.json").read_text())

# cas-dc geometry (measured from the compiled document: \textwidth and \columnwidth)
TEXT_W = 7.05
COL_W = 3.43

ARM_NAME = {
    "none": "No intervention", "retry": "Retry once", "fixed": "Fixed per-task repair", "risk": "Risk gate",
    "benefit": "Benefit gate (ours)", "full": "Ours + selective retry", "inverse": "Inverse-skill rule",
    "prior": "Task-prior repair", "benefit_noretry": "Ours, no retry", "home_all": "Home before every target",
    "noloc": "Ours, locator-free", "benefit_m5": "Ours, no prior input", "benefit_alt": "Ours, no prior input",
}
# Task display names; identical to the task names in the manuscript and the task rows of the tables.
TASK_NAME = {
    "push_blue_block_right": "push blue block right", "push_red_block_right": "push red block right",
    "push_pink_block_right": "push pink block right", "lift_blue_block_slider": "lift blue block (slider)",
    "lift_pink_block_slider": "lift pink block (slider)", "lift_red_block_slider": "lift red block (slider)",
    "lift_pink_block_drawer": "lift pink block (drawer)", "rotate_red_block_left": "rotate red block left",
    "rotate_pink_block_right": "rotate pink block right", "push_into_drawer": "push into drawer", "stack_block": "stack blocks",
}
OURS = ROLE["ours"]
GREY = "#7A7A7A"
LIGHT = "#BDBDBD"
INK = ROLE["ink"]
MUTED = ROLE["muted"]
C_STATE = OKABE_ITO["blue"]
C_NOISE = "#A9C9E3"
C_TASK = "#D9D9D9"
C_RULE = OKABE_ITO["green"]
C_PRIOR = OKABE_ITO["orange"]


def setup():
    use_style({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.0, "xtick.labelsize": 7.0,
               "ytick.labelsize": 7.0, "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
               "savefig.bbox": "standard"})


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    out = save_figure(fig, FIG / f"{stem}.pdf", also_png=True, close=False)
    fig.savefig(FIG / f"{stem}.svg")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def pp(x, sign=True, nd=1):
    v = 100 * x
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}").replace("-", "\u2212")
