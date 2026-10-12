# -*- coding: utf-8 -*-
"""Matplotlib style used for the quantitative figures of the paper (rcParams, palette, axis helpers, export).

Fonts: Calibri / Calibri Math are registered when present (Windows); otherwise matplotlib falls back to Arial,
Helvetica or DejaVu Sans, which changes glyph shapes but not the plotted data.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

GRID_COLOR = "#D9D9D9"
EDGE_COLOR = "#333333"
LABEL_COLOR = "#111111"
FONT_STACK = ["Calibri Math", "Calibri", "Arial", "Helvetica", "DejaVu Sans"]
FONT_STATUS = {"calibri_math": False, "calibri": False}

# Okabe & Ito (2008) colour-blind-safe palette
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
ROLE = {"ours": OKABE_ITO["vermillion"], "baseline": "#4D4D4D", "oracle": OKABE_ITO["green"], "blue": OKABE_ITO["blue"],
        "sky": OKABE_ITO["sky"], "orange": OKABE_ITO["orange"], "purple": OKABE_ITO["purple"], "gray": "#9E9E9E",
        "safe": OKABE_ITO["blue"], "fail": OKABE_ITO["vermillion"], "ambiguous": "#BDBDBD", "grid": "#D9D9D9",
        "grid_alt": "#C9D0DA", "annot": "#3A3A3A", "spine": "#333333", "ink": "#111111", "muted": "#6B7280",
        "rule": "#9A9A9A", "infeasible": "#EFEFEF", "mist": "#F4F6F8"}

DEFAULT_RCPARAMS = {
    "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 9.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0, "figure.titlesize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 6,
    "svg.fonttype": "none", "axes.linewidth": 0.6, "axes.edgecolor": EDGE_COLOR, "axes.labelcolor": LABEL_COLOR,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 4.0,
    "axes.labelpad": 2.5, "axes.axisbelow": True, "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.color": EDGE_COLOR, "ytick.color": EDGE_COLOR, "xtick.top": False, "ytick.right": False,
    "grid.color": GRID_COLOR, "grid.linewidth": 0.4, "grid.alpha": 0.9, "grid.linestyle": ":", "lines.linewidth": 1.1,
    "lines.markersize": 3.6, "lines.markeredgewidth": 0.5, "errorbar.capsize": 2.0, "scatter.edgecolors": "none",
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "legend.labelspacing": 0.32,
    "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "figure.dpi": 130, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "none", "savefig.transparent": False,
}


def register_preferred_fonts():
    """Register Calibri Math / Calibri from the Windows font directory if the files exist."""
    from matplotlib import font_manager as fm
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    candidates = {"calibri_math": ("Calibri Math.ttf", "calibri math.ttf", "CalibriMath.ttf", "CalibriMath.otf"),
                  "calibri": ("calibri.ttf", "Calibri.ttf")}
    for key, names in candidates.items():
        hit = next((os.path.join(fonts_dir, n) for n in names if os.path.isfile(os.path.join(fonts_dir, n))), None)
        FONT_STATUS[key] = False
        if hit:
            try:
                fm.fontManager.addfont(hit)
                FONT_STATUS[key] = True
            except (OSError, RuntimeError):
                pass
    return dict(FONT_STATUS)


def _font_rc():
    params = {"font.family": "sans-serif", "font.sans-serif": list(FONT_STACK), "mathtext.fontset": "dejavusans",
              "mathtext.fallback": "stixsans"}
    if FONT_STATUS.get("calibri_math"):
        params.update({"mathtext.fontset": "custom", "mathtext.rm": "Calibri Math", "mathtext.it": "Calibri Math:italic",
                       "mathtext.bf": "Calibri Math:bold",
                       "mathtext.sf": "Calibri" if FONT_STATUS.get("calibri") else "Calibri Math",
                       "mathtext.fallback": "stixsans"})
    return params


def use_style(extra=None):
    """Install the rcParams; call once before building figures."""
    register_preferred_fonts()
    params = dict(DEFAULT_RCPARAMS)
    params.update(_font_rc())
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)


def save_figure(fig, path, *, dpi=300, also_png=True, close=True):
    """Write a vector PDF and, optionally, a PNG next to it. Returns the PDF path."""
    path = str(path).replace("\\", "/")
    root, ext = os.path.splitext(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if ext.lower() in {".pdf", ".png", ".svg", ".eps"}:
        fig.savefig(path, dpi=dpi)
        out = path
        if also_png and ext.lower() == ".pdf":
            fig.savefig(root + ".png", dpi=dpi)
    else:
        out = root + ".pdf"
        fig.savefig(out, dpi=dpi)
        if also_png:
            fig.savefig(root + ".png", dpi=dpi)
    if close:
        plt.close(fig)
    return out


def apply_spines(ax, top=False, right=False, left=True, bottom=True, color=None, linewidth=0.6):
    vis = {"top": top, "right": right, "left": left, "bottom": bottom}
    for name, sp in ax.spines.items():
        sp.set_visible(vis.get(name, False))
        if vis.get(name, False):
            sp.set_color(color or ROLE["spine"])
            sp.set_linewidth(linewidth)
    return ax


def light_grid(ax, axis="y", which="major", *, color=None, linestyle="-", alpha=0.9, linewidth=0.4):
    ax.grid(True, axis=axis, which=which, color=color or ROLE["grid"], linewidth=linewidth, alpha=alpha,
            linestyle=linestyle, zorder=0)
    ax.set_axisbelow(True)
    return ax


def style_ax(ax, grid=None, *, top=False, right=False):
    """Outward ticks, no top/right spines, optional light grid ('x' or 'y')."""
    apply_spines(ax, top=top, right=right)
    ax.tick_params(length=2.6, pad=2.0, direction="out")
    if grid:
        light_grid(ax, axis=grid)
    return ax


def panel_tag(ax, text, dx=-0.085, dy=1.045, size=9.5, weight="bold"):
    """Bold (a)/(b) tag in axes-fraction coordinates."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="top", fontsize=size, fontweight=weight, color="#000000")
