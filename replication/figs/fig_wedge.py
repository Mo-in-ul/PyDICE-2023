#!/usr/bin/env python3
"""
Figure 2 -- observed SCC-cost gap under binding vs interior solutions.
Restyled to match fig_mechanism / fig_scatter: serif/CM mathtext, grayscale,
print-legible fonts, USETEX toggle.

Panel (a): Howard-Sterner High (binding). Reported SCC, marginal abatement
           cost, and the shaded OBSERVED GAP Delta = SCC - p^gross.
Panel (b): Dietz-Stern (interior). SCC and carbon price coincide (wedge = 0).

The shaded band is the OBSERVED GAP Delta (= SCC - p^gross), NOT the deferral
wedge w_nu. Do not relabel the band "deferral wedge".
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------- parameters ----------------
HS_FILE = "results/ramp_ceiling_experiment_clean/runs/ramp005_standard_dice2023_hs_high.csv"
DS_FILE = "results/ramp_ceiling_experiment_clean/runs/ramp007_standard_dice2023_dietz_stern.csv"
YEARS   = list(range(2030, 2051, 5))
OUT     = "fig_wedge"
USETEX  = False   # <-- set True locally for Computer Modern (matches the paper)

# ---------------- house style (shared with the other two figures) ----------------
mm = 1 / 25.4
mpl.rcParams.update({
    "text.usetex": USETEX, "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 11, "axes.linewidth": 0.9,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLACK = "0.0"; CP_C = "0.45"; GAP_FC = "0.85"; GAP_EC = "0.6"
TITLE = 11.5; LAB = 11; ANN = 10; LEG = 9.5
WBOX = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none")

# ---------------- data ----------------
def load(fn):
    d = pd.read_csv(fn)
    d["YEAR"] = 2020 + 5 * (d["PERIOD"] - 1)
    return d

def series(d):
    x = d[d.YEAR.isin(YEARS)].sort_values("YEAR")
    return x.YEAR.values, x.SCC.values, x.CPRICE.values

hs = load(HS_FILE); ds = load(DS_FILE)

# ---------------- figure ----------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(174*mm, 82*mm), sharey=True)

# --- (a) Howard-Sterner High (binding) ---
X, S, P = series(hs)
axa.fill_between(X, P, S, facecolor=GAP_FC, edgecolor=GAP_EC, hatch="////",
                 linewidth=0.0, zorder=1)
axa.plot(X, S, color=BLACK, lw=1.6, marker="o", ms=6, mfc=BLACK, mec=BLACK, zorder=3)
axa.plot(X, P, color=CP_C, lw=1.5, ls=(0, (5, 2)), marker="s", ms=5.5,
         mfc="white", mec=CP_C, zorder=3)
axa.set_title(r"(a) Howard-Sterner High (binding)", fontsize=TITLE, loc="left")

axa.annotate(r"\$158", (2030, S[0]), textcoords="offset points", xytext=(8, 7),
             fontsize=ANN, ha="left", bbox=WBOX, zorder=5)
axa.annotate(r"\$64", (2030, P[0]), textcoords="offset points", xytext=(8, -14),
             fontsize=ANN, color=CP_C, ha="left", zorder=5)
axa.annotate(r"\$273", (2050, S[-1]), textcoords="offset points", xytext=(-6, 8),
             fontsize=ANN, ha="right", zorder=5)
# 2030 gap arrow + label (moved a bit right and a bit up)
axa.annotate("", (2030, S[0]), (2030, P[0]),
             arrowprops=dict(arrowstyle="<->", lw=1.1, color="0.3"), zorder=4)
axa.annotate(r"observed gap" + "\n" + r"$\Delta = \$93$",
             (2030, (S[0] + P[0]) / 2), textcoords="offset points", xytext=(17, 15),
             fontsize=ANN, va="center", ha="left", color="0.15", bbox=WBOX, zorder=6)

axa.set_ylabel(r"$\$$/tCO$_2$", fontsize=LAB)
axa.set_xlabel("year", fontsize=LAB)
axa.set_ylim(0, 300); axa.set_xticks(YEARS)
axa.spines[["top", "right"]].set_visible(False)

# --- (b) Dietz-Stern (interior) ---
X, S, P = series(ds)
axb.plot(X, S, color=BLACK, lw=1.6, marker="o", ms=6, mfc=BLACK, mec=BLACK, zorder=3)
axb.plot(X, P, color=CP_C, lw=1.5, ls=(0, (5, 2)), marker="s", ms=5.5,
         mfc="white", mec=CP_C, zorder=3)
axb.set_title(r"(b) Dietz-Stern (interior)", fontsize=TITLE, loc="left")
axb.annotate(r"\$45", (2030, S[0]), textcoords="offset points", xytext=(6, 10),
             fontsize=ANN, ha="left", zorder=5)
axb.annotate("SCC and carbon price" + "\n" + r"coincide: wedge $=0$",
             (2042, 150), fontsize=ANN, ha="center", va="center", color="0.2")
axb.set_xlabel("year", fontsize=LAB); axb.set_xticks(YEARS)
axb.spines[["top", "right"]].set_visible(False)

# --- shared legend ---
leg = [
    Line2D([0], [0], marker="o", color=BLACK, ms=6, lw=1.6,
           label="reported SCC (constrained shadow price)"),
    Line2D([0], [0], marker="s", color=CP_C, mfc="white", ls=(0, (5, 2)), ms=5.5, lw=1.5,
           label=r"$p^{\mathrm{gross}}$ (marginal abatement cost)"),
    Patch(facecolor=GAP_FC, edgecolor=GAP_EC, hatch="////",
          label=r"observed gap $\Delta$ (SCC $-$ $p^{\mathrm{gross}}$)"),
]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.03), fontsize=LEG,
           columnspacing=1.6, handletextpad=0.5)

plt.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.5)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=300)
print(f"wrote {OUT}.pdf and {OUT}.png")
