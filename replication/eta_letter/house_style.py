"""House figure style: serif Times 8 pt (ticks and legend 7 pt), ticks pointing in, Elsevier widths
(90 mm single column, 190 mm double), colours BLUE #2166ac and ORANGE #d6604d, embedded TrueType fonts."""
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2166ac', '#d6604d'
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42, 'savefig.dpi': 500,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix', 'lines.linewidth': 1.0})


def panel_label(ax, s):
    ax.text(0.02, 0.96, s, transform=ax.transAxes, va='top')
