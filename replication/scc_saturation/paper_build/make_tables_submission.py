"""Tables added or rearranged for the submission revision (pydice2023_v6_submission_revised.tex).
  tex/tab_sub_mc.tex      Table 7: regression coefficients and Gaussian-process Sobol indices (S1, ST) of the share,
                          the same estimator as Figure 6 (values from results/sobol.json, unchanged)
  tex/tab_sub_sobol.tex   Appendix E: Sobol indices of log SCC and the decile-based S1 of the share (cross-check)
  tex/tab_sub_fairown.tex Appendix C: FaIR v2 sets with DICE-2023 welfare weights and with each set's own baseline
Reads results/mc2_summary.json, results/sobol.json, results/fair_own_weights.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(HERE, 'results', f)))
W = lambda f, s: open(os.path.join(HERE, 'tex', f), 'w').write(s)
KEYS = ['rho', 'elasmu', 'ecs', 'psi2', 'a3', 'irC', 'irT', 'eland0']
LAB = {'rho': '$\\rho$', 'elasmu': '$\\varphi$', 'ecs': 'ECS', 'psi2': '$\\psi_2$', 'a3': 'Damage exponent',
       'irC': '$\\sigma_C$', 'irT': '$\\sigma_T$', 'eland0': '$E_{\\mathrm{land}}(0)$'}
f2 = lambda x: f"{x:.2f}".replace('-0.00', '0.00')

m, s = R('mc2_summary.json'), R('sobol.json')
L = []
for k in KEYS:
    L.append(f"{LAB[k]} & " + ' & '.join(f"{m[y]['coef_pp_per_sd'][k]:+.2f} & {f2(s['share_' + y]['S1'][k])} & {f2(s['share_' + y]['ST'][k])}"
                                          for y in ('2025', '2100')) + " \\\\")
mc = lambda txt: "\\multicolumn{3}{c}{" + txt + "}"
L += ['\\midrule',
      "$R^2$ (linear) & " + mc(f"{m['2025']['R2']:.2f}") + " & " + mc(f"{m['2100']['R2']:.2f}") + " \\\\",
      "Surrogate CV $R^2$ & " + mc(f"{s['share_2025']['cv_r2_mean']:.2f}") + " & " + mc(f"{s['share_2100']['cv_r2_mean']:.2f}") + " \\\\",
      "Median share (\\%) & " + mc(f"{m['2025']['p50']:.1f}") + " & " + mc(f"{m['2100']['p50']:.1f}") + " \\\\",
      "5th--95th percentile & " + mc(f"{m['2025']['p5']:.1f} to {m['2025']['p95']:.1f}") + " & " + mc(f"{m['2100']['p5']:.1f} to {m['2100']['p95']:.1f}") + " \\\\",
      "Draws with negative share (\\%) & " + mc(f"{100 * m['2025']['share_negative']:.1f}") + " & " + mc(f"{100 * m['2100']['share_negative']:.1f}") + " \\\\"]
W('tab_sub_mc.tex', "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Pulse in 2025} & \\multicolumn{3}{c}{Pulse in 2100} \\\\\n"
  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nParameter & Coef. (pp/s.d.) & $S_1$ & $S_T$ & Coef. (pp/s.d.) & $S_1$ & $S_T$ \\\\\n\\midrule\n"
  + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")

L = []
for k in KEYS:
    L.append(f"{LAB[k]} & {f2(s['logscc_2025']['S1'][k])} & {f2(s['logscc_2025']['ST'][k])} & {m['2025']['S1'][k]:.2f} & {m['2100']['S1'][k]:.2f} \\\\")
L += ['\\midrule', "Surrogate CV $R^2$ & \\multicolumn{2}{c}{" + f"{s['logscc_2025']['cv_r2_mean']:.2f}" + "} & & \\\\"]
W('tab_sub_sobol.tex', "\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{log SCC 2025} & \\multicolumn{2}{c}{Share, deciles} \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nParameter & $S_1$ & $S_T$ & 2025 & 2100 \\\\\n\\midrule\n" + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")

f = R('fair_own_weights.json')
L = []
for name, o in f['sets'].items():
    fw, ow = o['fixed_weights'], o['own_as_used']
    nm = name.replace('FaIR v2.2 AR6 default', 'FaIR v2.2 AR6').replace('FaIRv2.0.0 default', 'FaIRv2.0.0')
    if name == 'FaIRv2.0.0 default':
        L.append('\\midrule')
    L.append(f"{nm} & {100 * fw['share_2025']:.1f} & {100 * ow['share']['2025']:.1f} & {fw['scc_2025']:.2f} & {ow['scc']['2025']:.2f} & {ow['peak_T_2020_2215']:.2f} \\\\")
W('tab_sub_fairown.tex', "\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Feedback share, 2025 (\\%)} & \\multicolumn{2}{c}{SCC, 2025 (\\$/tCO$_2$)} & \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nParameter set & DICE weights & Own baseline & DICE weights & Own baseline & Peak $T$ (\\textdegree C) \\\\\n\\midrule\n"
  + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
print(open(os.path.join(HERE, 'tex', 'tab_sub_mc.tex')).read()); print(open(os.path.join(HERE, 'tex', 'tab_sub_fairown.tex')).read())
