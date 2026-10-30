"""
Compare a PyDICE run with the GAMS reference solution for the same scenario.

    from dice.compare import run_scenario
    results = run_scenario(9)            # optimise, save results, compare with GAMS

or, after your own run:

    from dice.compare import compare_with_gams
    summary, figs = compare_with_gams(model, output, meta)

The GAMS files are read from dice/validation/gams_reference by default; pass gams_dir
to use another folder.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dice.model import Dice2023Model, diceTrajectory
from dice.irf import solve_alpha

GAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation', 'gams_reference')
RESULTS_DIR = './results'


# PyDICE scenario number -> GAMS reference file in GAMS_DIR
GAMS_FILES = {
    1: 'dice_scenario_R1.xlsx', 2: 'dice_scenario_R2.xlsx', 3: 'dice_scenario_R3.xlsx',
    4: 'dice_scenario_R4.xlsx', 5: 'dice_scenario_R5.xlsx', 6: 'dice_scenario_T1.5.xlsx',
    7: 'dice_scenario_T2.xlsx', 8: 'dice_scenario_paris.xlsx', 9: 'dice_scenario_opt.xlsx',
    10: 'dice_scenario_base.xlsx',
}
SCENARIO_NAMES = {
    1: '1% discounting', 2: '2% discounting', 3: '3% discounting', 4: '4% discounting',
    5: '5% discounting', 6: '1.5°C limit', 7: '2°C limit', 8: 'Paris', 9: 'Cost-benefit optimal',
    10: 'Baseline',
}

# variable key: (GAMS column prefix, PyDICE output column, label, unit)
COMPARE_VARS = {
    'TATM':    ('Atmospheric temperatur',        3,  'Atmospheric temperature',  '°C'),
    'MIU':     ('Emissions control rate',        8,  'Emissions control rate',   'fraction'),
    'ECO2':    ('Total CO2 Emissions',           1,  'Total CO2 emissions',      'GtCO2/yr'),
    'CO2PPM':  ('Atmospheric concentration C',   2,  'CO2 concentration',        'ppm'),
    'FORC':    ('Total forcings',                31, 'Total forcing',            'W/m²'),
    'SCC':     ('Social cost of carbon',         39, 'Social cost of carbon',    '$/tCO2'),
    'CPRICE':  ('Carbon price',                  7,  'Carbon price',             '$/tCO2'),
    'S':       ('Savings rate',                  15, 'Savings rate',             'fraction'),
    'K':       ('Capital stock',                 14, 'Capital stock',            'trn $'),
    'CPC':     ('Consumption per capita',        6,  'Consumption per capita',   'thousand $'),
    'DAMFRAC': ('Climate damages, fraction',     5,  'Damages',                  'share of output'),
    'ALPHA':   ('Alpha',                         36, 'Carbon-cycle alpha',       '-'),
}


def load_gams(scenario, gams_dir=None):
    """Load the GAMS reference solution for a scenario. Returns a dict of 1-D arrays."""
    gams_dir = GAMS_DIR if gams_dir is None else gams_dir
    path = os.path.join(gams_dir, GAMS_FILES[scenario])
    if not os.path.exists(path):
        raise FileNotFoundError(f"GAMS reference not found: {path}. Set GAMS_DIR to the folder with the .xlsx files.")
    df = pd.read_excel(path, header=0)

    def col(prefix):
        hits = [c for c in df.columns if str(c).strip().lower().startswith(prefix.lower())]
        if not hits:
            raise KeyError(f"Column starting with '{prefix}' not found in {path}")
        return pd.to_numeric(df[hits[0]], errors='coerce').values

    g = {k: col(v[0]) for k, v in COMPARE_VARS.items()}
    g['Year'] = col('Year')
    g['C'] = pd.to_numeric(df[[c for c in df.columns if str(c).strip() == 'Consumption'][0]], errors='coerce').values
    g['I'] = col('Gross investment')
    return g


def _welfare(params, C):
    """PyDICE welfare function evaluated on a consumption path C (trillion $/yr, 0-based)."""
    n = len(C)
    cpc = 1000.0 * C / params['L'][1:n + 1]
    el = params['elasmu']
    U = ((cpc ** (1.0 - el)) - 1.0) / (1.0 - el) - 1.0
    return params['tstep'] * params['scale1'] * np.sum(U * params['L'][1:n + 1] * params['RR'][1:n + 1]) + params['scale2']


def model_only_check(params, gams):
    """
    Run the GAMS control and savings paths through the PyDICE forward model (alpha solved from
    the IRF condition). This separates differences in the model equations from differences
    in how the two optimisers converged. A small value means the equations agree.
    """
    N = params['num_periods']
    MIU = np.zeros(N + 1); S = np.zeros(N + 1)
    MIU[1:] = gams['MIU'][:N]; S[1:] = gams['S'][:N]
    alpha, _, res = solve_alpha(params, MIU, S)
    TATM = diceTrajectory(params, MIU, S, alpha)[4][1:]
    return TATM, res


def capital_disposal(params, gams, horizon=40):
    """Largest one-period shortfall of GAMS capital below the accumulation equation, as % of the stock."""
    K, I = gams['K'], gams['I']
    implied = (1.0 - params['dk']) ** params['tstep'] * K[:-1] + params['tstep'] * I[:-1]
    gap = implied - K[1:]
    share = 100.0 * gap[:horizon - 1] / np.maximum(implied[:horizon - 1], 1e-12)
    return float(np.max(share)), int(params['yr0'] + params['tstep'] * (np.argmax(share) + 1))


def compare_with_gams(model, output, meta=None, gams_dir=None, results_dir=None,
                      show=True, horizon_year=2200, save=True):
    """
    Compare a finished PyDICE run with the GAMS reference solution for the same scenario.

    Prints a summary table, and draws two figures:
      1. PyDICE (solid) vs GAMS (dashed) for 12 key variables
      2. PyDICE minus GAMS for the same variables
    Both are saved as PDF and PNG in results_dir. Returns (summary dict, list of figures).
    """
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    os.makedirs(results_dir, exist_ok=True)
    p = model.params
    sc = model.scenario
    g = load_gams(sc, gams_dir)
    N = output.shape[0]
    years = p['yr0'] + p['tstep'] * np.arange(N)
    n_h = int(np.searchsorted(years, horizon_year, side='right'))
    i2030, i2050, i2100 = [int(np.searchsorted(years, y)) for y in (2030, 2050, 2100)]
    n2100 = i2100 + 1
    n2215 = min(40, N)

    py = {k: output[:, v[1]] for k, v in COMPARE_VARS.items()}
    T_model_only, irf_res_mo = model_only_check(p, g)
    disposal_pct, disposal_year = capital_disposal(p, g)

    W_py = _welfare(p, output[:, 27])
    W_g = _welfare(p, g['C'][:N])
    summary = {
        'scenario': sc,
        'name': SCENARIO_NAMES[sc],
        'welfare_gap_pct': 100.0 * (W_py - W_g) / abs(W_g),
        'max_dT_to_2100': float(np.max(np.abs(py['TATM'][:n2100] - g['TATM'][:n2100]))),
        'max_dT_to_2215': float(np.max(np.abs(py['TATM'][:n2215] - g['TATM'][:n2215]))),
        'peak_T_pydice': float(py['TATM'][:n2215].max()),
        'peak_T_gams': float(g['TATM'][:n2215].max()),
        'MIU_2030': (float(py['MIU'][i2030]), float(g['MIU'][i2030])),
        'MIU_2050': (float(py['MIU'][i2050]), float(g['MIU'][i2050])),
        'SCC_2025': (float(py['SCC'][1]), float(g['SCC'][1])),
        'SCC_2050': (float(py['SCC'][i2050]), float(g['SCC'][i2050])),
        'ECO2_2030': (float(py['ECO2'][i2030]), float(g['ECO2'][i2030])),
        'model_only_max_dT_to_2215': float(np.max(np.abs(T_model_only[:n2215] - g['TATM'][:n2215]))),
        'gams_capital_disposal_pct': disposal_pct,
        'gams_capital_disposal_year': disposal_year,
        'irf_max_residual': (meta or {}).get('irf_max_residual'),
        'infeasible': (meta or {}).get('infeasible', False),
    }

    # ── printed summary ──
    line = '─' * 64
    print(f"\n{line}\n PyDICE vs GAMS  |  scenario {sc}: {SCENARIO_NAMES[sc]}\n{line}")
    print(f" Welfare gap (PyDICE − GAMS)/|GAMS|      {summary['welfare_gap_pct']:+.4f} %")
    print(f" Max |ΔT| 2020–2100 / 2020–2215          {summary['max_dT_to_2100']:.4f} / {summary['max_dT_to_2215']:.4f} °C")
    print(f" Peak T (to 2215)  PyDICE / GAMS         {summary['peak_T_pydice']:.3f} / {summary['peak_T_gams']:.3f} °C")
    for k, lab in (('MIU_2030', 'Control rate 2030'), ('MIU_2050', 'Control rate 2050'),
                   ('SCC_2025', 'SCC 2025 ($/tCO2)'), ('SCC_2050', 'SCC 2050 ($/tCO2)'),
                   ('ECO2_2030', 'CO2 emissions 2030 (Gt)')):
        a, b = summary[k]
        print(f" {lab:<28}PyDICE / GAMS   {a:9.3f} / {b:9.3f}")
    print(f" Model-only check (GAMS controls in PyDICE): max |ΔT| {summary['model_only_max_dT_to_2215']:.4f} °C")
    if disposal_pct > 1.0:
        print(f" Note: GAMS capital falls {disposal_pct:.1f}% below its accumulation path in {disposal_year}"
              f" (free disposal of capital). PyDICE does not allow this.")
    if summary['infeasible']:
        print(" Note: this scenario is infeasible with equality capital; PyDICE returned a best-effort path.")
    print(line)

    # ── figures ──
    keys = list(COMPARE_VARS.keys())
    yrs = years[:n_h]
    title = (f"Scenario {sc}: {SCENARIO_NAMES[sc]}   |   welfare gap {summary['welfare_gap_pct']:+.4f}%   |   "
             f"max |ΔT| to 2100 {summary['max_dT_to_2100']:.3f}°C   |   peak T {summary['peak_T_pydice']:.2f} vs {summary['peak_T_gams']:.2f}°C")

    fig1, axes = plt.subplots(3, 4, figsize=(18, 11), facecolor='white')
    for ax, k in zip(axes.flat, keys):
        lab, unit = COMPARE_VARS[k][2], COMPARE_VARS[k][3]
        sl = slice(1, n_h) if k == 'SCC' else slice(0, n_h)     # SCC 2020 is not computed
        ax.plot(years[sl], py[k][sl], color='#1f5fa8', lw=2.0, label='PyDICE')
        ax.plot(years[sl], g[k][:N][sl], color='#d1495b', lw=1.6, ls='--', label='GAMS')
        if k == 'TATM' and sc in (6, 7):
            ax.axhline(1.5 if sc == 6 else 2.0, color='0.5', lw=0.8, ls=':')
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9, frameon=False)
    fig1.suptitle(title, fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    fig2, axes2 = plt.subplots(3, 4, figsize=(18, 11), facecolor='white')
    for ax, k in zip(axes2.flat, keys):
        lab, unit = COMPARE_VARS[k][2], COMPARE_VARS[k][3]
        sl = slice(1, n_h) if k == 'SCC' else slice(0, n_h)
        d = py[k][sl] - g[k][:N][sl]
        ax.axhline(0, color='0.4', lw=0.8)
        ax.plot(years[sl], d, color='#2a7f62', lw=1.8)
        if k == 'TATM':
            ax.plot(years[:n_h], T_model_only[:n_h] - g['TATM'][:n_h], color='0.3', lw=1.2, ls=':',
                    label='GAMS controls in PyDICE')
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{lab}: PyDICE − GAMS", fontsize=11)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    fig2.suptitle(f"Scenario {sc}: {SCENARIO_NAMES[sc]}  |  differences (PyDICE − GAMS)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        base = os.path.join(results_dir, f"gams_comparison_scen{sc}")
        with PdfPages(base + '.pdf') as pdf:
            pdf.savefig(fig1); pdf.savefig(fig2)
        fig1.savefig(base + '_levels.png', dpi=110)
        fig2.savefig(base + '_differences.png', dpi=110)
        pd.DataFrame({'Year': yrs, **{f'{k}_pydice': py[k][:n_h] for k in keys},
                      **{f'{k}_gams': g[k][:n_h] for k in keys}}).to_csv(base + '_series.csv', index=False)
        print(f"Comparison saved: {base}.pdf, {base}_levels.png, {base}_differences.png, {base}_series.csv")
    if show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig2)
    return summary, [fig1, fig2]


def run_scenario(scenario, compare=True, show=True, plots=True, gams_dir=None):
    """Run one scenario end to end: optimise, save results, and compare with GAMS."""
    model = Dice2023Model(num_times=81, scenario=scenario)
    model.dump_parameters()
    years = np.arange(model.params['yr0'],
                      model.params['yr0'] + model.params['tstep'] * model.num_periods,
                      model.params['tstep'])
    x_opt, output, meta = model.run_model()
    model.dump_state(years, output, None, scenario)
    if plots:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        model.plot_state_to_file(os.path.join(RESULTS_DIR, f"dice2023_plots_scen{scenario}.pdf"), years, output, x_opt)
    summary = None
    if compare:
        summary, _ = compare_with_gams(model, output, meta, gams_dir=gams_dir, show=show)
    return {'model': model, 'x_opt': x_opt, 'output': output, 'meta': meta, 'summary': summary}

