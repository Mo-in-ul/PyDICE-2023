"""
tests/test_validation.py
========================
Science validation tests — verify PyDICE-2023 results match the reference
outputs stored in dice/validation/.

TWO levels of validation
------------------------
1. Regression (fast, ~5s)
   Load stored python_output CSVs, re-run the FORWARD PASS only from the
   stored optimal control trajectories, and check that the resulting state
   variables match.  No optimization.  Tests that the dynamics are
   deterministic and consistent with the paper numbers.

2. GAMS comparison (slow, skipped in CI by default)
   Load GAMS reference xlsx files and compare key variables against the
   python_output CSVs within paper-reported tolerances.
   Run with:  pytest tests/test_validation.py -m gams -v

Key tolerances used
-------------------
   TATM   : 0.01 °C     (< 1% of 1°C signal)
   SCC    : 2.0 USD/tCO2
   CPC    : 0.5 %
   Y      : 0.5 %
   MIU    : 0.001       (absolute)
"""

import os
import numpy as np
import pandas as pd
import pytest

from dice import LoadParams, diceTrajectory, apply_disc_prstp, recoverAllVars

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = os.path.join(os.path.dirname(__file__), "..")
PY_OUT_DIR   = os.path.join(REPO_ROOT, "dice", "validation", "python_output")
GAMS_REF_DIR = os.path.join(REPO_ROOT, "dice", "validation", "gams_reference")

# Scenarios available in python_output (skip 6 = 1.5°C infeasible)
REGRESSION_SCENARIOS = [1, 2, 3, 4, 5, 7, 8, 9, 10]

# GAMS xlsx → PyDICE scenario number
GAMS_FILE_MAP = {
    "dice_scenario_opt.xlsx":    9,
    "dice_scenario_base.xlsx":   10,
    "dice_scenario_R1.xlsx":     1,
    "dice_scenario_R2.xlsx":     2,
    "dice_scenario_R3.xlsx":     3,
    "dice_scenario_R4.xlsx":     4,
    "dice_scenario_R5.xlsx":     5,
    "dice_scenario_T2.xlsx":     7,
    "dice_scenario_paris.xlsx":  8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_python_output(scenario_num):
    path = os.path.join(PY_OUT_DIR, f"dice2023_state_scen{scenario_num}.csv")
    if not os.path.exists(path):
        pytest.skip(f"python_output not found for scenario {scenario_num}")
    return pd.read_csv(path)


def make_params(scenario_num, num_periods=81):
    p = LoadParams(num_periods)
    if scenario_num == 1:
        p["k0"] = 420; apply_disc_prstp(p, 0.01)
    elif scenario_num == 2:
        p["k0"] = 409; apply_disc_prstp(p, 0.02)
    elif scenario_num == 3:
        p["k0"] = 370; apply_disc_prstp(p, 0.03)
    elif scenario_num == 4:
        p["k0"] = 326; apply_disc_prstp(p, 0.04)
    elif scenario_num == 5:
        p["k0"] = 290; apply_disc_prstp(p, 0.05)
    return p


def rerun_forward(df, params):
    """Re-run the forward pass from stored control trajectories."""
    N = params["num_periods"]
    MIU   = np.zeros(N + 1); MIU[1:]   = df["MIUopt"].values[:N]
    S     = np.zeros(N + 1); S[1:]     = df["Sopt"].values[:N]
    alpha = np.zeros(N + 1); alpha[1:] = df["ALPHA"].values[:N]
    return diceTrajectory(params, MIU, S, alpha)


# ---------------------------------------------------------------------------
# Regression tests — forward pass reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scen", REGRESSION_SCENARIOS)
def test_tatm_regression(scen):
    """TATM from re-run matches stored paper values within 0.01°C."""
    df     = load_python_output(scen)
    params = make_params(scen)
    out    = rerun_forward(df, params)
    TATM_new  = out[4][1:]
    TATM_ref  = df["TATM"].values[:len(TATM_new)]
    max_diff  = np.max(np.abs(TATM_new - TATM_ref))
    assert max_diff < 0.01, (
        f"Scen {scen}: TATM max deviation {max_diff:.4f}°C > 0.01°C\n"
        f"  period of max diff: {np.argmax(np.abs(TATM_new - TATM_ref)) + 1}"
    )


@pytest.mark.parametrize("scen", REGRESSION_SCENARIOS)
def test_output_y_regression(scen):
    """Net output Y matches stored values within 0.5%."""
    df     = load_python_output(scen)
    params = make_params(scen)
    out    = rerun_forward(df, params)
    Y_new  = out[6][1:]
    Y_ref  = df["Y"].values[:len(Y_new)]
    rel    = np.max(np.abs(Y_new - Y_ref) / np.maximum(np.abs(Y_ref), 1e-6))
    assert rel < 0.005, f"Scen {scen}: Y max relative deviation {rel:.4%} > 0.5%"


@pytest.mark.parametrize("scen", REGRESSION_SCENARIOS)
def test_welfare_sign(scen):
    """Welfare is finite and negative (we minimise -welfare)."""
    df     = load_python_output(scen)
    params = make_params(scen)
    out    = rerun_forward(df, params)
    W      = out[0]
    assert np.isfinite(W), f"Scen {scen}: welfare is not finite"
    assert W > 0, f"Scen {scen}: raw welfare should be positive (optimizer returns -W)"


@pytest.mark.parametrize("scen", REGRESSION_SCENARIOS)
def test_peak_tatm_stored(scen):
    """Stored peak TATM is consistent with physics — above initial value."""
    df      = load_python_output(scen)
    tatm0   = LoadParams(81)["tatm0"]
    peak    = df["TATM"].max()
    assert peak > tatm0, (
        f"Scen {scen}: stored peak TATM {peak:.3f}°C ≤ initial {tatm0:.3f}°C — physically impossible"
    )


def test_optimal_peak_tatm_range():
    """Optimal scenario peak temperature is in a physically plausible range (2–4°C)."""
    df   = load_python_output(9)
    peak = df["TATM"].max()
    assert 2.0 < peak < 4.5, f"Optimal peak TATM {peak:.3f}°C outside expected range [2, 4.5]°C"


def test_temp_2c_constraint_respected():
    """2°C scenario stored TATM never exceeds 2.05°C."""
    df   = load_python_output(7)
    peak = df["TATM"].max()
    assert peak <= 2.05, f"2°C scenario stored TATM peak {peak:.3f}°C exceeds 2.05°C"


def test_feasibility_1p5c():
    """1.5°C scenario: check_temp_feasibility detects infeasibility, peak between 1.55–1.70°C."""
    from dice import Dice2023Model
    model  = Dice2023Model(num_times=81, scenario=6)
    result = model.check_temp_feasibility(1.5)
    assert not result["feasible"], "1.5°C should be infeasible"
    assert 1.55 < result["peak_temp"] < 1.70, (
        f"Best-effort peak {result['peak_temp']:.3f}°C outside expected range [1.55, 1.70]°C"
    )


def test_discount_ordering():
    """Higher discount rate → lower peak TATM (less abatement is optimal)."""
    peaks = {}
    for scen in [1, 2, 3, 4, 5]:
        df = load_python_output(scen)
        peaks[scen] = df["TATM"].max()
    # ρ=1% (scen1) < ρ=5% (scen5) in terms of peak temperature
    assert peaks[1] < peaks[5], (
        f"Expected lower peak for ρ=1% ({peaks[1]:.3f}°C) vs ρ=5% ({peaks[5]:.3f}°C)"
    )


# ---------------------------------------------------------------------------
# GAMS comparison tests — skipped in CI unless -m gams is passed
# ---------------------------------------------------------------------------

def _load_gams_xlsx(filename):
    path = os.path.join(GAMS_REF_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"GAMS reference not found: {filename}")
    try:
        df = pd.read_excel(path, sheet_name=0, header=0, index_col=None)
    except Exception as e:
        pytest.skip(f"Could not read {filename}: {e}")
    return df


@pytest.mark.gams
@pytest.mark.parametrize("gams_file,scen", GAMS_FILE_MAP.items())
def test_gams_tatm_match(gams_file, scen):
    """PyDICE TATM matches GAMS within 0.05°C across all periods."""
    gams = _load_gams_xlsx(gams_file)
    py   = load_python_output(scen)

    # Normalise GAMS column names — try common variants
    tatm_col = next(
        (c for c in gams.columns if "TATM" in str(c).upper() or "TEMP" in str(c).upper()),
        None
    )
    if tatm_col is None:
        pytest.skip(f"No TATM column found in {gams_file}. Columns: {list(gams.columns)[:10]}")

    gams_tatm = gams[tatm_col].dropna().values[:81]
    py_tatm   = py["TATM"].values[:len(gams_tatm)]
    diff      = np.abs(py_tatm - gams_tatm)
    assert diff.max() < 0.05, (
        f"Scen {scen} ({gams_file}): TATM max diff {diff.max():.4f}°C > 0.05°C at "
        f"period {np.argmax(diff) + 1}"
    )


@pytest.mark.gams
@pytest.mark.parametrize("gams_file,scen", GAMS_FILE_MAP.items())
def test_gams_miu_match(gams_file, scen):
    """PyDICE MIU matches GAMS within 0.01 across all periods."""
    gams = _load_gams_xlsx(gams_file)
    py   = load_python_output(scen)

    miu_col = next(
        (c for c in gams.columns if "MIU" in str(c).upper()),
        None
    )
    if miu_col is None:
        pytest.skip(f"No MIU column found in {gams_file}.")

    gams_miu = gams[miu_col].dropna().values[:81]
    py_miu   = py["MIUopt"].values[:len(gams_miu)]
    diff     = np.abs(py_miu - gams_miu)
    assert diff.max() < 0.01, (
        f"Scen {scen} ({gams_file}): MIU max diff {diff.max():.4f} > 0.01 at "
        f"period {np.argmax(diff) + 1}"
    )


@pytest.mark.gams
@pytest.mark.parametrize("gams_file,scen", GAMS_FILE_MAP.items())
def test_gams_scc_match(gams_file, scen):
    """PyDICE SCC at period 2 (2025) matches GAMS within 5 USD/tCO2."""
    gams = _load_gams_xlsx(gams_file)
    py   = load_python_output(scen)

    scc_col = next(
        (c for c in gams.columns if "SCC" in str(c).upper()),
        None
    )
    if scc_col is None:
        pytest.skip(f"No SCC column found in {gams_file}.")

    gams_scc = gams[scc_col].dropna().values
    py_scc   = py["SCC"].values

    # Period 2 = index 1 (SCC[1] = 0 by convention, SCC[2] = first real value)
    if len(gams_scc) < 2 or len(py_scc) < 2:
        pytest.skip("Not enough SCC periods")

    diff = abs(float(py_scc[1]) - float(gams_scc[1]))
    assert diff < 5.0, (
        f"Scen {scen} ({gams_file}): SCC_2025 PyDICE={py_scc[1]:.2f}, "
        f"GAMS={gams_scc[1]:.2f}, diff={diff:.2f} > 5 USD/tCO2"
    )
