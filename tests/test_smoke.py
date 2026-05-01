import numpy as np
import pytest
from dice import (
    LoadParams,
    apply_disc_prstp,
    Dice2023Model,
    diceTrajectory,
    DiceFunc,
    compute_SCC,
    recoverAllVars,
    COLUMNS,
)


@pytest.fixture(scope="module")
def base_params():
    return LoadParams(81)


@pytest.fixture(scope="module")
def flat_policy(base_params):
    p = base_params
    N = p["num_periods"]
    MIU   = np.full(N + 1, 0.5);  MIU[1] = p["miu1"]
    S     = np.full(N + 1, p["optlrsav"])
    alpha = np.full(N + 1, p["a0"])
    return MIU, S, alpha


# ── LoadParams ────────────────────────────────────────────────────────────────

def test_loadparams_keys(base_params):
    required = ["num_periods", "tstep", "a0", "miu1", "optlrsav",
                 "L", "RR", "miuup", "eco2Param"]
    for k in required:
        assert k in base_params, f"Missing key: {k}"


def test_loadparams_a0_positive(base_params):
    assert base_params["a0"] > 0


def test_loadparams_arrays_length(base_params):
    N = base_params["num_periods"]
    for arr in ["L", "RR", "miuup", "eco2Param", "sigma", "eland"]:
        assert len(base_params[arr]) == N + 1, f"{arr} wrong length"


def test_apply_disc_prstp(base_params):
    import copy
    p = copy.deepcopy(base_params)
    apply_disc_prstp(p, prstp_value=0.03)
    assert abs(p["prstp"] - 0.03) < 1e-12
    assert p["RR"][1] == pytest.approx(1.0)
    assert p["RR"][2] == pytest.approx(1.0 / 1.03 ** 5, rel=1e-6)


# ── diceTrajectory ────────────────────────────────────────────────────────────

def test_trajectory_returns_finite(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    out = diceTrajectory(base_params, MIU, S, alpha)
    UTILITY, C, CCATOT, MAT, TATM = out[0], out[1], out[2], out[3], out[4]
    assert np.isfinite(UTILITY)
    assert np.all(np.isfinite(C[1:]))
    assert np.all(np.isfinite(TATM[1:]))


def test_trajectory_tatm_positive(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    out = diceTrajectory(base_params, MIU, S, alpha)
    TATM = out[4]
    assert np.all(TATM[1:] > 0)


def test_trajectory_shape(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    out = diceTrajectory(base_params, MIU, S, alpha)
    N = base_params["num_periods"]
    assert len(out[1]) == N + 1   # C
    assert len(out[4]) == N + 1   # TATM


def test_trajectory_no_pulse_vs_zero_pulse(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    N = base_params["num_periods"]
    out_no_pulse   = diceTrajectory(base_params, MIU, S, alpha)
    out_zero_pulse = diceTrajectory(base_params, MIU, S, alpha,
                                    pulse_GtCO2_per_year=np.zeros(N + 1))
    assert out_no_pulse[0] == pytest.approx(out_zero_pulse[0], rel=1e-10)


# ── compute_SCC ───────────────────────────────────────────────────────────────

def test_scc_shape(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    scc = compute_SCC(base_params, MIU, S, alpha)
    assert len(scc) == base_params["num_periods"] + 1


def test_scc_period1_zero(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    scc = compute_SCC(base_params, MIU, S, alpha)
    assert scc[1] == 0.0


def test_scc_positive_early_periods(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    scc = compute_SCC(base_params, MIU, S, alpha)
    assert scc[2] > 0, "SCC at period 2 should be positive"


# ── recoverAllVars ────────────────────────────────────────────────────────────

def test_recover_shape(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    N = base_params["num_periods"]
    x = np.concatenate([MIU[1:], S[1:], alpha[1:]])
    output = recoverAllVars(x, base_params)
    assert output.shape == (N, 46)


def test_recover_columns_aligned(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    x = np.concatenate([MIU[1:], S[1:], alpha[1:]])
    output = recoverAllVars(x, base_params)
    assert COLUMNS[3]  == "TATM"
    assert COLUMNS[39] == "SCC"
    assert COLUMNS[8]  == "MIUopt"
    assert np.all(output[:, 3] > 0), "TATM column should be positive"


def test_recover_finite(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    x = np.concatenate([MIU[1:], S[1:], alpha[1:]])
    output = recoverAllVars(x, base_params)
    assert np.all(np.isfinite(output))


# ── Dice2023Model ─────────────────────────────────────────────────────────────

def test_model_init_cb_optimal():
    model = Dice2023Model(num_times=81, scenario=9)
    assert model.num_periods == 81
    assert model.TempUpperConstraint == 20.0


def test_model_init_temp_scenarios():
    assert Dice2023Model(num_times=81, scenario=7).TempUpperConstraint == 2.0
    assert Dice2023Model(num_times=81, scenario=6).TempUpperConstraint == 1.5


def test_model_feasibility_check():
    model = Dice2023Model(num_times=81, scenario=6)
    result = model.check_temp_feasibility(1.5)
    assert "feasible" in result
    assert "peak_temp" in result
    assert "TATM_maxabate" in result
    assert result["peak_temp"] > 1.5, "1.5°C should be infeasible"


def test_model_feasibility_check_2c():
    model = Dice2023Model(num_times=81, scenario=7)
    result = model.check_temp_feasibility(2.0)
    assert result["peak_temp"] > 0


# ── DiceFunc ──────────────────────────────────────────────────────────────────

def test_dicefunc_objective_finite(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    N = base_params["num_periods"]
    prob = DiceFunc(N, base_params)
    x = np.concatenate([MIU[1:], S[1:], alpha[1:]])
    val = prob.objective(x)
    assert np.isfinite(val)
    assert val < 0, "Negative because we minimize -welfare"


def test_dicefunc_irf_residual_shape(base_params, flat_policy):
    MIU, S, alpha = flat_policy
    N = base_params["num_periods"]
    prob = DiceFunc(N, base_params)
    x = np.concatenate([MIU[1:], S[1:], alpha[1:]])
    res = prob.irf_residual(x)
    assert len(res) == N
