"""Regression tests against the published DICE-2023 GAMS runs and the numbers in the paper."""
import numpy as np
import pytest
from common import dm, scenario_params, gams_controls, load_json
from scc_decomposition import scc_decomposition


@pytest.fixture(scope='module')
def opt():
    _, p = scenario_params(9)
    MIU, S, g = gams_controls(9, p)
    return p, MIU, S, g


def test_forward_model_reproduces_gams_temperature(opt):
    """GAMS controls through the PyDICE forward model (JAX copy), alpha solved from the IRF condition."""
    import jax.numpy as jnp, dice_jax as dj
    p, MIU, S, g = opt
    alpha, _ = dm.solve_alpha(p, MIU, S)
    T = np.asarray(dj.trajectory(dj.pack_params(p), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha))['TATM'])[1:]
    Tg = np.asarray(g['TATM'], float)
    n = min(40, len(Tg))
    assert np.max(np.abs(T[:n] - Tg[:n])) < 1e-6


def test_scc_decomposition_2025(opt):
    p, MIU, S, _ = opt
    o = scc_decomposition(p, MIU, S)
    assert o['total'][1] == pytest.approx(59.067, abs=5e-3)
    assert o['direct'][1] == pytest.approx(51.364, abs=5e-3)
    assert o['feedback'][1] == pytest.approx(7.703, abs=5e-3)
    assert o['share'][1] == pytest.approx(0.1304, abs=5e-4)


def test_decomposition_matches_finite_differences(opt):
    """Exact total vs responsive Richardson SCC, exact direct vs fixed-alpha SCC (2025-2100)."""
    p, MIU, S, _ = opt
    o = scc_decomposition(p, MIU, S)
    a = o['alpha']
    fr = np.asarray(dm.compute_SCC(p, MIU, S, a, alpha_response=True, last_period=18))
    ff = np.asarray(dm.compute_SCC(p, MIU, S, a, alpha_response=False, last_period=18))
    k = np.arange(1, 17)
    assert np.max(np.abs(o['total'][k] / fr[k + 1] - 1)) < 1e-6
    assert np.max(np.abs(o['direct'][k] / ff[k + 1] - 1)) < 1e-6


def test_feedback_sign_change(opt):
    p, MIU, S, _ = opt
    o = scc_decomposition(p, MIU, S)
    neg = o['years'][:41][o['feedback'][:41] < 0]
    assert neg[0] == 2110


def test_published_results_files():
    """The result files behind the paper's tables carry the reported values."""
    assert load_json('exp_c_fair_range.json')['cmip6_summary']['median'] == pytest.approx(0.1241, abs=1e-3)
    b = load_json('exp_b_physical_policy.json')
    assert b['baseline']['share_2025'] == pytest.approx(0.1304, abs=5e-4)
    s = load_json('mc2_summary.json')
    assert s['n_ipopt'] == 493 and s['n_slsqp_fallback'] == 7 and s['n_flag_bound_active'] == 19
