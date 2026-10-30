"""
IRF condition helpers: solve the carbon-cycle parameter alpha(t) for given policies.

The DICE-2023 IRF100 condition links alpha(t) to the model state in the same period
(cumulative uptake and temperature), so alpha(t) and the period-t state are determined
together. For fixed control and savings paths, the residual in period t depends only on
alpha(1..t), which lets us solve alpha one period at a time.
"""
import numpy as np
from scipy.optimize import newton, brentq

from dice.model import diceTrajectory


def irf_rhs(params, a):
    """Right-hand side of the IRF100 condition for alpha a (scalar or array)."""
    out = 0.0
    for j in range(4):
        tau = params[f'tau{j}']
        out = out + a * params[f'emshare{j}'] * tau * (1.0 - np.exp(-100.0 / (a * tau)))
    return out


def irf_lhs(params, CCATOT, MAT, TATM):
    """Left-hand side of the IRF100 condition from the model state."""
    return params['irf0'] + params['irC'] * (CCATOT - (MAT - params['mateq'])) + params['irT'] * TATM


def solve_alpha(params, MIU, S, pulse=None, alpha_start=None, first_period=2):
    """
    Solve alpha(t), t = 2..N, so that the IRF condition holds in every period.

    Parameters
    ----------
    params       : dict from LoadParams
    MIU, S       : 1-based arrays of length N+1 (fixed control and savings paths)
    pulse        : optional emissions pulse in GtCO2/yr, 1-based array
    alpha_start  : optional starting path; periods before first_period are kept from it
    first_period : first period to solve (periods before it are left unchanged)

    Periods where alpha would have to fall below AlphaLowerBound (possible in the far
    future under large net-negative emissions) are held at the bound, matching the clamp
    in the forward model, and are excluded from the reported residual.

    Returns
    -------
    alpha    : 1-based array of length N+1
    ok       : False if some period could not be solved (other than at the lower bound)
    max_res  : max |IRF residual| over the periods not held at the bound
    """
    N = params['num_periods']
    MIU = np.ascontiguousarray(MIU, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)
    lo, hi = max(params['AlphaLowerBound'], 1e-3), 50.0
    if alpha_start is None:
        a = np.full(N + 1, params['a0'])
    else:
        a = np.array(alpha_start, dtype=np.float64).copy()
    a[1] = params['a0']
    at_bound = np.zeros(N + 1, dtype=bool)
    ok = True

    def resid_k(val, k):
        a[k] = val
        o = diceTrajectory(params, MIU, S, a, pulse)
        return irf_rhs(params, val) - irf_lhs(params, o[2][k], o[3][k], o[4][k])

    for k in range(max(first_period, 2), N + 1):
        x0 = a[k] if alpha_start is not None else a[k - 1]
        try:
            # secant iteration from a good start converges in a few steps
            val = newton(resid_k, x0, args=(k,), tol=1e-13, maxiter=50)
            if not (lo <= val <= hi) or abs(resid_k(val, k)) > 1e-8:
                raise RuntimeError
            a[k] = val
        except (RuntimeError, OverflowError, ZeroDivisionError, FloatingPointError):
            g_lo, g_hi = resid_k(lo, k), resid_k(hi, k)
            if g_lo * g_hi < 0:
                a[k] = brentq(resid_k, lo, hi, args=(k,), xtol=1e-14)
            elif g_lo > 0:      # even the smallest allowed alpha is too large
                a[k] = lo
                at_bound[k] = True
            else:
                ok = False
                a[k] = hi

    o = diceTrajectory(params, MIU, S, a, pulse)
    r = irf_rhs(params, a[1:]) - irf_lhs(params, o[2][1:], o[3][1:], o[4][1:])
    keep = ~at_bound[1:]
    max_res = float(np.max(np.abs(r[keep]))) if np.any(keep) else 0.0
    return a, ok, max_res
