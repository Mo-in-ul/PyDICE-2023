"""Guards against the v4 alpha-floor mismatch: the forward-model clamp on alpha must equal the solver bound (0.1,
as in GAMS), so that a path with alpha between 0.1 and 0.2 is evaluated as it is, in both forward models."""
import numpy as np
import jax.numpy as jnp
from common import dm, gams_controls
import mc_v2 as V
import mc_ift as M
import dice_jax as dj
from ipopt_model import GAMS_ALPHA

N = 81


def test_floor_matches_gams():
    p, _ = V.draw_params(np.array([V.DICE[k] for k in V.KEYS]))
    assert GAMS_ALPHA[0] == 0.1
    assert p['AlphaLowerBound'] == GAMS_ALPHA[0]


def test_alpha_just_above_floor_not_clamped():
    p, _ = V.draw_params(np.array([V.DICE[k] for k in V.KEYS]))
    MIU, S, _ = gams_controls(9, p)      # published optimal controls, inside every clamp
    alpha = np.r_[0, np.full(N, 0.105)]
    clamped = dj.trajectory(M.full(M.qdyn(p)), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha), clamp=True)
    free = dj.trajectory(M.full(M.qdyn(p)), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha), clamp=False)
    assert np.allclose(np.asarray(clamped['MAT'][1:]), np.asarray(free['MAT'][1:]), rtol=0, atol=1e-10)
    numba_mat = dm.diceTrajectory(p, MIU, S, alpha)[3][1:]
    assert np.allclose(numba_mat, np.asarray(free['MAT'][1:]), rtol=1e-12, atol=1e-9)


def test_numba_default_floor_is_gams():
    """The Numba model's own default must equal the GAMS bound, so direct users of the code get 0.1, not 0.2."""
    p = dm.LoadParams()
    assert p['AlphaLowerBound'] == GAMS_ALPHA[0]
