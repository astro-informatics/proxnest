import pytest
import numpy as np
import ProxNest.utils as utils
import ProxNest.optimisations as opts

from skimage.metrics import structural_similarity as ssim


@pytest.mark.parametrize("tight", [True, False])
@pytest.mark.parametrize("pos", [True, False])
def test_l2_ball_projection(tight: bool, pos: bool):
    # Define the noise level.
    sigma = 1

    # Create random truth and observation set.
    x = np.ones((64, 64))
    x0 = np.random.randn(64, 64)
    data = x + sigma * np.random.randn(64, 64)

    # Define likelihood lambda function and evaluate L2-ball radius tau.
    LogLikeliL = lambda sol: -np.linalg.norm(data - sol, "fro") ** 2 / (2 * sigma**2)
    tau = -LogLikeliL(x0) * 1e-1

    # Create a parameters structure.
    params = utils.create_parameters_dict(y=data, tight=tight, pos=pos, reality=True)

    # Evaluate the projection algorithm
    z = opts.l2_ball_proj.sopt_fast_proj_B2(x0, tau, params)

    assert np.linalg.norm(data - z) < tau


@pytest.mark.parametrize("tight", [True, False])
@pytest.mark.parametrize("pos", [True, False])
def test_l1_norm_projection_extremes(tight: bool, pos: bool):
    # Create random signal
    x = np.random.randn(64, 64)

    # Create a parameters structure.
    params = utils.create_parameters_dict(tight=tight, pos=pos, reality=True)

    if tight:
        # Evaluate the l1-norm sub-iterations lambda=0
        z = opts.l1_norm_prox.l1_norm_prox(x, 0, params)
        np.testing.assert_allclose(z, x, 1e-14)

    # Evaluate the l1-norm sub-iterations lambda >> 0
    z = opts.l1_norm_prox.l1_norm_prox(x, 1e10, params)
    np.testing.assert_allclose(z, np.zeros_like(x), 1e-14)


@pytest.mark.parametrize("tight", [True, False])
@pytest.mark.parametrize("pos", [True, False])
def test_l1_norm_projection_specific(tight: bool, pos: bool):
    # Create random signal
    lamb = 0.5
    x = np.ones((64, 64))
    xpred = x / 2
    obj_pred = (3 / 8) * len(x.flatten("C"))

    # Create a parameters structure.
    params = utils.create_parameters_dict(tight=tight, pos=pos, reality=True)

    # Evaluate the l1-norm sub-iterations
    z = opts.l1_norm_prox.l1_norm_prox(x, lamb, params)
    np.testing.assert_allclose(z, xpred, 1e-10)

    # Minimised solution
    obj_z = 0.5 * np.linalg.norm(x - z) ** 2 + lamb * np.linalg.norm(np.ravel(z), ord=1)
    assert obj_z == pytest.approx(obj_pred)
