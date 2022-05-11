from ProxNest.operators import proximal_operators as prox_ops
from ProxNest.operators import sensing_operators as sense_ops
from ProxNest.operators import wavelet_operators as wav_ops
import numpy as np
import pytest


def test_identity_linear_operator():
    x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    phi = sense_ops.Identity()
    np.testing.assert_allclose(phi.dir_op(x), x, 1e-14)
    np.testing.assert_allclose(phi.adj_op(x), x, 1e-14)


def test_maskedfourier_linear_operator():
    ratio = 0.5
    rdim = 10
    fdim = int(rdim**2 * ratio)

    x = np.random.randn(rdim, rdim) + 1j * np.random.randn(rdim, rdim)
    y = np.random.randn(fdim) + 1j * np.random.randn(fdim)
    phi = sense_ops.MaskedFourier(rdim, ratio)

    xx = phi.dir_op(x)
    yy = phi.adj_op(y)

    a = abs(np.vdot(x, yy))
    b = abs(np.vdot(y, xx))

    assert a * rdim**2 == pytest.approx(b)


def test_wavelet_linear_operator():
    dim = 64
    x = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    psi = wav_ops.db_wavelets("db6", levels=2, shape=(dim, dim))
    xx = psi.dir_op(x)
    np.testing.assert_allclose(x, psi.adj_op(xx), 1e-8)


def test_soft_thresholding():
    x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    x_thresh = np.zeros_like(x)
    threshold = 0.1

    for i in range(10):
        for j in range(10):
            xx = x[i, j]
            xabs = np.abs(xx)
            if xabs - threshold > 0:
                x_thresh[i, j] = np.sign(xx) * (xabs - threshold)

    np.testing.assert_allclose(x_thresh, prox_ops.soft_thresh(x, threshold), 1e-14)


def test_hard_thresholding():
    x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    x_thresh = np.zeros_like(x)
    threshold = 0.1

    for i in range(10):
        for j in range(10):
            xx = x[i, j]
            xabs = np.abs(xx)
            if xabs - threshold > 0:
                x_thresh[i, j] = xx

    np.testing.assert_allclose(x_thresh, prox_ops.hard_thresh(x, threshold), 1e-14)


def test_l1_projection():
    x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    x_backproj = np.zeros_like(x)
    threshold = 0.1

    np.testing.assert_allclose(
        prox_ops.soft_thresh(x, threshold),
        prox_ops.l1_projection(x, threshold, 2),
        1e-14,
    )


def test_l2_projection():
    x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

    np.testing.assert_allclose(
        np.zeros_like(x), prox_ops.l2_projection(x, 1, 0.25), 1e-14
    )
