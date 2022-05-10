import numpy as np


def augmented_TV_norm_prox(x, lamb, params):
    r"""Compute the augmented total variation proximal operator

    Compute the TV proximal operator when an additional linear operator A is
    incorporated in the TV norm, i.e. solve

    .. math::

        x^* = \min_{x} ||y - x||_2^2 + \lambda * ||A x||_{TV}

    where :math:`y` is the input vector and the solution :math:`x^*` is returned as sol.

    Args:
        x (np.ndarray): A sample position :math:`x` in the posterior space.

        lamb (float): Regularisation parameter.

        params (dict): Dictionary of parameters defining the optimisation.

    Returns:
        np.ndarray: Optimal solution :math:`x^*` of proximal operator.

    Notes:
        [1] A. Beck and  M. Teboulle, "Fast gradient-based algorithms for constrained Total Variation Image Denoising and Deblurring Problems", IEEE Transactions on Image Processing, VOL. 18, NO. 11, 2419-2434, November 2009.
    """
    return 0
