import numpy as np
from . import sensing_operators as sense


def soft_thresh(x, T, delta=2):
    r"""Compute the element-wise soft-thresholding of :math:`x`.

    Args:
        x (np.ndarray): Array to threshold.

        T (float): Soft-thresholding level (regularisation parameter)

        delta (float): Weighting parameter (default = 2).

    Returns:
        np.ndarray: Thresholded coefficients of :math:`x`.
    """
    return np.sign(x) * np.maximum(np.abs(x) - T * delta / 2, 0)


def hard_thresh(x, T):
    r"""Compute the element-wise hard-thresholding of :math:`x`.

    Args:
        x (np.ndarray): Array to threshold.

        T (float): Hard-thresholding level (regularisation parameter)

        delta (float): Weighting parameter.

    Returns:
        np.ndarray: Thresholded coefficients of :math:`x`.
    """
    return x * (np.abs(x) > T).astype(float)


def l1_projection(x, T, delta, Psi=sense.Identity()):
    r"""Compute the l1 proximal operator wrt dictionary :math:`\Psi`.

    Args:
        x (np.ndarray): Array to threshold.

        T (float): Soft-thresholding level (regularisation parameter)

        delta (float): Weighting parameter.

        Psi (LinearOperator): Prior dictionary (default = Identity)

    Returns:
        np.ndarray: Thresholded coefficients of :math:`x`.
    """
    u = Psi.dir_op(x)
    return x + Psi.adj_op(soft_thresh(u, T, delta) - u)


def l2_projection(x, T, delta, Psi=sense.Identity()):
    r"""Compute the l2 gradient step wrt dictionary :math:`\Psi`.

    Args:
        x (np.ndarray): Array to threshold.

        T (float): Soft-thresholding level (regularisation parameter)

        delta (float): Weighting parameter.

        Psi (LinearOperator): Prior dictionary (default = Identity)

    Returns:
        np.ndarray: Thresholded coefficients of :math:`x`.
    """
    return x - 2 * T * Psi.adj_op(Psi.dir_op(x)) * 2 * delta
