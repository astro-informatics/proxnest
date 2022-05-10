import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


class Identity:
    """Identity sensing operator

    Notes:
        Implemented originally in optimus-primal.
    """

    def dir_op(self, x):
        """Computes the forward operator of the identity class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        return x

    def adj_op(self, x):
        """Computes the forward adjoint operator of the identity class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        return x


class MaskedFourier:
    """
    Masked fourier sensing operator i.e. MRI/Radio imaging.
    """

    def __init__(self, dim, ratio):
        """Initialises the masked fourier sensing operator.

        Args:
            dim (int): Dimension of square pixel-space image.

            ratio (float): Fraction of measurements observed.
        """
        mask = np.full(dim**2, False)
        mask[: int(ratio * dim**2)] = True
        np.random.shuffle(mask)
        self.mask = mask.reshape((dim, dim))
        self.shape = (dim, dim)

    def dir_op(self, x):
        """Computes the forward operator of the class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        out = np.fft.fft2(x)
        return self.__mask(out)

    def adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        out = self.__mask_adjoint(x)
        return np.fft.ifft2(out)

    def __mask(self, x):
        """Applies observational mask to image.

        Args:
            x (np.ndarray): Vector to apply mask to.

        Returns:
            np.ndarray: slice of masked coefficients
        """
        return x[self.mask]

    def __mask_adjoint(self, x):
        """Applies adjoint of observational mask to image.

        Args:
            x (np.ndarray): Vector to apply adjoint mask to.

        Returns:
            np.ndarray: Projection of masked coefficients onto image.
        """
        xx = np.zeros(self.shape, dtype=complex)
        xx[self.mask] = x
        return xx
