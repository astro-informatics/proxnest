import numpy as np
import pywt


class db_wavelets:
    """Constructs a linear operator for abstract Daubechies Wavelets.

    Notes:
        Stripped back version of optimus-primal linear operator.
    """

    def __init__(self, wav, levels, shape, axes=None):
        """Initialises Daubechies Wavelet linear operator class.

        Args:
            wav (string): Wavelet type (see https://tinyurl.com/5n7wzpmb).

            levels (list[int]): Wavelet levels (scales) to consider.

            shape (list[int]): Dimensionality of input to wavelet transform.

            axes (int): Which axes to perform wavelet transform (default = all axes).

        Raises:
            ValueError: Raised when levels are not positive definite.
        """

        if np.any(levels <= 0):
            raise ValueError("Wavelet levels must be positive definite")
        if axes is None:
            axes = range(len(shape))
        self.axes = axes
        self.wav = wav
        self.levels = np.array(levels, dtype=int)
        self.shape = shape
        self.coeff_slices = None
        self.coeff_shapes = None

        self.adj_op(self.dir_op(np.ones(shape)))

    def dir_op(self, x):
        r"""Evaluates the forward abstract wavelet transform of :math:`x`.

        Args:
            x (np.ndarray): Array to wavelet transform.

        Raises:
            ValueError: Raised when the shape of x is not even in every dimension.

        Returns:
            np.ndarray: Flattened array of wavelet coefficients.
        """
        if self.wav == "dirac":
            return np.ravel(x)

        if self.shape[0] % 2 == 1:
            raise ValueError("Signal shape should be even dimensions.")

        if len(self.shape) > 1:
            if self.shape[1] % 2 == 1:
                raise ValueError("Signal shape should be even dimensions.")

        coeffs = pywt.wavedecn(
            x, wavelet=self.wav, level=self.levels, mode="periodic", axes=self.axes
        )
        arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(
            coeffs, axes=self.axes
        )
        return arr

    def adj_op(self, x):
        r"""Evaluates the forward adjoint abstract wavelet transform of :math:`x`.

        Args:
            x (np.ndarray): Array to adjoint wavelet transform.

        Returns:
            np.ndarray: Array of pixel-space coefficients.
        """
        if self.wav == "dirac":
            return np.reshape(x, self.shape)

        coeffs_from_arr = pywt.unravel_coeffs(
            x, self.coeff_slices, self.coeff_shapes, output_format="wavedecn"
        )
        return pywt.waverecn(
            coeffs_from_arr, wavelet=self.wav, mode="periodic", axes=self.axes
        )
