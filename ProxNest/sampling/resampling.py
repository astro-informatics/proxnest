import numpy as np


def reorder_samples(samples, likelihood_values):
    r"""This program is to find the sample with the smallest likelihood and move it to the end of the list

    Args:
        samples (np.ndarray): given sample list
        likeliVal (np.ndarray): corresponding likelihood

    Returns:
        tuple: Reordered version of (samples, likelihood_values)

    Notes:
        MATLAB version: Xiaohao Cai (30/01/2019)

        Python version: Matthew Price (10/05/2022)
    """

    # find the smallest likelihood and corresponding index
    minSamIdx = np.argmin(likelihood_values)

    # swap the sample wit the smallest likelihood to the end of the list
    tempSample = samples[minSamIdx]
    samples[minSamIdx] = samples[-1]
    samples[-1] = tempSample

    # swap the likelihood accordingly
    tempL = likelihood_values[minSamIdx]
    likelihood_values[minSamIdx] = likelihood_values[-1]
    likelihood_values[-1] = tempL

    return samples, likelihood_values
