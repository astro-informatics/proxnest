import numpy as np
from tqdm import tqdm
import ProxNest.logs as lg
from . import resampling


def ProxNestedSampling(X0, LikeliL, proxH, proxB, params, options):
    r"""Executes the proximal nested sampling algorithm

    Args:
        X0 (np.ndarray): initialisation of the sample chain.

        LikeliL (lambda): function to compute the likelihood value of a sample.

        proxH (lambda): proximity operator of the prior.

        proxB (lambda): proximity operator of the constraint :math:`\ell_2`-ball.

        params (dict): parameters for prior resampling subject to likelihood isocontour.

        options (dict): parameters about number of samples, thinning factor, burnning numbers.

    Returns:
        tuple: (Evidence, sample trace).

    Notes:
        MATLAB version: Xiaohao Cai (21/02/2019)

        Python version: Matthew Price (9/05/2022)
    """
    sigma = options["sigma"]
    Phi = params["Phi"]
    y = params["y"]

    lg.info_log("Constructing lambda functions for resampling projections...")

    # Simulation setup
    # Use backward-forward splitting to approximate proxPi using proxH and gradF
    driftIniN = lambda X, delta, lamb: np.real(
        (1 - delta / (2 * lamb)) * X + delta / (2 * lamb) * proxH(X, lamb)
    )
    drift = lambda X, delta, lamb, tau: np.real(
        (1 - delta / lamb) * X
        + delta
        / (2 * lamb)
        * (proxH(X, lamb) + proxB(X, np.sqrt(tau * 2 * sigma**2)))
    )

    # Initialize variables
    delta = options[
        "delta"
    ]  # delta controls the proposal variance, the step-length and Moreau approximation
    lamb = 5 * delta  # lamb \in [4*delta, 10*delta]
    Xcur = X0  # set initial state as current state
    tau_0 = -LikeliL(Xcur) * 1e-1

    lg.info_log("Allocating memory and populating initial live-samples...")

    # Initialise arrays to store samples
    # Indexing: sample, likelihood, weights
    NumLiveSetSamples = options["samplesL"]
    NumDiscardSamples = options["samplesD"]

    Xtrace = {}

    Xtrace["LiveSet"] = np.zeros((NumLiveSetSamples, Xcur.shape[0], Xcur.shape[1]))
    Xtrace["LiveSetL"] = np.zeros(NumLiveSetSamples)

    Xtrace["Discard"] = np.zeros((NumDiscardSamples, Xcur.shape[0], Xcur.shape[1]))
    Xtrace["DiscardL"] = np.zeros(NumDiscardSamples)
    Xtrace["DiscardW"] = np.zeros(NumDiscardSamples)
    Xtrace["DiscardPostProb"] = np.zeros(NumDiscardSamples)

    # Generate initialisation
    j = 0
    for ii in tqdm(range(200), desc="ProxNest || Initialise"):
        # P-ULA -- MARKOV CHAIN generating initialisation
        Xcur = drift(Xcur, delta, lamb, tau_0) + np.sqrt(delta) * np.random.randn(
            Xcur.shape[0], Xcur.shape[1]
        )

    # Obtain samples from priors
    for ii in tqdm(
        range(2, NumLiveSetSamples * options["thinning"] + options["burn"]),
        desc="ProxNest || Populate",
    ):

        # P-ULA -- MARKOV CHAIN generating live samples
        Xcur = driftIniN(Xcur, delta, lamb) + np.sqrt(delta) * np.random.randn(
            Xcur.shape[0], Xcur.shape[1]
        )

        # Save sample (with thinning)
        if (ii > options["burn"]) and not (
            (ii - options["burn"]) % options["thinning"]
        ):
            # Record the current sample in the live set and its likelihood
            Xtrace["LiveSet"][j] = Xcur
            Xtrace["LiveSetL"][j] = LikeliL(Xcur)

            j += 1

    lg.info_log("Executing primary nested resampling iterations...")

    # Reorder samples TODO: Make this more efficient!
    Xtrace["LiveSet"], Xtrace["LiveSetL"] = resampling.reorder_samples(
        Xtrace["LiveSet"], Xtrace["LiveSetL"]
    )

    # Update samples using the proximal nested sampling technique
    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Sample"):
        # Compute the smallest threshould wrt live samples' likelihood
        tau = -Xtrace["LiveSetL"][-1]  # - 1e-2

        # Randomly select a sample in the live set as a starting point
        indNewSample = (
            np.floor(np.random.rand() * (NumLiveSetSamples - 1)).astype(int) - 1
        )
        Xcur = Xtrace["LiveSet"][indNewSample]

        # Generate a new sample with likelihood larger than given threshould
        Xcur = drift(Xcur, delta, lamb, tau) + np.sqrt(delta) * np.random.randn(
            Xcur.shape[0], Xcur.shape[1]
        )

        # check if the new sample is inside l2-ball (metropolis-hasting); if
        # not, force the new sample into L2-ball
        if np.sum(np.sum(np.abs(y - Phi.dir_op(Xcur)) ** 2)) > tau * 2 * sigma**2:
            Xcur = proxB(Xcur, np.sqrt(tau * 2 * sigma**2))

        # Record the sample discarded and its likelihood
        Xtrace["Discard"][k] = Xtrace["LiveSet"][-1]
        Xtrace["DiscardL"][k] = Xtrace["LiveSetL"][-1]

        # Add the new sample to the live set and its likelihood
        Xtrace["LiveSet"][-1] = Xcur
        Xtrace["LiveSetL"][-1] = LikeliL(Xcur)

        # Reorder the live samples TODO: Make this more efficient!
        Xtrace["LiveSet"], Xtrace["LiveSetL"] = resampling.reorder_samples(
            Xtrace["LiveSet"], Xtrace["LiveSetL"]
        )

    lg.info_log(
        "Estimating Bayesian evidence (with variance), posterior probabilies, and posterior mean..."
    )

    # Bayesian evidence calculation
    BayEvi = np.zeros(2)
    Xtrace["DiscardW"][0] = 1 / NumLiveSetSamples

    # Compute the sample weight
    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Compute Weights"):
        Xtrace["DiscardW"][k] = np.exp(-(k + 1) / NumLiveSetSamples)

    # Compute the volumn length for each sample using trapezium rule
    discardLen = np.zeros(NumDiscardSamples)
    discardLen[0] = (1 - np.exp(-2 / NumLiveSetSamples)) / 2

    for i in tqdm(
        range(1, NumDiscardSamples - 1), desc="ProxNest || Trapezium Integrate"
    ):
        discardLen[i] = (Xtrace["DiscardW"][i - 1] - Xtrace["DiscardW"][i + 1]) / 2

    discardLen[-1] = (
        np.exp(-(NumDiscardSamples - 1) / NumLiveSetSamples)
        - np.exp(-(NumDiscardSamples + 1) / NumLiveSetSamples)
    ) / 2
    # volume length of the last discarded sample

    liveSampleLen = np.exp(-(NumDiscardSamples) / NumLiveSetSamples)
    # volume length of the living sample

    # Apply the disgarded sample for Bayesian evidence value computation
    vecDiscardLLen = Xtrace["DiscardL"] + np.log(discardLen)

    # Apply the final live set samples for Bayesian evidence value computation
    vecLiveSetLLen = Xtrace["LiveSetL"] + np.log(liveSampleLen / NumLiveSetSamples)

    # #   ------- Way 1: using discarded and living samples --------
    # # Get the maximum value of the exponents for all the samples
    # maxAllSampleLLen = max(max(vecDiscardLLen),max(vecLiveSetLLen))

    # # Compute the Bayesian evidence value using discarded and living samples
    # BayEvi[0] = maxAllSampleLLen + np.log(np.sum(np.exp(vecDiscardLLen-maxAllSampleLLen)) + np.sum(np.exp(vecLiveSetLLen-maxAllSampleLLen)))

    # ------- Way 2: using discarded samples --------
    # Get the maximum value of the exponents for the discarded samples
    maxDiscardLLen = np.max(vecDiscardLLen)

    # Compute the Bayesian evidence value using discarded and living samples
    BayEvi[0] = maxDiscardLLen + np.log(np.sum(np.exp(vecDiscardLLen - maxDiscardLLen)))

    # Extimate the error of the computed Bayesian evidence
    entropyH = 0

    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Estimate Variance"):
        temp1 = np.exp(Xtrace["DiscardL"][k] + np.log(discardLen[k]) - BayEvi[0])
        entropyH = entropyH + temp1 * (Xtrace["DiscardL"][k] - BayEvi[0])

    # Evaluate the evidence variance
    BayEvi[1] = np.sqrt(np.abs(entropyH) / NumLiveSetSamples)

    # Compute the posterior probability for each discarded sample
    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Compute Posterior Mean"):
        Xtrace["DiscardPostProb"][k] = np.exp(
            Xtrace["DiscardL"][k] + np.log(discardLen[k]) - BayEvi[0]
        )

    # Compute the posterior mean of the discarded samples -- optimal solution
    Xtrace["DiscardPostMean"] = np.zeros((Xcur.shape[0], Xcur.shape[1]))
    for k in range(NumDiscardSamples):
        Xtrace["DiscardPostMean"] += Xtrace["DiscardPostProb"][k] * Xtrace["Discard"][k]

    return BayEvi, Xtrace
