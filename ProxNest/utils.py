def create_parameters_dict(
    y=0,
    Phi=None,
    Psi=None,
    epsilon=1e-3,
    tight=True,
    nu=1,
    tol=1e-3,
    max_iter=200,
    verbose=1,
    u=0,
    pos=False,
    reality=False,
    l1weights=1,
    rel_obj=0,
):
    r"""Compiles a dictionary of parameters for code simplicity

    Args:
        y (np.ndarray): Measurements (default = 0).

        Phi (linear operator): Sensing operator (default = None).

        Psi (linear operator): Redundant dictionary (default = None).

        epsilon (float): Radius of the :math:`\ell_2` ball (default = 1e-3).

        tight (bool): True if A is a tight frame or False otherwise (default = 1).

        nu (float): Bound on the squared-norm of the operator A, i.e. :math:`||A x||^2 <= \nu ||x||^2` (default = 1).

        tol (float): Tolerance, i.e. the algorithms stops if :math:`\epsilon/(1-tol) <= ||y - A z||_2 <= \epsilon/(1+tol)` (default = 1e-3).

        max_iter (int): Maximum number of iterations (default: 200).

        verbose (int): Verbosity level (0 = no log, 1 = summary at convergence, 2 = print main steps; default = 1).

        u (np.ndarray): Initial vector for the dual problem, same dimension as y (default = 0).

        pos (bool): Positivity flag (True = positive solution, False (default) general case).

        reality (bool): Reality flag (True = real solution, 0 (default) = general complex case).

        l1weights (np.ndarray): Reweighting of thresholding of :math:`\ell_1`-norm (default = 1).

        rel_obj (float): Stopping criterion for :math:`\ell_1` proximal sub-iterations (default = 0).

    Returns:
        dict: Dictionary of parameters.
    """
    params = {}
    params["y"] = y
    params["Phi"] = Phi
    params["Psi"] = Psi
    params["epsilon"] = epsilon
    params["tight"] = tight
    params["nu"] = nu
    params["tol"] = tol
    params["max_iter"] = int(max_iter)
    params["verbose"] = verbose
    params["u"] = u
    params["pos"] = pos
    params["reality"] = reality
    params["l1weights"] = l1weights
    params["rel_obj"] = rel_obj

    return params


def create_options_dict(
    samplesL=1e3, samplesD=1e4, thinning=1e2, delta=1e-8, burn=1e2, sigma=1
):
    r"""Compiles a dictionary of option parameters for sampling

    Args:
        samplesL (int): Number of live samples (default = 1e3).

        samplesD (int): Number of discarded samples (default = 1e4).

        thinning (int): Thinning factors (i.e. iterations per sample, default =1 1e2).

        delta (float): Discretisation stepsize (< Lipschitz constant of :math:`\nabla F`, default = 1e-8).

        burn (int): Number of burn in samples to be discarded (default = 1e2).

        sigma (float): Noise std of degraded image (default = 1).

    Returns:
        dict: Dictionary of sampling options.
    """
    options = {}
    options["samplesL"] = int(samplesL)
    options["samplesD"] = int(samplesD)
    options["thinning"] = int(thinning)
    options["delta"] = delta
    options["burn"] = int(burn)
    options["sigma"] = sigma

    return options
