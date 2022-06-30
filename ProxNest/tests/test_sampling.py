import numpy as np
import pytest

import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators

def test_against_analytic_gaussian():
    """ Tests ProxNest against analytic Gaussian """
    
    # A simple identity forward model and redundant dictionary
    phi = operators.sensing_operators.Identity()
    psi = operators.sensing_operators.Identity()  
    sigma = 1
    iterations = 20
    delta = 1/2
    error = 0
    dim = 32
    image = np.random.rand(dim, 1)
    
    # Parameter dictionary associated with optimisation problem of resampling from the prior subject to the likelihood iso-ball
    params = utils.create_parameters_dict(
              y = image,                # Measurements i.e. data
            Phi = phi,                  # Forward model
        epsilon = 1e-3,                 # Radius of L2-ball of likelihood 
          tight = True,                 # Is Phi a tight frame or not?
             nu = 1,                    # Bound on the squared-norm of Phi
            tol = 1e-10,                # Convergence tolerance of algorithm
       max_iter = 200,                  # Maximum number of iterations
        verbose = 0,                    # Verbosity level
              u = 0,                    # Initial vector for the dual problem
            pos = True,                 # Positivity flag
        reality = True                  # Reality flag
    )

    # Options dictionary associated with the overall sampling algorithm
    options = utils.create_options_dict(
     samplesL = 2e3,                  # Number of live samples
     samplesD = 3e4,                  # Number of discarded samples 
     thinning = 1e1,                  # Thinning factor (to mitigate correlations)
        delta = 1e-2,                 # Discretisation stepsize
         burn = 1e2,                  # Number of burn in samples
        sigma = sigma                 # Noise standard deviation of degraded image
    )

    for iter in range(iterations):
        # Generate a vector drawn from a Uniform distribution
        image = np.random.rand(dim, 1)

        # Simulate some unit variance Gaussian noise on this random vector
        n = sigma*np.random.randn(dim, 1)
        image = image + n  

        params["y"] = image

        # Lambda functions to evaluate cost function
        LogLikeliL = lambda sol : - np.linalg.norm(image-phi.dir_op(sol))**2/(2*sigma**2)

        # Lambda function for L2-norm identity prior backprojection steps
        proxH = lambda x, T : x - 2*T*psi.adj_op(psi.dir_op(x))*2*delta

        # Lambda function for L2-ball likelihood projection during resampling
        proxB = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params)

        # Select a starting position
        X0 = np.abs(phi.adj_op(image))

        # Perform proximal nested sampling
        NS_BayEvi, NS_Trace = sampling.proximal_nested.ProxNestedSampling(X0, LogLikeliL, proxH, proxB, params, options)
        rescaled_evidence_estimate = NS_BayEvi[0] + np.log(np.pi/delta)*(dim/2)

        detPar = 1/(2*delta+1/sigma**2)
        ySquare= np.linalg.norm(image,'fro')**2
        BayEvi_Val_gt_log = np.log(np.sqrt(((2*np.pi)**dim)*(detPar**dim))) + (-ySquare/(2*sigma**2)) + (detPar/2)*(ySquare/sigma**4)

        error += (rescaled_evidence_estimate - BayEvi_Val_gt_log)/BayEvi_Val_gt_log
    
    assert error / iterations == pytest.approx(0, abs=1, rel=1)