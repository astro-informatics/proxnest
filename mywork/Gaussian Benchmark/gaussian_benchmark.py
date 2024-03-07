import numpy as np
import _pickle as pickle
import argparse
import os
import time
import matplotlib.pyplot as plt
import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators


def main(args):

    dimensions = np.linspace(args.dims[0], args.dims[1], args.dims[2], dtype=int)
    print(f"DIMENSIONS: {dimensions}")

    if not args.save:
        save_dir = os.getcwd()+"/"+args.label+"/"+time.strftime("%Y_%m_%d")+"/"+time.strftime("%H_%M_%S")+"/"
        os.makedirs(save_dir)
    else:
        save_dir = args.save+args.label
    
    predictions = []
    residuals = []

    for i in dimensions:
        # Dimension of Gaussian
        dimension = i
        print(f"Calculating for dimension: {dimension}")

        # A simple identity forward model and redundant dictionary
        phi = operators.sensing_operators.Identity()
        psi = operators.sensing_operators.Identity()

        # Generate a vector drawn from a Uniform distribution
        if args.seed:
            np.random.seed(int(args.seed))
        image = np.random.rand(dimension, 1)

        # Simulate some unit variance Gaussian noise on this random vector
        sigma = 1
        n = sigma*np.random.randn(dimension, 1)
        image = image + n

        if args.seed:
            np.random.seed()

        # Define a regularisation parameter (this should be tuned for a given problem)
        delta = 1/2 # This is mu in the paper! == 1/2 for Gaussian Benchmark in Cai et. al.

        # Parameter dictionary associated with optimisation problem of resampling from the prior subject to the likelihood iso-ball
        params = utils.create_parameters_dict(
                y = image,                # Measurements i.e. data
                Phi = phi,                  # Forward model
            epsilon = 1e-3,                 # Radius of L2-ball of likelihood                   ## Not used
            tight = True,                 # Is Phi a tight frame or not?
                nu = 1,                    # Bound on the squared-norm of Phi                   ## Should be 1
                tol = 1e-10,                # Convergence tolerance of algorithm
            max_iter = 200,                  # Maximum number of iterations
            verbose = 0,                    # Verbosity level
                u = 0,                    # Initial vector for the dual problem                 ## Not used
                pos = True,                 # Positivity flag                                   ## True in Cai et. al.
            reality = True                  # Reality flag
        )

        # Options dictionary associated with the overall sampling algorithm
        options = utils.create_options_dict(
            samplesL = 2e2,                  # Number of live samples                           ## 2e2 in Cai et. al.
            samplesD = 3e3,                  # Number of discarded samples                      ## 3e3 in Cai et. al.
            thinning = 1e1,                  # Thinning factor (to mitigate correlations)       ## 1e1 in Cai et. al.
            delta = 1e-2,                 # Discretisation stepsize
                burn = 1e2,                  # Number of burn in samples
            sigma = sigma                 # Noise standard deviation of degraded image          ## Should be 1
        )

        # Lambda functions to evaluate cost function
        LogLikeliL = lambda sol : - np.linalg.norm(image-phi.dir_op(sol))**2/(2*sigma**2)

        # Lambda function for L2-norm identity prior backprojection steps
        proxH = lambda x, T : x - 2*T*psi.adj_op(psi.dir_op(x))*2*delta

        # Lambda function for L2-ball likelihood projection during resampling
        proxB = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params)

        # Select a starting position
        X0 = np.abs(phi.adj_op(image)) # image has negative values, so this is different to just image

        # Perform proximal nested sampling
        #np.seterr(invalid='raise')
        NS_BayEvi, _ = sampling.proximal_nested.ProxNestedSampling(X0, LogLikeliL, proxH, proxB, params, options)
        rescaled_evidence_estimate = NS_BayEvi[0] + np.log(np.pi/delta)*(dimension/2) # == log_V_x_Z

        detPar = 1/(2*delta+1/sigma**2)
        ySquare= np.linalg.norm(image,'fro')**2

        # Ground truth
        BayEvi_Val_gt_log = np.log(np.sqrt(((2*np.pi)**dimension)*(detPar**dimension))) + (-ySquare/(2*sigma**2)) + (detPar/2)*(ySquare/sigma**4)

        #V = np.sqrt(((2*np.pi)**dimension)/((2*delta)**dimension)) # Appendix A1 Cai et. al.

        predictions.append([dimension,BayEvi_Val_gt_log,rescaled_evidence_estimate])

        res = np.abs(rescaled_evidence_estimate - BayEvi_Val_gt_log)
        residuals.append((dimension,res))
    
    predictions = np.array(predictions)

    plt.figure(dpi=300)
    plt.plot(dimensions, predictions[:,2], color="r", marker="x", linewidth=0.5, markersize=3)
    plt.plot(dimensions, predictions[:,1], color="black", linewidth=0.5)
    plt.ylim(-250,100)
    plt.xlim(0,200)
    plt.scatter(dimensions, log_V_x_Z, color="r", marker="x", linewidth=0.5)
    plt.xlabel("Dimensions")
    plt.ylabel(r"$\log (V \times \mathcal{Z})$")
    plt.savefig(save_dir+"plot")
    
    with open(save_dir+"config.txt", 'w') as file: 
        print(dict(list(params.items())[3:]), file=file)
        print(options, file=file)
    
    np.savetxt(save_dir+"residuals.csv", residuals, delimiter=",")
    np.savetxt(save_dir+"predictions.csv", predictions, delimiter=",")

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ProxNest Gaussian Benchmark')
    parser.add_argument(
        '-d', '--dims',
        required=True,
        type=int,
        nargs='+',
        help="min, max dimensions followed number of samples"
    )

    parser.add_argument(
        '-s', '--save',
        required=False,
        type=str,
        help="Folder to save results to"
    )

    parser.add_argument(
        '-r', '--runs',
        required=False,
        default=1,
        type=int,
        help="Number of runs to perform"
    )

    parser.add_argument(
        '-l', '--label',
        required=False,
        default="",
        type=str,
        help="Label to add to run output folder"
    )

    parser.add_argument(
        '--seed',
        required=False,
        type=str,
        default=None,
        help="Seed for reproducibility"
    )
    
    args = parser.parse_args()

    for i in range(args.runs):
        main(args)