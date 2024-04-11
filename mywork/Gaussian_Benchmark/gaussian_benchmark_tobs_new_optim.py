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
    start = time.time()
    dimensions = np.linspace(args.dims[0], args.dims[1], args.dims[2], dtype=int)
    print(f"DIMENSIONS: {dimensions}")
    print(f"RUNS: {args.runs}")

    if not args.save:
        save_dir = (
            os.getcwd()
            + "/"
            + args.label
            + "/"
            + time.strftime("%Y_%m_%d")
            + "/"
            + time.strftime("%H_%M_%S")
            + "/"
        )
        os.makedirs(save_dir)
    else:
        save_dir = args.save + args.label

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # predictions = []
    # residuals = []

    array_predictions = np.zeros((args.runs, dimensions.shape[0], 3))
    array_residuals = np.zeros((args.runs, dimensions.shape[0], 2))

    for it_run in range(args.runs):

        for it_dim in range(dimensions.shape[0]):
            # Dimension of Gaussian
            dimension = dimensions[it_dim]
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
            n = sigma * np.random.randn(dimension, 1)
            image = image + n

            if args.seed:
                np.random.seed()

            # Define a regularisation parameter (this should be tuned for a given problem)
            delta = (
                1 / 2
            )  # This is mu in the paper! == 1/2 for Gaussian Benchmark in Cai et. al.

            # Parameter dictionary associated with optimisation problem of resampling from the prior subject to the likelihood iso-ball
            params = utils.create_parameters_dict(
                y=image,  # Measurements i.e. data
                Phi=phi,  # Forward model
                epsilon=1e-3,  # Radius of L2-ball of likelihood                            ## Not used
                tight=False,  # Is Phi a tight frame or not?
                nu=1,  # Bound on the squared-norm of Phi                           ## Should be 1
                tol=1e-10,  # Convergence tolerance of algorithm
                max_iter=200,  # Maximum number of iterations                               ## 200 in src_proxnest
                verbose=0,  # Verbosity level
                u=0,  # Initial vector for the dual problem                        ## Not used
                pos=False,  # Positivity flag                                            ## True in Cai et. al.
                reality=False,  # Reality flag
            )

            # Options dictionary associated with the overall sampling algorithm
            options = utils.create_options_dict(
                samplesL=1e3,  # Number of live samples                                      ## 2e2 in Cai et. al.
                samplesD=1e4,  # Number of discarded samples                                 ## 3e3 in Cai et. al.
                lv_thinning_init=1e1,  # Thinning factor in initialisation
                lv_thinning=1e0,  # Thinning factor in the sample update                        ## 1e1 in Cai et. al.
                MH_step=False,  # Metropolis-Hastings step
                warm_start_coeff=1e0,  # Warm start coefficient
                delta=1e-2,  # Discretisation stepsize                                     ## 10*1e-1 in src_proxnest
                lamb=5e-2,  # Moreau-Yosida approximation parameter, usually `5 * delta`
                burn=1e2,  # Number of burn in samples                                   ## 1e2 in src_proxnest
                sigma=sigma,  # Noise standard deviation of degraded image                  ## Should be 1
            )

            # Lambda functions to evaluate cost function
            LogLikeliL = lambda sol: -np.linalg.norm(image - phi.dir_op(sol)) ** 2 / (
                2 * sigma**2
            )

            # Lambda function for L2-norm identity prior backprojection steps
            proxH = lambda x, T: x - 2 * T * psi.adj_op(psi.dir_op(x)) * 2 * delta

            # Lambda function for L2-ball likelihood projection during resampling
            # proxB = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params)
            proxB = lambda x, tau: optimisations.l2_ball_proj.new_sopt_fast_proj_B2(
                x, tau, params
            )

            # Select a starting position
            X0 = np.abs(
                phi.adj_op(image)
            )  # image has negative values, so this is different to just image

            # Perform proximal nested sampling
            # np.seterr(invalid='raise')
            NS_BayEvi, _ = sampling.proximal_nested.ProxNestedSampling(
                X0, LogLikeliL, proxH, proxB, params, options
            )
            rescaled_evidence_estimate = NS_BayEvi[0] + np.log(np.pi / delta) * (
                dimension / 2
            )  # == log_V_x_Z

            detPar = 1 / (2 * delta + 1 / sigma**2)
            ySquare = np.linalg.norm(image, "fro") ** 2
            # Ground truth
            # BayEvi_Val_gt_log = np.log(np.sqrt(((2*np.pi)**dimension)*(detPar**dimension))) + (-ySquare/(2*sigma**2)) + (detPar/2)*(ySquare/sigma**4)
            BayEvi_Val_gt_log = (
                0.5 * dimension * np.log(2 * np.pi * detPar)
                + (-ySquare / (2 * sigma**2))
                + (detPar / 2) * (ySquare / sigma**4)
            )

            # V = np.sqrt(((2*np.pi)**dimension)/((2*delta)**dimension)) # Appendix A1 Cai et. al.

            # predictions.append([dimension,BayEvi_Val_gt_log,rescaled_evidence_estimate])
            array_predictions[it_run, it_dim, :] = np.array(
                [dimension, BayEvi_Val_gt_log, rescaled_evidence_estimate]
            )

            res = np.abs(rescaled_evidence_estimate - BayEvi_Val_gt_log)
            # residuals.append([dimension, res])
            array_residuals[it_run, it_dim, :] = np.array([dimension, res])

    end = time.time()
    elapsed = end - start

    # predictions = np.array(predictions)

    mean_predictions = np.mean(array_predictions, axis=0)
    std_dev_predictions = np.std(array_predictions, axis=0)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.figure(dpi=200)
    plt.errorbar(
        x=dimensions,
        y=mean_predictions[:, 2],
        yerr=std_dev_predictions[:, 2],
        color="tomato",
        marker="x",
        linewidth=0.5,
        markersize=2,
        label="ProxNest",
    )
    plt.plot(
        dimensions,
        mean_predictions[:, 1],
        color="black",
        marker="o",
        linewidth=0.5,
        markersize=2,
        label="Ground truth",
    )
    plt.ylim(0, np.max(mean_predictions[:, 1:]) + 10)
    plt.xlim(0, args.dims[1] + 10)
    plt.xlabel("Dimensions")
    plt.ylabel(r"$\log (V \times \mathcal{Z})$")
    title_str = "delta = {:.1e}, runs = {:d}, samplesL = {:.1e}, samplesD = {:.1e}, MH_step = {}, \n lv_thinning_init = {:.1e}, lv_thinning = {:.1e}, warm_start_coeff = {:.1e}, lamb = {:.1e}, time = {:.1e}".format(
        options["delta"],
        args.runs,
        options["samplesL"],
        options["samplesD"],
        options["MH_step"],
        options["lv_thinning_init"],
        options["lv_thinning"],
        options["warm_start_coeff"],
        options["lamb"],
        elapsed,
    )
    plt.title(title_str, fontsize=7)
    plt.savefig(save_dir + "plot1.pdf")
    plt.close()

    with open(save_dir + "config.txt", "w") as file:
        print(dict(list(params.items())[3:]), file=file)
        print(options, file=file)

    with open(save_dir + "predictions.csv", "ab") as f:
        for it_dim in range(dimensions.shape[0]):
            # for it_run in range(args.runs):
            np.savetxt(f, array_predictions[:, it_dim, :], delimiter=",")

    with open(save_dir + "mean_predictions.csv", "ab") as f:
        np.savetxt(f, mean_predictions, delimiter=",")

    with open(save_dir + "residuals.csv", "ab") as f:
        for it_dim in range(dimensions.shape[0]):
            # for it_run in range(args.runs):
            np.savetxt(f, array_residuals[:, it_dim, :], delimiter=",")

    # np.savetxt(save_dir+"residuals.csv", residuals, delimiter=",")
    # np.savetxt(save_dir+"predictions.csv", predictions, delimiter=",")


if __name__ == "__main__":
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description="ProxNest Gaussian Benchmark")
    parser.add_argument(
        "-d",
        "--dims",
        required=True,
        type=int,
        nargs="+",
        help="min, max dimensions followed number of samples",
    )

    parser.add_argument(
        "-s", "--save", required=False, type=str, help="Folder to save results to"
    )

    parser.add_argument(
        "-r",
        "--runs",
        required=False,
        default=1,
        type=int,
        help="Number of runs to perform",
    )

    parser.add_argument(
        "-l",
        "--label",
        required=False,
        default="",
        type=str,
        help="Label to add to run output folder",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=str,
        default=None,
        help="Seed for reproducibility",
    )

    args = parser.parse_args()

    # for i in range(args.runs):
    main(args)
