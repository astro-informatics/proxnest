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
import pandas as pd


def main(args):
    start = time.time()
    dimensions = np.linspace(args.dims[0], args.dims[1], args.dims[2], dtype=int)
    print(f"DIMENSIONS: {dimensions}")

    if not args.save:
        save_dir = os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
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
             epsilon = 1e-3,                 # Radius of L2-ball of likelihood                            ## Not used
               tight = False,                # Is Phi a tight frame or not?
                  nu = 1,                    # Bound on the squared-norm of Phi                           ## Should be 1
                 tol = 1e-10,                # Convergence tolerance of algorithm
            max_iter = 200,                  # Maximum number of iterations                               ## 200 in src_proxnest
             verbose = 0,                    # Verbosity level
                   u = 0,                    # Initial vector for the dual problem                        ## Not used
                 pos = False,                # Positivity flag                                            ## True in Cai et. al.
             reality = False                 # Reality flag
        )

        # Options dictionary associated with the overall sampling algorithm
        options = utils.create_options_dict(
            samplesL = 1e3,                  # Number of live samples                                      ## 2e2 in Cai et. al.
            samplesD = 1e4,                  # Number of discarded samples                                 ## 3e3 in Cai et. al.
    lv_thinning_init = a,                  # Thinning factor in initialisation
         lv_thinning = b,                  # Thinning factor in the sample update                        ## 1e1 in Cai et. al.
             MH_step = False,                 # Metropolis-Hastings step
    warm_start_coeff = c,                  # Warm start coefficient
               delta = 1e-2,                 # Discretisation stepsize                                     ## 10*1e-1 in src_proxnest
                lamb = d * 1e-2,                 # Moreau-Yosida approximation parameter, usually `5 * delta`
                burn = 1e2,                  # Number of burn in samples                                   ## 1e2 in src_proxnest
               sigma = sigma                 # Noise standard deviation of degraded image                  ## Should be 1
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
    
    end = time.time()
    elapsed = end - start
    predictions = np.array(predictions)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.figure(dpi=300)
    plt.plot(dimensions, predictions[:,2], color="tomato", marker="x", linewidth=0.5, markersize=2, 
             label="ProxNest")
    plt.plot(dimensions, predictions[:,1], color="black", linewidth=0.5, label="Ground truth")
    plt.ylim(-10,60)
    plt.xlim(0,200)
    plt.xlabel("Dimensions")
    plt.ylabel(r"$\log (V \times \mathcal{Z})$")
    plt.title(f"delta = {options["delta"]}, "+"samplesL = {:.1e}, samplesD = {:.1e}, MH_step = {}, \n lv_thinning_init = {:.1e}, lv_thinning = {:.1e}, warm_start_coeff = {:.1e}, lamb = {:.1e}, time = {:.1e}"
              .format(options["samplesL"], options["samplesD"], options["MH_step"], options["lv_thinning_init"], options["lv_thinning"], options["warm_start_coeff"], options["lamb"], elapsed), 
              fontsize=7)
    plt.savefig(save_dir+"plot")
    
    with open(save_dir+"config.json", 'w') as file: 
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

    parser.add_argument(
        '-p, --params',
        required=False,
        type=str,
        default=None,
        nargs='+',
        help="Params to test"
    )
    
    args = parser.parse_args()

    
    for i in range(args.runs):
        defaults = {
            "lv_thinning_init": 1e1,
            "lv_thinning": 1e1,
            "warm_start_coeff": 1e-1,
            "lamb": 5
        }
        a, b, c, d = list(defaults.values())
        param = "baseline_new2"
        value = ""
        main(args)
        baseline = pd.read_csv(os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"+predictions.csv, names=("dim", "gt", "pred"))

        if "MH_step" in args.params:
            mh = True

        param = "lv_thinning_init"
        if param in args.params:
            lv_thinning_inits = np.logspace(np.log10(1e0, np.log10(1e3), num=10, endpoint=True, base=10.0))
            lv_thinnings_init.remove(10)

            color = cm.rainbow(np.linspace(0, 1, len(lv_thinning_inits)))
            plt.figure(dpi=300)
            for n, a in enumerate(lv_thinning_inits):
                value = a
                main(args)
                save_dir = os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"
                df = pd.read_csv(save_dir+"predictions.csv",  names=("dim", "gt", "pred"))
                if n == 0:
                    plt.plot(df["dim"], df["gt"], color='black', linewidth=0.5)
                    plt.plot(baseline["dim"], baseline["pred"], linewidth=0.5, linestyle="dashed", color="black")
                plt.plot(df["dim"], df["pred"], color=color[n], markersize=2, marker="x", label=str(a))

            plt.ylim(-10,60)
            plt.xlim(0,200)
            plt.xlabel("Dimensions")
            plt.ylabel(r"$\log (V \times \mathcal{Z})$")
            plt.legend()
            plt.title(f"Varying {param}")

            a = defaults["lv_thinning_init"]

        param = "lv_thinning"
        if param in args.params:
            lv_thinnings = np.logspace(np.log10(1e0, np.log10(1e2), num=10, endpoint=True, base=10.0))

            color = cm.rainbow(np.linspace(0, 1, len(lv_thinnings)))
            plt.figure(dpi=300)
            for n,b in enumerate(lv_thinnings):
                value = b
                main(args)
                save_dir = os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"
                df = pd.read_csv(save_dir+"predictions.csv",  names=("dim", "gt", "pred"))
                if n == 0:
                    plt.plot(df["dim"], df["gt"], color='black', linewidth=0.5)
                    plt.plot(baseline["dim"], baseline["pred"], linewidth=0.5, linestyle="dashed", color="black")
                plt.plot(df["dim"], df["pred"], color=color[n], markersize=2, marker="x", label=str(b))

            plt.ylim(-10,60)
            plt.xlim(0,200)
            plt.xlabel("Dimensions")
            plt.ylabel(r"$\log (V \times \mathcal{Z})$")
            plt.legend()
            plt.title(f"Varying {param}")

            b = defaults["lv_thinning"]

        param = "warm_start_coeff"    
        if param in args.params:
            warm_start_coeff_list = np.logspace(np.log10(1e-2), np.log10(5e1), num=20, endpoint=True, base=10.0)
            warm_start_coeff_list = np.append(warm_start_coeff_list, [1e0, 5e0, 1e1])
            warm_start_coeff_list.sort()

            color = cm.rainbow(np.linspace(0, 1, len(warm_start_coeff_list)))
            plt.figure(dpi=300)
            for n,c in enumerate(warm_start_coeff_list):
                value = c
                main(args)
                save_dir = os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"
                df = pd.read_csv(save_dir+"predictions.csv",  names=("dim", "gt", "pred"))
                if n == 0:
                    plt.plot(df["dim"], df["gt"], color='black', linewidth=0.5)
                    plt.plot(baseline["dim"], baseline["pred"], linewidth=0.5, linestyle="dashed", color="black")
                plt.plot(df["dim"], df["pred"], color=color[n], markersize=2, marker="x", label=str(c))

            plt.ylim(-10,60)
            plt.xlim(0,200)
            plt.xlabel("Dimensions")
            plt.ylabel(r"$\log (V \times \mathcal{Z})$")
            plt.legend()
            plt.title(f"Varying {param}")

            c = defaults["warm_start_coeff"]
        
        param = "lamb"
        if param in args.params:
            lambs = [2,4,6,7,8,9,10]

            color = cm.rainbow(np.linspace(0, 1, len(lambs)))
            plt.figure(dpi=300)
            for n,d in enumerate(lambs):
                value = d
                main(args)
                save_dir = os.getcwd()+"/"+args.label+"/"+param+"/"+value+"/"
                df = pd.read_csv(save_dir+"predictions.csv",  names=("dim", "gt", "pred"))
                if n == 0:
                    plt.plot(df["dim"], df["gt"], color='black', linewidth=0.5)
                    plt.plot(baseline["dim"], baseline["pred"], linewidth=0.5, linestyle="dashed", color="black")
                plt.plot(df["dim"], df["pred"], color=color[n], markersize=2, marker="x", label=str(d))

            plt.ylim(-10,60)
            plt.xlim(0,200)
            plt.xlabel("Dimensions")
            plt.ylabel(r"$\log (V \times \mathcal{Z})$")
            plt.legend()
            plt.title(f"Varying {param}")

            d = defaults["lamb"]
