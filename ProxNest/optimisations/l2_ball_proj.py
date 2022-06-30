import numpy as np


def sopt_fast_proj_B2(x, tau, params):
    r"""Fast projection algorithm onto the :math:`\ell_2`-ball.

    Compute the projection onto the :math:`\ell_2` ball, i.e. solve

    .. math::

        z^* = \min_{z} ||x - z||_2^2   s.t.  ||y - \Phi z||_2 < \tau

    where :math:`x` is the input vector and the solution :math:`z^*` is returned as sol.

    Args:
        x (np.ndarray): A sample position :math:`x` in the posterior space.

        tau (float): Radius of likelihood :math:`\ell_2`-ball.

        params (dict): Dictionary of parameters defining the optimisation.

    Returns:
        np.ndarray: Optimal solution :math:`z^*` of proximal projection.

    Notes:
        [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for optimization problems in sparse recovery" , IEEE ICIP, Cairo, Egypt, 2009.

        [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences 2 (2009), no. 1, 183--202.
    """
    # Lambda function for scaling, used for tight frames only
    sc = lambda z: z*np.minimum(tau / np.linalg.norm(z), 1)

    # TIGHT FRAMES
    if (params["tight"]) and (params["pos"] or params["reality"]):

        temp = params["Phi"].dir_op(x) - params["y"]
        sol = x + 1 / params["nu"] * params["Phi"].adj_op(sc(temp) - temp)
        crit_B2 = "TOL_EPS"
        iter = 0
        u = 0

    # NON-TIGHT FRAMES
    else:

        # Initializations
        sol = x
        # u = params['u']
        u = params["Phi"].dir_op(sol)
        v = u
        iter = 1
        told = 1

        # Tolerance onto the L2 ball
        epsilon_low = tau / (1 + params["tol"])
        epsilon_up = tau / (1 - params["tol"])

        # Check if we are in the L2 ball
        dummy = params["Phi"].dir_op(sol)
        norm_res = np.linalg.norm(params["y"] - dummy, 2)
        if norm_res <= epsilon_up:
            crit_B2 = "TOL_EPS"
            true = 0

        # Projection onto the L2-ball
        if params["verbose"] > 1:
            print("  Proj. B2:")

        while 1:

            # Residual
            res = params["Phi"].dir_op(sol) - params["y"]
            norm_res = np.linalg.norm(res)

            # Scaling for the projection
            res = u * params["nu"] + res
            norm_proj = np.linalg.norm(res)

            # Log
            if params["verbose"] > 1:
                print(
                    "   Iter {}, epsilon = {}, ||y - Phi(x)||_2 = {}".format(
                        iter, tau, norm_res
                    )
                )

            # Stopping criterion
            if (norm_res >= epsilon_low) and (norm_res <= epsilon_up):
                crit_B2 = "TOL_EPS"
                break
            elif iter >= params["max_iter"]:
                crit_B2 = "MAX_IT"
                break

            # Projection onto the L2 ball
            t = (1 + np.sqrt(1 + 4 * told**2)) / 2
            ratio = np.minimum(1, tau / norm_proj)
            u = v
            v = 1 / params["nu"] * (res - res * ratio)
            u = v + (told - 1) / t * (v - u)

            # Current estimate
            sol = x - params["Phi"].adj_op(u)

            # Projection onto the non-negative orthant (positivity constraint)
            if params["pos"]:
                sol = np.real(sol)
                sol[sol < 0] = 0

            # Projection onto the real orthant (reality constraint)
            if params["reality"]:
                sol = np.real(sol)

            # Increment iteration labels
            told = t
            iter = iter + 1

    # Log after the projection onto the L2-ball
    if params["verbose"] >= 1:
        temp = params["Phi"].dir_op(sol)
        print(
            "  Proj. B2: epsilon = {}, ||y - Phi(x)||_2 = {}, {}, iter = {}".format(
                tau, np.linalg.norm(params["y"] - temp), crit_B2, iter
            )
        )

    return sol
