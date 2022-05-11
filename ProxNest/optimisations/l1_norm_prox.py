import numpy as np
import ProxNest.operators as ops


def l1_norm_prox(x, lamb, params):
    r"""Proximal operator associated with L1 norm.

    Compute the L1 proximal operator, i.e. solve

    .. math::

        z^* = \min_{z} \frac{1}{2}||x - z||_2^2 + \lambda * ||\Psi^{\dagger} z||_1,

    where :math:`x` is the input vector and the solution :math:`z^*` is returned as sol.

    Args:
        x (np.ndarray): A sample position :math:`x` in the posterior space.

        lamb (float): Regularisation parameter.

        params (dict): Dictionary of parameters defining the optimisation.

    Returns:
        np.ndarray: Optimal solution :math:`z^*` of proximal operator.

    Notes:
        [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for optimization problems in sparse recovery" , IEEE ICIP, Cairo, Egypt, 2009.

        [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences 2 (2009), no. 1, 183--202.
    """

    # TIGHT FRAMES
    if (params["tight"]) and (params["pos"] or params["reality"]):

        temp = params["Psi"].dir_op(x)
        sol = x + 1 / params["nu"] * params["Psi"].adj_op(
            ops.proximal_operators.soft_thresh(temp, lamb * params["nu"] * params["l1weights"]) - temp
        )
        dummy = params["Psi"].dir_op(sol)
        norm_l1 = np.sum(params["l1weights"] * np.abs(dummy))
        crit_L1 = "REL_OBJ"
        iter_L1 = 1

    # NON TIGHT FRAME CASE OR CONSTRAINT INVOLVED
    else:

        # Initializations
        sol = x
        if params["pos"] or params["reality"]:
            sol = np.real(sol)

        dummy = params["Psi"].dir_op(sol)
        u_l1 = np.zeros(len(dummy))
        prev_obj = 0
        iter_L1 = 0

        # Soft-thresholding
        if params["verbose"] > 1:
            print("  Proximal L1 operator:")

        while 1:

            # L1 norm of the estimate
            norm_l1 = np.sum(params["l1weights"] * np.abs(dummy))
            obj = 0.5 * np.linalg.norm(x - sol, 2) ** 2 + lamb * norm_l1
            rel_obj = np.abs(obj - prev_obj) / obj

            # Log
            if params["verbose"] > 1:
                print(
                    "   Iter {}, prox_fval = {}, rel_fval = {}".format(
                        iter_L1, obj, rel_obj
                    )
                )

            # Stopping criterion
            if rel_obj < params["rel_obj"]:
                crit_L1 = "REL_OB"
                break
            elif iter_L1 >= params["max_iter"]:
                crit_L1 = "MAX_IT"
                break

            # Soft-thresholding
            res = u_l1 * params["nu"] + dummy
            dummy = ops.proximal_operators.soft_thresh(res, lamb * params["nu"] * params["l1weights"])
            u_l1 = 1 / params["nu"] * (res - dummy)
            sol = x - params["Psi"].adj_op(u_l1)

            if params["pos"]:
                sol = np.real(sol)
                sol[sol < 0] = 0

            if params["reality"]:
                sol = np.real(sol)

            # Update
            prev_obj = obj
            iter_L1 = iter_L1 + 1
            dummy = params["Psi"].dir_op(sol)

    # Log after the projection onto the L2-ball
    if params["verbose"] >= 1:
        print(
            "  prox_L1: prox_fval = {}, {}, iter = {}".format(norm_l1, crit_L1, iter_L1)
        )

    return sol
