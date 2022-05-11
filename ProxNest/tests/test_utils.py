import ProxNest.utils as utils
import pytest


def test_parameter_dict_creation():

    y = 0
    epsilon = 1e-3
    tight = True
    nu = 1
    tol = 1e-3
    max_iter = 200
    verbose = 1
    u = 0
    pos = False
    reality = False
    rel_obj = 0

    params = utils.create_parameters_dict(
        y=y,
        epsilon=epsilon,
        tight=tight,
        nu=nu,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        u=u,
        pos=pos,
        reality=reality,
        rel_obj=rel_obj,
    )

    assert params["y"] == y
    assert params["epsilon"] == epsilon
    assert params["tight"] == tight
    assert params["nu"] == nu
    assert params["tol"] == tol
    assert params["max_iter"] == max_iter
    assert params["verbose"] == verbose
    assert params["u"] == u
    assert params["reality"] == reality
    assert params["pos"] == pos
    assert params["rel_obj"] == rel_obj


def test_options_dict_creation():

    samplesL = 1e3
    samplesD = 1e4
    thinning = 1e2
    delta = 1e-8
    burn = 1e2
    sigma = 1

    options = utils.create_options_dict(
        samplesL=samplesL,
        samplesD=samplesD,
        thinning=thinning,
        delta=delta,
        burn=burn,
        sigma=sigma,
    )

    assert options["samplesL"] == samplesL
    assert options["samplesD"] == samplesD
    assert options["thinning"] == thinning
    assert options["delta"] == delta
    assert options["burn"] == burn
    assert options["sigma"] == sigma
