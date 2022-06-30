.. image:: https://img.shields.io/badge/GitHub-ProxNest-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/proxnest
.. image:: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml
.. image:: https://github.com/astro-informatics/proxnest/actions/workflows/docs.yml/badge.svg
    :target: https://astro-informatics.github.io/proxnest
.. image:: https://codecov.io/gh/astro-informatics/proxnest/branch/main/graph/badge.svg?token=oGowwdoMRN
    :target: https://codecov.io/gh/astro-informatics/proxnest
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-2106.03646-orange.svg?style=flat
    :target: https://arxiv.org/abs/2106.03646
    
|logo| Proximal nested sampling for high-dimensional Bayesian model selection
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/ProxNestLogo.png" align="center" height="80" width="100">

``ProxNest`` is an open source, well tested and documented Python implementation of the *proximal nested sampling* algorithm (`Cai et al. 2022 <https://arxiv.org/pdf/2106.03646.pdf>`_) which is uniquely suited for sampling from very high-dimensional posteriors that are log-concave and potentially not smooth (*e.g.* Laplace priors). This is achieved by exploiting tools from proximal calculus and Moreau-Yosida regularisation (`Moreau 1962 <https://hal.archives-ouvertes.fr/hal-01867195/file/Fonctions_convexes_duales_points_proximaux_Moreau_CRAS_1962.pdf>`_) to efficiently sample from the prior subject to the hard likelihood constraint. The resulting Markov chain iterations include a gradient step, approximating (with arbitrary precision) an overdamped Langevin SDE that can scale to very high-dimensional applications.

Basic Usage
===========

The following is a straightforward example application to image denoising (Phi = I), regularised with Daubechies wavelets (DB6). 

.. code-block:: Python

    # Import relevant modules.
    import numpy as np 
    import ProxNest 

    # Load your data and set parameters.
    data = np.load(<path to your data.npy>)
    params = params    # Parameters of the prior resampling optimisation problem.
    options = options  # Options associated with the sampling strategy.

    # Construct your forward model (phi) and wavelet operators (psi).
    phi = ProxNest.operators.sensing_operators.Identity()
    psi = ProxNest.operators.wavelet_operators.db_wavelets(["db6"], 2, (dim, dim))

    # Define proximal operators for both your likelihood and prior.
    proxH = lambda x, T : ProxNest.operators.proximal_operators.l1_projection(x, T, delta, Psi=psi)
    proxB = lambda x, tau: ProxNest.optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params)

    # Write a lambda function to evaluate your likelihood term (here a Gaussian)
    LogLikeliL = lambda sol : - np.linalg.norm(y-phi.dir_op(sol), 'fro')**2/(2*sigma**2)

    # Perform proximal nested sampling
    BayEvi, XTrace = ProxNest.sampling.proximal_nested.ProxNestedSampling(
        np.abs(phi.adj_op(data)), LogLikeliL, proxH, proxB, params, options
        )

At this point you have recovered the tuple **BayEvi** and dict **Xtrace** which contain 

.. code-block:: python

    Live = options["samplesL"] # Number of live samples
    Disc = options["samplesD"] # Number of discarded samples

    # BayEvi is a tuple containing two values:
    BayEvi[0] = 'Estimate of Bayesian evidence (float).'
    BayEvi[1] = 'Variance of Bayesian evidence estimate (float).'

    # XTrace is a dictionary containing the np.ndarrays:
    XTrace['Liveset'] = 'Set of live samples (shape: Live, dim, dim).'
    XTrace['LivesetL'] = 'Likelihood of live samples (shape: Live).'

    XTrace['Discard'] = 'Set of discarded samples (shape: Disc, dim, dim).'
    XTrace['DiscardL'] = 'Likelihood of discarded samples (shape: Disc).'
    XTrace['DiscardW'] = 'Weights of discarded samples (shape: Disc).'

    XTrace['DiscardPostProb'] = 'Posterior probability of discarded samples (shape: Disc)'
    XTrace['DiscardPostMean'] = 'Posterior mean solution (shape: dim, dim)'

from which one can perform *e.g.* Bayesian model comparison.

Installation
============

Brief installation instructions are given below (for further details see the full installation documentation).  

Quick install (PyPi)
--------------------
The ``ProxNest`` package can be installed by running

.. code-block:: bash
    
    pip install ProxNest

Install from source (GitHub)
----------------------------
The ``ProxNest`` package can also be installed from source by running

.. code-block:: bash

    git clone https://github.com/astro-informatics/proxnest
    cd harmonic

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_proxnest.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

    pytest --black ProxNest/tests/

Contributors
============
`Matthew Price <https://cosmomatt.github.io>`_, `Xiaohao Cai <https://xiaohaocai.netlify.app>`_, `Jason McEwen <http://www.jasonmcewen.org>`_, `Marcelo Pereyra <https://www.macs.hw.ac.uk/~mp71/about.html>`_, and contributors.

Attribution
===========
A BibTeX entry for ``ProxNest`` is:

.. code-block:: 

     @article{Cai:ProxNest:2021, 
        author = {Cai, Xiaohao and McEwen, Jason~D. and Pereyra, Marcelo},
         title = {"High-dimensional Bayesian model selection by proximal nested sampling"},
       journal = {ArXiv},
        eprint = {arXiv:2106.03646},
          year = {2021}
     }

License
=======

``ProxNest`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/proxnest/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/proxnest/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     ProxNest
     Copyright (C) 2022 Matthew Price, Xiaohao Cai, Jason McEwen, Marcelo Pereyra & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
