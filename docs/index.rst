|GitHub| |Build Status| |Docs| |CodeCov| |GPL license| |ArXiv|

.. |GitHub| image:: https://img.shields.io/badge/GitHub-ProxNest-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/proxnest
.. |Build Status| image:: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml
.. |Docs| image:: https://github.com/astro-informatics/proxnest/actions/workflows/docs.yml/badge.svg
    :target: https://astro-informatics.github.io/proxnest
.. |CodeCov| image:: https://codecov.io/gh/astro-informatics/proxnest/branch/main/graph/badge.svg?token=oGowwdoMRN
    :target: https://codecov.io/gh/astro-informatics/proxnest
.. |GPL License| image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. |ArXiv| image:: http://img.shields.io/badge/arXiv-2106.03646-orange.svg?style=flat
    :target: https://arxiv.org/abs/2106.03646

ProxNest
========

``ProxNest`` is an open source, well tested and documented Python implementation of the *proximal nested sampling* algorithm (`Cai et al. 2022 <https://arxiv.org/pdf/2106.03646.pdf>`_) which is uniquely suited for sampling from very high-dimensional posteriors that are log-concave and potentially not smooth (*e.g.* Laplace priors). This is achieved by exploiting tools from proximal calculus and Moreau-Yosida regularisation (`Moreau 1962 <https://hal.archives-ouvertes.fr/hal-01867195/file/Fonctions_convexes_duales_points_proximaux_Moreau_CRAS_1962.pdf>`_) to efficiently sample from the prior subject to the hard likelihood constraint. The resulting Markov chain iterations include a gradient step, approximating (with arbitrary precision) an overdamped Langevin SDE that can scale to very high-dimensional applications.

Referencing
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

.. bibliography:: 
    :notcited:
    :list: bullet

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   background/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Interactive Tutorials
   
   tutorials/galaxy_denoising.nblink

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index



