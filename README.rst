.. image:: https://img.shields.io/badge/GitHub-PyTemplate-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/proxnest
.. image:: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/astro-informatics/proxnest/actions/workflows/tests.yml
.. image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/proxnest
.. image:: https://codecov.io/gh/astro-informatics/proxnest/branch/main/graph/badge.svg?token=oGowwdoMRN
    :target: https://codecov.io/gh/astro-informatics/proxnest
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-2106.03646-orange.svg?style=flat
    :target: https://arxiv.org/abs/2106.03646
    
|logo| Proximal nested sampling for high-dimensional Bayesian inference
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/ProxNestLogo.png" align="center" height="80" width="80">

This is a very loose framework for people to start python projects from. To get up and running, go through the code carefully replacing ``Project-name`` with your 
desired project name (check all documents!), don't forget to change the directory project_name as well! You will also need to update the links in all badges!


Auto-formatting code
====================
To keep the code readable and organised you should (strongly) consider using the ``black`` package. Whenever you are finished updating a file, just run 

.. code-block:: bash

    black <file_to_tidy.py>

or alternatively format everything by running

.. code-block:: bash

    black project_name/*

This is important as the CI enforces black formatting (this can be disabled by removing the --black flag in pytest) so your unit tests will fail if you don't do this!

CodeCov
============
To set up code coverage you will need to enter this  

.. code-block:: bash

    https://codecov.io/gh/{account-name}/{desired-repo} 

into any browser, then go to settings and activate the repository. You will then need to find the ``repository upload token`` which 
should be added to the github actions script (roughly line 29)

.. code-block::

    codecov --token <add your token here>

Next time CI runs on main branch it will automatically update codecov. Now go back to codecov, copy the badge and put it in the readme, .pipreadme, and 
the root index of the documentation!

PyPi
=====
To deploy the code on PyPi first test the deployment on PyPi's mirror site by, first making an account on https://test.pypi.org and then running 

.. code-block:: bash 

    python setup.py bdist_wheel --universal
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    pip install -i https://test.pypi.org/simple/ project_name

From the root directory. Keep in mind that installing from the mirror site won't automatically find dependencies, so if you have an error because the pacakge can't find numpy that's probably why, and may not be an issue on the main PyPi site. To deploy the main PyPi site simply remove the --repostiry-url name, note that you can add multiple wheels to dist/*, to provide a package which may be pip installed for multiple python version, and on multiple machine architectures.

Contributors
============
Xiaohao Cai, Jason McEwen, Marcelo Pereyra, Matthew Price, and contributors.

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
     Copyright (C) 2022 Xiaohao Cai, Jason McEwen, Marcelo Pereyra, Matthew Price & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
