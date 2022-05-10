.. _install:

Installation
============
We recommend installing ``ProxNest`` through `PyPi <https://pypi.org>`_ , however in some cases one may wish to install ``ProxNest`` directly from source, which is also relatively straightforward.

Quick install (PyPi)
--------------------
Install ``ProxNest`` from PyPi with a single command

.. code-block:: bash

    pip install ProxNest

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate ProxNest.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n proxnest_env python=3.9
    conda activate proxnest_env

Once within a fresh environment ``ProxNest`` may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/proxnest
    cd proxnest

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_proxnest.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest --black ProxNest/tests/ 

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
