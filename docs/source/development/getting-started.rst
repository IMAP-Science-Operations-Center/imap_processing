.. _getting-started:

Getting Started
===============

Installing requirements
-----------------------

Poetry Installation and Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``imap-processing`` uses :ref:`poetry-link` for dependency management. Check out our :ref:`style guide <poetry-environment>` for more information on specific IMAP Poetry usage. If you are upgrading from Poetry 1.x, see :ref:`upgrading-poetry`.

If you're running locally, you can install the Python requirements with Poetry.

To setup versioning *(recommended for developers)*

.. code-block:: bash

    poetry self add "poetry-dynamic-versioning[plugin]"

To install without the extras

.. code-block:: bash

    poetry install

To install all extras

.. code-block:: bash

    poetry install --all-extras

This will install the dependencies from `poetry.lock`, ensuring that consistent versions are used.
Poetry manages a virtual environment for the project, which you will need to activate.

.. note::

    ``poetry shell`` was removed in Poetry 2. Use one of the following to activate
    the virtual environment instead:

    .. code-block:: bash

        # Activate via Poetry (outputs the activation command)
        source $(poetry env info --path)/bin/activate

        # Or on Windows (PowerShell)
        & (poetry env info --path)\Scripts\activate.ps1

    Alternatively, you can use the ``venv`` module to create and manage your
    own virtual environment independently:

    .. code-block:: bash

        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e ".[dev,test,tools]"

.. note::

    Some dependencies include compiled extensions and require system libraries
    if no pre-built wheel is available for your platform. If installation fails
    with a PEP 517 build error, install the relevant system libraries below and
    then re-run the install command.

    **netcdf4** requires the NetCDF-C and HDF5 libraries:

    .. code-block:: bash

        # Debian/Ubuntu
        sudo apt-get install -y libnetcdf-dev libhdf5-dev

        # macOS (Homebrew)
        brew install netcdf hdf5

    **scipy** provides pre-built wheels for Python 3.10–3.13. If you are using
    Python 3.14 or later, no wheel is available yet and scipy must be compiled
    from source, which requires a Fortran compiler and BLAS/LAPACK. The
    simplest fix is to use Python 3.12 or 3.13.

    If you need to build from source, install the following:

    .. code-block:: bash

        # Debian/Ubuntu
        sudo apt-get install -y gfortran libopenblas-dev pkg-config

        # macOS (Homebrew)
        brew install openblas

    On Apple Silicon (M1/M2/M3), Homebrew installs to ``/opt/homebrew`` rather
    than ``/usr/local``, so pkg-config cannot find OpenBLAS automatically. Set
    ``PKG_CONFIG_PATH`` before running the install:

    .. code-block:: bash

        export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"

    If you do not need the ``test`` extras, you can avoid these system
    dependencies entirely by installing without them:

    .. code-block:: bash

        poetry install --extras "dev"

In summary, the expected setup of ``Poetry`` for a development environment is:

.. code-block:: bash

    poetry self add "poetry-dynamic-versioning[plugin]"
    poetry install --all-extras
    source $(poetry env info --path)/bin/activate


Using IMAP processing
---------------------

The main way to run IMAP processing locally is through the :ref:`cli`.

Our tests are run using pytest:

.. code-block:: bash

    source $(poetry env info --path)/bin/activate
    pytest
