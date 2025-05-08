.. _getting-started:

Getting Started
===============

Installing requirements
-----------------------

Poetry Installation and Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``imap-processing`` uses :ref:`poetry-link` for dependency management. Check out our :ref:`style guide <poetry-environment>` for more information on specific IMAP Poetry usage.

If you're running locally, you can install the Python requirements with Poetry.

To setup versioning *(recommended for developers)*

.. code-block:: bash

    poetry self add poetry-dynamic-versioning

To install without the extras

.. code-block:: bash

    poetry install

To install all extras

.. code-block:: bash

    poetry install --all-extras

This will install the dependencies from `poetry.lock`, ensuring that consistent versions are used. Poetry also provides a virtual environment, which you will have to activate.

.. code-block:: bash

    poetry shell


In summary, the expected setup of ``Poetry`` for a development environment is:

.. code-block:: bash

    poetry self add poetry-dynamic-versioning
    poetry install --all-extras


Using IMAP processing
---------------------

The main way to run IMAP processing locally is through the :ref:`cli`.

Our tests are run using pytest:

.. code-block:: bash

    poetry shell
    pytest

Related Information
-------------------

For more information relating to infrastructure, see `sds-data-manager <https://sds-data-manager.readthedocs.io/en/latest/>`_.