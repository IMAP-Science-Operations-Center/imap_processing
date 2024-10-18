.. _getting-started:

Getting Started
===============

Installing requirements
-----------------------

``imap-processing`` uses :ref:`poetry-link` for dependency management. Check out our :ref:`style guide <poetry-environment>` for specific IMAP Poetry usage.

To install using Poetry for development:

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

Other Information
-----------------

Another source of useful links is the `internal IMAP help page <https://lasp.colorado.edu/galaxy/pages/viewpage.action?pageId=193401231>`_, or the
`infrastructure documentation <https://sds-data-manager.readthedocs.io/en/latest/>`_.