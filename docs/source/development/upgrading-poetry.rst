.. _upgrading-poetry:

Upgrading Poetry 1.x to 2.x
============================

Poetry 2.x introduces breaking changes, and it is not possible to run Poetry 1.x against a Poetry 2 project. This page covers two ways to
upgrade an existing Poetry 1.8 installation.


Option 1: In-place upgrade with ``poetry self update``
------------------------------------------------------

::

    poetry self update

If you have the ``poetry-dynamic-versioning`` plugin installed, reinstall it
after upgrading::

    poetry self add "poetry-dynamic-versioning[plugin]"

Option 2: Uninstall and reinstall
----------------------------------

If the in-place upgrade fails or produces errors, a clean reinstall is more
reliable.

**1. Uninstall the existing Poetry installation:**

.. code-block:: bash

    # If installed via the official installer
    curl -sSL https://install.python-poetry.org | python3 - --uninstall

    # If installed via pipx
    pipx uninstall poetry

**2. Install Poetry 2.x:**

.. code-block:: bash

    # Via the official installer (recommended)
    curl -sSL https://install.python-poetry.org | python3 -

    # Via pipx
    pipx install poetry

After install
-------------

After install, reinstall the project using `poetry install --all-extras`.

If you see any issues, please reach out to the SDC team (imap-sdc@lists.lasp.colorado.edu) for troubleshooting help.