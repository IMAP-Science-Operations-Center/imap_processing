.. _poetry-link:

Poetry
======

We are using `Poetry <https://python-poetry.org/docs/>`_ for the dependency management in both ``sds-data-manager`` and ``imap_processing``. All dependency management should go through Poetry, following our standard :ref:`poetry-environment`.

.. _poetry-overview-link:

Poetry overview
---------------

Poetry is a tool which provides the ability to manage :ref:`dependencies <poetry-dependency-link>` in Python. Although there are lots of tools which do this, including the built-in pip, Poetry has several specific advantages:

#. Poetry allows you to "lock" dependency versions in place.
#. Poetry will automatically determine if dependencies are compatible.
#. Poetry can provide dependency groups, which allow you to specify which subset of dependencies you would like to install.

These advantages ensure that we can all install the same versions of tools, that these versions are compatible with each other, and that we can manage dependencies in a fine tuned way. When you run ``poetry install`` to install your dependencies, you can be confident that they are the dependency versions that other people have been testing with.

Poetry also provides tools for automatically bumping dependency versions only where it makes sense. You can specify when a dependency should get a version change using Poetry's extensive `dependency specification <https://python-poetry.org/docs/dependency-specification/>`_ formatting.

Finally, Poetry manages a virtual environment tied to the project directory, which allows developers to install dependency versions in an isolated environment. This means that, for example, if you have a different project using the same package with a different version, you can have each project with its own version kept separate from conflicts. See :ref:`poetry-shell-link` for how to activate it.

We have a :ref:`Poetry style guide<poetry-environment>` for specific recommendations about using Poetry in these projects.

.. _using-poetry-link:

Using Poetry
------------

Poetry can be installed with the `official instructions <https://python-poetry.org/docs/master/>`_.

The dependencies for the project are all specified in the ``pyproject.toml`` file in the base level of the repository.

The Poetry project has a great `basic usage guide <https://python-poetry.org/docs/master/basic-usage/>`_ which introduces the basic capabilities. This covers the basic commands which are used in installing and managing Poetry.

You can use Poetry in addition to other Python tools. Unlike with Conda, installing Poetry should not interfere with other projects which may use other tools. You can also use different methods for virtual environments if you prefer, which is covered under the :ref:`Poetry shell section <poetry-shell-link>`. Poetry uses pip to install under the hood.

.. _poetry-dependency-link:

Adding Dependencies
-------------------

Poetry has a command to `add a new dependency <https://python-poetry.org/docs/master/cli/#add>`_, with an optional version specification:

::

    # Add a new dependency
    poetry add pendulum@^2.0.5

    # Add a new dependency with default latest version
    poetry add pendulum

These dependencies are then added automatically to the ``pyproject.toml`` file. The main project dependencies go under the ``[project]`` table as a ``dependencies`` list, using `PEP 508 <https://peps.python.org/pep-0508/>`_ version specifiers (e.g. ``numpy>=1.24,<2``). Dependencies in ``[tool.poetry.group.*]`` sections continue to use Poetry's caret notation. The main project dependencies are always installed.

You can also update the ``pyproject.toml`` file directly, using the existing formatting or the `Poetry documentation on it <https://python-poetry.org/docs/pyproject/>`_ as a guide.

After you update any dependencies, you will need to update the lock file::
    poetry lock

This will create a new version of the ``poetry.lock`` file, which should be committed to the repository. Our pre-commit tools also do this step automatically if needed.

.. _poetry-dependency-groups-link:

Dependency groups
^^^^^^^^^^^^^^^^^^

Poetry also provides dependency groups for separating dependencies into logical separations. If you are installing the project as an end user, you do not need the development tools. The testing environment does not need the documentation generation dependencies. In our case, the AWS Lambda environment does not need the same dependencies as the CDK deployment. Before you add a dependency to the main group, ask yourself if it would make more sense in one of the other existing dependency groups.

To add a dependency to an existing group, you can use the ``--group`` flag::

    poetry add mkdocs --group docs

These groups can be made optional as well, meaning they will not be installed by default when the user runs ``poetry install``. You can specify what groups to install using the ``--with`` or ``--without`` flags.

Pip also provides a standard for optional dependencies. These can be installed when using ``pip`` instead of Poetry to install the dependencies. This goes under the ``[project.optional-dependencies]`` section in ``pyproject.toml``, using PEP 508 version specifiers. These are separate from, but similar to, the Poetry dependency groups. They can only be all installed or all not installed per extra, with no splitting out into specific groups like the Poetry dependency groups.

.. _poetry-shell-link:

Installing and Activating the Virtual Environment
-------------------------------------------------

To install the Poetry project, you can use the `install <https://python-poetry.org/docs/cli/#install>`_ command::

    # We use dynamic versioning, which requires a plugin to be installed first
    poetry self add "poetry-dynamic-versioning[plugin]"

    # Install main dependencies and any dependency groups which are installed by default
    poetry install

    # Install all extras
    poetry install --all-extras

    # install without specific dependency groups
    poetry install --without test,docs

    # Install with optional dependency groups
    poetry install --with lambda_dev

By default, this command will install dependencies out of the ``poetry.lock`` file.

.. note::

    ``poetry shell`` was removed in Poetry 2. To activate the Poetry-managed
    virtual environment, use:

    .. code-block:: bash

        source $(poetry env info --path)/bin/activate

        # To exit the virtual environment
        deactivate

    On Windows (PowerShell):

    .. code-block:: powershell

        & (poetry env info --path)\Scripts\activate.ps1

    You can also run a single command inside the environment without activating
    it using ``poetry run``::

        poetry run pytest

.. _venv-alternative-link:

Using a plain ``venv`` instead of Poetry's virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to manage your virtual environment with the standard Python
``venv`` module rather than Poetry, you can do so by creating the environment
first and then running ``poetry install`` (or ``pip install``) inside it.
Poetry will detect the active virtual environment and install into it instead
of creating its own:

.. code-block:: bash

    # Create and activate a venv
    python3 -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate.ps1

    # Install via Poetry (uses the active venv)
    poetry self add "poetry-dynamic-versioning[plugin]"
    poetry install --all-extras

    # Or install directly with pip using the pyproject.toml extras
    pip install -e ".[dev,test,tools]"

The ``venv`` approach avoids the need for ``poetry shell`` entirely: your
shell is already in the virtual environment after running ``source .venv/bin/activate``.

Poetry will, by default, not create a new virtual environment if it detects that it is running in a virtual environment already. So, for example, you can use a `Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ by activating the environment first, and then running ``poetry install``.

There are also `settings <https://python-poetry.org/docs/configuration/#virtualenvscreate>`_ surrounding the virtual environment that you can change to suit your workflow.
