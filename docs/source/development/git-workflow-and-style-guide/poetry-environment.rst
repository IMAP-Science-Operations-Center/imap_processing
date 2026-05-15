.. _poetry-environment:

Poetry Environment
------------------

:ref:`Poetry <poetry-link>` is used for dependency management within this project. To update dependencies, you can either update
``pyproject.toml`` manually, or use ``poetry add <dependency>``.

If you do add dependencies, please make sure you define the version numbers mindfully. The required syntax depends on
which section of ``pyproject.toml`` you are editing:

* **``[project] dependencies``** (main runtime deps) and **``[project.optional-dependencies]``** (extras) use
  `PEP 508 <https://peps.python.org/pep-0508/>`_ specifiers. For example, if you have tested with numpy
  ``1.24.2`` and want to allow patch and minor updates up to the next major version::

      "numpy>=1.24,<2"

* **``[tool.poetry.group.*]``** sections (dev/test groups used only by Poetry) may continue to use
  `Poetry caret notation <https://python-poetry.org/docs/dependency-specification/#caret-requirements>`_::

      numpy = "^1.24"

  Both forms allow updates to the rightmost non-zero version component. Use your best judgement for when packages can
  upgrade automatically, but try to avoid pinning a single exact version unless it is absolutely required.

The ``poetry.lock`` file contains the existing dependencies for the project. These are the dependencies that you should
install to ensure you're getting the accepted versions for every package. If the ``poetry.lock`` file is not up-to-date,
there is an automatic process within the repo to update ``poetry.lock`` to match ``pyproject.toml``. To resolve any
dependency issues before the pre-commit hook, users should run ``poetry lock`` and resolve any dependency problems ahead
of time. The ``poetry.lock`` file should be part of the files committed in this case.