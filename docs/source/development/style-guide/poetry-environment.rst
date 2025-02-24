.. _poetry-environment:

Poetry Environment
------------------

:ref:`Poetry <poetry-link>` is used for dependency management within this project. To update dependencies, you can either update
``pyproject.toml`` manually, or use ``poetry add <dependency>``.

If you do add dependencies, please make sure you define the version numbers mindfully. The best way to do this is to use
`caret notation <https://python-poetry.org/docs/dependency-specification/#caret-requirements>`_ to allow for minor
version updates. For example, if you have tested the code using numpy version ``1.24.2``, the best way to specify the
dependency is to allow for updates to the right most version number:

::

    numpy = "^1.24"

This will allow for patches for security reasons without upgrading to ``1.25`` or beyond. Use your best judgement for
when packages can upgrade automatically, but try and avoid specifying only one specific version unless it is absolutely
required.

The ``poetry.lock`` file contains the existing dependencies for the project. These are the dependencies that you should
install to ensure you're getting the accepted versions for every package. If the ``poetry.lock`` file is not up-to-date,
there is an automatic process within the repo to update ``poetry.lock`` to match ``pyproject.toml``. To resolve any
dependency issues before the pre-commit hook, users should run ``poetry lock`` and resolve any dependency problems ahead
of time. The ``poetry.lock`` file should be part of the files committed in this case.