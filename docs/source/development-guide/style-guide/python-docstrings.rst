.. _python-docstrings:

Python Docstrings
-----------------

Python code documentation shall adhere to the `PEP257 <https://peps.python.org/pep-0257/>`_ and `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ conventions.

The following are further recommendations:

* Each module should have at minimum a description and a ``Use`` section.
* Each function/method should have at minimum a description, ``Parameters`` (if necessary), and ``Returns`` (if
  necessary) sections.

When a new module is added to the repository, make sure to add the module to the appropriate ``docs/source/*.rst`` file,
so that the API documentation can be part of the automated documentation build.