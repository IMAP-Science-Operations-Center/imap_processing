.. _style-guide-overview:

Style Guide
===========

This document serves as a style guide for all software development on the `imap-processing
<https://github.com/IMAP-Science-Operations-Center/imap_processing>`_ and `sds-data-manager
<https://github.com/IMAP-Science-Operations-Center/sds-data-manager>`_ repositories. Any potential contribution to
the these repositories should be checked against this guide, and any violation of the guide should be fixed
before the code is committed to the ``main`` or ``dev`` branches.


tl;dr
-----

The following is a short version of this style guide to be used as a quick reference. Further details about each of
these items are provided below in the guide.

#. Use a :ref:`forking workflow <git-and-github-workflow>` for git/GitHub contributions.
#. Use ``PEP8`` for :ref:`python coding conventions <python-coding>` (with a few exceptions).
#. Use ``PEP257`` and ``numpydocs`` for :ref:`docstring conventions <python-docstrings>` (with a few exceptions), and
   update the documentation builds where applicable.
#. Update the :ref:`poetry environment <poetry-environment>` when dependencies change.
#. Be mindful of committing credentials and other :ref:`sensitive information <security>`.
#. Follow agreed-upon :ref:`naming conventions <naming-conventions>` where applicable.
#. Use specific :ref:`tools and libraries <tools-and-library-recommendations>` where applicable.
#. Use nominal semantic :ref:`versioning <versioning>` for version numbers.

Contributors can refer to the :ref:`checklist for contributors and reviewers of pull requests
<checklist-for-contributors-and-reviewers-of-pull-requests>` for assistance in making sure pull requests are adhering
to these conventions.

.. toctree::
    :maxdepth: 1

    git-and-github-workflow
    python-coding
    python-docstrings
    poetry-environment
    security
    naming-conventions
    tools-and-library-recommendations
    versioning
    checklist-for-pull-requests
    review-standards