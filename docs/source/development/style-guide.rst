Style Guide
===========

This document serves as a style guide for all software development on the `imap-processing
<https://github.com/IMAP-Science-Operations-Center/imap_processing>`_ and `sds-data-manager
<https://github.com/IMAP-Science-Operations-Center/sds-data-manager>`_ repositories. Any potential contribution to
the these repositories should be checked against this guide, and any violation of the guide should be fixed
before the code is committed to the ``main`` or ``dev`` branches.


.. _tl;dr:

tl;dr
-----

The following is a short version of this style guide to be used as a quick reference. Further details about each of
these items are provided below in the guide.

#. Use a :ref:`forking workflow <git-and-github-workflow>` for git/GitHub contributions.
#. Use ``PEP8`` for :ref:`python coding conventions <python-coding>` (with a few exceptions).
#. Use ``PEP257`` and ``numpydocs`` for :ref:`docstring conventions <api-documentation>` (with a few exceptions), and
   update the documentation builds where applicable.
#. Update the :ref:`poetry environment <poetry-environment>` when dependencies change.
#. Be mindful of committing credentials and other :ref:`sensitive information <security>`.
#. Follow agreed-upon :ref:`naming conventions <naming-conventions>` where applicable.
#. Use specific :ref:`tools and libraries <tools-and-library-recommendations>` where applicable.
#. Use nominal semantic :ref:`versioning <versioning>` for version numbers.
#. Follow a specific :ref:`release workflow <release-workflow>` when making releases.


.. _checklist-for-contributors-and-reviewers-of-pull-requests:

Checklist for Contributors and Reviewers of Pull Requests
---------------------------------------------------------

The following is a guide to be used for contributors and reviewers of ``imap_processing`` pull requests. Note that this
is only a guide; it should not be treated as a fully comprehensive, foolproof list that must be used in all situations,
and parts of it are subjective.

If the contributor/reviewer can answer "yes" to all the following questions, then conceivably the proposed changes are
acceptable and the PR can be reviewed and merged.


.. _Checklist-for-Contributors:

Checklist for Contributors
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _Pertaining-to-the-code:

Pertaining to the code:
"""""""""""""""""""""""

* Does the code conform to this style guide?
* Is any of the code functionality not already available via native or third-party python libraries?
* Does the code execute successfully?
    * Do all the tests pass in the existing test suite?
    * Does the newly added functionality run without errors?
* Is the code documented and commented sufficiently such that it is easy to read and follow?
    * Are docstrings included for all new modules, classes, and functions?
    * Are in-line comments included to provide necessary context?
    * Are any documentation files in other locations updated?
* Have all debugging/print statements been removed?
* Does the code contain sufficient exception handling?
* Does the code contain no deprecation warnings?
* Does the code include all necessary unit tests?
* Are any new dependencies correctly added to the ``pyproject.toml`` file?


.. _pertaining-to-the-pull-request:

Pertaining to the pull request:
"""""""""""""""""""""""""""""""

* Is the PR excessively long and/or covers multiple issues? If so, consider breaking it up into multiple PRs.
* Does the PR have a concise, descriptive title?
* Does the PR link to and close the relevant issue?
* Does the PR have a sufficient description as to make it clear what the reasons for the changes are?
* Is the PR merging into ``upstream/dev`` from ``<username>/<branchname>`` (in most cases)?
* Are you listed as an assignee to the PR?
* Does the PR have proper labels?
* Do all the automated checks pass?


.. _checklist-for-reviewers:

Checklist for Reviewers
^^^^^^^^^^^^^^^^^^^^^^^


.. _pertaining-to-the-pull-request-review:

Pertaining to the pull request:
"""""""""""""""""""""""""""""""

* Does the PR have a concise, descriptive title?
* Does the PR have a sufficient description as to make it clear what the reasons for the changes are?
* Is the PR merging into ``upstream/dev`` from ``<username>/<branchname>`` (in most cases)?
* Does the PR have at least one assignee?
* Does the PR have proper labels?
* Is the PR no longer a work in progress?
* Do all the automated checks pass?


.. _pertaining-to-the-code-review:

Pertaining to the code:
"""""""""""""""""""""""

* Does the code conform to this style guide?
* Is the code documented and commented sufficiently such that it is easy to read and follow?
* Does the code contain sufficient exception handling?
* Does the code contain no sensitive data or information?
* Does the code contain any necessary unit tests?
* If there are new dependencies, are they all necessary? Are they correctly added to the ``pyproject.toml`` file?
