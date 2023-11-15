.. _checklist-for-contributors-and-reviewers-of-pull-requests:

Checklist for Contributors and Reviewers of Pull Requests
---------------------------------------------------------

The following is a guide to be used for contributors and reviewers of ``imap_processing`` pull requests. Note that this
is only a guide; it should not be treated as a fully comprehensive, foolproof list that must be used in all situations,
and parts of it are subjective.

If the contributor/reviewer can answer "yes" to all the following questions, then conceivably the proposed changes are
acceptable and the PR can be reviewed.

Before merging, make sure you have addressed all the comments in the review adhering to the :ref:`PR standards <pull-request-review-standards>`.


.. _Checklist-for-Contributors:

Checklist for Contributors
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _contributing-to-the-code:

Contributing to the code:
"""""""""""""""""""""""""

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


.. _contributing-to-the-pull-request:

Contributing to the pull request:
"""""""""""""""""""""""""""""""""

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

Reviewing the pull request:
""""""""""""""""""""""""""""

* Does the PR have a concise, descriptive title?
* Does the PR have a sufficient description as to make it clear what the reasons for the changes are?
* Is the PR merging into ``upstream/dev`` from ``<username>/<branchname>`` (in most cases)?
* Does the PR have at least one assignee?
* Does the PR have proper labels?
* Is the PR no longer a work in progress?
* Do all the automated checks pass?
* Does this post indicate the issue or explain the issue it is solving?


.. _pertaining-to-the-code-review:

Reviewing the code:
""""""""""""""""""""

* Does the code conform to this style guide?
* Is the code documented and commented sufficiently such that it is easy to read and follow?
* Does the code contain sufficient exception handling?
* Does the code contain no sensitive data or information?
* Does the code contain any necessary unit tests?
* If there are new dependencies, are they all necessary? Are they correctly added to the ``pyproject.toml`` file?
