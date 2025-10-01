.. _release-workflow:

Release Workflow
----------------

This page describes how to make releases of the software within the ``imap_processing``, ``sds-data-manager``, and
``imap-data-access`` repositories, and the steps needed to push said software to the production environment and update
the data products.

Software Releases
^^^^^^^^^^^^^^^^^

This project uses a workflow in which releases are made off of a 'version branch' (e.g. ``v0.1.x``), as depicted in the
diagram shown in the :ref:`git & GitHub Workflow <git-and-github-workflow>` page. This includes both 'nominal releases'
(i.e. new features to be released to the public) and 'patches' (i.e. bug fixes to the version branch). These workflows
are described below. It is assumed that steps (1) through (4) in the :ref:`git & GitHub Workflow <git-and-github-workflow>`
are already completed.

The three repositories are expected to be released at different cadences:

* The ``imap_processing`` repository is released on a monthly cadence (typically the last day of the month, but can
  vary depending on the availability of developers time to make the release), and also release on an as-needed basis.
* The ``sds_data_manager`` repository is released on a as-needed basis, typically when enough changes have been made
  since the last release to justify a new release.
* The ``imap-data-access`` repository is released whenever a new feature or bug fix is implemented, since this
  repository acts as a dependency of ``imap_processing`` and is used by external users.


Note: We use the ``imap_processing`` repository as an example in these instructions, but this can be applied to the
other repositories as well.

.. _nominal-releases:

Nominal releases
""""""""""""""""

#. Make sure your local ``dev`` branch is up-to-date with any changes you want included in the release (i.e. merge in
   any feature branches and update the branch against ``upstream`` using the nominal
   :ref:`git & GitHub Workflow <git-and-github-workflow>`).
#. Create a new version branch off of ``dev``.  The name of the branch should match the version number to be used for
   the release, which should follow the :ref:`software versioning <versioning>` conventions. The name should be
   prepended with ``v`` (e.g. ``v0.1.0``).
#. Make any release-specific commits to the new version branch using the nominal ``git add``/``git commit`` cycle. This
   may include commits that add release notes, or update version numbers in various configurations. (Currently, the
   ``imap_processing`` repo does not require any release-specific commits, as the versioning for that repo is fully
   automated. The ``sds_data_manager`` and ``imap-data-access`` repositories currently require an update to the
   hard-coded version number in the ``pyproject.toml`` file.)
#. Push the version branch to the main ``IMAP-Science-Operations-Center`` ``imap_processing`` repo (i.e. ``upstream``).
#. If there have been release-specific commits, In GitHub, create a pull request that merges the version branch into
   ``dev``. Proceed with the nominal review & merge process described in steps (10) and (11) in the :ref:`git & GitHub
   Workflow <git-and-github-workflow>` section.
#. Create a `new release <https://github.com/IMAP-Science-Operations-Center/imap_processing/releases>`_. Under "Choose a
   tag", create a new tag with the same name as the version branch (e.g. ``v0.1.0``). For "Release title", also use the
   name of the version branch (e.g. ``v0.1.0``). In the description box, include appropriate highlights and release
   notes (you can use a previous release for an example of how this should be formatted), or use the "Generate Release
   Notes" button.


Patches
"""""""

This workflow assumes that a version branch has already been created (and released at least once) for the minor version
which you want to patch (e.g. ``v0.1.x``).

#. Create a new branch off of the version branch that you want to patch (hereby called the 'bug-fix branch'). Commit any
   relevant bug fixes to this bug-fix branch using the nominal ``git add``/``git commit`` cycle.
#. Create a new pull request that merges the bug-fix branch into the version branch. Proceed with the nominal review &
   merge process described in steps (10) and (11) in the :ref:`git & GitHub Workflow <git-and-github-workflow>` section.
#. Once the changes are merged into the version branch, create a `new release
   <https://github.com/IMAP-Science-Operations-Center/imap_processing/releases>`_, using the version branch as the
   ``Target`` branch. Assign a new tag with the specific version number that increases the patch number by one from the
   most recent tag (e.g. ``v0.1.1``).
#. In GitHub, create a pull request that merges the version branch into ``dev``. Proceed with the nominal review & merge
   process described in steps (10) and (11) in the :ref:`git & GitHub Workflow <git-and-github-workflow>` section.



Deployment
^^^^^^^^^^

Once a release is created in GitHub, a GitHub action workflow is used to automatically deploy the changes. In the case
of ``sds_data_manager``, a ``Deploy`` workflow is used to deploy the software to AWS. For ``imap_processing`` and
``imap-data-access``, a ``Build and upload to PyPi`` workflow is used to build the package and upload it to PyPi. The
person performing the release should check that these workflows succeeded successfully.