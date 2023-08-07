.. _release-workflow:

Release Workflow
----------------

This page describes how to make releases of the software within the ``imap_processing`` and ``sds-data-manager``
repositories, and the steps needed to push said software to the production environment and update the data products.

Software Releases
^^^^^^^^^^^^^^^^^

Any changes merged or pushed to the ``main`` branch should follow a specific release workflow which follows the diagram
shown in the :ref:`git & GitHub Workflow <git-and-github-workflow>` page. This includes both 'nominal releases' (i.e.
new features being merged into ``main``) and 'hotfixes' (i.e. bug fixes made directly to ``main``). These workflows are
described below. It is assumed that steps (1) through (4) in the :ref:`git & GitHub Workflow <git-and-github-workflow>`
are already completed.

*Note: We use the 'imap_processing' repository as an example here, but this can be applied to 'sds-data-manager' as
well.*


.. _nominal-releases:

Nominal releases
""""""""""""""""

#. Make sure the ``dev`` branch is up-to-date with any changes you want included in the release (i.e. merge in any
   feature branches using the nominal :ref:`git & GitHub Workflow <git-and-github-workflow>`).
#. Create a new 'release' branch off of ``dev``.  The name of the branch should match the version number to be used for
   the release, which should follow the :ref:`software versioning <versioning>` conventions.
#. Make any release-specific commits to the new release branch using the nominal ``git add``/``git commit`` cycle. This
   may include commits that add release notes, or update version numbers in various configurations.
#. Push the release branch to the main ``IMAP-Science-Operations-Center`` ``imap_processing`` repo (i.e. ``upstream``).
#. In GitHub, create two pull requests: one that merges the release branch into ``main``, and one that merges the
   release branch into ``dev``. Proceed with the nominal review & merge process described in steps (10) and (11) in the
   :ref:`git & GitHub Workflow <git-and-github-workflow>` section.
#. Once the changes are merged into the ``main`` branch, create a `new release
   <https://github.com/IMAP-Science-Operations-Center/imap_processing/releases>`_ for the merge commit and assign the
   appropriate version number.


.. _Making-hotfixes:

Making hotfixes
"""""""""""""""

*Note: A hotfix should generally be avoided unless there is some specific reason to bypass the nominal ``dev`` branch
workflow; for example, if a bug fix is very time critical.*

#. Create a new branch named ``hotfix-<description>`` off of the ``main`` branch, and commit any necessary changes
   following the nominal ``git add``/``git commit`` cycle.
#. Push the hotfix branch to the main ``IMAP-Science-Operations-Center`` ``imap_processing`` repo (i.e. ``upstream``),
   and open two separate pull requests: one that merges the hotfix branch into the ``main`` branch, and one that merges
   the hotfix branch into the ``dev`` branch.
#. For each of these pull requests, proceed with the nominal review & merge process described in steps (10) and (11) in
   the :ref:`git & GitHub Workflow <git-and-github-workflow>`.
#. Once the changes are merged into the ``main`` branch, create a `new release
   <https://github.com/IMAP-Science-Operations-Center/imap_processing/releases>`_ or a `new tag
   <https://github.com/IMAP-Science-Operations-Center/imap_processing/tags>`_ for the merge commit and assign the
   appropriate :ref:`version number <versioning>`.


Deployment
^^^^^^^^^^

Once a release is created in GitHub, a SDC dev team member can follow the `CDK deployment steps
<https://sds-data-manager.readthedocs.io/en/latest/cdk/cdk-deployment.html>`_ to deploy the software to AWS. Once the
software is deployed, a user should be able to call the APIs.