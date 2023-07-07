`imap_processing` Style Guide
=============================

This document serves as a style guide for all `imap_processing` software
development.  Any potential contribution to the `imap_processing` repository
should be checked against this guide, and any violation of the guide should be
fixed before the code is committed to the `main` or `dev` branches.


git & GitHub Workflow
---------------------

The best method for contributing software to the `imap_processing` repository is
a workflow that involves forking the repository, developing changes on "feature"
branches, and opening pull requests through GitHub.

The following diagram depicts this workflow:

![workflow](https://wac-cdn.atlassian.com/dam/jcr:cc0b526e-adb7-4d45-874e-9bcea9898b4a/04%20Hotfix%20branches.svg?cdnVersion=1089)

As such, all feature branches should be branched off of and merged back into the
`dev` branch.

### Contributing new features

1. Create a personal fork of the `imap_processing` repository by visiting the
   repository main `IMAP-Science-Operations-Center` `imap_processing`
   [repository](https://github.com/IMAP-Science-Operations-Center/imap_processing)
   and clicking the `Fork` button near the top-right of the page.  Follow the
   various steps to create a fork under your GitHub account.
2. Make a local copy of your personal fork by cloning the repository, using the
   URL found by clicking the green "clone" button.
3. Ensure that the personal fork is pointing to the `upstream` `imap_processing`
   repository:

       git remote add upstream <url>

4. Retrieve the `upstream` `dev` branch:

       git fetch upstream
       git checkout -b dev upstream/dev

5. Create a feature branch off of the `dev` branch to develop changes on.
   Branch names should be short but descriptive (e.g.
   `update-codice-unit-tests`) and not too generic (e.g. `bug-fix`, `updates`).
   Consistent use of hyphens is encouraged.

        git checkout -b <branchname>

6. Make changes to the branch using the nominal `git add`/`git commit` cycle:

        git add <new or changed files you want to commit>
        git commit -m 'Explaination of the changes'

7. Push the feature branch to your personal fork's GitHub repository:

        git push origin <branchname>

8. On the `IMAP-Science-Operations-Center` `imap_processing`
   [repository](https://github.com/IMAP-Science-Operations-Center/imap_processing)
   create a new pull request.  Click on the "compare across forks" link to
   enable the pull request to use your fork.  Set the "base repository" to
   `IMAP-Science-Operations-Center` and "base" to `dev`.  Set the "head
   repository" to the `imap_processing` repository under your personal fork
   and "compare" to your feature branch. If the feature branch is still under
   development, you can click the "Convert to draft" button under the
   "Reviewers" section, or add a "[WIP]" at the beginning of the pull request
   title to signify that the pull request is not ready to be merged.

9. Assign at least one reviewer to the pull request.  They will review your
   pull request and either accept the request or ask for additional changes.
   If additional changes are needed, iterate through steps (6) and (7) until
   you and the reviewer(s) are satisfied.

10. Once the pull request has been accepted, you can merge the pull request and
    delete the feature branch.


### Making hotfixes

As shown in the diagram above, sometimes hotfixes need to be directly to the
`main` branch. The workflow for this scenario is as follows:

1. Assuming steps (1) through (3) in the previous section are already completed,
   create a new branch named `hotfix-<description>` off of the `main` branch,
   and commit any necessary changes following the nominal `git add`/`git commit`
   cycle.
2. Push the hotfix branch to your personal fork, and open two separate pull
   requests: one that merges the hotfix branch into the
   `IMAP-Science-Operations-Center` `imap_processing` `main` branch, and one
   that merges the hotfix branch into the `IMAP-Science-Operations-Center`
   `imap_processing` `dev` branch.
3. For each of these pull requests, proceed with the nominal review & merge
   process described in steps (9) and (10) in the previous section.


### Keeping your fork updated

You can keep your personal fork up-to-date with the
`IMAP-Science-Operations-Cetner` `imap_processing` repository by fetching and
pulling the `upstream` remote:

    git checkout dev
    git fetch upstream dev
    git pull upstream/dev

### Collaborating on someone else's fork

To contribute to a branch on another person's personal fork, add a new `remote`
that points to their fork, and use the nominal workflow for contributing:

    git remote add <username> <remote url>
    git fetch <username>
    git checkout -b <branchname> <username>/<branchname>
    # Make some changes via add/commit cycle
    git push <username> <branchname>

Versioning
----------

Any changes pushed to the `main` branch should be tagged with a version number.
The version number convention is `x.y.z`, where

    x = The main version number.  Increase when making incompatible API changes.
    y = The feature number.  Increase when change contains a new feature with or without bug fixes.
    z = The hotfix number. Increase when change only contains bug fixes.


Security
--------

The following items should never be committed in the `imap_processing` source
code or GitHub issues/pull requests:

- Account credentials of any kind (e.g. database usernames/passwords, AWS
credentials, etc.)
- Internal directory structures or filepaths
- Machine names
- Proprietary data

If `imap_processing` code needs access to this information, it should be stored
in a configuration file that is not part of the repository.

Additionally, developers of this project should be mindful of application
security risks, and should adhere to the
[OWASP Top 10](https://www.owasp.org/images/7/72/OWASP_Top_10-2017_%28en%29.pdf.pdf)
as much as possible.


Python Coding
-------------

`jwql` code shall adhere to the [PEP8](https://www.python.org/dev/peps/pep-0008/)
conventions save for the following exceptions:

 - Lines of code need not be restricted to 79 characters.  However, it is
encouraged to break up excessively long lines into several lines if it benefits
the overall readability of the code

 Additionally, the code shall adhere to the following special guidelines:

 - Function and class definitions should be placed in alphabetical order in the
module
 - It is encouraged to annotate variables and functions using the
[`typing`](https://docs.python.org/3/library/typing.html) library (see
[PEP 483](https://www.python.org/dev/peps/pep-0483/),
[PEP 484](https://www.python.org/dev/peps/pep-0484/), and
[PEP 526](https://www.python.org/dev/peps/pep-0526/)).


Documentation
-------------

`imap_processing` code shall adhere to the [PEP257](https://www.python.org/dev/peps/pep-0008/)
and [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) conventions.

The following are further recommendations:

- Each module should have at minimum a description, `Authors`, and `Use`
sections.
- Each function/method should have at minimum a description, `Parameters` (if
necessary), and `Returns` (if necessary) sections.


Logging
-------

TBD

Naming Conventions
------------------

TBD

Tools and Library Recommendations
---------------------------------

TBD

Releases
--------

TBD
