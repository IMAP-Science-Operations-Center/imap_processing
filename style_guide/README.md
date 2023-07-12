`imap_processing` Style Guide
=============================

This document serves as a style guide for all `imap_processing` software development.  Any potential contribution to the
`imap_processing` repository should be checked against this guide, and any violation of the guide should be fixed before
the code is committed to the `main` or `dev` branches.


## tl;dr

The following is a short version of this style guide to be used as a quick reference.  Further details about each of
these items are provided below in the guide.

1. Use a [forking workflow](#git-and-github-workflow) for git/GitHub contributions.
2. Use `PEP8` for [python coding conventions](#python-coding) (with a few exceptions).
3. Use `PEP257` and `numpydocs` for [docstring conventions](#api-documentation) (with a few exceptions), and update the
   documentation builds where applicable.
4. Update the [`poetry` environment](#poetry-environment) when dependencies change.
5. Be mindful of committing credentials and other [sensitive information](#security).
6. Follow agreed-upon [naming conventions](#naming-conventions) where applicable.
7. Use specific [tools and libraries](#tools-and-library-recommendations) where applicable.
8. Use nominal semantic versioning for [version numbers](#versioning).
9. Follow a specific [release workflow](#release-workflow) when making releases.


## git and GitHub Workflow

The best method for contributing software to the `imap_processing` repository is a workflow that involves forking the
repository, developing changes on "feature" branches, and opening pull requests through GitHub.

The following diagram depicts this workflow (credit to
[Atlassian](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)):

<img src="https://wac-cdn.atlassian.com/dam/jcr:cc0b526e-adb7-4d45-874e-9bcea9898b4a/04%20Hotfix%20branches.svg?cdnVersion=1089" alt="git and GitHub Workflow" width="650" height="650">

As such, all feature branches should be branched off of and merged back into the `dev` branch.

### Contributing new features

Before contributing, it is generally encouraged to
[open a GitHub issue](https://github.com/IMAP-Science-Operations-Center/imap_processing/issues/new), if one does not
already exist, that describes the feature to be implemented, or the problem that is trying to be solved. This allows the
contributor and the team as a whole to keep track of how the feature relates to the project and track its progress.  Any
contributors to the feature should be assigned to the issue, and proper label(s) should also be assigned.

*Note: Steps (1) through (5) only to be completed once.*

1. Create a personal fork of the `imap_processing` repository by visiting the main `IMAP-Science-Operations-Center`
   `imap_processing` [repository](https://github.com/IMAP-Science-Operations-Center/imap_processing) and clicking the
   `Fork` button near the top-right of the page.  Follow the various steps to create a fork under your GitHub account.
2. Make a local copy of your personal fork by cloning the repository, using the URL found by clicking the green "clone"
   button:

       git clone https://github.com/<username>/imap_processing.git  # for HTTPS
       git clone git@github.com:<username>/imap_processing.git  # for SSH

3. Ensure that the personal fork is pointing to the `upstream` `imap_processing` repository:

       git remote add upstream https://github.com/IMAP-Science-Operations-Center/imap_processing.git  # for HTTPS
       git remote add upstream git@github.com:IMAP-Science-Operations-Center/imap_processing.git  # for SSH

4. Install the development dependencies and pre-commit hooks. These hooks will automatically run the tools
   [`black`](https://black.readthedocs.io/en/stable/) and [`ruff`](https://beta.ruff.rs/docs/) on your code changes
   before commits:

       pip install .[dev]
       pre-commit install

5. Retrieve the `upstream` `dev` branch:

       git fetch upstream
       git checkout -b dev upstream/dev

6. Create a feature branch off of the `dev` branch to develop changes on. Branch names should be short but descriptive
   (e.g. `update-codice-unit-tests`) and not too generic (e.g. `bug-fix`, `updates`). Consistent use of hyphens is
   encouraged.

        git checkout -b <branchname>

7. Make changes to the branch using the nominal `git add`/`git commit` cycle.

        git add <new or changed files you want to commit>
        git commit -m 'Explaination of the changes'

   With the pre-commit hooks installed, you may need to make fixes to any warnings/errors reported by `black` and/or
   `ruff` and try committing your changes again.

8. Push the feature branch to your personal fork's GitHub repository:

        git push origin <branchname>

9. On the `IMAP-Science-Operations-Center` `imap_processing`
   [repository](https://github.com/IMAP-Science-Operations-Center/imap_processing) create a new pull request.  Click on
   the "compare across forks" link to enable the pull request to use your fork.  Set the "base repository" to
   `IMAP-Science-Operations-Center` and "base" to `dev`.  Set the "head repository" to the `imap_processing` repository
   under your personal fork and "compare" to your feature branch. If the feature branch is still under development, you
   can click the "Convert to draft" button under the "Reviewers" section, or add a "[WIP]" at the beginning of the pull
   request title to signify that the pull request is not ready to be merged.

10. Assign at least one reviewer to the pull request, though two or more reviewers are highly encouraged, especially for
    significant changes. They will review your pull request and either accept the request or ask for additional changes.
    If additional changes are needed, iterate through steps (7) and (8) until you and the reviewer(s) are satisfied.

11. Once the pull request has been accepted, you can merge the pull request and delete the feature branch.

### Keeping your fork updated

You can keep your personal fork up-to-date with the `IMAP-Science-Operations-Center` `imap_processing` repository by
fetching and pulling the `upstream` remote:

    git checkout dev
    git fetch upstream dev
    git pull upstream/dev

### Collaborating on someone else's fork

To contribute to a branch on another person's personal fork, add a new `remote` that points to their fork, and use the
nominal workflow for contributing:

    git remote add <username> <remote url>
    git fetch <username>
    git checkout -b <branchname> <username>/<branchname>
    # Make some changes via add/commit cycle
    git push <username> <branchname>


## Python Coding

`imap_processing` code shall adhere to the [PEP8](https://peps.python.org/pep-0008/) conventions save for the following
exceptions:

 - Lines of code need to be restricted to 88 characters to adhere to the `black` code formatter.
 - Avoid one-letter or unnecessarily-abbreviated variable names (e.g. use `temperature` instead of `temp`, `filename`
   instead of `fn`).
 - Function and class definitions should be placed in alphabetical order in the module.
 - It is encouraged to annotate variables and functions using the
   [`typing`](https://docs.python.org/3/library/typing.html) library.


## API Documentation

`imap_processing` code shall adhere to the [PEP257](https://peps.python.org/pep-0257/) and
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) conventions.

The following are further recommendations:

- Each module should have at minimum a description and a `Use` section.
- Each function/method should have at minimum a description, `Parameters` (if necessary), and `Returns` (if necessary)
  sections.

When a new module is added to the repository, make sure to add the module to the appropriate `docs/source/*.rst` file,
so that the API documentation can be part of the automated documentation build.


## Poetry Environment

Poetry is used for dependency management within this project. To update dependencies, you can either update
`pyproject.toml` manually, or use `poetry add <dependency>`.

If you do add dependencies, please make sure you define the version numbers mindfully. The best way to do this is to use
[caret notation](https://python-poetry.org/docs/dependency-specification/#caret-requirements) to allow for minor version
updates. For example, if you have tested the code using numpy version `1.24.2`, the best way to specify the dependency
is to allow for updates to the right most version number:

    numpy = "^1.24"`

This will allow for patches for security reasons without upgrading to `1.25` or beyond. Use your best judgement for when
packages can upgrade automatically, but try and avoid specifying only one specific version unless it is absolutely
required.

The `poetry.lock` file contains the existing dependencies for the project. These are the dependencies that you should
install to ensure you're getting the accepted versions for every package. If the `poetry.lock` file is not up-to-date,
there is an automatic process within the repo to update `poetry.lock` to match `pyproject.toml`. To resolve any
dependency issues before the pre-commit hook, users should run `poetry lock` and resolve any dependency problems ahead
of time. The `poetry.lock` file should be part of the files committed in this case.


## Security

The following items should never be committed in the `imap_processing` source code or GitHub issues/pull requests:

- Account credentials of any kind (e.g. database usernames/passwords, AWS credentials, etc.)
- Internal directory structures or filepaths
- Machine names
- Proprietary data

If `imap_processing` code needs access to this information, it should be stored in a configuration file that is not part
of the repository.


## Naming Conventions

*To be determined*


## Tools and Library Recommendations

- `black` for checking python code formatting
- `poetry` for dependency management and virtual environments
- `pytest` for unit testing
- `ruff` for checking docstring formatting
- `spiceypy` for using SPICE kernels
- `space-packet-parser` to unpack CCSDS packets


## Versioning

Any changes pushed to the `main` branch should be tagged with a version number. The version number convention is
`x.y.z`, where

    x = The main version number.  Increase when making incompatible API changes.
    y = The feature number.  Increase when change contains a new feature with or without bug fixes.
    z = The hotfix number. Increase when change only contains bug fixes.


## Release Workflow

Any changes merged or pushed to the `main` branch should follow a specific release workflow which follows the diagram
shown in the [git & GitHub Workflow](#git-and-github-workflow) section. This includes both 'nominal releases' (i.e. new
features being merged into `main`) and 'hotfixes' (i.e. bug fixes made directly to `main`). These workflows are
described below. It is assumed that steps (1) through (4) in the [git & GitHub Workflow](#git-and-github-workflow) section
are already completed.

### Nominal releases

1. Make sure the `dev` branch is up-to-date with any changes you want included in the release (i.e. merge in any feature
   branches using the nominal [git/GitHub workflow](#git-and-github-workflow)).
2. Create a new 'release' branch off of `dev`.  The name of the branch should match the version number to be used for
   the release, which should follow the [versioning](#versioning) conventions.
3. Make any release-specific commits to the new release branch using the nominal `git add`/`git commit` cycle. This may
   include commits that add release notes, or update version numbers in various configurations.
4. Push the release branch to the main `IMAP-Science-Operations-Center` `imap_processing` repo (i.e. `upstream`).
5. In GitHub, create two pull requests: one that merges the release branch into `main`, and one that merges the release
   branch into `dev`. Proceed with the nominal review & merge process described in steps (10) and (11) in the
   [git/GitHub workflow](#git-and-github-workflow) section.
6. Once the changes are merged into the `main` branch, create a
   [new release](https://github.com/IMAP-Science-Operations-Center/imap_processing/releases) for the merge commit and
   assign the appropriate version number.

### Making hotfixes

*Note: A hotfix should generally be avoided unless there is some specific reason to bypass the nominal `dev` branch
workflow; for example, if a bug fix is very time critical.*

1. Create a new branch named `hotfix-<description>` off of the `main` branch, and commit any necessary changes following
   the nominal `git add`/`git commit` cycle.
2. Push the hotfix branch to the main `IMAP-Science-Operations-Center` `imap_processing` repo (i.e. `upstream`), and
   open two separate pull requests: one that merges the hotfix branch into the `main` branch, and one that merges the
   hotfix branch into the `dev` branch.
3. For each of these pull requests, proceed with the nominal review & merge process described in steps (10) and (11) in
   the [git/GitHub workflow](#git-and-github-workflow) section.
4. Once the changes are merged into the `main` branch, create a
   [new release](https://github.com/IMAP-Science-Operations-Center/imap_processing/releases) or a
   [new tag](https://github.com/IMAP-Science-Operations-Center/imap_processing/tags) for the merge commit and assign the
   appropriate [version number](#versioning).


## Checklist for Contributors and Reviewers of Pull Requests

The following is a guide to be used for contributors and reviewers of `imap_processing` pull requests. Note that this is
only a guide; it should not be treated as a fully comprehensive, foolproof list that must be used in all situations, and
parts of it are subjective.

If the contributor/reviewer can answer "yes" to all the following questions, then conceivably the proposed changes are
acceptable and the PR can be reviewed and merged.

### Checklist for Contributors

#### Pertaining to the code:

- Does the code conform to this style guide?
- Is any of the code functionality not already available via native or third-party python libraries?
- Does the code execute successfully?
    - Do all the tests pass in the existing test suite?
    - Does the newly added functionality run without errors?
- Is the code documented and commented sufficiently such that it is easy to read and follow?
    - Are docstrings included for all new modules, classes, and functions?
    - Are in-line comments included to provide necessary context?
    - Are any documentation files in other locations updated?
- Have all debugging/print statements been removed?
- Does the code contain sufficient exception handling?
- Does the code contain no deprecation warnings?
- Does the code include all necessary unit tests?
- Are any new dependencies correctly added to the `pyproject.toml` file?

#### Pertaining to the pull request:

- Is the PR excessively long and/or covers multiple issues? If so, consider breaking it up into multiple PRs.
- Does the PR have a concise, descriptive title?
- Does the PR link to and close the relevant issue?
- Does the PR have a sufficient description as to make it clear what the reasons for the changes are?
- Is the PR merging into `upstream/dev` from `<username>/<branchname>` (in most cases)?
- Are you listed as an assignee to the PR?
- Does the PR have proper labels?
- Do all the automated checks pass?


### Checklist for Reviewers

##### Pertaining to the pull request:

- Does the PR have a concise, descriptive title?
- Does the PR have a sufficient description as to make it clear what the reasons for the changes are?
- Is the PR merging into `upstream/dev` from `<username>/<branchname>` (in most cases)?
- Does the PR have at least one assignee?
- Does the PR have proper labels?
- Is the PR no longer a work in progress?
- Do all the automated checks pass?

#### Pertaining to the code:

- Does the code conform to this style guide?
- Is the code documented and commented sufficiently such that it is easy to read and follow?
- Does the code contain sufficient exception handling?
- Does the code contain no sensitive data or information?
- Does the code contain any necessary unit tests?
- If there are new dependencies, are they all necessary? Are they correctly added to the `pyproject.toml` file?
