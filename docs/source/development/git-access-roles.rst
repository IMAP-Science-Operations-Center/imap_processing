.. _git-access-roles:

GitHub Access & Permissions Guide
======================================

This document outlines the different permission levels available in IMAP
repositories and what each role can do.

Overview
========

GitHub provides five repository roles. Below is a detailed breakdown of
permissions for each role, including what GitHub users can access and
and what access requests must be submitted to IMAP SDC.

---

What GitHub Users Can Access
=============================

Any GitHub user (without invitation) can:

- ✅ View **public repositories**
- ✅ Create issues in public repos (if enabled)
- ✅ Comment on public issues/PRs
- ✅ Fork public repositories
- ❌ **Cannot** push, merge, or modify anything
- ❌ **Cannot** access private repositories

---

Role Permissions & Responsibilities
=====================================

Read
----

Full Permission List
^^^^^^^^^^^^^^^^^^^^

+----------------------------+-----------+
| Action                     | Allowed   |
+============================+===========+
| View repository content    | ✅        |
+----------------------------+-----------+
| Create issues              | ✅        |
+----------------------------+-----------+
| Comment on issues and PRs  | ✅        |
+----------------------------+-----------+
| View pull requests         | ✅        |
+----------------------------+-----------+
| Push code                  | ❌        |
+----------------------------+-----------+
| Create PRs                 | ❌        |
+----------------------------+-----------+
| Merge anything             | ❌        |
+----------------------------+-----------+

**When to Use**: Add people who only need to report issues and view code.

---

Triage
------

Full Permission List
^^^^^^^^^^^^^^^^^^^^

+----------------------------+-----------+
| Action                     | Allowed   |
+============================+===========+
| Everything in Read         | ✅        |
+----------------------------+-----------+
| Manage labels              | ✅        |
+----------------------------+-----------+
| Manage assignees           | ✅        |
+----------------------------+-----------+
| Manage milestones          | ✅        |
+----------------------------+-----------+
| Mark as duplicate          | ✅        |
+----------------------------+-----------+
| Close/reopen issues        | ✅        |
+----------------------------+-----------+
| Push code                  | ❌        |
+----------------------------+-----------+
| Create PRs                 | ❌        |
+----------------------------+-----------+
| Merge anything             | ❌        |
+----------------------------+-----------+

**When to Use**: Add people who manage the issue/ticket workflow but don't
write code.

---

Write
-----

Full Permission List
^^^^^^^^^^^^^^^^^^^^

+----------------------------------+-------------------------------------------+
| Action                           | Allowed                                   |
+==================================+===========================================+
| Everything in Triage             | ✅                                        |
+----------------------------------+-------------------------------------------+
| Push code to branches            | ✅                                        |
+----------------------------------+-------------------------------------------+
| Create pull requests             | ✅                                        |
+----------------------------------+-------------------------------------------+
| Review PRs                       | ✅                                        |
+----------------------------------+-------------------------------------------+
| Approve PRs                      | ✅                                        |
+----------------------------------+-------------------------------------------+
| Merge PRs                        | ❌ (controlled by branch protection)      |
+----------------------------------+-------------------------------------------+
| Delete branches                  | ❌                                        |
+----------------------------------+-------------------------------------------+
| Manage settings                  | ❌                                        |
+----------------------------------+-------------------------------------------+

**When to Use**: Add developers who create PRs but need approval before
merging.

**Branch Protection Required**:

.. code-block::

   ✅ Require pull request reviews before merging (1+ approval)
   ✅ Require status checks to pass before merging
   ✅ Require branches to be up to date

---

Maintain
--------

Full Permission List
^^^^^^^^^^^^^^^^^^^^

+----------------------------------+-------------------------------------------+
| Action                           | Allowed                                   |
+==================================+===========================================+
| Everything in Write              | ✅                                        |
+----------------------------------+-------------------------------------------+
| Merge pull requests              | ✅                                        |
+----------------------------------+-------------------------------------------+
| Manage branches and protections  | ✅                                        |
+----------------------------------+-------------------------------------------+
| Create releases                  | ✅                                        |
+----------------------------------+-------------------------------------------+
| Dismiss pull request reviews     | ❌ (controlled by branch protection)      |
+----------------------------------+-------------------------------------------+
| Override branch protections      | ❌ (controlled by branch protection)      |
+----------------------------------+-------------------------------------------+
| Delete repository                | ❌                                        |
+----------------------------------+-------------------------------------------+
| Manage access/members            | ❌                                        |
+----------------------------------+-------------------------------------------+

**When to Use**: Add team leads who can approve, merge, and manage the PR
workflow.

**Branch Protection Required** (for "merge only when tests pass, no override"):

.. code-block::

   ✅ Require pull request reviews before merging
   ✅ Require status checks to pass before merging
   ✅ Require branches to be up to date
   ✅ Include administrators (enforces restrictions on all)
   ❌ Allow force pushes

---

Admin
-----

Full Permission List
^^^^^^^^^^^^^^^^^^^^

+----------------------------------+-------------------------------------------+
| Action                           | Allowed                                   |
+==================================+===========================================+
| Everything in Maintain           | ✅                                        |
+----------------------------------+-------------------------------------------+
| Full repository control          | ✅                                        |
+----------------------------------+-------------------------------------------+
| Override all branch protections  | ✅                                        |
+----------------------------------+-------------------------------------------+
| Delete repository                | ✅                                        |
+----------------------------------+-------------------------------------------+
| Manage all repository settings   | ✅                                        |
+----------------------------------+-------------------------------------------+
| Manage repository access         | ✅                                        |
+----------------------------------+-------------------------------------------+

**When to Use**: Only for repository owners.

---

How to Add Users to This Repository
====================================

1. Go to **Settings → Collaborators and teams** (or **Access** in newer
   GitHub UI)
2. Click **Add people**
3. Search for the GitHub username
4. Select the appropriate role from the dropdown
5. Click **Add [username] to the repository**

---

Permission Assignment Examples
===============================

Example 1: New Team Member (Contributor)
----------------------------------------
- **Role**: `Write`
- **Access**: Can create PRs and review code
- **Restrictions**: Cannot merge until approved and tests pass
- **Branch Protection**: Required approvals enforce this

Example 2: IMAP SDC Manager(s)
------------------------------
- **Role**: `Triage`
- **Access**: Can manage issue labels, assignees, milestones
- **Restrictions**: Cannot write code or modify PRs
- **Use Case**: Triaging bugs and managing workflow

Example 3: Team Lead
--------------------
- **Role**: `Maintain`
- **Access**: Can merge PRs, manage releases
- **Restrictions**: Cannot override failed tests (with branch protection)
- **Use Case**: Merges reviewed and tested code

Example 4: Repository Owner
---------------------------
- **Role**: `Admin`
- **Access**: Full control
- **Use Case**: Repository administration and settings

---

Questions?
==========

If you have questions about your repository access or need a different permission level, please contact the IMAP SDC team.