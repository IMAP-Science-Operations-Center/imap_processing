.. _database-migrations:

Database Migrations with Alembic
=================================

We use Alembic for database migrations. Alembic compares the database schema defined in the models to the current
state of the database and generates revision files containing the changes needed to update the schema.

Docs: https://alembic.sqlalchemy.org/en/latest/index.html

----

**Migration Workflow (Separated: DEV Test and PROD Deploy)**
-------------------------------------------------------------

.. warning::

    Be **extremely careful** to never mix up DEV and PROD.
    ``DATABASE_URL`` is what determines which database you are operating on.
    Before any command that can change schema/data, verify:
    1) ``DATABASE_URL`` points to the intended RDS instance (DEV or PROD)
    2) ``alembic current`` matches the intended environment

A) DEV Migration + Testing (Steps 1–6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Make your changes to the models**

   Make your desired changes to the models in ``sds_data_manager/lambda_code/SDSCode/database/models.py``.

2. **Create a revision (DEV ONLY)**

   This compares the database RDS instance in aws to the DEV schema and generates a migration file with ``upgrade`` and ``downgrade``.

   .. important::

      Never create revisions on PROD.

   .. code-block:: bash

       export DATABASE_URL="postgresql://user_name:password@host:5432/db_name"

       alembic revision --autogenerate -m "description"

   This generates a new file in ``alembic/versions/``. **Always review the file before applying** — Alembic cannot
   detect all changes. See what it misses here:
   https://alembic.sqlalchemy.org/en/latest/autogenerate.html#what-does-autogenerate-detect-and-what-does-it-not-detect

3. **Preview the SQL (Dry Run)**

   To see the SQL that would be run without actually applying it:

   .. code-block:: bash

       export DATABASE_URL="postgresql://user_name:password@host:5432/db_name"

       alembic upgrade head --sql

4. **Apply migration to DEV**

   .. code-block:: bash

       export DATABASE_URL="postgresql://user_name:password@host:5432/db_name"

       alembic current
       alembic upgrade head

5. **Test on DEV**

   Test your changes thoroughly to ensure the migration works as expected. Connect to the database on DataGrip and
    verify the schema changes.

6. **Downgrade DEV to verify downgrade path**

   .. code-block:: bash

       export DATABASE_URL="postgresql://user_name:password@host:5432/db_name"

       alembic downgrade -1

   Make sure the downgrade works as expected.

B) PROD Verification + Deployment (Steps 7–10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7. **Commit and push migration file + code changes**

8. **Open PR, review, approve, merge**

   First re-check DEV after merge:

   .. code-block:: bash

       export DATABASE_URL="postgresql://user_name:password@host:5432/db_name"

       alembic current
       alembic upgrade head

   Then repeat **Step 3 (Dry Run)**, but on PROD:

   .. warning::

      PROD safety gate (must all be true):
      - ``DATABASE_URL=prod_database_url``
      - URL host/DB clearly match the PROD RDS instance
      - ``alembic current`` is the expected PROD revision

   .. code-block:: bash

       export DATABASE_URL=prod_database_url
       alembic current
       alembic upgrade head --sql
       alembic upgrade head

10. **Post-deploy monitoring (DEV + PROD)**

    Confirm both environments are healthy and at expected revision:
    run ``alembic current`` and functional checks.

----

Setup (This is performed once)
------------------------------

Ensure you are in the root of the ``sdc-data-manager`` repo.

.. code-block:: bash

    alembic init alembic

Edit ``alembic/env.py`` and add the following at the top:

.. code-block:: python

    # Override the URL from environment variable
    # Do not hardcode the DATABASE_URL in the config file for security reasons
    config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

Import the database schema and point ``target_metadata`` to ``Base.metadata``:

.. code-block:: python

    from sds_data_manager.lambda_code.SDSCode.database.models import Base
    target_metadata = Base.metadata

----

Connecting to the Database
---------------------------

The ``DATABASE_URL`` follows this format:

.. code-block:: text

    driver://username:password@host:port/database_name

We have different databases for DEV and PROD, so you will need to set the ``DATABASE_URL`` environment variable before
running any Alembic commands.

To find the credentials:
1. Log into AWS account (dev or prod)
2. Go to **Secrets Manager** → **Secrets** → ``sdp-database-cred``
3. Click **Retrieve secret value**
4. Construct the URL from those values
5. Verify the URL points to the intended RDS instance (DEV vs PROD)

Export the URL:

.. code-block:: bash

    export DATABASE_URL=postgresql://username:password@host:port/database_name

Test the connection:

.. code-block:: bash

    alembic current

.. warning::

    Always double check which database your ``DATABASE_URL`` is pointing to before running any commands.
    ``DATABASE_URL`` is the environment boundary between DEV and PROD.
    Always test on DEV first before applying to PROD.


Downtime Considerations
-----------------------

Most schema changes in PostgreSQL are non-blocking, but some operations will lock tables and cause downtime:

- **Adding a NOT NULL column without a default** — locks the table
- **Dropping a column** — locks the table briefly
- **Renaming a column or table** — locks the table
- **Adding a non-concurrent index** — locks the table for writes
- **Changing a column type** — locks the table and may require a full rewrite

To minimize downtime on PROD, consider running heavy migrations during off-hours or using ``CONCURRENTLY``
for index operations where possible.

----

Troubleshooting
----------------

Can't locate revision identified by 'xxxx'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The database has a revision recorded that no longer has a corresponding file. This usually means a migration
file was deleted after being applied. Connect directly to the database and clear the version table:

.. code-block:: sql

    DELETE FROM alembic_version;

Then stamp to a known good revision:

.. code-block:: bash

    alembic stamp head

Target database is not up to date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are pending migrations. Run ``alembic current`` to see where the DB is, then ``alembic upgrade head``
to apply.
