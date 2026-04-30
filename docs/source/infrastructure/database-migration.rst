.. _database-migrations:

Database Migrations with Alembic
=================================

We use Alembic for database migrations. Alembic compares the database schema defined in the models to the current
state of the database and generates revision files containing the changes needed to update the schema.

Docs: https://alembic.sqlalchemy.org/en/latest/index.html

----

**Full Migration Workflow (DEV and PROD)**
------------------------------------------

.. warning::

    Be **extremely careful** to never mix up DEV and PROD environments.
    Double-check your ``DATABASE_URL`` before running any commands.
    Mistakes can cause data loss or downtime.

Follow these steps **in order** for every migration:

1. **Make your changes to the models**

   Make your desired changes to the models in ``sds_data_manager/lambda_code/SDSCode/database/models.py``

2. **Create a revision (DEV ONLY)**

    This step compares the models in the code to the current database schema in DEV and generates a migration file.
    A Migration file is a Python script that defines the changes to be made to the database schema. It contains an
    "upgrade" function that applies the changes and a "downgrade" function that reverts them.

   .. important::

      Never create migrations on PROD. Only do this in your DEV environment.

   .. code-block:: bash

       alembic revision --autogenerate -m "description"

   This generates a new file in ``alembic/versions/``. **Always review the file before applying** — Alembic cannot
   detect all changes. See what it misses here:
   https://alembic.sqlalchemy.org/en/latest/autogenerate.html#what-does-autogenerate-detect-and-what-does-it-not-detect

3. **Preview the SQL (Dry Run)**

   To see the SQL that would be run without actually applying it:

   .. code-block:: bash

       alembic upgrade head --sql

   This is safe to run at any time — it does not modify the database.

4. **Apply to DEV**

   .. warning::

       Always test on DEV first before applying to PROD.

   .. code-block:: bash

       export DATABASE_URL=dev_database_url
       alembic upgrade head

   The first time ``alembic upgrade head`` is run it will create an ``alembic_version`` table in the database
   to track the current schema version.

5. **Test on DEV**

   Test your changes thoroughly to ensure the migration works as expected. Connect to the database on DataGrip and
    verify the schema changes.

6. **Downgrade DEV to verify downgrade path**

   .. code-block:: bash

       alembic downgrade -1

   Make sure the downgrade works as expected.

7. **Commit and Push Migration Files**

   Commit your code changes **and** the new migration file(s).

8. **Open a PR and have it reviewed**

   Open a pull request for your changes, have them reviewed and approved, and merge into the main branch.

9. **Apply to DEV and then PROD**

   - Once your changes are merged, rebase against dev to pull in the new migration file.
   - Set the correct ``DATABASE_URL`` for DEV.

   .. code-block:: bash

       export DATABASE_URL=dev_database_url
       alembic upgrade head

    - Test on DEV again to confirm everything works after merging.

    - Then set the ``DATABASE_URL`` for PROD and apply.

   .. code-block:: bash

       export DATABASE_URL=dev_database_url
       alembic upgrade head



10. **Monitor both databases after applying**

    Confirm there are no issues in either environment. Both Dev and Prod should be synced to the same schema version in
    the code.

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

Export the URL:

.. code-block:: bash

    export DATABASE_URL=postgresql://username:password@host:port/database_name

Test the connection:

.. code-block:: bash

    alembic current

.. warning::

    Always double check which database your ``DATABASE_URL`` is pointing to before running any commands.
    It is easy to accidentally run against the wrong database. Always test on DEV first before applying to PROD.


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
