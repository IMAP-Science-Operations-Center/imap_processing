.. _development:

Development
===========

To contribute to imap_processing, you can fork the GitHub repository,
add your code, and make a pull request. If you are adding additional functionality,
you should also include a test with your enhancement.

imap_processing is designed to be run in a containerized environment, so
the dependencies will be locked in after a release and the data products will
be versioned appropriately to correspond with the code that produced them.

A typical development workflow might look like the following:

.. code:: bash

    # Install the development dependencies
    pip install .[dev]

    # Install the pre-commit hooks
    pre-commit install

    # Update the code on a feature branch
    git checkout -b my-cool-feature

    # Run the tests
    pytest

    # Commit the changes and push to your remote repository
    git add my-file
    git commit
    git push -u origin my-cool-feature

    # Go to GitHub and open a pull request!

.. toctree::
    :maxdepth: 1

    doc-overview
    release-workflow

.. toctree::
    :maxdepth: 2

    style-guide