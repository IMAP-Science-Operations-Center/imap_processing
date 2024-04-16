Versioning
----------

Software Versioning
^^^^^^^^^^^^^^^^^^^

Any changes pushed to the ``main`` branch (regardless of if the change is instrument-specific, or project-wide) should
be tagged with a version number. This project uses `semantic versioning <https://semver.org/>`_, in which version
numbers should follow a convention of ``x.y.z``, where:

::

    x = The main version number. Increase when making incompatible API changes.
    y = The feature number. Increase when change contains a new feature with or without bug fixes.
    z = The hotfix number. Increase when change only contains bug fixes.

This convention was discussed and decided on in `this GitHub issue
<https://github.com/IMAP-Science-Operations-Center/imap_processing/issues/43>`__.


Data Product Versioning
^^^^^^^^^^^^^^^^^^^^^^^

IMAP data products are versioned with the convention ``v<num>``, where ``<num>`` is a 3 digit integer. The data version
is increased when data are reprocessed due to changes in the processing algorithm, or changes in ancillary files or
calibrations (for example, updated SPICE kernels).

Version numbers are specific to an instrument. In other words, each IMAP instrument has their own version number that
is only updated when that instrument's algorithm is changed, or that instrument's data are reprocessed. The version is
updated independently of the other IMAP instruments.

This convention was discussed and decided on in `this GitHub issue
<https://github.com/IMAP-Science-Operations-Center/imap_processing/issues/48>`__.
