.. _versioning:

Versioning
----------

Any changes pushed to the ``main`` branch should be tagged with a version number. The version number convention is
``x.y.z``, where

::

    x = The main version number.  Increase when making incompatible API changes.
    y = The feature number.  Increase when change contains a new feature with or without bug fixes.
    z = The hotfix number. Increase when change only contains bug fixes.