Code for creating CDF attributes
================================

The IMAP project has tools for organizing CDF file attributes in Python code. If the code
is using :ref:`xarray-cdf` to generate CDF files, the classes in ``imap_processing/cdf``
can be used to set attributes for the files.

Overview
--------

The main classes are in ``global_attrs.py``. These classes can be used to describe the
attributes of the file and the datasets, including automatically generating attribute
dictionaries for :ref:`xarray-cdf`

The classes beginning with "Global" are used for global file attributes, and should be
included with each file. The other classes are optional and can be used in combination
depending on each file's requirements.


