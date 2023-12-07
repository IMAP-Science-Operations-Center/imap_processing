.. _naming-conventions:

Naming Conventions
------------------

Data Product File Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following describes the filename conventions for IMAP data products. These conventions follow the `recommendations
provided by the SPDF <https://spdf.gsfc.nasa.gov/guidelines/filenaming_recommendations.html>`_.

The general filename convention is as follows::

    imap_<instrument>_<datalevel>_<descriptor>_<startdate>_<enddate>_<version>.<format>

* ``<instrument>`` is the IMAP instrument associated with the data product. Acceptable values include: ``codice``,
  ``glows``, ``hi-45``, ``hi-90``, ``hit``, ``idex``, ``lo``, ``mag``, ``swapi``, ``swe``, ``ultra-45``, and
  ``ultra-90``

* ``<datalevel>`` is the data level for the data product.  Acceptable values depend on the instrument:

  * ``codice``: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``
  * ``glows``: TBD
  * ``hi-45``: TBD
  * ``hi-90``: TBD
  * ``hit``: TBD
  * ``idex``: TBD
  * ``lo``: TBD
  * ``mag``: TBD
  * ``swapi``: TBD
  * ``swe``: TBD
  * ``ultra-45``: TBD
  * ``ultra-90``: TBD

* ``<descriptor>`` stores information specific to the instrument and can store any information that is relevant. For
  example, it could store the orbit number, pointing number, or data product name.
* ``<startdate>`` is the date of the 'earliest' data within the data product, in the format of ``YYYYMMDD``.
* ``<enddate>`` is the date of the 'latest' data within the data product if the data product spans more than one day.
  If the data does not span multiple days, then the ``<enddate>`` will be the same as the ``<startdate>``. The format
  is also ``YYYYMMDD``.
* ``<version>`` stores the version of the data product in the format is ``vXX-YY``.  See the versioning conventions
  described in the :ref:`data product versioning <data-product-versioning>` documentation for further details.
* ``<format>`` is the data format. For ``l0`` data products, the format is ``pkts``. For data levels ``l1`` and higher,
  the format is ``cdf``.

Here are a few examples of acceptable filenames:

* ``imap_codice_l1a_lo_20261206_20261207_v01-01.cdf``
* ``imap_swe_l2_descriptor_20261206_20261206_v01-02.cdf``


Repository Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The repository shall adhere to specific naming conventions to allow for more consistency:

* Repositories shall have branches named ``main`` and ``dev``, instead of ``master`` and ``development``
* Repositories shall be named with underscores instead of hyphens (e.g. ``imap_processing`` instead of
  ``imap-processing``)