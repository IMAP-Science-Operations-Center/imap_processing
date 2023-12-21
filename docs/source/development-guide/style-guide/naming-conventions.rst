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

  * CoDICE: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``
  * GLOWS: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3a``, ``l3b``, ``l3c``, ``l3d``
  * HIT: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``
  * IDEX: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``, ``l3``
  * IMAP-Hi: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``, ``l3``
  * IMAP-Lo: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``
  * IMAP-Ultra: ``l0``, ``l1a``, ``l1b``, ``l1ca``, ``l1cb``, ``l1c``, ``l1d``, ``l2``, ``l3``
  * MAG: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2pre``, ``l2``
  * SWAPI: ``l0``, ``l1``, ``l2``, ``l3a``, ``l3b``
  * SWE: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``

* ``<descriptor>`` stores information specific to the instrument and can store any information that is relevant. For
  example, it will typically contain the data product name, and additionally may contain the pointing number (e.g.
  ``burst-7`` for 'burst' mode and the 7th repointing). This field can have any text or numbers as long as it doesn't
  include underscores. The following is a list of expected descriptors for each instrument:

  * CoDICE: ``hi``, ``lo``
  * GLOWS: ``histogram``, ``de``, ``<pointing number>``
  * HIT: TBD
  * IDEX: TBD
  * IMAP-Hi: TBD
  * IMAP-Lo: TBD
  * IMAP-Ultra: ``<pointing number>``
  * MAG: ``normal``, ``burst``
  * SWAPI: TBD
  * SWE: TBD

* ``<startdate>`` is the date of the 'earliest' data within the data product, in the format of ``YYYYMMDD``.
* ``<enddate>`` is the date of the 'latest' data within the data product if the data product spans more than one day.
  If the data does not span multiple days, then the ``<enddate>`` will be the same as the ``<startdate>``. The format
  is also ``YYYYMMDD``.
* ``<version>`` stores the version of the data product in the format is ``vXX-YY``.  See the versioning conventions
  described in the :ref:`data product versioning <Data Product Versioning>` documentation for further details.
* ``<format>`` is the data format. For ``l0`` data products, the format is ``pkts``. For data levels ``l1`` and higher,
  the format is ``cdf``.

Here are a few examples of acceptable filenames:

* ``imap_idex_l0_20261206_20261206_v01-01.pkts``

  An IDEX L0 packet file containing data from the single day of ``2026-12-06``

* ``imap_codice_l1a_lo_20261206_20261207_v01-01.cdf``

  A version ``01-01`` CoDICE L1a data product called ``lo`` containing data from ``2026-12-06`` to ``2026-12-07``

* ``imap_swe_l2_burst_20261206_20261206_v01-02.cdf``

  A MAG L1c data product for 'burst' mode containing a single day's worth of data on ``2026-12-06``. The version ``01-02``
  indicates the software/algorithm version is ``01`` and the data dependency information has been updated once to
  make the data version ``02``


Repository Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The repository shall adhere to specific naming conventions to allow for more consistency:

* Repositories shall have branches named ``main`` and ``dev``, instead of ``master`` and ``development``
* Repositories shall be named with underscores instead of hyphens (e.g. ``imap_processing`` instead of
  ``imap-processing``)
