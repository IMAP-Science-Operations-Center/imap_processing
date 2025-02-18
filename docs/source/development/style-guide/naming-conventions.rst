.. _naming-conventions:

Naming Conventions
------------------

Data Product File Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following describes the filename conventions for IMAP data products. These conventions follow the `recommendations
provided by the SPDF <https://spdf.gsfc.nasa.gov/guidelines/filenaming_recommendations.html>`_.

The general filename convention is as follows::

    imap_<instrument>_<data_level>_<descriptor>_<startdate>(-<repointing>)_<version>.<extension>

* ``<instrument>`` is the IMAP instrument associated with the data product. Acceptable values include: ``codice``,
  ``glows``, ``hi``, ``hit``, ``idex``, ``lo``, ``mag``, ``swapi``, ``swe``, and ``ultra``

* ``<datalevel>`` is the data level for the data product.  Acceptable values depend on the instrument:

  * CoDICE: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``
  * GLOWS: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3a``, ``l3b``, ``l3c``, ``l3d``
  * HIT: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``
  * IDEX: ``l0``, ``l1a``, ``l1b``, ``l2a``, ``l2b``, ``l3``
  * IMAP-Hi: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``, ``l3``
  * IMAP-Lo: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``
  * IMAP-Ultra: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2``, ``l3``
  * MAG: ``l0``, ``l1a``, ``l1b``, ``l1c``, ``l2pre``, ``l2``
  * SWAPI: ``l0``, ``l1``, ``l2``, ``l3a``, ``l3b``
  * SWE: ``l0``, ``l1a``, ``l1b``, ``l2``, ``l3``

* ``<descriptor>`` stores information specific to the instrument and can store any information that is relevant. For
  example, it will typically contain the data product name, the sensor name, the timespan of the data, etc. This field
  can have any text or numbers as long as it doesn't include underscores. The following is a list of expected
  descriptors for each instrument:

  * CoDICE: ``hskp``, ``lo-counters-aggregated``, ``lo-counters-singles``, ``hi-counters-aggregated``,
    ``hi-counters-singles``, ``hi-priorities``, ``lo-sw-priority``, ``lo-nsw-priority``, ``lo-sw-angular``,
    ``lo-nsw-angular``, ``lo-pha``, ``hi-pha``, ``lo-sw-species``, ``lo-nsw-species``, ``hi-omni``, ``hi-sectored``,
    ``lo-ialirt``, ``hi-ialirt``
  * GLOWS: ``histogram``, ``de``
  * HIT: TBD
  * IDEX: ``sci``
  * IMAP-Hi: ``45sensor-counts``, ``90sensor-counts``, ``45sensor-flux-map-3month`` (more TBD)
  * IMAP-Lo: TBD
  * IMAP-Ultra: ``45sensor-counts``, ``90sensor-counts`` (more TBD)
  * MAG: ``normal``, ``burst``
  * SWAPI: ``sci-1min``, ``hk``
  * SWE: ``sci``

* ``<startdate>`` is the date of the 'earliest' data within the data product, in the format of ``YYYYMMDD``.
* ``<repointing>`` is an optional component that describes the repointing number, in the format of ``repoint<num>``,
  where ``<num>`` is a 5 digit integer (e.g. ``repoint00001``).
* ``<version>`` stores the version of the data product in the format is ``v<num>``, where ``<num>`` is a 3 digit
  integer (e.g. ``v001``).  See the versioning conventions described in the
  :ref:`data product versioning <Data Product Versioning>` documentation for further details.
* ``<extension>`` is the data extension. For ``l0`` data products, the extension is ``pkts``. For data levels ``l1`` and
  higher, the extension is ``cdf``.

Here are a few examples of acceptable filenames:

* ``imap_idex_l0_raw_20261206_v001.pkts``

  An IDEX L0 packet file containing data from the single day of ``2026-12-06``

* ``imap_codice_l1a_lo-pha_20261206_v001.cdf``

  A version ``001`` CoDICE-Lo L1a data product called ``pha`` containing data from ``2026-12-06``

* ``imap_glows_l1a_counts_20261206-repoint00002_v001.cdf``

  A version ``001`` GLOWS L1a data product for ``counts`` containing data from ``2026-12-06`` taken from repointing
  position ``00002``

* ``imap_ultra_l2_45sensor-flux-map-3month_20261206_v001.cdf``

  A version ``001`` ULTRA L2 flux map for the 45 sensor head for data taken over a 3 month span starting on ``20261206``


Repository Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The repository shall adhere to specific naming conventions to allow for more consistency:

* Repositories shall have branches named ``main`` and ``dev``, instead of ``master`` and ``development``
* Repositories shall be named with underscores instead of hyphens (e.g. ``imap_processing`` instead of
  ``imap-processing``)
