SPICE Files
===========

The IMAP SDC provides two REST API endpoints for accessing SPICE kernel data:
``/spice-query`` for querying kernel metadata and ``/metakernel`` for retrieving
a ready-to-use metakernel (or a list of kernel filenames) that covers a
requested time range.

Both endpoints are accessible from the base URL: ``https://api.imap-mission.com``

.. _spice-query-endpoint:

SPICE Query
-----------

The ``/spice-query`` endpoint returns metadata for SPICE kernels stored in the
SDC database. Results can be filtered by kernel type, time range, ingestion
date, or filename.

.. openapi:: openapi.yml
   :group:
   :include: /spice-query

**Example Usage:**

.. code-block:: bash

   # Query for all attitude_history kernels covering a time range
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/spice-query?start_time=315576066&end_time=4575787269&type=attitude_history"

   # Query for a specific kernel by filename
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/spice-query?file_name=naif0012.tls"

   # Query for the latest version of the spacecraft_clock kernel
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/spice-query?type=spacecraft_clock&latest=True"

**Possible Responses:**

.. code-block:: json

   [
      {
        "file_name": "ck/imap_2025_274_2025_276_001.ah.bc",
        "file_root": "imap_2025_274_2025_276_.ah.bc",
        "kernel_type": "attitude_history",
        "version": 1,
        "min_date_j2000": 812631669.2765242,
        "max_date_j2000": 812725268.1910387,
        "file_intervals_j2000": [
          [
            812631669.2765242,
            812725268.1910387
          ]
        ],
        "min_date_datetime": "2025-10-01, 23:00:00",
        "max_date_datetime": "2025-10-03, 00:59:59",
        "file_intervals_datetime": [
          [
            "2025-10-01T23:00:00.094178+00:00",
            "2025-10-03T00:59:59.008694+00:00"
          ]
        ],
        "min_date_sclk": "1/0497055600:00000",
        "max_date_sclk": "1/0497149199:00000",
        "file_intervals_sclk": [
          [
            "1/0497055600:00000",
            "1/0497149199:00000"
          ]
        ],
        "sclk_kernel": "/tmp/naif0012.tls",
        "lsk_kernel": "/tmp/imap_sclk_0147.tsc",
        "ingestion_date": "2026-04-06, 23:57:54",
        "timestamp": 1775519874
      },
   ]

.. code-block:: json

   {"statusCode": 400, "body": "<param> is not a valid query parameter. Valid query parameters are: ['file_name', 'start_time', 'end_time', 'type', 'latest', 'start_ingest_date', 'end_ingest_date']"}

.. _metakernel-endpoint:

Metakernel
----------

The ``/metakernel`` endpoint builds and returns a SPICE metakernel covering a
requested time range. By default the response is a ``.tm`` metakernel file that
can be loaded directly with ``spiceypy.furnsh()``. Passing ``list_files=True``
returns a JSON list of kernel filenames instead.

.. openapi:: openapi.yml
   :group:
   :include: /metakernel

**Example Usage:**

.. code-block:: bash

   # Retrieve a metakernel covering a time range for selected kernel types
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/metakernel?start_time=797949057&end_time=798035454&file_types=leapseconds,attitude_history"

   # Get only a list of kernel filenames (no metakernel wrapper)
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/metakernel?start_time=797949057&end_time=798035454&file_types=leapseconds,spacecraft_clock&list_files=True"

   # Retrieve a metakernel with a custom spice_path prefix in the output
   curl -X GET -H "Accept: application/json" \
     "https://api.imap-mission.com/metakernel?start_time=797949057&end_time=798035454&spice_path=/my/path/imap"

**Possible Responses:**

Metakernel file (default, ``list_files`` omitted or ``False``):

.. code-block:: text

    \begintext

    This is the most up to date Metakernel as of
    2026-04-13 18:19:21.912444+00:00.

    This attempts to cover data from
    797949057 to 798035454
    seconds since J2000.

    \begindata

      KERNELS_TO_LOAD = ( 'lsk/naif0012.tls',
                          'fk/imap_130.tf',
                          'fk/imap_science_120.tf',
                          'sclk/imap_sclk_0147.tsc',
                          'spk/de440.bsp'
                        )

    \begintext

List of filenames (``list_files=True``):

.. code-block:: json

   ["naif0012.tls", "imap_130.tf", "imap_science_120.tf", "imap_sclk_0147.tsc", "de440.bsp"]

Coverage gap error (``require_coverage=True`` and gaps exist):

.. code-block:: json

   {
     "statusCode": 422,
     "body": {
       "leapseconds_category": [],
       "planetary_constants_category": [],
       "imap_frames_category": [],
       "science_frames_category": [],
       "spacecraft_clock_category": [],
       "earth_attitude_category": [],
       "planetary_ephemeris_category": [],
       "spacecraft_ephemeris_category": [[797990001, 798035454]],
       "spacecraft_attitude_category": [[797949057, 797990000]],
       "pointing_attitude_category": []
     }
   }

Each key is a SPICE kernel category. Empty lists indicate full coverage; non-empty
lists contain one or more ``[start_time, end_time]`` intervals (in TDB seconds
since J2000) where no kernel was found.

No files found:

.. code-block:: json

   {"statusCode": 404, "body": "No files found."}

Kernel Types
------------

The following kernel type values are accepted by both the ``type`` parameter of
``/spice-query`` and the ``file_types`` parameter of ``/metakernel``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Kernel Type
     - Description
   * - ``leapseconds``
     - Leapseconds kernel (LSK)
   * - ``planetary_constants``
     - Planetary constants kernel (PCK)
   * - ``imap_frames``
     - IMAP spacecraft frames kernel (FK)
   * - ``science_frames``
     - Science instrument frames kernel (FK)
   * - ``spacecraft_clock``
     - Spacecraft clock kernel (SCLK)
   * - ``earth_attitude``
     - Earth attitude kernel
   * - ``planetary_ephemeris``
     - Planetary ephemeris kernel (SPK)
   * - ``ephemeris_reconstructed``
     - Reconstructed spacecraft ephemeris (SPK)
   * - ``ephemeris_nominal``
     - Nominal spacecraft ephemeris (SPK)
   * - ``ephemeris_predicted``
     - Predicted spacecraft ephemeris (SPK)
   * - ``ephemeris_90days``
     - 90-day spacecraft ephemeris (SPK)
   * - ``ephemeris_long``
     - Long-term spacecraft ephemeris (SPK)
   * - ``ephemeris_launch``
     - Launch ephemeris (SPK)
   * - ``attitude_history``
     - Historical spacecraft attitude kernel (CK)
   * - ``attitude_predict``
     - Predicted spacecraft attitude kernel (CK)
   * - ``pointing_attitude``
     - Pointing attitude kernel (CK)
