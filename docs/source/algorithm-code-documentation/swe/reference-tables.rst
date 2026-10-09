.. _swe-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Rule of thumb
-------------

Big tables are **not** reproduced in these pages. They go stale, and every one
of them already exists in a machine-readable form that the code actually reads.
This page tells you which file, and which page of the algorithm document if you
have a copy.

The one exception is the **count decompression table**, reproduced in full in
:ref:`swe-decompression`, because it is small, frozen in flight software, and
you cannot debug a count without it.

Authority order, highest first:

#. **The code and its ancillary files.** What runs.
#. **The XTCE** (``swe_packet_definition.xml``). What is parsed.
#. **The algorithm document.** What the instrument team intends.

Appendix B of the algorithm document says so itself: "Note: packet definitions
continue to change as the instrument development progresses."

Machine-readable tables in the repository
-----------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - File
     - Contains
   * - ``imap_processing/swe/packet_definitions/swe_packet_definition.xml``
     - ``SWE_APP_HK`` (APID 1330), ``SWE_CEM_RAW`` (APID 1334),
       ``SWE_SCIENCE`` (APID 1344). Field names, bit widths, and derived-value
       conversions.
   * - ``imap_processing/ialirt/packet_definitions/ialirt_swe.xml``
     - The SWE I-ALiRT fields: ``SWE_CEM<1-7>_E<1-4>``, ``SWE_SHCOARSE``,
       ``SWE_ACQ_SEC``, ``SWE_ACQ_SUB``, ``SWE_SEQ``, ``SWE_NOM_FLAG``,
       ``SWE_OPS_FLAG``.

APIDs
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 14 26 60

   * - APID
     - ``SWEAPID`` member
     - Packet
   * - 1330
     - ``SWE_APP_HK``
     - Application housekeeping. Feeds ``imap_swe_l1a_hk`` and
       ``imap_swe_l1b_hk``.
   * - 1334
     - ``SWE_CEM_RAW``
     - Engineering-mode 1-second CEM counts. Feeds
       ``imap_swe_l1a_cem-raw``.
   * - 1344
     - ``SWE_SCIENCE``
     - One quarter cycle. Feeds everything else.

The SWE I-ALiRT packet is not in ``SWEAPID``; it is handled by the I-ALiRT
packet machinery.

``SWE_SCIENCE`` field list
^^^^^^^^^^^^^^^^^^^^^^^^^^

From the XTCE, in order. This is the full set; every one of them reaches L1B as
an ``(epoch, cycle)`` metadata variable.

.. code-block:: text

   SHCOARSE               ACQ_START_COARSE       ACQ_START_FINE
   CEM_NOMINAL_ONLY       SPIN_PERIOD_VALIDITY   SPIN_PHASE_VALIDITY
   SPIN_PERIOD_SOURCE     SETTLE_DURATION        ACQ_DURATION
   SPIN_PHASE             SPIN_PERIOD            REPOINT_WARNING
   HIGH_COUNT             STIM_ENABLED           QUARTER_CYCLE
   ESA_TABLE_NUM          ESA_ACQ_CFG            THRESHOLD_DAC
   STIM_CFG_REG           SCIENCE_DATA           CKSUM

Ancillary CSVs
^^^^^^^^^^^^^^

Operational copies come from the SDC ancillary store; representative fixtures
live in ``imap_processing/tests/swe/lut/``. Described in full in
:ref:`swe-ancillary`.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Fixture
     - Columns
   * - ``imap_swe_esa-lut_20250301_v000.csv``
     - ``table_idx, esa_step, esa_v, v_index, ialirt``. 8 tables x 48 rows.
   * - ``imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv``
     - ``met_time, cem1 ... cem7``.
   * - ``imap_swe_eu-conversion_20240510_v000.csv``
     - ``index, packetName, mnemonic, convertAs, segNumber, lowValue,
       highValue, c0 ... c7``. 63 rows.
   * - ``checker-board-indices.csv``
     - The expected ``(24, 30)`` checkerboard index map, used only as a test
       oracle for ``get_checker_board_pattern()``.

Validation and test data
^^^^^^^^^^^^^^^^^^^^^^^^

``imap_processing/tests/swe/``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - What it is
   * - ``l0_data/2024051010_SWE_SCIENCE_packet.bin``
     - Real ``SWE_SCIENCE`` packets, 2024-05-10.
   * - ``l0_data/2024051010_SWE_HK_packet.bin``
     - Real ``SWE_APP_HK`` packets.
   * - ``l0_data/2024051011_SWE_CEM_RAW_packet.bin``
     - Real ``SWE_CEM_RAW`` packets.
   * - ``l0_validation_data/idle_export_raw.SWE_SCIENCE_*.csv``
     - GSEOS export of the science packets in **raw** units.
   * - ``l0_validation_data/idle_export_eu.SWE_SCIENCE_*.csv``
     - Same in **engineering** units. The raw/EU pair is what validates
       decompression.
   * - ``l0_validation_data/idle_export_eu.SWE_APP_HK_*.csv``
     - HK validation.
   * - ``l0_validation_data/idle_export_eu.SWE_CEM_RAW_*.csv``
     - CEM raw validation.

Constants in code
^^^^^^^^^^^^^^^^^

``imap_processing/swe/utils/swe_constants.py`` holds every SWE magic number
except the deadtime (local to ``deadtime_correction``) and the I-ALiRT
thresholds (default arguments in ``process_swe.py``). The full list with
values is in :ref:`swe-ancillary`.

CDF attribute configs
^^^^^^^^^^^^^^^^^^^^^

``imap_processing/cdf/config/``: ``imap_swe_global_cdf_attrs.yaml`` (the
``Logical_source`` authority) plus ``imap_swe_l1a_variable_attrs.yaml``,
``imap_swe_l1b_variable_attrs.yaml`` and ``imap_swe_l2_variable_attrs.yaml``.

Document page index
-------------------

Against CN102D-D0001, Issue Draft, 15 June 2026 (38 pages). Put your copy in
``docs/reference/IMAP_SWE_Algorithms_v8.pdf`` - that directory is gitignored.

.. list-table::
   :header-rows: 1
   :widths: 10 14 30 46

   * - Pages
     - Section
     - Title
     - Worth reading for
   * - 4
     - 1
     - Introduction
     - Scope. Two paragraphs.
   * - 5-8
     - 2
     - SWE Instrument Description
     - Sensor head, ESA + 7 CEMs, HVPS ranges, CDH. Figures 1-3.
   * - 8-9
     - 3.1
     - Operating Modes
     - The six FSW modes and the MET coarse/fine counters. Figure 4.
   * - 9-11
     - 3.2
     - Nominal Science Operations
     - **The measurement cycle.** Figure 5 is the single most useful page in
       the document - it is the picture of the checkerboard.
   * - 11-15
     - 3.3
     - Data Products
     - Science, HK, engineering, in-flight calibration and I-ALiRT product
       definitions. Figure 6 is the pipeline diagram.
   * - 13
     - 3.3.4
     - In-Flight Calibration Data
     - The calibration file format, the filename date convention, and the
       far-future padding row. Read this before touching calibration code.
   * - 16
     - 3.4.1
     - Heritage data processing
     - Why the code looks the way it does, and the 8-table caveat.
   * - 16-18
     - 3.4.2
     - L0 to L1A
     - The ``SCIENCE_DATA`` byte layout and the **decompression table** with a
       worked example. Also the I-ALiRT, HK and CEM raw packet notes.
   * - 19-21
     - 3.4.3
     - L1A to L1B
     - Deadtime C fragment, counts-to-rate, and the ``electron_cal()``
       interpolation C fragment.
   * - 21-25
     - 3.4.4
     - L1B to L2
     - Analyzer constant, polar angles, the ``fspace()`` C fragment
       (**including the end-detector loop**), the flux conversion, the
       acquisition time formula, and the spin angle binning specification.
   * - 25-29
     - 3.4.5
     - I-ALiRT Processing
     - The BDE algorithm in full, with the worked bin-offset examples.
       Figure 7 overlays the I-ALiRT subset on Figure 5.
   * - 29-32
     - 3.4.6
     - L2 to L3
     - **Not our work.** Spacecraft potential, pitch angle, moments. See
       :ref:`swe-l3-scope`.
   * - 33-34
     - Appendix A
     - Telemetry packet conversions
     - Every raw-to-engineering polynomial, including the ``HVPS_ESA_DAC``
       dual-range note.
   * - 35-38
     - Appendix B
     - Telemetry Packet Definitions
     - Sample byte/bit layouts for ``SWE_SCIENCE`` and ``SWE_IALIRT``.
       **Superseded by the XTCE**; the appendix says so.

Figures worth knowing about
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - Figure
     - Page
     - What it shows
   * - 1
     - 5
     - Front and back views of the instrument: sensor head cylinder, FEE box,
       EBOX.
   * - 2
     - 6
     - Sensor cross section: the spherical ESA plates and the seven gold CEM
       cones.
   * - 3
     - 7
     - Electronics block diagram, colour-coded by assembly.
   * - 4
     - 9
     - Mode transition diagram. (The text extraction of this figure is
       garbled; you need the PDF.)
   * - 5
     - 11
     - **Science data acquisition timing and flow.** The energy/spin-angle
       grid with the quarter-cycle colouring. If you only look at one figure,
       this is it.
   * - 6
     - 15
     - The SWE data processing pipeline, L0 through L3, as defined in the
       SDMP.
   * - 7
     - 25
     - Figure 5 again with a yellow box around the 8 I-ALiRT ESA steps.

Where else to look
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Question
     - Answer
   * - "What does this packet field mean?"
     - The XTCE, then Appendix B.
   * - "What is this CDF variable?"
     - ``imap_processing/cdf/config/imap_swe_*_variable_attrs.yaml``.
   * - "Where does this number come from?"
     - ``swe_constants.py``, then :ref:`swe-ancillary`.
   * - "Is this implemented?"
     - :ref:`swe-implementation-status`.
   * - "Should this be implemented here at all?"
     - :ref:`swe-l3-scope`.
