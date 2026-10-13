.. _mag-data-products:

Data Products and Pipeline
==========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the map of **what exists, what feeds what, and what it is called**.
Use it to find the right module and the right ``Logical_source`` before diving
into an algorithm page.

Level definitions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 8 30 62

   * - Level
     - Frame / units
     - Meaning for MAG
   * - L0
     - MFO, MFI. Engineering units.
     - Raw CCSDS packets, NM and BM unseparated. Not produced by this
       repository.
   * - L1A
     - MFO, MFI. Engineering units.
     - Packets separated by mode and by **physical sensor**, decommutated and
       decompressed into per-vector time series. No calibration.
   * - L1B
     - URFO, URFI. **nT**.
     - Engineering (ground) calibration applied; compression rescaling applied;
       per-sensor time shift applied.
   * - L1C
     - URFO, URFI. nT.
     - **Normal mode only.** Gaps caused by burst-mode operation are filled with
       burst data interpolated onto a synthesised normal-mode timeline. Carries
       a per-sample flag for measured vs. interpolated.
   * - L1D
     - DSRF, SRF, RTN, GSE. nT.
     - **Preliminarily calibrated data for other instrument teams.** Uses
       predicted offsets, spin averaging and gradiometry rather than the MAG
       team's after-the-fact offsets, so it is available quickly but is less
       precise than L2. Formerly called "L2Pre".
   * - L2
     - DSRF, SRF, RTN, GSE, GSM. nT.
     - **Fully calibrated data for release.** Uses MAG-team per-vector offsets,
       time deltas and quality flags.
   * - L3
     - --
     - **Does not exist for MAG.** L2 is the final released product.

Source packets
--------------

**[CODE]** ``imap_processing/mag/l0/mag_l0_data.py``, ``Mode`` IntEnum. Fields
are defined in ``imap_processing/mag/packet_definitions/MAG_SCI_COMBINED.xml``.

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - APID
     - Name
     - Contents
   * - 1052
     - ``Mode.NORMAL``
     - Normal rate science telemetry. Up to 32 vectors/s per sensor.
   * - 1068
     - ``Mode.BURST``
     - Burst rate science telemetry. Up to 128 vectors/s per sensor.
   * - --
     - ``MAG_SCI_IALIRT``
     - Real-time stream, defined in
       ``imap_processing/ialirt/packet_definitions/ialirt_mag.xml`` and handled
       entirely outside ``imap_processing/mag`` (see :ref:`mag-ialirt`).

Housekeeping APIDs are not processed by this repository.

The ``MagL0`` header fields you will actually use
-------------------------------------------------

**[CODE]** ``MagL0`` in ``mag/l0/mag_l0_data.py``. The load-bearing fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``SHCOARSE``
     - Packet mission elapsed time (whole seconds).
   * - ``PUS_SSUBTYPE``
     - **Seconds of data in the packet, minus one.** ``seconds_per_packet =
       PUS_SSUBTYPE + 1``.
   * - ``COMPRESSION``
     - 1 if the vector block is Fibonacci/zig-zag compressed.
   * - ``MAGO_ACT`` / ``MAGI_ACT``
     - Whether each sensor is active. Also referred to as FOB / FIB.
   * - ``PRI_SENS``
     - ``0`` = MAGo is PRIMARY, ``1`` = MAGi is PRIMARY
       (``PrimarySensor`` enum).
   * - ``PRI_VECSEC`` / ``SEC_VECSEC``
     - Encoded rate 0-7. ``MagL0.__post_init__`` converts these **in place** to
       actual rates via ``2 ** value``, giving 1, 2, 4, ..., 128. Do not decode
       them a second time.
   * - ``PRI_COARSETM`` / ``PRI_FNTM``
     - MET seconds and 16-bit sub-second counter for the **first PRIMARY
       vector**. ``SEC_COARSETM``/``SEC_FNTM`` likewise for SECONDARY.
   * - ``VECTORS``
     - The raw bit-packed vector block, converted to a big-endian ``uint8``
       numpy array on construction.

``MagL0`` defines ``__eq__``/``__hash__`` on ``(SHCOARSE, APID, SRC_SEQ_CTR)``,
which is how ``decom_packets`` **de-duplicates** repeated packets.

Products produced by this repository
------------------------------------

**[CODE]** These are the exact ``Logical_source`` strings, taken from
``imap_processing/cdf/config/imap_mag_global_cdf_attrs.yaml``. Anything not in
these tables does not exist.

.. note::

   The document writes product names as ``imap_mag_l1a_raw_normal_...``. The
   code follows the IMAP filename convention instead, where the descriptor is a
   single hyphenated token: ``imap_mag_l1a_norm-raw``. **Use the code's
   strings.**

L1A
^^^

Produced by ``mag_l1a.mag_l1a(packet_file)`` - one call handles both APIDs in
the file and emits between 2 and 6 datasets.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``Logical_source``
     - Contents
   * - ``imap_mag_l1a_norm-raw``
     - One row **per packet**. ``raw_vectors`` is the undecoded byte block,
       zero-padded to the longest packet in the file, plus every header field as
       its own variable. Epoch is ``SHCOARSE``.
   * - ``imap_mag_l1a_burst-raw``
     - Same, for burst packets.
   * - ``imap_mag_l1a_norm-mago``
     - One row **per vector**: ``vectors`` (x, y, z, range) and
       ``compression_flags`` (is_compressed, compression_width).
   * - ``imap_mag_l1a_norm-magi``
     - Same, MAGi.
   * - ``imap_mag_l1a_burst-mago``
     - Same, burst MAGo.
   * - ``imap_mag_l1a_burst-magi``
     - Same, burst MAGi.

L1B
^^^

``mag_l1b.mag_l1b(input_dataset, day_to_process, calibration_dataset)`` - **one
L1A input, one L1B output**. Raw L1A files raise ``ValueError``.

* ``imap_mag_l1b_norm-mago``
* ``imap_mag_l1b_norm-magi``
* ``imap_mag_l1b_burst-mago``
* ``imap_mag_l1b_burst-magi``

L1C
^^^

``mag_l1c.mag_l1c(first, day, second, previous_day_dataset)``. **Normal mode
only** - burst data is an *input* used to fill gaps, not an output.

* ``imap_mag_l1c_norm-mago``
* ``imap_mag_l1c_norm-magi``

L1D
^^^

``mag_l1d.mag_l1d(science_data, calibration_dataset, day)`` produces **all
frames and both modes in a single call**, plus ancillary offset files.

Science products (8 when burst data is available, 4 otherwise):

* ``imap_mag_l1d_norm-srf``, ``imap_mag_l1d_norm-dsrf``,
  ``imap_mag_l1d_norm-gse``, ``imap_mag_l1d_norm-rtn``
* ``imap_mag_l1d_burst-srf``, ``imap_mag_l1d_burst-dsrf``,
  ``imap_mag_l1d_burst-gse``, ``imap_mag_l1d_burst-rtn``

Ancillary products, written by ``Mag.post_processing`` in ``cli.py`` with
``istp=False`` (they bypass ``write_cdf``):

* ``imap_mag_l1d_spin-offsets``
* ``imap_mag_l1d_gradiometry-offsets-norm``
* ``imap_mag_l1d_gradiometry-offsets-burst``

.. note::

   There is **no** ``imap_mag_l1d_*-gsm``. L1D outputs SRF, DSRF, GSE and RTN
   only; GSM appears at L2 and in I-ALiRT.

L2
^^

``mag_l2.mag_l2(calibration, offsets, input_data, day, mode, frames)``. One call
produces all frames **for a single mode**; the mode comes from the CLI
descriptor.

* ``imap_mag_l2_norm-srf``, ``imap_mag_l2_norm-gse``, ``imap_mag_l2_norm-gsm``,
  ``imap_mag_l2_norm-rtn``, ``imap_mag_l2_norm-dsrf``
* ``imap_mag_l2_burst-srf``, ``imap_mag_l2_burst-gse``,
  ``imap_mag_l2_burst-gsm``, ``imap_mag_l2_burst-rtn``,
  ``imap_mag_l2_burst-dsrf``

``DEFAULT_L2_FRAMES`` orders DSRF **last** deliberately: some vectors may become
NaN/FILLVAL after that rotation, and ``rotate_frame`` mutates the dataclass in
place.

Dependency graph
----------------

.. code-block:: text

   L0 packets (APID 1052 + 1068, 25 h)
     |
     +-- mag_l1a --> norm-raw, burst-raw
     |               norm-mago, norm-magi, burst-mago, burst-magi
     |
     +-- mag_l1b  (+ l1b-calibration ancillary)
     |     each L1A sensor/mode file -> matching L1B file
     |
     +-- mag_l1c  (norm L1B + burst L1B + previous day's L1C)
     |     -> norm-mago, norm-magi  [normal mode only]
     |
     +-- mag_l1d  (L1C norm mago+magi REQUIRED,
     |             L1B burst mago+magi OPTIONAL,
     |             + l1d-calibration ancillary, + SPICE)
     |     -> 4 or 8 science files + spin/gradiometry ancillary files
     |
     +-- mag_l2   (L1C norm OR L1B burst, chosen via the offsets file's Parents,
                   + l2-calibration ancillary
                   + l2-{norm,burst}-offsets ancillary, + SPICE)
           -> 5 science files for the requested mode

.. important::

   **L1D and L2 are siblings, not sequential.** Both consume L1B burst and L1C
   normal data directly. L2 does **not** consume L1D. L1D exists purely to get a
   usable product out fast.

CLI wiring
----------

**[CODE]** ``class Mag(ProcessInstrument)`` in ``imap_processing/cli.py``, and
``PROCESSING_LEVELS["mag"] = ["l1a", "l1b", "l1c", "l1d", "l2"]`` in
``imap_processing/__init__.py``.

.. list-table::
   :header-rows: 1
   :widths: 10 44 46

   * - Level
     - Science dependencies
     - Ancillary dependencies
   * - ``l1a``
     - Exactly one ``mag`` ``l0`` file.
     - None.
   * - ``l1b``
     - Exactly one ``mag`` ``l1a`` file.
     - Exactly one ``l1b-calibration``.
   * - ``l1c``
     - One or two ``mag`` ``l1b`` files valid for the start date, plus
       optionally the previous day's ``l1c`` file.
     - None.
   * - ``l1d``
     - All ``mag`` ``l1c`` files plus all ``mag`` ``l1b`` files.
     - ``l1d-calibration``.
   * - ``l2``
     - Retrieved from the offsets file's ``Parents``, **not** from the
       dependency list (falls back to passed-in L1B/L1C with a warning).
     - Exactly one ``l2-{norm,burst}-offsets`` **and** one ``l2-calibration``.

A few CLI behaviours that surprise people:

* ``day_buffer = start_date + 3 days`` is passed to ``MagAncillaryCombiner`` so
  that open-ended calibration files get a synthetic end date at least three days
  past the processing day.
* At L2 the descriptor is split on ``-``; the first token
  (``norm``/``burst``) selects both the offsets descriptor and the
  ``DataMode``.
* After L2, ``Parents`` is rewritten to record the L1 file that was actually
  pulled from the offsets file, dropping any passed-in ``imap_mag_l1b_`` /
  ``imap_mag_l1c_`` names, so provenance matches the data.
* Every MAG dataset is checked for monotonically increasing epochs (warning
  only) and for epochs within +/-24 h of the processing day (hard error).

Global attributes carried through the pipeline
-----------------------------------------------

**[CODE]** These are set at L1A and propagated (sometimes lossily) forward. They
are not in the algorithm document but the pipeline depends on them.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Attribute
     - Meaning and lifetime
   * - ``is_mago``
     - ``"True"``/``"False"`` string. L1A -> L1B -> L1C.
   * - ``is_active``
     - Whether the sensor was active. L1A -> L1B -> L1C.
   * - ``all_vectors_primary``
     - True when the sensor was the PRIMARY sensor in **every** packet. L1D uses
       this to decide whether gradiometry may be applied at all.
   * - ``vectors_per_second``
     - String of the form ``"ttj2000ns:rate,ttj2000ns:rate"``, recording every
       rate change and when it happened. Parsed by
       ``constants.vectors_per_second_from_string``. **This is how L1C knows the
       expected cadence**, so it is load-bearing, and L1B shifts the embedded
       timestamps when it applies the time shift.
   * - ``missing_sequences``
     - List of missing CCSDS source sequence counters, or ``"None"``. Empty
       arrays are dropped by cdflib, hence the string.
   * - ``interpolation_method``
     - Set at L1C to the ``InterpolationFunction`` name used.
