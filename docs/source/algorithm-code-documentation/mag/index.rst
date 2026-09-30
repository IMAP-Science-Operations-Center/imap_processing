.. _mag:

MAG
===

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.mag

This is the MAG (magnetometer) instrument module, which contains the code for
processing data from the MAG instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the MAG
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the algorithm document below plus what the code in
``imap_processing/mag`` actually does. Where the two disagree, that is called
out explicitly in :ref:`mag-implementation-status`. The MAG team's upstream
calibration and cleaning, which the public CMAD documents instead, is summarised
in :ref:`mag-cmad`.

.. _mag-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the MAG instrument team at Imperial College London or the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - IMAP-MAG-SW-009-01B, *MAG Science Algorithm Document*, Issue 5
       Revision 2, 01 June 2026. Prepared by Alastair Crabtree, approved by
       Tim Horbury, Imperial College London. 44 pages. The primary source for
       these pages.

       Unlike most instruments' algorithm documents, this one is **not**
       embedded in the public CMAD (below).
   * - **CMAD**
     - *Calibration and Measurement Algorithms Document for NASA's IMAP
       Mission*, version 1.1 (preliminary), ``IMAP_CMAD_20260722.pdf``. Public.
       For MAG it embeds two Imperial technical notes instead of the algorithm
       document:

       * IMAP-OPS-TN-ICL-017, *IMAP MAG Calibration Inputs Description*,
         Issue 2, 4 June 2026 (CMAD section 3.5.2, printed pages 152-163);
       * IMAP-OPS-TN-ICL-013, *IMAP MAG Data Cleaning Processes*, Issue 4,
         4 June 2026 (CMAD section 4.4, printed pages 912-929).

       Section 5.4.5 (printed pages 1163-1166) holds the authoritative L2 quality
       flag and bitmask definitions. All of this is summarised in
       :ref:`mag-cmad`.
   * - **TLM_MAG** ([RD01])
     - MAG telemetry definition spreadsheet. Superseded in practice by
       ``imap_processing/mag/packet_definitions/MAG_SCI_COMBINED.xml``, which
       is what the code parses.
   * - **MAG TMTC / SDMP**
     - Referenced by the algorithm document for exhaustive packet and product
       structure. Not needed for any code in this repository.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`mag-reference-tables` is written against that location.
   Even though the CMAD is public, do not commit it: it is ~128 MB and far over
   the repository's file-size limit.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/mag``.

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. note::

   **This repository stops at L2.** For MAG that is not a limitation - L2 *is*
   the final released science product in the algorithm document. There is no MAG
   L3. The one thing to be careful about is that MAG's **L1D** looks like a
   near-L2 product (it is calibrated, despun and delivered in science frames)
   but it is a Level 1 product produced here, on purpose, for other instrument
   teams to use before the MAG team's full offset determination is available.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`mag-overview`
     - What MAG physically is, the two sensors, ranging, sample rates, science
       modes and the reference-frame chain. **Start here if you are new.**
   * - :ref:`mag-data-products`
     - The full product inventory, exact ``Logical_source`` strings, what feeds
       what, and how the CLI is wired. **The "what goes into what" map.**
   * - :ref:`mag-l1a`
     - Packet decommutation, vector unpacking, and the Fibonacci/zig-zag
       decompression algorithm.
   * - :ref:`mag-l1b`
     - Compression rescaling, the measurement-frame to unit-reference-frame
       rotation, and the per-sensor time shift.
   * - :ref:`mag-l1c`
     - Gap filling: reconstructing a continuous normal-mode timeline from burst
       mode data, and the six interpolation methods.
   * - :ref:`mag-l1d`
     - The rapid near-L2 product: spin-average offsets, gradiometry, despinning
       and multi-frame output.
   * - :ref:`mag-l2`
     - The released science product: MAG-team offsets, calibration matrices,
       quality flags and frame transforms.
   * - :ref:`mag-ialirt`
     - The real-time space-weather stream, which reuses L1A-L1D steps on a
       four-packet-per-sample telemetry format.
   * - :ref:`mag-ancillary`
     - Every calibration and offset file: who delivers it, what variables it
       contains, and which level consumes it.
   * - :ref:`mag-cmad`
     - How the MAG team produces the L2 offsets and matrices (cleaning and
       calibration at Imperial), the **authoritative quality bitmask**, and the
       artifacts left in released L2. Summarises the public CMAD.
   * - :ref:`mag-implementation-status`
     - What is implemented, where the code deviates from the document, known
       and suspected bugs, and what is not written at all. **Read before
       proposing work.**
   * - :ref:`mag-reference-tables`
     - Where the big tables live (XTCE, CDF attribute YAML, validation data).
       Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1a
   l1b
   l1c
   l1d
   l2
   ialirt
   ancillary
   cmad
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* MAG is a **conventional dual fluxgate magnetometer**. Two sensors on the
  spacecraft boom: **MAGo** (outboard, far from the spacecraft, the primary
  science sensor) and **MAGi** (inboard, closer to the spacecraft, used to
  characterise and remove spacecraft-generated fields).
* Every measurement is a 3-component vector plus a **range** (0-3) plus a
  timestamp. Calibration is different for every (sensor, range) pair, so range
  is carried all the way through the pipeline.
* Two science telemetry streams: **normal mode (NM)**, APID 1052, nominally
  2 vectors/s from each sensor for ~23 h/day; and **burst mode (BM)**,
  APID 1068, nominally 64 vectors/s from MAGo and 8 from MAGi for ~1 h/day.
  Only one is transmitted at a time.
* Processing is **vector-by-vector**, on UTC-day windows with a **30 minute
  buffer on each side** (a 25 hour file). The buffer is stripped at L1D and L2.
* Processing chain::

      CCSDS packets
        -> L1A  raw + per-sensor per-mode timeseries, measurement frame (MFO/MFI)
        -> L1B  engineering calibration, unit reference frame (URFO/URFI), nT
        -> L1C  normal-mode gaps filled from burst mode (norm only)
        -> L1D  rapid near-L2 quality: SPICE frames, spin offsets, gradiometry
        -> L2   released science: MAG-team offsets, SRF/DSRF/RTN/GSE/GSM

* A separate **I-ALiRT** path takes a fixed 1 vector / 4 s real-time stream all
  the way to an L1D-equivalent product for space weather forecasting.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/mag/
     constants.py                          DataMode, Sensor, VecSec, FIBONACCI_SEQUENCE, tolerances
     imap_mag_sdc_configuration_v001.py    SDC config: interpolation method, ALWAYS_OUTPUT_MAGO
     packet_definitions/MAG_SCI_COMBINED.xml   XTCE for APIDs 1052 and 1068
     l0/
       mag_l0_data.py                      MagL0 dataclass, Mode (APID) IntEnum
       decom_mag.py                        packets -> MagL0 list, raw L1A dataset
     l1a/
       mag_l1a.py                          MagL0 -> MAGo/MAGi datasets
       mag_l1a_data.py                     the big one; vector unpacking + decompression
     l1b/
       mag_l1b.py                          rescale, calibrate to URF, time shift
       imap_mag_l1b-calibration_20240229_v002.cdf   bundled fallback calibration
     l1c/
       mag_l1c.py                          gap finding, timeline generation, gap filling
       interpolation_methods.py            6 interpolation methods + CIC filter
     l1d/
       mag_l1d.py                          orchestration, frame loop, ancillary output
       mag_l1d_data.py                     MagL1d + MagL1dConfiguration; spin/gradiometry
     l2/
       mag_l2.py                           orchestration, calibration matrix selection
       mag_l2_data.py                      MagL2 + MagL2L1dBase + ValidFrames

   imap_processing/ialirt/l0/parse_mag.py           the entire I-ALiRT MAG algorithm
   imap_processing/ialirt/l0/mag_l0_ialirt_data.py  I-ALiRT L0 dataclass
   imap_processing/ialirt/packet_definitions/ialirt_mag.xml

   imap_processing/ancillary/ancillary_dataset_combiner.py   MagAncillaryCombiner
   imap_processing/cdf/config/imap_mag_*.yaml                CDF attributes
   imap_processing/tests/mag/                                tests + validation data
   imap_processing/cli.py (class Mag)                        dependency wiring per level

API reference
-------------

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    constants
    l0.decom_mag
    l0.mag_l0_data
    l1a.mag_l1a
    l1a.mag_l1a_data
    l1b.mag_l1b
    l1c.mag_l1c
    l1c.interpolation_methods
    l1d.mag_l1d
    l1d.mag_l1d_data
    l2.mag_l2
    l2.mag_l2_data
