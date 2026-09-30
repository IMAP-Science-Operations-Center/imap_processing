.. _codice:

CoDICE
======

.. currentmodule:: imap_processing.codice

This is the CoDICE (Compact Dual Ion Composition Experiment) instrument module,
which contains the code for processing data from the CoDICE instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the CoDICE
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the source document below plus what the code in
``imap_processing/codice`` actually does. Where the two disagree, that is called
out explicitly in :ref:`codice-implementation-status`.

.. _codice-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the CoDICE instrument team at SwRI or the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - 26850.03-DPA-01, *Data Products and Algorithms (DPA) for the IMAP Compact
       Dual Ion Composition Experiment (CoDICE)*, **Rev 3 Chg 1, June 2026**.
       Prepared by Joey Mukherjee and Michael Starkey, Southwest Research
       Institute, SDRL SW-009. The primary source for these pages.

       This revision is publicly released as **section 4.3.2 of the IMAP
       Calibration and Measurement Algorithms Document (CMAD)**,
       ``IMAP_CMAD_20260722.pdf``, PDF pages 623-709. Rev 3 Chg 1 differs from
       Rev 3 Chg 0 only by removing the EAR export-control statement; the
       substantive changes listed under Rev 3 Chg 0 (May 2026) are summarised in
       :ref:`codice-doc-revisions` below.
   * - **SCI_LUT spreadsheet**
     - ``26850.03-SCI-LUT-01.xls`` (and successors). The plan / ESA-sweep /
       stepping / views / collapse tables. **This is not optional** - CoDICE
       science packets cannot be unpacked without it. The SDC consumes a JSON
       rendering of it as an ancillary file with descriptor ``l1a-sci-lut``.
   * - **Packet ICD**
     - Field-level packet definitions. Superseded in practice by the two XTCE
       files in ``imap_processing/codice/packet_definitions/``, which are what
       the code parses.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`codice-reference-tables` is written against that location.
   Even though the CMAD is public, do not commit it: it is ~128 MB and far over
   the repository's file-size limit.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/codice``.

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. note::

   **This repository stops at L2.** CoDICE has a substantial L3 program -
   partial densities, abundance and charge-state ratios, 3-D VDFs, pitch-angle
   distributions and the combined Hi+Lo L3c products - all described in section
   13 of the algorithm document. **None of that belongs here.** It is produced
   by a separate repository closer to the science team. Section 13 is summarised
   on :ref:`codice-l3-scope` only so that you can recognise an L3 request when
   one arrives and redirect it.

   The one exception is I-ALiRT: the real-time stream computes abundance and
   charge-state ratios that look like L3 quantities, but they are produced here
   because the whole I-ALiRT chain lives in this repository. See
   :ref:`codice-ialirt`.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`codice-overview`
     - What CoDICE physically is, the two sensors, the coordinate frames and
       angle conventions, the ESA stepping scheme, RGFO/NSO modes, the
       commissioning timeline and the instrument team's data caveats. **Start
       here if you are new.**
   * - :ref:`codice-data-products`
     - The full product inventory, exact ``Logical_source`` strings, APID
       mapping, what feeds what, and how the CLI is wired. **The "what goes into
       what" map.**
   * - :ref:`codice-l1a`
     - The SCI-LUT unpacking machinery (plan / view / collapse tables),
       decompression, de-spinning, NSO masking and direct-event bit unpacking.
       **The hardest part of CoDICE.**
   * - :ref:`codice-l1b`
     - Counts to rates: acquisition times, spin-sector counts, and the
       ``energy_table`` derivation.
   * - :ref:`codice-l2`
     - Rates to intensities: geometric factors, RGFO Full/Reduced selection,
       efficiencies, and the direct-event unit conversions.
   * - :ref:`codice-ialirt`
     - The real-time space-weather stream and the pseudo-density ratios it
       produces.
   * - :ref:`codice-ancillary`
     - Every lookup table and calibration file: who delivers it, what is in it,
       and which level consumes it.
   * - :ref:`codice-implementation-status`
     - What is implemented, what is missing, where the code deviates from the
       document, and known/suspected bugs. **Read before proposing or
       estimating work.**
   * - :ref:`codice-l3-scope`
     - What section 13 asks for, and why none of it is built here.
   * - :ref:`codice-reference-tables`
     - Where the big tables live (SCI-LUT, XTCE, CDF attribute YAML, PDF page
       ranges). Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1a
   l1b
   l2
   ialirt
   ancillary
   implementation-status
   l3-scope
   reference-tables

Ten-second orientation
----------------------

* CoDICE is **two ion sensors sharing one time-of-flight / energy (TOF-E)
  subsystem**, with two separate apertures.

  * **CoDICE-Lo** measures ~0.5-80 keV/q ions. An electrostatic analyzer (ESA)
    selects energy-per-charge; a -15 kV post-acceleration carbon foil produces
    start electrons; 24 avalanche photodiodes (APDs) around 360 degrees of
    azimuth measure residual energy. (E/q, TOF, E) gives mass, charge state and
    m/q.
  * **CoDICE-Hi** measures ~0.03-5 MeV/nuc ions through 12 collimators onto 12
    solid-state detectors (SSDs). (E, TOF) gives mass.

* The fundamental Lo cadence is a **16-spin (32 half-spin, ~4 minute) cycle**
  over which the ESA steps through **128 energy-per-charge steps**. Different
  half-spins sample different numbers of ESA steps (1 to 6). Hi products
  accumulate over 4 or 16 spins.

* Almost every science product is **"collapsed" on board** - angles and spins
  summed together per a configurable table - and then optionally compressed
  (table-based lossy and/or LZMA lossless). Ground unpacking is impossible
  without the **SCI_LUT** tables, which are keyed by ``(table_id, plan_id,
  plan_step, view_id)`` carried in every science packet.

* Two on-board protective modes change the meaning of the data and both must be
  handled on the ground: **RGFO** (Reduced Geometric Factor Operation, changes
  which geometric factor applies) and **NSO** (No-Scan Operation, stops the
  energy sweep - affected data must be set to fill).

* Processing chain::

      CCSDS packets
        -> L1A  decompressed, un-collapsed, de-spun raw counts + metadata
        -> L1B  counts / acquisition time = rates (plus energy_per_charge)
        -> L2   rates / (G * efficiency * energy passband) = intensities;
                direct events converted to physical units
        -> L3   (elsewhere) densities, ratios, VDFs, pitch angles, Hi+Lo combos

* A separate **I-ALiRT** path takes a trickle-fed subset of the Lo SW species
  counts and the Hi sectored H counts all the way to abundance ratios and
  intensities for space-weather forecasting.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/codice/
     constants.py                     APIDs, species lists, lossy tables, angle tables
     utils.py                         SCI-LUT reading, collapse patterns, acq times, epochs
     decompress.py                    lossy A/B, LZMA, 24-bit pack
     codice_l1a.py                    APID dispatch for L1A (+ the L1B housekeeping product)
     codice_l1a_de.py                 direct events (Lo + Hi), segmented packet reassembly
     codice_l1a_lo_species.py         Lo SW species counts (also used by I-ALiRT Lo)
     codice_l1a_lo_priority.py        Lo SW/NSW priority counts
     codice_l1a_lo_counters_aggregated.py
     codice_l1a_lo_counters_singles.py
     codice_l1a_hi_omni.py            Hi omni-directional species counts
     codice_l1a_hi_sectored.py        Hi sectored species counts
     codice_l1a_hi_priority.py        Hi priority counts
     codice_l1a_hi_counters_aggregated.py
     codice_l1a_hi_counters_singles.py
     codice_l1a_ialirt_hi.py          Hi I-ALiRT counts (called only by the I-ALiRT pipeline)
     codice_l1b.py                    counts -> rates for every descriptor
     codice_l2.py                     intensities, geometric factors, DE unit conversions
     data/esa_sweep_values.csv        historical ESA sweep values (not read by the pipeline)
     data/lo_stepping_values.csv      historical Lo stepping values (not read by the pipeline)
     packet_definitions/
       imap_codice_packet-definition_20250101_v001.xml   pre-2026-01-29 FSW
       imap_codice_packet-definition_20260129_v001.xml   post-2026-01-29 FSW
       P_COD_NHK.xml                                     housekeeping only

   imap_processing/ialirt/l0/process_codice.py   the entire I-ALiRT CoDICE algorithm
   imap_processing/ialirt/utils/constants.py     I-ALiRT CoDICE field/energy definitions

   imap_processing/cdf/config/imap_codice_*.yaml   CDF global + variable attributes
   imap_processing/tests/codice/                   tests + validation data
   imap_processing/cli.py (class Codice)           dependency wiring per level

API reference
-------------

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    codice_l1a
    codice_l1b
    codice_l2
    constants
    utils
    decompress
