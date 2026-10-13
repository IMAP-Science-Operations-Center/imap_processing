:orphan:

.. _swe-index:

SWE
===

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.swe

This is the SWE (Solar Wind Electron) instrument module, which contains the code
for processing data from the SWE instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the SWE
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the source document below plus what the code in
``imap_processing/swe`` actually does. Where the two disagree, that is called
out explicitly in :ref:`swe-implementation-status`.

.. _swe-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the SWE instrument team at Los Alamos National Laboratory or the SDC document
store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - CN102D-D0001, *IMAP SWE Science Data Product Algorithms*, Issue Draft,
       dated 15 June 2026. Prepared by Ruth Skoug. Los Alamos National
       Laboratory (Triad National Security, LLC). 38 pages. The primary source
       for these pages. Referred to below as "the algorithm document".
   * - **Science Data Management Plan (SDMP)**
     - Defines the SWE product inventory and the L1A/L1B/L2/L3 split that the
       algorithm document's Figure 6 reproduces. Not needed by any code here.
   * - **Packet ICD / Appendix B**
     - Appendix B of the algorithm document gives sample packet field layouts
       and warns that "packet definitions continue to change as the instrument
       development progresses." Superseded in practice by
       ``imap_processing/swe/packet_definitions/swe_packet_definition.xml``,
       which is what the code parses.
   * - **7516-9054 GSW-FSW ICD**
     - Cited in ``imap_processing/ialirt/l0/process_swe.py`` for the
       spacecraft SCLK sub-second convention (LSB = 1/256 s).
   * - **Heritage codes**
     - ACE/SWEPAM (C), Ulysses/SWOOPS (Fortran) and Genesis/GEM. The algorithm
       document reproduces SWEPAM C fragments verbatim for deadtime, gain
       calibration and phase space density. SWEPAM is the most current and is
       the stated basis for SWE processing (algorithm document section 3.4.1).
   * - **Genesis BDE algorithm**
     - Neugebauer et al., 2003. The basis for the I-ALiRT bidirectional
       electron parameter.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`swe-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/swe``
     (or ``imap_processing/ialirt/l0/process_swe.py`` for the real-time
     product).

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. warning::

   **This repository stops at L2.** SWE's L3 (spacecraft potential, pitch angle
   and gyrophase distributions, and electron moments) is produced by a separate
   repository run closer to the science team. Section 3.4.6 of the algorithm
   document - four of its 38 pages, and by far the most algorithmically dense
   part - describes work that does **not** belong here. See
   :ref:`swe-l3-scope` before starting anything that looks like a Maxwellian
   fit, a pitch angle, or a moment.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`swe-overview`
     - What SWE physically is, how a measurement happens, and the vocabulary
       (CEM, ESA step, quarter cycle, full cycle, spin sector, spin angle bin,
       checkerboard). **Start here if you are new.**
   * - :ref:`swe-data-products`
     - The full product inventory, exact ``logical_source`` strings, what feeds
       what, and how the CLI is wired. **The "what goes into what" map.**
   * - :ref:`swe-l1`
     - Packet decommutation, count decompression, the checkerboard
       reorganization, acquisition timing, deadtime correction, in-flight gain
       calibration, counts-to-rate, and the L1A/L1B CDF contents.
   * - :ref:`swe-l2`
     - ESA voltage to electron energy, phase space density, number flux, spin
       phase from SPICE, and 12-degree spin angle binning.
   * - :ref:`swe-ancillary`
     - The ESA LUT, the in-flight calibration file, the EU conversion table,
       the geometric factors, the analyzer constant, and the SPICE
       dependencies.
   * - :ref:`swe-ialirt`
     - The 30-second real-time space-weather product and the bidirectional
       electron (BDE) search.
   * - :ref:`swe-l3-scope`
     - What L3 is, why it is not in this repository, and exactly what L2 has to
       hand it. **Read before writing any fitting or moments code.**
   * - :ref:`swe-implementation-status`
     - What is implemented, what is stubbed, where the code deviates from the
       document, and what is not written at all. **Read before proposing
       work.**
   * - :ref:`swe-reference-tables`
     - Where the big tables live (XTCE, ancillary CSVs, PDF page ranges).
       Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1
   l2
   ancillary
   ialirt
   l3-scope
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* SWE is a **spherical-section electrostatic analyzer followed by seven channel
  electron multipliers (CEMs)**, a direct descendant of Ulysses/SWOOPS,
  ACE/SWEPAM and Genesis/GEM. It measures the 3D distribution of solar wind
  thermal and suprathermal **electrons** from 1 eV to 5 keV.
* The 7 CEMs sit at fixed polar angles **0, ±21, ±42, ±63 degrees** relative to
  the aperture normal, which points perpendicular to the spin axis. Polar angle
  is a property of *which detector*, not of anything you compute. Spacecraft
  spin sweeps the fan-shaped FOV over >95% of 4π sr.
* One **ESA step** (an ESA voltage setting) lasts nominally 83.333 ms =
  3.333 ms settle + 80 ms accumulate, and covers ~2 degrees of spin. Each step
  yields 7 numbers, one per CEM.
* A **quarter cycle** is one telemetry packet: 15 seconds, 180 ESA steps
  (12 per second), 1260 compressed bytes. Four quarter cycles make a **full
  cycle** (~1 minute), which is the L1B/L2 record. A full cycle covers
  **24 energies × 30 spin angles × 7 CEMs**.
* Within a quarter cycle only 6 of the 24 energies are sampled per spin-angle
  bin, and odd- and even-numbered spin bins get *different* sets of 6. Sorting
  the 4 × 180 measurements back into a (24, 30) grid is the **checkerboard**
  reorganization and is the single most confusing piece of SWE code.
* Counts are telemetered **8-bit compressed** (SWEPAM scheme) and must be
  expanded to 16 bits through a 16-entry base/step_size table.
* Processing chain in this repository:
  ``CCSDS packets -> L1A (decompressed counts, per packet) -> L1B (checkerboard,
  deadtime, gain cal, count rates) -> L2 (phase space density, flux, spin
  angle bins)``. L3 is not ours.
* There is also a **30-second I-ALiRT** product (8 of the 24 energies,
  normalized counts and a bidirectional-electron flag) living under
  ``imap_processing/ialirt/``.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/swe/
     utils/swe_constants.py              N_ESA_STEPS=24, N_ANGLE_SECTORS=30, N_CEMS=7,
                                         N_QUARTER_CYCLES=4, N_QUARTER_CYCLE_STEPS=180,
                                         GEOMETRIC_FACTORS, ENERGY_CONVERSION_FACTOR=4.75,
                                         CEM_DETECTORS_ANGLE, ESA_VOLTAGE_ROW_INDEX_DICT
     utils/swe_utils.py                  SWEAPID, acquisition-time helpers
     packet_definitions/
       swe_packet_definition.xml         SWE_APP_HK + SWE_CEM_RAW + SWE_SCIENCE
     l1a/swe_l1a.py                      APID dispatch for science / HK / CEM raw
     l1a/swe_science.py                  count decompression, L1A science dataset
     l1b/swe_l1b.py                      checkerboard, deadtime, in-flight cal, rates
     l2/swe_l2.py                        phase space density, flux, spin angle binning

   imap_processing/ialirt/
     l0/process_swe.py                   the whole SWE I-ALiRT algorithm (BDE)
     utils/constants.py                  swe_energy (the 8 I-ALiRT energies in eV)
     packet_definitions/ialirt_swe.xml   SWE I-ALiRT packet fields

   imap_processing/cdf/config/imap_swe_global_cdf_attrs.yaml
   imap_processing/cdf/config/imap_swe_l1a_variable_attrs.yaml
   imap_processing/cdf/config/imap_swe_l1b_variable_attrs.yaml
   imap_processing/cdf/config/imap_swe_l2_variable_attrs.yaml
   imap_processing/quality_flags.py (class SweL1bFlags)
   imap_processing/cli.py (class Swe)   dependency wiring per level
   imap_processing/tests/swe/           tests, L0 test data, validation CSVs, LUTs