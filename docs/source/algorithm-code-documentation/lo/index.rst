.. _lo:

IMAP-Lo
=======

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.lo

This is the IMAP-Lo instrument module, which contains the code for processing
data from the IMAP-Lo instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the
IMAP-Lo processing algorithms, written so that a developer (human or AI agent)
can get productive without reading the full ~230-page algorithm document.

They are a summary of the source documents below plus what the code in
``imap_processing/lo`` actually does. Where the two disagree, that is called
out explicitly in :ref:`lo-implementation-status`.

.. _lo-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the IMAP-Lo instrument team or the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - UNH-IMAP-Lo-27850-6002, *IMAP-Lo Data Product Algorithms*, rev. 10
       (with appendices). Prepared by F. Rahmanifard, University of New
       Hampshire, for SwRI project 27850. The primary source for these pages.
   * - **Appendix A**
     - N. A. Schwadron, F. Rahmanifard, E. Moebius, A. Galli, K. Fairchild,
       H. Islam, J. Bower, M. Shen, *The IMAP-Lo Mapping Algorithms*, v6,
       submitted to ApJ. Bound into the algorithm document as Appendix A.
       **Authoritative for all L2/L3 map algorithms** - it supersedes section
       12 of the main body wherever they differ.
   * - **Packet ICD**
     - Field-level packet definitions. Superseded in practice by
       ``imap_processing/lo/packet_definitions/lo_xtce.xml``, which is what the
       code parses.
   * - **Compression tables**
     - ``IMAP-Lo_Compression_Tables.xlsx``. Already vendored as CSVs in
       ``imap_processing/lo/l0/decompression_tables/``.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the page
   index in :ref:`lo-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/lo``.

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`lo-overview`
     - What IMAP-Lo physically is, how a measurement happens, and the
       vocabulary (spin, ESA step, science cycle, pointing, pivot angle,
       coincidence type, ram/anti-ram). **Start here if you are new.**
   * - :ref:`lo-data-products`
     - The full product inventory, exact ``logical_source`` strings, what
       feeds what, and how the CLI is wired. **The "what goes into what" map.**
   * - :ref:`lo-l1a`
     - Packet decommutation, bit unpacking, decompression tables, direct-event
       case decoding.
   * - :ref:`lo-l1b`
     - The bulk of the real per-event physics: annotation, timing, species ID,
       look directions, rates, exposure, badtimes/goodtimes, background rates.
   * - :ref:`lo-l1c`
     - Pointing sets: the 3600 x 40 x 7 sky-binned counts/exposure grid.
   * - :ref:`lo-l2`
     - Sky maps and every science correction (sputter, bootstrap,
       Compton-Getting, ISN mask), with the equations.
   * - :ref:`lo-ancillary`
     - Calibration and ancillary files: geometric factors, efficiency,
       backgrounds, sweep table, correction factors, pointing files.
   * - :ref:`lo-implementation-status`
     - What is implemented, what is stubbed, where the code deviates from the
       document, and what is not written at all. **Read before proposing work.**
   * - :ref:`lo-reference-tables`
     - Where the big tables live (XTCE, decompression CSVs, PDF page ranges).
       Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1a
   l1b
   l1c
   l2
   ancillary
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* IMAP-Lo is a single-pixel neutral atom camera. It converts a neutral atom to
  a negative ion on a conversion surface, energy-selects it with an
  electrostatic analyzer (ESA), and identifies it with a triple-coincidence
  time-of-flight (TOF) telescope.
* The spacecraft spins (~4 RPM). The instrument sits on a **pivot platform**
  that can point its 9 degree FWHM boresight anywhere from 60 to 165 degrees
  off the spin axis. One spin sweeps a great circle on the sky.
* The ESA steps through **7 energy levels**, 4 spins per level, so one
  **science cycle is 28 spins (~420 s)**. That cycle is the fundamental
  aggregation unit at L1B and above.
* Two science goals: **ENA** maps (heliospheric hydrogen, 40 eV - 1 keV) and
  **ISN** observations (interstellar H, He, D, O, Ne). ENAs are reported as
  fluxes/intensities; ISN is reported as rates.
* Processing chain:
  ``CCSDS packets -> L1A (CDF of raw fields) -> L1B (annotated events + rates)
  -> L1C (pointing sets) -> L2 (sky maps / fluxes) -> L3 (survival-probability
  corrected)``.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/lo/
     constants.py                     LoConstants: every tunable number
     lo_ancillary.py                  CSV ancillary reader
     ancillary_data/                  calibration CSVs shipped with the package
     packet_definitions/lo_xtce.xml   the packet field definitions (~7200 lines)
     l0/
       lo_apid.py                     APID enum
       lo_science.py                  histogram + direct event unpacking
       lo_star_sensor.py              star sensor unpacking
       utils/bit_decompression.py     log decompression via lookup tables
       decompression_tables/          the lookup CSVs + direct event CASE_DECODER
     l1a/lo_l1a.py                    packets -> L1A datasets
     l1b/lo_l1b.py                    the big one; all L1B descriptors
     l1b/tof_conversions.py           TOF ADC -> ns coefficients
     l1c/lo_l1c.py                    pointing sets
     l2/lo_l2.py                      sky maps and corrections

   imap_processing/cdf/config/imap_lo_*.yaml   CDF variable/global attributes
   imap_processing/tests/lo/                   tests + validation data
   imap_processing/cli.py (class Lo)           dependency wiring per level

API reference
-------------

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    l0.lo_science
    l0.lo_star_sensor
    l1a.lo_l1a
    l1b.lo_l1b
    l1c.lo_l1c
    l2.lo_l2
    constants
    lo_ancillary
