.. _swapi:

SWAPI
=====

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.swapi

This is the SWAPI (Solar Wind and Pickup Ion) instrument module, which contains
the code for processing data from the SWAPI instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the SWAPI
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the source document below plus what the code in
``imap_processing/swapi`` actually does. Where the two disagree, that is called
out explicitly in :ref:`swapi-implementation-status`.

.. _swapi-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the SWAPI instrument team at Princeton or the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - 05899-Algorithms_AN, *IMAP SWAPI Instrument Algorithms Document*,
       Version 07, 2026-06-03. Prepared by Bishwas Shrestha (SWAPI Algorithms
       Lead) and Margaret Shaw-Lecerf; approved by Eric Zirnstein (Data
       Analysis Lead), Jamie Rankin (Instrument Lead) and Scott Weidner
       (Project Manager). Princeton University, Department of Astrophysical
       Sciences, Space Physics Group. 80 pages. The primary source for these
       pages.
   * - **ESA Unit Conversion ADP**
     - ``imap_swapi_esa-unit-conversion_<start>_<end>_<version>.xlsx``,
       delivered by the SWAPI team. The workbook's main sheet and its
       ``LUT_Notes_vx`` sheet are both required to derive L2 energies. The SDC
       ingests them as two CSV ancillary files (``esa-unit-conversion`` and
       ``lut-notes``) - see :ref:`swapi-ancillary`.
   * - **Packet ICD**
     - Field-level packet definitions. Superseded in practice by
       ``imap_processing/swapi/packet_definitions/swapi_packet_definition.xml``,
       which is what the code parses.
   * - **SDC ICDs**
     - LASP 167441 (POC-to-Instrument-Team) and LASP 167442
       (SDC-to-Instrument-Team). Referenced by the algorithm document for
       filenames and L0 delivery; not needed by any code here.
   * - **Instrument paper**
     - Rankin et al. 2025, SSRv, 221, 108. The authority for the calibration
       numbers (geometric factor, passbands, ``k`` factor) the algorithm
       document quotes.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`swapi-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/swapi``
     (or ``imap_processing/ialirt`` for the real-time product).

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. warning::

   **This repository stops at L2.** SWAPI's L3a (solar wind proton, alpha and
   pickup-helium plasma parameters) and L3b (combined differential flux) are
   produced by the SWAPI team's own code, delivered as a separate Docker
   container. Section 13 of the algorithm document states this explicitly.
   Roughly two thirds of the algorithm document (sections 7.1.1, 9.5, 10.4,
   10.5) describes work that does **not** belong here. See
   :ref:`swapi-l3-scope` before starting anything that looks like a fit.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`swapi-overview`
     - What SWAPI physically is, how a measurement happens, and the vocabulary
       (sweep, ESA step, coarse/fine, sweep plan, PCEM/SCEM/COIN, sunglasses,
       open aperture, live time). **Start here if you are new.**
   * - :ref:`swapi-data-products`
     - The full product inventory, exact ``logical_source`` strings, what feeds
       what, and how the CLI is wired. **The "what goes into what" map.**
   * - :ref:`swapi-l1`
     - Packet decommutation, sweep grouping, count decompression, quality
       flags, count uncertainty, and the L1 CDF contents.
   * - :ref:`swapi-l2`
     - Counts to rates, the live-time constant, and the ESA step-to-energy
       solve (the only real algorithm at L2).
   * - :ref:`swapi-ancillary`
     - The ESA unit conversion ADP, the LUT notes table, the efficiency and
       gain-test LUTs, the instrument response CSVs, and the external
       dependencies (MAG, SPICE, thruster history).
   * - :ref:`swapi-ialirt`
     - The 12-second real-time space-weather product and its analytical
       pseudo-moment fit.
   * - :ref:`swapi-l3-scope`
     - What L3a/L3b are, why they are not in this repository, and exactly what
       L2 has to hand them. **Read before writing any fitting code.**
   * - :ref:`swapi-implementation-status`
     - What is implemented, what is stubbed, where the code deviates from the
       document, and what is not written at all. **Read before proposing
       work.**
   * - :ref:`swapi-reference-tables`
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

* SWAPI is a **top-hat electrostatic analyzer with a coincidence detector**,
  derived from New Horizons' SWAP. It measures solar wind protons
  (H\ :sup:`+`) and alphas (He\ :sup:`2+`) plus interstellar pickup ions
  (H\ :sup:`+` and He\ :sup:`+`, helium dominated).
* Solar wind enters through a **0.1% transmissive grid** ("the sunglasses") so
  the bright core beam does not saturate the detector. Pickup ions enter
  through the **open aperture** unattenuated. Both populations land on the same
  detector; which aperture a count came from is an *angular* distinction, not a
  separate channel.
* Two channel electron multipliers - **PCEM** (primary, sees the ion) and
  **SCEM** (secondary, sees carbon-foil secondary electrons). Hits within
  100 ns are also counted as **COIN** (coincidence). Every science product is
  three numbers per energy step: PCEM, SCEM, COIN counts.
* The **sweep is the fundamental unit**: 12 seconds, 72 ESA steps, telemetered
  as 12 one-second packets of 6 steps each. One L1/L2 record is one sweep.
* The 72 steps are **1 ramp-up + 62 coarse (0.1-20 keV/q) + 9 fine** steps. The
  fine steps move with the solar wind peak and are defined by the active
  **sweep plan**; their energies must be *solved for* at L2, they are not in a
  fixed table.
* Processing chain in this repository:
  ``CCSDS packets -> L1 (counts/sweep, CDF) -> L2 (count rates + energies)``.
  L3a/L3b are the SWAPI team's, not ours.
* There is also a **12-second I-ALiRT** product (coincidence counts only,
  pseudo speed/density/temperature) built from a different APID and living
  under ``imap_processing/ialirt/``.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/swapi/
     constants.py                        NUM_PACKETS_PER_SWEEP=12, NUM_ENERGY_STEPS=72
     swapi_utils.py                      SWAPIAPID, SWAPIMODE, read_swapi_lut_table()
     packet_definitions/
       swapi_packet_definition.xml       SWP_HK (102 fields) + SWP_SCI (46 fields)
     l1/swapi_l1.py                      packets -> L1 science + L1A/L1B housekeeping
     l2/swapi_l2.py                      counts -> rates, ESA step -> energy

   imap_processing/ialirt/
     l0/process_swapi.py                 the whole I-ALiRT algorithm
     constants.py (IalirtSwapiConstants)  A_eff, dE/E, azimuth FOV, density factor
     packet_definitions/ialirt_swapi.xml  SWAPI I-ALiRT packet (APID 1187)

   imap_processing/cdf/config/imap_swapi_global_cdf_attrs.yaml
   imap_processing/cdf/config/imap_swapi_variable_attrs.yaml
   imap_processing/quality_flags.py (class SWAPIFlags)
   imap_processing/cli.py (class Swapi)  dependency wiring per level
   imap_processing/tests/swapi/          tests, L0 test data, validation CSVs, LUTs
