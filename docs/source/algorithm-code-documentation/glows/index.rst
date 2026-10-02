:orphan:
.. _glows-index:

GLOWS
=====

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.glows

This is the GLOWS (GLObal solar Wind Structure) instrument module, which
contains the code for processing data from the GLOWS instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the GLOWS
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full ~120-page data-products document.

They are a summary of the source document below plus what the code in
``imap_processing/glows`` actually does. Where the two disagree, that is called
out explicitly in :ref:`glows-implementation-status`.

.. _glows-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the GLOWS instrument team at CBK PAN (Warsaw) or from the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - M. Bzowski, M. Strumik, I. Kowalska-Leszczyńska, M. A. Kubiak,
       R. Wawrzaszek, K. Ber, P. Orleański, *GLOWS data products*, revision
       4.4.7, 25 July 2025. CBK PAN, Bartycka 18a, Warsaw. 123 pages. The
       primary source for these pages. Referred to below simply as "the
       document".
   * - **Telemetry definition**
     - ``TLM_GLX_YYYY_MM_DD.xlsx``, sheets ``P_GLX_TMSCHIST`` (histograms) and
       ``P_GLX_TMSCDE`` (direct events). Superseded in practice by the XTCE in
       ``imap_processing/glows/packet_definitions/``, which is what the code
       parses.
   * - **Python-script bundle**
     - The GLOWS team's own reference implementation of the L0-to-L3A pipeline,
       delivered to the SDC together with validation data sets. It emits JSON
       for every level. **When the document is ambiguous, this bundle is the
       tie-breaker** (document §3.11). The JSON outputs in
       ``imap_processing/tests/glows/validation_data/`` come from it.
   * - **Supporting reports**
     - *GLOWS Signal Evolution Report* (Kowalska-Leszczyńska et al., 2021),
       *PSF Report* / *PSF Definition Report* (Strumik, 2020), *Baffle design
       report* (Kaźmierczak et al., 2021), *GLOWS Entrance System Writeup*
       (Bzowski et al., 2021), MICD (Kowalski, 2020). Referenced by the
       document for instrument physics; not needed for any code here.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`glows-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/glows``.

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. note::

   **This repository stops at L2.** GLOWS has a rich L3A-L3E chain (low-res
   lightcurves, ionization rates, solar-wind latitude profiles, and ENA survival
   probabilities for Lo/Hi/Ultra), but none of it is produced here. It is the
   responsibility of a separate repository closer to the science team. Sections
   4 and 13 of the algorithm document describe L3; they are summarised only
   briefly in :ref:`glows-data-products` so that you know what your L2 output is
   feeding.

.. warning::

   GLOWS has a **flag polarity trap**. The document says a bad-time flag value
   of ``true`` means "there is a problem". The code writes the opposite:
   ``1 = good, 0 = bad``. See :ref:`glows-l1b` before you touch anything
   flag-related.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`glows-overview`
     - What GLOWS physically is, how a measurement happens, and the vocabulary
       (helioglow, spin block, histogram bin, observational day, spin angle vs.
       position angle, day/night mode). **Start here if you are new.**
   * - :ref:`glows-data-products`
     - The full product inventory, exact ``Logical_source`` strings, APIDs, what
       feeds what, and how the CLI is wired. **The "what goes into what" map.**
   * - :ref:`glows-l1a`
     - Packet decommutation, the direct-event compression markers, packet
       merging, and the L1A CDF variables.
   * - :ref:`glows-l1b`
     - Integer-to-physical decoding, the 17 bad-time flags, the 4 bad-angle
       flags, sky masking, and everything SPICE contributes.
   * - :ref:`glows-l2`
     - The daily lightcurve: good-time selection, day/night offsets, co-adding,
       exposure, the Rayleigh calibration, and the position-angle conversion.
   * - :ref:`glows-ancillary`
     - Every instrument-team-supplied file: format, descriptor, who reads it,
       and which of them are currently ignored.
   * - :ref:`glows-implementation-status`
     - What is implemented, what is stubbed, where the code deviates from the
       document, and what is not written at all. **Read before proposing work.**
   * - :ref:`glows-reference-tables`
     - Where the big tables live (XTCE, ancillary ``.dat``, CDF attribute YAML,
       validation JSON) and the document's section index. Deliberately *not*
       reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1a
   l1b
   l2
   ancillary
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* GLOWS is a **single-pixel, non-imaging Lyman-α photometer**. It has no
  imaging optics and no energy or mass analysis. It counts photons. Everything
  else is bookkeeping.
* The spacecraft spins at ~4 RPM (15 s period). GLOWS' boresight is fixed at an
  angle to the spin axis, so one spin sweeps a **small circle of 75° angular
  radius** on the sky. The daily product is the **modulation of the helioglow
  brightness around that circle** - a "lightcurve".
* Photon arrival times ("direct events") are histogrammed **onboard** in the
  spin-angle domain: **3600 bins of 0.1°**, accumulated over a **block of 8
  spins (~2 minutes)**. One CCSDS packet = one block histogram. That is the
  primary science telemetry and all of it is downlinked.
* Direct events are **supporting data**, not science. Only ~13 blocks per day
  are downlinked. This repository takes them to L1B and stops.
* Data are organised **per pointing** ("observational day"), i.e. the interval
  between IMAP repointing maneuvers, nominally 24 h.
* Processing chain::

      CCSDS packets (APID 1480 hist, 1481 DE)
        -> L1A  unpacked telemetry; still integer-encoded; DEs decompressed
        -> L1B  physical units, ancillary from SPICE, bad-time + bad-angle flags
        -> L2   daily lightcurve of photon flux in Rayleighs
        -> L3A..L3E  (NOT in this repository)

* The whole scientific point of L1B/L2 is **culling**: deciding which blocks
  (bad *times*) and which bins (bad *angles*) are trustworthy. The arithmetic is
  trivial; the flag bookkeeping is where the complexity and the bugs live.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/glows/
     __init__.py                       BAD_TIME_FLAG_NAMES (17 names), FLAG_LENGTH
     utils/constants.py                TimeTuple, DirectEvent, GlowsConstants
     packet_definitions/
       GLX_COMBINED.xml                master XTCE, loaded by decom_packets
       P_GLX_TMSCHIST.xml              APID 1480 histogram packet
       P_GLX_TMSCDE.xml                APID 1481 direct-event packet
     l0/
       decom_glows.py                  GlowsParams APID enum; decom_packets()
       glows_l0_data.py                HistogramL0, DirectEventL0 dataclasses
     l1a/
       glows_l1a.py                    orchestration + both L1A xr.Datasets
       glows_l1a_data.py               StatusData, HistogramL1A, DirectEventL1A
     l1b/
       glows_l1b.py                    orchestration, apply_ufunc wiring, CDF assembly
       glows_l1b_data.py               the big one: PipelineSettings,
                                       AncillaryExclusions, AncillaryParameters,
                                       DirectEventL1B, HistogramL1B
     l2/
       glows_l2.py                     orchestration + L2 xr.Dataset assembly
       glows_l2_data.py                DailyLightcurve, HistogramL2
     ancillary/                        bundled example instrument-team files

   imap_processing/ancillary/ancillary_dataset_combiner.py   GlowsAncillaryCombiner
   imap_processing/quality_flags.py                         GLOWSL1bFlags (bad-angle)
   imap_processing/cdf/config/imap_glows_*.yaml              CDF attributes
   imap_processing/tests/glows/                              tests + validation JSON
   imap_processing/cli.py (class Glows)                      dependency wiring per level

.. note::

   ``imap_processing/glows/l1a/``, ``l1b/``, ``l2/`` and ``ancillary/`` have **no**
   ``__init__.py``. Only ``l0/`` and ``utils/`` do. Imports work because the
   package is installed and Python treats these as namespace packages, but be
   aware of it if you are debugging an import or packaging problem.
