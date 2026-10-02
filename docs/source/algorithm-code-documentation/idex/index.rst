:orphan:

.. _idex-index:

IDEX
====

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.idex

This is the IDEX (Interstellar Dust Experiment) instrument module, which contains
the code for processing data from the IDEX instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the IDEX
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the source document below plus what the code in
``imap_processing/idex`` actually does. Where the two disagree, that is called
out explicitly in :ref:`idex-implementation-status`.

.. _idex-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the IDEX instrument team at LASP / CU Boulder or the SDC document store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - *IDEX Algorithms Document*, dated 1 June 2026. Ethan Ayari, Alex Doner,
       Jamey Szalay, Scott Knappmiller, Mihály Horányi. 39 numbered pages
       (41 PDF pages). The primary source for these pages. Referred to below
       as "the algorithm document".
   * - **Science Data Management Plan (SDMP)**
     - Defines the IDEX product inventory and the L1A/L1B/L2A/L2B/L2C split
       that the algorithm document's Figure 4.1 reproduces. Not read by any
       code here.
   * - **XTCE packet definitions**
     - ``imap_processing/idex/packet_definitions/idex_science_packet_definition.xml``
       and ``idex_housekeeping_packet_definition.xml``. The algorithm document
       explicitly defers to these as authoritative for field names, bit widths,
       encodings and enumerations, and reproduces only their first 100 lines
       in Appendix B.
   * - **Heritage instruments**
     - SUDA (electronics heritage), and LDEX on LADEE for the impact-charge
       pulse model. Horányi et al. (2014), *The Lunar Dust Experiment (LDEX)*,
       Space Sci. Rev. 185(1-4), 93-113, doi:10.1007/s11214-014-0118-7 is cited
       directly in ``idex_l2a.fit_impact``.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`idex-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It describes
     the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/idex``.

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match the
   other without asking.

.. warning::

   **IDEX is moving faster than its algorithm document.** The 1 June 2026
   document is unusually code-aware - it was clearly written against a snapshot
   of this repository - but the code has since moved on in ways that matter:

   * **Event classification and saturation flags do not appear in the document
     at all.** ``imap_processing/idex/idex_event_flags.py`` (~480 lines) adds
     ten per-event flags at L1A that gate what L2A and L2B are allowed to
     publish. See :ref:`idex-event-classification`.
   * **The TOF DN-to-engineering-unit factors changed, and so did their
     units.** The document's Table 4.2 gives pC/DN for all six channels; the
     code converts the three TOF channels to **mA**, with different numbers.
     See :ref:`idex-l1`.
   * **Ion-grid velocity and mass are implemented**, though the document lists
     them under "Future Work" (section 4.7.9).
   * **L2B and L2C are implemented** (``idex_l2b.py``, ~840 lines) and are
     barely more than a flowchart box in the document.

   Treat the document as the statement of intent for L0-L2A, and this page set
   plus the code as the statement of fact.

.. warning::

   **Product cadences are 10 days and 1 month, and they have moved before.**
   L1A, L1B and L2A are **10-day** products cut on a fixed calendar table
   (``idex_10_day_CDF_names.csv``); L2B and L2C are **monthly**. If the
   instrument team changes the cadence again, the changes land in that CSV,
   in ``get_10_day_window_end_date()``, in the ``logical_source`` strings, and
   in the ``descriptor`` strings the CLI branches on - four places, none of
   which validate each other. See :ref:`idex-data-products`.

.. warning::

   **L0 files are not tagged by event time.** A dust event's epoch comes from
   the FPGA metadata header (when the impact happened), but the L0 ``.pkts``
   file is organized by the packet's own downlink/creation time. Events from
   one day can therefore be spread across several L0 files, and one L0 file can
   contain events from several days. Every IDEX level deals with this by
   over-querying its inputs and then filtering on the reconstructed event
   epoch. Do not "optimize" that away. See :ref:`idex-data-products`.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`idex-overview`
     - What IDEX physically is, how one dust impact becomes six waveforms, and
       the vocabulary (event, block, fragment, science type, high/low rate,
       dead time). **Start here if you are new.**
   * - :ref:`idex-data-products`
     - The full product inventory, exact ``logical_source`` strings, the 10-day
       and monthly windowing model, the L0 time-tagging problem, and how the
       CLI is wired. **The "what goes into what" map.**
   * - :ref:`idex-l1`
     - Packet decommutation, event assembly from fragments, event time
       reconstruction, waveform decoding and Rice-Golomb decompression, time
       axes, event-message rendering, then the L1B DN conversions, instrument
       setting unpacking, dead time, trigger decoding and SPICE geometry.
   * - :ref:`idex-event-classification`
     - The ten L1A event and saturation flags: how an event is labelled
       science / pulser / noise-capture, how a dust hit is detected, and what
       downstream code refuses to publish because of them. **Not in the
       algorithm document at all.**
   * - :ref:`idex-l2`
     - L2A waveform fits, impact charge, the rise-time-to-velocity inversion,
       the charge-yield mass estimate, the TOF mass scale and peak fits, and
       the deliberate NaN block. Then L2B/L2C daily counts, uptime-corrected
       rates and rectangular maps.
   * - :ref:`idex-ancillary`
     - The 10-day window table, the EU conversion table, the two L2A
       calibration curves, the atomic mass table, the event-message
       dictionaries, and the SPICE dependencies.
   * - :ref:`idex-l3-scope`
     - Why there is no IDEX L3 here, and what sits downstream of L2C.
   * - :ref:`idex-implementation-status`
     - What is implemented, what is deliberately withheld, where the code
       deviates from the document, and the suspected defects.
       **Read before proposing work.**
   * - :ref:`idex-reference-tables`
     - Where the big tables live (XTCE, ancillary CSVs, CDF attribute YAML,
       PDF page ranges). Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1
   event-classification
   l2
   ancillary
   l3-scope
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* IDEX is a **high-resolution impact-ionization time-of-flight mass
  spectrometer** for interstellar dust (ISD) and interplanetary dust particles
  (IDP). A dust grain hits a +3 kV target, the impact ionizes it, reflectron
  ion optics focus the ions onto a central detector, and the flight times give
  a mass spectrum.
* IDEX is **event-driven, not a scanner.** There is no sweep, no spin binning
  at the instrument level, no accumulation interval. Nothing happens until a
  grain arrives. **[DOC]** The mission expectation is roughly **16 events per
  day**, which is why the products are 10 days and a month long rather than a
  day.
* Every event produces **six waveforms**: ``TOF_High``, ``TOF_Mid``,
  ``TOF_Low`` (three gain stages of the same TOF signal, 260 MHz), and
  ``Target_High``, ``Target_Low``, ``Ion_Grid`` (charge-sensitive amplifiers,
  4.0625 MHz).
* One event is **many CCSDS packets**: one FPGA metadata header packet followed
  by waveform fragments, routed by ``IDX__SCI0TYPE`` and ordered by
  ``IDX__SCI0FRAGOFF``. Reassembling them is the single most fiddly part of
  IDEX L1A.
* Processing chain in this repository::

      CCSDS .pkts
        -> L1A  event-level, raw DN waveforms + metadata + event flags (10 days)
        -> L1B  engineering units, dead time, trigger decode, SPICE geometry (10 days)
        -> L2A  waveform fits, impact charge, velocity, mass, TOF mass scale (10 days)
        -> L2B  daily counts and uptime-corrected rates vs spin phase (1 month)
        -> L2C  the same counts and rates as rectangular sky maps (1 month)

  There is also a parallel **event-message** chain
  (``l1a_msg-10days -> l1b_msg-10days``) that turns instrument log entries into
  ``science_on`` / ``pulser_on`` state, and a **catalog-list** (``catlst``)
  passthrough product.
* **A large fraction of the L2A and L2B product is deliberately NaN or
  fill-valued.** The variables exist so the CDF schema is stable; the science
  team has not validated the TOF mass spectrum or the derived mass, so those
  values are overwritten just before the dataset is returned. This is on
  purpose, it is tested, and it is not a bug. See :ref:`idex-l2`.
* **This repository goes all the way to the map product.** Unlike most IMAP
  instruments, IDEX has no separate L3 repository to hand off to.
  See :ref:`idex-l3-scope`.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/idex/
     idex_constants.py               IDEXAPID, ConversionFactors, DT_BLOCK,
                                     IDEX_10_DAY_RANGES_PATH, SPICE_ARRAYS,
                                     ION_GRID_VELOCITY_*, IDEX_SPACING_DEG,
                                     IDEX_EVENT_REFERENCE_FRAME
     idex_utils.py                   get_idex_attrs, setup_dataset,
                                     get_10_day_window_end_date
     idex_l0.py                      decom_packets() - science vs housekeeping split
     idex_l1a.py                     idex_l1a(), PacketParser, RawDustEvent, Scitype
     decode.py                       rice_decode() - Rice-Golomb decompression
     evt_msg_decode_utils.py         render_event_template()
     idex_event_flags.py             classify_event_flags(), saturation flags
     idex_l1b.py                     idex_l1b_science(), idex_l1b_msg(), TriggerMode,
                                     TriggerOrigin, EventMessage, get_spice_data()
     idex_l2a.py                     idex_l2a(), estimate_dust_mass(), fit_impact(),
                                     time_to_mass(), analyze_peaks(), calibration curves
     idex_l2b.py                     idex_l2b() - produces BOTH L2B and L2C

     packet_definitions/
       idex_science_packet_definition.xml       APID 1424
       idex_housekeeping_packet_definition.xml  APIDs incl. 1418 (EVT), 1419 (CATLST)

     idex_10_day_CDF_names.csv                  444 product windows, 2025-2036
     idex_variable_unpacking_and_eu_conversion.csv   31 instrument settings
     idex_evt_msg_parsing_dictionaries.json     event-message templates
     atomic_masses.csv                          21 reference ion masses

   imap_processing/cdf/config/imap_idex_global_cdf_attrs.yaml
   imap_processing/cdf/config/imap_idex_l1a_variable_attrs.yaml
   imap_processing/cdf/config/imap_idex_l1b_variable_attrs.yaml
   imap_processing/cdf/config/imap_idex_l2a_variable_attrs.yaml
   imap_processing/cdf/config/imap_idex_l2b_variable_attrs.yaml
   imap_processing/cdf/config/imap_idex_l2c_variable_attrs.yaml
   imap_processing/cli.py (class Idex)        dependency wiring per level
   imap_processing/tests/idex/                tests, L0 test data, calibration CSVs