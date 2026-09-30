.. _lo-data-products:

Data Products and Pipeline
==========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the map of **what exists, what feeds what, and what it is called**.
Use it to find the right module and the right ``logical_source`` before diving
into an algorithm page.

Level definitions
-----------------

**[DOC]** IMAP follows the NASA PDS4 v1.18.0.0 processing-level definitions.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Level
     - Meaning for IMAP-Lo
   * - L0
     - Reconstructed, unprocessed CCSDS packets. Duplicates removed. A
       ``.pkts`` file, not produced by this repository.
   * - L1A
     - The same telemetry fields, unpacked and decompressed, written to CDF.
       **No calibration, no unit conversion.** One product per APID.
   * - L1B
     - Partially processed and *annotated*: engineering units, science-cycle
       aggregation, per-event timing / direction / species, count rates with
       exposure times, and the badtimes/goodtimes bookkeeping. Mostly for
       internal use.
   * - L1C
     - Pointing sets. Counts and exposure accumulated over one repointing
       period into a fixed sky-relative grid.
   * - L2
     - Calibrated science: ENA intensities and sky maps, ISN rates and rate
       maps, with the instrumental corrections applied.
   * - L3
     - Derived science requiring external models or other instruments, chiefly
       survival-probability-corrected fluxes at the termination shock.

Source packets (APIDs)
----------------------

**[CODE]** ``imap_processing/lo/l0/lo_apid.py``. Fields for all of these are
defined in ``imap_processing/lo/packet_definitions/lo_xtce.xml``.

.. list-table::
   :header-rows: 1
   :widths: 10 12 26 52

   * - Dec
     - Hex
     - Name
     - Contents / cadence
   * - 673
     - 0x2A1
     - ``ILO_BOOT_HK``
     - Boot memory dump. **Not processed by the pipeline.**
   * - 676
     - 0x2A4
     - ``ILO_APP_SHK``
     - Static housekeeping: versions, LUT identities, settings that do not
       change. On change.
   * - 677
     - 0x2A5
     - ``ILO_APP_NHK``
     - Nominal housekeeping: ~157 fields of temperatures, voltages, currents,
       modes, counters. Periodic.
   * - 705
     - 0x2C1
     - ``ILO_SCI_CNT``
     - Science counts: monitor rates (singles, TOF coincidences, discarded,
       triples, positions) and the on-board H/O histograms. One packet per
       science cycle (~420 s).
   * - 706
     - 0x2C2
     - ``ILO_SCI_DE``
     - Science direct events, bit-packed and variable length. One record per
       detected particle, ~6/s. May be segmented across CCSDS packets.
   * - 707
     - 0x2C3
     - ``ILO_STAR``
     - Star sensor, 720 compressed samples per spin.
   * - 708
     - 0x2C4
     - ``ILO_SPIN``
     - Spin start times, ESA DAC settings, validity flags. One packet per
       science cycle carrying 28 spins.
   * - 725
     - 0x2D5
     - ``ILO_DIAG_PCC``
     - Pivot platform control: potentiometers, motor currents, setpoints,
       temperatures.

Products actually produced by this repository
---------------------------------------------

**[CODE]** These are the exact ``Logical_source`` strings. Anything not in
this table does not exist yet.

L1A
^^^

Produced by ``lo_l1a.lo_l1a(l0_file)``, one call handles every APID in the file.

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - ``logical_source``
     - APID
     - Notes
   * - ``imap_lo_l1a_shk``
     - 676
     - Static housekeeping, raw field values.
   * - ``imap_lo_l1a_nhk``
     - 677
     - Nominal housekeeping, raw field values.
   * - ``imap_lo_l1a_histogram``
     - 705
     - 22 histogram/monitor-rate arrays, decompressed.
   * - ``imap_lo_l1a_de``
     - 706
     - Unpacked direct events; segmented packets reassembled.
   * - ``imap_lo_l1a_star``
     - 707
     - Star sensor, decompressed 8 -> 12 bit.
   * - ``imap_lo_l1a_spin``
     - 708
     - 28 spins per epoch, 7 fields per spin.
   * - ``imap_lo_l1a_pcc``
     - 725
     - Pivot platform housekeeping.

Also produced by the L1A entry point, despite the name:

* ``imap_lo_l1b_instrument-status-summary`` - merged from NHK + SHK + PCC.
  This is the code's version of the document's "Instrument State Vector".
* ``imap_lo_l1b_nhk``, ``imap_lo_l1b_shk`` - housekeeping converted to
  engineering units.

L1B
^^^

``lo_l1b.lo_l1b(sci_dependencies, anc_dependencies, descriptor)`` is a router.
The **descriptor selects the branch**, and each branch returns one or two
datasets.

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - descriptor
     - produces
     - main inputs
   * - ``badtimes``
     - ``imap_lo_l1b_badtimes``
     - Spin data (thruster firings and spin validity). May be an empty
       dataset.
   * - ``de``
     - ``imap_lo_l1b_de``
     - ``imap_lo_l1a_de``, ``imap_lo_l1a_spin``, sweep-table ancillary,
       SPICE. The annotated direct events.
   * - ``all-rates``
     - ``imap_lo_l1b_histrates``, ``imap_lo_l1b_monitorrates``
     - ``imap_lo_l1a_histogram``, ``imap_lo_l1a_spin``.
   * - ``derates``
     - ``imap_lo_l1b_derates``
     - ``imap_lo_l1b_de``, ``imap_lo_l1a_spin``.
   * - ``prostar``
     - ``imap_lo_l1b_prostar``
     - ``imap_lo_l1a_star``, ``imap_lo_l1a_nhk``.
   * - ``goodtimes``
     - ``imap_lo_l1b_bgrates``, ``imap_lo_l1b_goodtimes``
     - ``imap_lo_l1b_histrates``, pivot angle from housekeeping.

L1C
^^^

``lo_l1c.lo_l1c(sci_dependencies, anc_dependencies)``. No descriptor branching.

* ``imap_lo_l1c_pset`` - the pointing set. One per repointing.
* ``imap_lo_l1c_goodtimes`` - a reference copy of the goodtimes list.

L2
^^

``lo_l2.lo_l2(sci_dependencies, anc_dependencies, descriptor)``. The descriptor
is a **map descriptor** parsed by
``imap_processing.ena_maps.utils.naming.MapDescriptor``:

.. code-block:: text

   l090-ena-h-sf-nsp-ram-hae-6deg-3mo
   |    |   | |  |   |   |   |    +-- duration of the accumulation window
   |    |   | |  |   |   |   +------- spatial resolution
   |    |   | |  |   |   +----------- coordinate system / frame
   |    |   | |  |   +--------------- spin phase (ram / anti-ram / full)
   |    |   | |  +------------------- survival probability (nsp = none)
   |    |   | +---------------------- sky tiling (sf = rectangular)
   |    |   +------------------------ species
   |    +----------------------------- principal data ("ena")
   +---------------------------------- pivot angle, "l090" = 90 degrees,
                                        or "ilo" for a combined map

Correction flags appear as extra tokens in the ``principal_data`` field and are
read via ``MapDescriptor.sputter_corrected``, ``.bootstrap_corrected``,
``.cg_corrected``, ``.isn_masked``.

Pipeline dependency graph
-------------------------

**[CODE]** How the CLI actually wires it (``cli.py``, ``class Lo``):

.. code-block:: text

   L0 .pkts
     |
     +--> lo_l1a  (exactly one L0 file in)
            |
            +--> imap_lo_l1a_{shk,nhk,histogram,de,star,spin,pcc}
            +--> imap_lo_l1b_{nhk,shk,instrument-status-summary}
     |
     +--> lo_l1b  (L1A + L1B products + ancillary, routed by descriptor)
            |
            +-- badtimes  --> imap_lo_l1b_badtimes
            +-- de        --> imap_lo_l1b_de
            +-- all-rates --> imap_lo_l1b_histrates, imap_lo_l1b_monitorrates
            +-- derates   --> imap_lo_l1b_derates
            +-- prostar   --> imap_lo_l1b_prostar
            +-- goodtimes --> imap_lo_l1b_bgrates, imap_lo_l1b_goodtimes
     |
     +--> lo_l1c  (needs descriptors: de, goodtimes, bgrates, histrates)
            |
            +--> imap_lo_l1c_pset, imap_lo_l1c_goodtimes
     |
     +--> lo_l2   (needs L1B goodtimes + bgrates + histrates, per repointing)
            |
            +--> imap_lo_l2_<map descriptor>

.. important::

   **L2 does not consume the L1C pointing set.** ``lo_l2.REQUIRED_PRODUCTS``
   is ``("goodtimes", "bgrates", "histrates")``, all L1B, grouped by
   repointing. The pointing set is produced but is currently a dead end in the
   pipeline. The algorithm document says maps are built from pointing sets.
   See :ref:`lo-implementation-status`. **[CODE]**

L2 input selection by pivot angle
---------------------------------

**[CODE]** ``Lo.pre_processing`` overrides the base class for L2 only. It:

1. Parses the map descriptor and reads its pivot angle from the ``sensor``
   field (``l090`` -> 90). A descriptor of ``ilo`` means "all pivot angles",
   and no filtering happens.
2. Opens every ``goodtimes`` file, reads its ``pivot`` variable, and keeps the
   repointing only if
   ``abs(pivot - map_pivot) < LoConstants.PSET_PIVOT_ANGLE_TOLERANCE`` (5
   degrees).
3. Drops *all* Lo science inputs for rejected repointings, so the map's
   ``Parents`` attribute lists only files it was actually built from.

Products defined in the document but not implemented
-----------------------------------------------------

**[DOC]** These appear in the algorithm document's product list. There is no
code for them. Details in :ref:`lo-implementation-status`.

* **SweepTable** as an L1B *product* (it exists in the code only as an
  ancillary *input*).
* **ISN Rates** (document section 12.4).
* **ISN 1AU Maps** (12.5).
* **ISN Pointing Event Lists** (12.6).
* **ENA Spectra in Selected Directions**.
* **L3 Survival Probability Corrected Fluxes** (section 13).
* **L3 off-nominal pivot-angle ("DDD-degree") maps**.

File naming
-----------

**[DOC]** Higher level files:

.. code-block:: text

   imap_{inst}_{level}_{descriptor}_{time}_{repoint}_v{NNN}.cdf

The descriptor uses hyphens, never underscores, to separate its own
components. Version is zero-padded. This repository builds filenames through
``imap_data_access``; do not hand-assemble them.

Audit trail
-----------

**[DOC]** Each stage is meant to append to a JSON audit trail recording
``revision``, ``date``, ``event`` (the level), ``files_input`` and
``files_output``. In this repository the equivalent is the CDF ``Parents``
global attribute plus the SDC's own bookkeeping outside the container; there is
no separate JSON audit file. **[CODE]**
