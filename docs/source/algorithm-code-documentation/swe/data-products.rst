.. _swe-data-products:

Data Products and What Feeds What
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This is the map. If you are trying to work out which file a variable comes from
or which CLI invocation produces it, start here.

Product inventory
-----------------

**[CODE]** Every ``Logical_source`` below is defined in
``imap_processing/cdf/config/imap_swe_global_cdf_attrs.yaml``. That file is the
authority; if you add a product, add it there first.

.. list-table::
   :header-rows: 1
   :widths: 26 10 64

   * - ``logical_source``
     - Level
     - Contents
   * - ``imap_swe_l1a_sci``
     - L1A
     - Decompressed 16-bit CEM counts, one record per ``SWE_SCIENCE`` packet
       (one quarter cycle), shaped ``(epoch, 180 spin_sector, 7 cem_id)``.
       Carries the raw 8-bit counts alongside, plus every science packet
       metadata field in raw (unconverted) units.
   * - ``imap_swe_l1a_hk``
     - L1A
     - ``SWE_APP_HK`` decommutated, **raw units**. No algorithm.
   * - ``imap_swe_l1a_cem-raw``
     - L1A
     - ``SWE_CEM_RAW`` decommutated. Engineering-mode 1-second CEM counts,
       latched and live, uncompressed. No algorithm.
   * - ``imap_swe_l1b_sci``
     - L1B
     - Deadtime-corrected, gain-calibrated **count rates** on the full-cycle
       checkerboard grid, shaped ``(epoch, 24 esa_step, 30 spin_sector,
       7 cem_id)``. Plus per-measurement acquisition times, ESA energies,
       counting uncertainty and a quality flag.
   * - ``imap_swe_l1b_hk``
     - L1B
     - ``SWE_APP_HK`` decommutated with **derived (engineering) units** applied
       by the XTCE. Note this is produced from the **L0 file**, not from
       ``imap_swe_l1a_hk``.
   * - ``imap_swe_l2_sci``
     - L2
     - Phase space density and number flux, both in the original
       ``(esa_step, spin_sector)`` organization **and** binned into 30 fixed
       12-degree spin angle bins. Plus statistical uncertainties, spin angles,
       acquisition times and the quality flag.

Not produced here
-----------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Product
     - Why not
   * - **SWE L3**
     - Separate repository, run closer to the science team. Spacecraft
       potential, core/halo break, pitch angle and gyrophase distributions,
       and moments (density, velocity, temperature, heat flux). See
       :ref:`swe-l3-scope`.
   * - **I-ALiRT SWE**
     - Produced here, but **not by** ``imap_processing/swe`` and **not as a
       SWE CDF**. It lives in ``imap_processing/ialirt/l0/process_swe.py`` and
       is merged with the other instruments' real-time records into a single
       I-ALiRT dataset. See :ref:`swe-ialirt`.
   * - **In-flight calibration analysis**
     - **[DOC]** Section 3.3.4: the weekly gain sweep data are examined **on
       the ground by a SWE science team member** to decide whether CEM bias
       needs raising. The SDC's job is only to keep those data out of L1B and
       above. The resulting factors arrive back as an ancillary file.
   * - **Quicklook**
     - Nothing SWE-specific exists in this repository.

The processing chain
--------------------

.. code-block:: text

   L0 CCSDS (.pkts)
     |
     |  swe_l1a()  -- packet_file_to_datasets, use_derived_value=False
     |              dispatch on APID
     +-- APID 1344 SWE_SCIENCE --> swe_science() --> imap_swe_l1a_sci
     |                             (8-bit -> 16-bit decompression)
     +-- APID 1330 SWE_APP_HK  ------------------> imap_swe_l1a_hk
     +-- APID 1334 SWE_CEM_RAW ------------------> imap_swe_l1a_cem-raw
           |
           |  (all three then pass through filter_day_boundary_data)
           v
   imap_swe_l1a_sci  +  eu-conversion  +  esa-lut  +  l1b-in-flight-cal
     |
     |  swe_l1b_science()
     |    1. convert_raw_to_eu on science metadata
     |    2. drop calibration-mode data (esa_table_num != 0)
     |    3. keep only complete 0,1,2,3 quarter-cycle runs
     |    4. checkerboard reorganization -> (n, 24, 30, 7)
     |    5. acquisition time per measurement
     |    6. deadtime correction
     |    7. in-flight gain calibration (+ LAST_CAL_INTERVAL flag)
     |    8. counts -> rate
     |    9. sqrt(counts) uncertainty, ESA energies
     v
   imap_swe_l1b_sci
     |
     |  swe_l2()
     |    1. phase space density from count rate
     |    2. number flux from phase space density
     |    3. spin phase from SPICE -> spin angle
     |    4. bin into 30 fixed 12-degree spin angle bins
     v
   imap_swe_l2_sci   ---->  (separate repository)  ---->  SWE L3

   L0 CCSDS (.pkts)  --  swe_l1b(), use_derived_value=True  -->  imap_swe_l1b_hk

Dependency wiring
-----------------

**[CODE]** ``imap_processing/cli.py``, ``class Swe``. The CLI checks the
*number* of dependencies before doing anything, so an extra or missing
ancillary file fails loudly rather than silently changing behavior.

.. list-table::
   :header-rows: 1
   :widths: 12 12 34 42

   * - Level
     - Descriptor
     - Dependencies (exact count enforced)
     - Notes
   * - ``l1a``
     - -
     - **2**: SWE L0 file, time kernels
     - ``swe_l1a(path)`` takes a plain path, not the collection. Returns a list
       of up to three datasets.
   * - ``l1b``
     - ``sci``
     - **5**: L1A science, ``l1b-in-flight-cal``, ``esa-lut``,
       ``eu-conversion``, time kernels
     - Exactly one science file; multiple is rejected.
   * - ``l1b``
     - ``hk``
     - **2**: SWE L0 file, time kernels
     - Reparses L0 with ``use_derived_value=True``. Does **not** read L1A HK.
   * - ``l2``
     - -
     - **2**: L1B science, spin data
     - Exactly one science file. Spin data is consumed indirectly through
       SPICE in ``get_instrument_spin_phase``.

Any other data level raises ``NotImplementedError``.

Ancillary inputs
----------------

Covered in full in :ref:`swe-ancillary`. Summary of descriptors:

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Descriptor
     - Used at
     - Purpose
   * - ``esa-lut``
     - L1B
     - The 8 onboard ESA stepping tables. Supplies both the checkerboard index
       map and the per-cell ESA voltage.
   * - ``l1b-in-flight-cal``
     - L1B, I-ALiRT
     - Per-CEM gain factors versus MET, one row per weekly gain sweep.
   * - ``eu-conversion``
     - L1B
     - Polynomial raw-to-engineering conversions for science packet metadata.
   * - SPICE kernels
     - L1A, L1B, L2
     - Time conversion everywhere; spin phase at L2.

CDF variable inventory
----------------------

L1A science (``imap_swe_l1a_sci``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinates: ``epoch``, ``spin_sector`` (0-179), ``cem_id`` (0-6), plus label
variables.

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Variable
     - Shape
     - Meaning
   * - ``science_data``
     - (epoch, 180, 7)
     - **Decompressed** 16-bit counts.
   * - ``raw_science_data``
     - (epoch, 180, 7)
     - The 8-bit values as telemetered. Dropped at L1B.
   * - science packet metadata
     - (epoch,)
     - ``shcoarse``, ``acq_start_coarse``, ``acq_start_fine``,
       ``acq_duration``, ``settle_duration``, ``spin_phase``, ``spin_period``,
       ``quarter_cycle``, ``esa_table_num``, ``esa_acq_cfg``, ``threshold_dac``,
       ``stim_enabled``, ``stim_cfg_reg``, ``cem_nominal_only``,
       ``spin_period_validity``, ``spin_phase_validity``,
       ``spin_period_source``, ``repoint_warning``, ``high_count``, ``cksum``.
       All in **raw** units at L1A.

**[CODE]** The CCSDS header fields (``version``, ``type``, ``sec_hdr_flg``,
``pkt_apid``, ``seq_flgs``, ``src_seq_ctr``, ``pkt_len``) are dropped in
``swe_science()`` before the merge, with a ``TODO`` noting they should not be
returned by ``packet_file_to_datasets`` in the first place. The APID is
preserved as the global attribute ``packet_apid``, which L1B reads back to pick
the EU conversion table.

L1B science (``imap_swe_l1b_sci``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinates: ``epoch`` (one per full cycle), ``esa_step`` (0-23),
``spin_sector`` (0-29), ``cem_id`` (0-6), ``cycle`` (0-3), plus labels.

.. list-table::
   :header-rows: 1
   :widths: 26 28 46

   * - Variable
     - Shape
     - Meaning
   * - ``science_data``
     - (epoch, 24, 30, 7)
     - **Count rate**, counts/s. Deadtime corrected and gain calibrated.
   * - ``counts_stat_uncert``
     - (epoch, 24, 30, 7)
     - ``sqrt`` of the decompressed counts. **In counts, not counts/s** - see
       :ref:`swe-implementation-status`.
   * - ``acquisition_time``
     - (epoch, 24, 30)
     - MET seconds at the **center** of each measurement's accumulation
       window.
   * - ``acq_duration``
     - (epoch, 24, 30)
     - Microseconds, per measurement.
   * - ``esa_energy``
     - (epoch, 24, 30)
     - Electron energy in eV for each cell: ESA voltage from the LUT times the
       analyzer constant.
   * - ``data_quality``
     - (epoch,)
     - ``SweL1bFlags`` bitfield. Currently only bit 2, ``LAST_CAL_INTERVAL``.
   * - packet metadata
     - (epoch, 4)
     - Every L1A metadata field, reshaped so the four quarter cycles of a full
       cycle sit along the ``cycle`` dimension. Engineering units where the EU
       table defines a conversion.

``epoch`` is ``met_to_ttj2000ns`` of the acquisition start time of the **third**
quarter cycle (index 2) of each full cycle, i.e. approximately the center of
the ~1 minute cycle.

L2 science (``imap_swe_l2_sci``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinates: ``epoch``, ``esa_step`` (0-23), ``energy`` (24 values in eV),
``spin_sector`` (0-29), ``inst_az`` (30 bin centers, 6...354 deg), ``cem_id``
(0-6), ``inst_el`` (the 7 CEM polar angles), plus labels.

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Variable
     - Dimensions
     - Meaning
   * - ``phase_space_density``
     - epoch, energy, inst_az, inst_el
     - Binned into 12-degree spin angle bins. **The primary L2 product.**
       Units s^3 / cm^6.
   * - ``flux``
     - epoch, energy, inst_az, inst_el
     - Same, as differential number flux. Units 1 / (eV cm^2 s ster).
   * - ``phase_space_density_spin_sector``
     - epoch, esa_step, spin_sector, cem_id
     - **Unbinned**, on the checkerboard grid. Retained explicitly for L3,
       which needs per-measurement resolution to compute pitch angles.
   * - ``flux_spin_sector``
     - epoch, esa_step, spin_sector, cem_id
     - Same.
   * - ``inst_az_spin_sector``
     - epoch, energy, inst_az (see note)
     - The **actual** SWE spin angle in degrees of each measurement, before
       binning. Needed by L3.
   * - ``psd_stat_uncert``, ``flux_stat_uncert``
     - epoch, esa_step, spin_sector, cem_id (see note)
     - Statistical uncertainties.
   * - ``acquisition_time``, ``acq_duration``, ``data_quality``
     - carried through from L1B
     - Carried "for L3 purposes" per comments in ``swe_l2.py``.

.. warning::

   **[CODE]** Two of the L2 variables carry dimension names that do not match
   the data in them. ``inst_az_spin_sector`` holds unbinned
   ``(epoch, esa_step, spin_sector)`` data but is declared
   ``(epoch, energy, inst_az)``; ``psd_stat_uncert`` and ``flux_stat_uncert``
   hold **binned** data but are declared ``(esa_step, spin_sector)``. The sizes
   happen to match (24 energies vs 24 ESA steps, 30 sectors vs 30 bins), so
   nothing raises. See :ref:`swe-implementation-status`.

Epoch convention
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Level
     - ``epoch``
   * - L1A
     - Straight from ``packet_file_to_datasets``: one per packet, derived from
       the packet's ``SHCOARSE``.
   * - L1B science
     - One per full cycle. ``met_to_ttj2000ns`` of
       ``ACQ_START_COARSE + ACQ_START_FINE/1e6`` of the **third** quarter cycle
       packet.
   * - L2
     - Inherited unchanged from L1B.
   * - I-ALiRT
     - ``met_to_ttj2000ns`` of the midpoint of each 30-second half cycle,
       stored as ``swe_epoch``.

Filenames and descriptors
-------------------------

Standard IMAP convention: ``imap_swe_<level>_<descriptor>_<YYYYMMDD>_v<NNN>.cdf``,
derived by ``write_cdf()`` from ``Logical_source`` and ``Data_version``.
Descriptors in use: ``sci``, ``hk``, ``cem-raw``.

**[CODE]** ``filter_day_boundary_data(ds, self.start_date)`` is applied to every
L1A dataset, so a file dated ``YYYYMMDD`` contains only packets from that UTC
day even when the L0 file spans a boundary. It is **not** applied at L1B or L2.
