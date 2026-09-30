.. _codice-l1a:

Level 1A - Unpacking
====================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Sections 7, 8 and 10. **[CODE]** ``codice_l1a.py``, ``utils.py``,
``decompress.py`` and the eleven ``codice_l1a_*`` product modules.

L1A is where essentially all of CoDICE's complexity lives. Nothing here is a
science calculation; it is entirely about reversing a very flexible on-board
packing scheme.

.. important::

   **You cannot unpack a CoDICE science packet without the SCI-LUT.** Every
   science packet carries four identifiers - ``Table_ID``, ``Plan_ID``,
   ``Plan_Step``, ``View_ID`` - which must be used *in tandem* to look up how
   the data was collapsed and compressed on board. The lookup tables originate
   as an Excel spreadsheet (``26850.03-SCI-LUT-01.xls``) and reach the SDC as a
   JSON ancillary file with descriptor ``l1a-sci-lut``.

Packet structure
----------------

**[DOC]** All CoDICE packets share the standard CCSDS primary header plus
``SHCOARSE`` (32-bit spacecraft seconds).

Non-PHA science packets then carry the fields below. The latest version of the CoDICE algorithm document 
dropped the "Max Length in Bits" column from these tables. The widths shown here come from a
January 2026 draft, and are included for reference only. The XTCE files are the authority.

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Mnemonic
     - Bits
     - Description
   * - ``Packet_Version``
     - 16
     - Incremented when the packet format changes. **The ground branches on
       this** (``<= 1`` = pre-2026-01-29 FSW).
   * - ``Spin_Period``
     - 16
     - Spin period in **spin ticks of 320 us**, active when the 16-spin cycle
       started.
   * - ``Acq_Start_Seconds``
     - 32
     - Whole seconds at the start of the 16-spin cycle.
   * - ``Acq_Start_Subseconds``
     - 20
     - Sub-seconds (16 for PHA packets).
   * - ``ST_Bias_Gain_Mode_Flag``
     - 2
     - Suprathermal high/low gain bias curve. Not used for Hi.
   * - ``SW_Bias_Gain_Mode_Flag``
     - 2
     - Solar-wind high/low gain bias curve. Not used for Hi.
   * - ``Table_ID``
     - 32
     - Which SCI-LUT version applies.
   * - ``Plan_ID``
     - 16
     - Plan table in use.
   * - ``Plan_Step``
     - 4
     - Plan step active during acquisition.
   * - ``View_ID``
     - 4
     - How the data was collapsed and/or compressed.
   * - ``RGFO_Half_Spin``
     - 6
     - Half spin at which RGFO activated. Not used for Hi.
   * - ``NSO_Half_Spin``
     - 6
     - Half spin at which NSO activated. Not used for Hi.
   * - ``Data_Quality_Flag``
     - 1
     - Errors during acquisition/processing (EDAC, timing violations, ...).
       **[CODE]** decommutated as ``suspect``, written as ``data_quality``.
   * - ``Compression_Flag``
     - 3
     - See the compression table below.
   * - ``Byte_Count``
     - 23
     - Length of the data array; the **compressed** length if compressed.
   * - ``RGFO_esa_step``, ``RGFO_spin_sector``, ``NSO_esa_step``,
       ``NSO_spin_sector``
     - -
     - **Added by the 2026-01-29 FSW update.** Pin the trigger to an exact bin.
       **[DOC]** "the ESA step / spin sector at which the RGFO (NSO) limit is
       exceeded. RGFO (NSO) mode is activated for all ESA steps (spin sectors)
       after this value." The spin-sector fields count over a **full spin
       (0-23)**, while COUNTS-product ``spin_sector`` is half-spin relative
       (0-11), so compare using ``% 12``. The same fields, plus
       ``RGFO_Half_Spin`` / ``NSO_Half_Spin``, are also in PHA packets since
       2026-01-29.

**[DOC]** PHA (direct-event) packets differ: 16-bit ``Acq_Start_Subseconds``, a
4-bit ``Priority``, a 32-bit ``Num_Events``, a 1-bit ``Compressed`` flag and a
31-bit ``Byte_Count``. Event data is Rice-compressed if the flag is set.
**[CODE]** ``DE_METADATA_FIELDS`` in ``constants.py`` lists the exact field
widths the code unpacks, and applies **LZMA** (``CoDICECompression.LOSSLESS``)
when ``compressed`` is set.

**[DOC]** When the data array is too large for one CCSDS packet, CoDICE uses the
**CCSDS grouping flags** to spread it across several packets. Fields are padded
to a 16-bit boundary; pad bits are counted in the CCSDS length field but **not**
in ``Byte_Count``.

Compression
-----------

**[DOC]** Section 7:

.. list-table::
   :header-rows: 1
   :widths: 12 28 60

   * - Flag
     - Name
     - Algorithm
   * - 0
     - No compression
     - -
   * - 1
     - Lossy A
     - Table-based 24 -> 8 bit compression per counter value; decompress by
       looking up the centre of each compression region.
   * - 2
     - Lossy B
     - As above with a different table.
   * - 3
     - Lossless
     - LZMA over the whole ``Data`` field as a single unit.
   * - 4
     - Lossy A + Lossless
     - Undo LZMA first, then the lossy table.
   * - 5
     - Lossy B + Lossless
     - As above.

**[CODE]** ``CoDICECompression`` in ``utils.py`` adds a sixth value the document
does not list:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Value
     - Name
     - Implementation
   * - 6
     - ``PACK_24_BIT``
     - ``_apply_pack_24_bit`` - reads 3-byte big-endian integers into 32-bit
       values. Used where the on-board counters are 24-bit packed rather than
       compressed.

``LOSSY_A_TABLE`` and ``LOSSY_B_TABLE`` are 256-entry dictionaries in
``constants.py``, transcribed from Greg Dunn's ``sohis_cdh_utils.v``. The values
are expected to change; the format is not.

``decompress(compressed_bytes, algorithm)`` dispatches on the enum and applies
LZMA before the lossy table for the combined modes.

.. note::

   **[DOC]** The 2026-01-29 FSW load made **Hi and Lo priority counts
   lossless-only** (no lossy stage). Because the algorithm is read from the
   SCI-LUT view table rather than hard-coded, no code change was needed.

The unpacking algorithm
-----------------------

**[DOC]** Section 7, verbatim in structure:

1. **Plan lookup.** ``PlanTables[Plan_ID][Plan_Step]`` gives ``Iterations``,
   ``ESA Sweep`` (0 or 1; nominally 0, descending in energy), ``Lo Stepping``
   (0-4), ``Hi Products`` and ``Lo Products`` table indices.
2. **Energies.** The ESA Sweep table has 128 entries of voltage; multiply by
   ``k_factor`` (5.76) and divide by 1000 for keV/e.
3. **Acquisition time.** Index the Lo Stepping table by ESA step; the "Acq Time
   (including Sector Margin)" column gives milliseconds.
4. **View lookup.** ``View_ID`` selects a row of the ``Views`` table giving the
   expected APID, compression scheme and collapse-table index.
5. **Decompress** in the correct order (lossless first, then lossy).
6. **Extract:**

   * **Lo (APIDs 0x480-0x48F):** 128 frames (one per ESA step); each frame
     started as a [12 spin angle x 24 position] matrix; use the collapse table
     to index into ``Collapse_Lo``.
   * **Hi (APIDs 0x490-0x49F):** 192 frames (one per counter); each frame is
     [n spins x 24 spin angles x 16 azimuths]; use ``Collapse_Hi``, and use the
     ``3D Collapse`` value to know how many spins went into the matrix.

7. **Write.** The collapse table ID determines which data product is being
   extracted and how it is written. **No transformations at L1A except writing
   Electrons as Flashed minus Unflashed.**

Worked example (section 8)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** ``Table_ID = 1``, ``Plan_ID = 0``, ``Plan_Step = 0``, ``View_ID = 3``:

* Table 1 -> use ``26850.03-SCI-LUT-01.xls``.
* Plan 0 occupies rows 7-10; plan step 0 is row 7: Iterations 1, ESA Sweep 0,
  Lo Stepping 0, Hi/Lo products 0.
* ESA Sweep 0 -> 128 decreasing voltages.
* Sweep step 74 with Lo Stepping 0 -> row 28 of the Lo Stepping tab -> 118.75 ms
  acquisition time.
* Views tab, Lo, row 10 for ``View_ID`` 3 -> lossy **and** lossless compressed,
  collapse table 3.
* Collapse_Lo table 3 is "SW Angular Counts" -> write 128 E/q x 12 spin sectors
  x 5 positions to ``imap_codice_l1a_lo-angular`` variable
  ``SunwardAngularCounts``.

How the code implements it
--------------------------

**[CODE]** ``codice_l1a.process_l1a`` does three things:

1. Picks the XTCE file by date (see :ref:`codice-modes`).
2. Calls ``packet_file_to_datasets(science_file, xtce_file)`` -> ``dict[apid,
   Dataset]``.
3. Dispatches each APID to a product module, wrapping non-PHA products in
   ``utils.process_by_table_id``.

``process_by_table_id``
^^^^^^^^^^^^^^^^^^^^^^^

The shared wrapper for every non-PHA, non-housekeeping product. It:

* reads ``view_id``, ``pkt_apid``, ``plan_id`` and ``plan_step`` from the
  **first** record (they are assumed uniform across a stream),
* splits the dataset by unique ``table_id``,
* calls the per-product function once per group with signature
  ``(group_ds, lut_file, table_id, view_id, apid, plan_id, plan_step)``,
* concatenates on ``epoch`` with ``data_vars="minimal", coords="minimal",
  compat="equals"`` and sorts by epoch.

The ``minimal`` settings keep 1-D support variables such as ``voltage_table``
and ``k_factor`` from being broadcast along ``epoch``.

.. warning::

   ``view_id``, ``plan_id`` and ``plan_step`` are taken from index 0 only. If
   any of them change mid-file the rest of the day will be unpacked with the
   wrong view. Only ``table_id`` is handled per-group.

SCI-LUT access helpers
^^^^^^^^^^^^^^^^^^^^^^

All in ``utils.py``:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Function
     - What it does
   * - ``read_sci_lut(path, table_id)``
     - Loads the JSON and returns the sub-dict for one ``table_id``. Raises if
       absent.
   * - ``get_view_tab_info(json, view_id, apid)``
     - Indexes ``view_tab`` by the literal key ``"(<view_id>, 0x<APID hex>)"``.
   * - ``get_view_tab_obj(...)``
     - Returns ``(sci_lut_data, ViewTabInfo)``. ``ViewTabInfo`` carries ``apid``,
       ``view_id``, ``sensor`` (0 = Lo, 1 = Hi), ``collapse_table``,
       ``compression`` and ``three_d_collapsed``.
   * - ``get_collapse_pattern_shape(json, sensor_id, collapse_table_id)``
     - Reads ``collapse_<lo|hi>[<id>]["matrix"]`` and derives the **reduced**
       shape. Returns ``(1,)`` when every non-zero entry is identical (fully
       collapsed), otherwise ``(unique_spin_sectors, unique_inst_azs)``.
   * - ``get_counters_aggregated_pattern(...)``
     - Reads ``collapse_<sensor>[<id>]["variables"]``, drops rows containing a
       zero (counter turned off in flight), sorts by first value, and returns
       ``{counter_name: n_spin_sectors}``.
   * - ``index_to_position(...)``
     - Indices of unique non-zero rows in the collapse matrix - i.e. which
       physical positions survived collapsing.
   * - ``calculate_acq_time_per_step(lo_stepping_tab)``
     - Appendix C timing, returns a 128-element array in **seconds**.
   * - ``get_energy_info(energy_table)``
     - Geometric bin centres ``sqrt(min*max)`` and the plus/minus deltas for Hi
       energy-per-nucleon bins.
   * - ``get_codice_epoch_time(...)``
     - Epoch centre and delta - see below.

Epoch construction
^^^^^^^^^^^^^^^^^^

**[DOC]** "an Epoch variable is written in CDF_TT2000 format which is the start
time of the acquisition ... For variables with an accumulation period, a
DELTA_EPOCH_PLUS is written with the units of seconds."

**[CODE]** ``get_codice_epoch_time`` writes the **centre** of the accumulation
window with symmetric ``epoch_delta_minus``/``epoch_delta_plus`` in **integer
nanoseconds**:

.. code-block:: python

   spin_period_ns = spin_period.astype(np.int64) * 320_000
   delta_times    = (num_spins * spin_period_ns) // 2
   center_times_seconds = (acq_start_seconds
                           + acq_start_subseconds / 65536
                           + delta_times / 1e9)

``num_spins`` is **16** for every Lo product and for Hi PHA; for other Hi
products it is the SCI-LUT ``3d_collapse`` value.

This is a deliberate deviation - centre-plus-delta is the IMAP project
convention - but note that the subsecond divisor is ``65536`` (2^16) even though
non-PHA packets declare a **20-bit** subseconds field. The CoDICE team specified
``subseconds / 65536``; see :ref:`codice-implementation-status`.

Per-product notes
-----------------

Lo species counts (``codice_l1a_lo_species.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.3.3. Data is collapsed on board over spin sectors and
positions to **two values per energy**: azimuths 1, 2, 3, 23, 24 sum to a
sunward product; 4-22 sum to a non-sunward product. Because the spin axis is
sun-facing these assignments are constant, so the two are separated on board
and telemetered as different products.

The section 9.2 caveats say the flight instrument
**actually bins by delay-line position**, which lowers the counts and
intensities. This is an on-board issue; the ground cannot undo it. See
:ref:`codice-data-caveats`.

**[CODE]** Handles ``COD_LO_SW_SPECIES_COUNTS`` and ``COD_LO_IAL``. Reshapes to
``(num_packets, num_species, 128, *collapsed_shape)`` where ``collapsed_shape``
is normally ``(1,)``. Output dims ``(epoch, esa_step, spin_sector)`` with
``spin_sector`` of length 1.

Species selection is defensive: the module takes the union of the SCI-LUT's
``desired_species_names`` and the hard-coded list, then for each wanted species
either takes its index in ``actual_species_names`` or logs a warning and fills
NaN. This absorbs the Fe highQ/lowQ label swap.

Uncertainty is ``sqrt(counts)``, matching **[DOC]**
:math:`\sigma_j(l) = \sqrt{C_j(l)}`.

NSO masking, **[DOC]** section 10.3.3, branches by date:

* **Launch - 2026-01-29 (P0-P2):** the instrument enters NSO on the half-spin
  *after* ``NSO_Half_Spin``, so set to NaN where ``half_spin > NSO_half_spin``.
* **2026-01-29 onward (P3+):** NSO now starts mid-half-spin at a specific (ESA
  step, spin sector). The summed species counts for that half-spin are not
  representative, so set to NaN where ``half_spin >= NSO_half_spin``.

**[CODE]** One rule for all dates:

.. code-block:: python

   nso_mask = (half_spin_per_esa_step >= nso_half_spin[:, None]) | \
              (half_spin_per_esa_step == HALF_SPIN_FILLVAL)

i.e. ``half_spin >= NSO_half_spin`` -> NaN, plus anything at a padded (never
sampled) ESA step. ``half_spin_per_esa_step`` is set to ``HALF_SPIN_FILLVAL``
(63) and ``acquisition_time_per_esa_step`` to NaN in the same places.

.. warning::

   **Discrepancy.** For pre-2026-01-29 data the code also NaNs the
   ``half_spin == NSO_half_spin`` half-spin, which the document says was still
   valid, so one extra half-spin of ESA steps is discarded per NSO cycle. This
   rule is unchanged between the January 2026 draft and Rev 3 Chg 1. See
   :ref:`codice-implementation-status`.

Lo priority counts (``codice_l1a_lo_priority.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.3.5. Summed over all positions; final arrays are
(128 ESA steps x 12 spin sectors), one variable per priority.

**[DOC]** CMAD adds an explicit NSO rule for this product:

* **Launch - 2026-01-29:** ``half_spin > NSO_half_spin`` -> NaN.
* **2026-01-29 onward:** the exact-bin rule, identical to the angular
  products (section 10.3.4):

  1. ``half_spin > nso_half_spin`` -> NaN
  2. ``half_spin == nso_half_spin``:

     a. ``spin_sector > nso_spin_sector`` -> NaN
     b. ``spin_sector == nso_spin_sector`` and ``esa_step > nso_esa_step`` ->
        NaN

  with ``nso_spin_sector`` taken **mod 12**, because it counts over the full
  spin (0-23) while ``spin_sector`` is half-spin relative (0-11).

**[CODE]** When ``packet_version > 1`` the code implements the exact-bin rule
as documented, including the ``% 12``.

For ``packet_version <= 1`` it uses ``half_spin >= nso_half_spin``. **That
disagrees with the new document rule** (``>``) and with the code's own comment
directly above it ("set all data to NaN where half_spin > nso_half_spin").
Compare ``codice_l1a_lo_counters_singles.py``, whose pre-FSW branch uses
``>``. See :ref:`codice-implementation-status`.

Lo instrument counters
^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.3.1. Two packets. ``AGGREGATED`` carries a selectable
subset of 30 rate types; ``SINGLES`` carries the 24 APD singles. The nominal
aggregated selection is 6 rates - TCR (type 0), DCR (1), Start-A (13), Start-B
(14), Stop (15) and Total Position Count (16 in the document's numbering) -
giving 128 ESA steps x 6 rates x 6 spin sectors. Adjacent 15 deg spin sectors
are paired into 30 deg bins, hence 6 rather than 12.

**[CODE]** ``LO_COUNTERS_AGGREGATED_VARIABLE_NAMES`` matches that nominal six.
Which counters are present is read from the collapse matrix's ``variables`` dict
via ``get_counters_aggregated_pattern``, so a re-configuration in flight does not
require a code change - but the six variable names are fixed, so a *different*
selection would not be written.

The singles module applies the same exact-bin NSO rule as priority counts for
``packet_version > 1``, with the extra step of ``nso_spin_sector % 12 // 2``
to map to the 6 paired spin-sector bins. Its pre-FSW branch uses
``half_spin > nso_half_spin``, which matches the document's priority/angular
rule, **unlike** the priority module. The aggregated module uses
``half_spin >= nso_half_spin`` for all dates, like the species module. Section
10.3.1 gives no counters-specific rule. Section 11.2 says only that
"esa-steps that are in half-spins after NSO_Half_Spin" are NaN, which reads as
``>``.

Hi omni-directional counts (``codice_l1a_hi_omni.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.2.3. The product is summed over all spin sectors and SSD
IDs, provided as a function of energy-per-nucleon in **sqrt(2)-spaced** bins.
Nominally summed over every **4 spins** at **1 min** cadence, so there are 4
values per species in each 16-spin packet. Species: H, He3, He4, C, O,
Ne/Mg/Si, Fe, UH and unknown ("junk"). **The number of energy bins varies by
species.** The CDF must carry each species' energy table with centres and
plus/minus deltas.

**[DOC]** CMAD names the deltas ``energy_<species>_delta_plus`` and
``energy_<species>_delta_minus`` (sections 10.2.3 and 10.2.4). **[CODE]** The
variables are ``energy_<species>_plus`` and ``energy_<species>_minus``, and
``codice_l2.py`` reads them by those names. The meaning is the same; renaming
would be a CDF-schema change touching L1A, L1B and L2.

**[CODE]** ``n_spins = int(16 / three_d_collapsed)``; each packet's epoch is
expanded into ``n_spins`` epochs:

.. code-block:: python

   epoch_times = (np.repeat(epoch_center, n_spins)
                  + np.tile(np.arange(n_spins), num_packets)
                  * np.repeat(deltas, n_spins) / 1e9 * 2)   # TODO: why multiply by 2?

The ``* 2`` is unexplained and flagged in the source. It is arithmetically the
undoing of the ``// 2`` in ``get_codice_epoch_time`` (which halves the full
window to make a symmetric delta), so the sub-epoch spacing is a full window
rather than a half - which is probably correct, but nobody has written that
down. See :ref:`codice-implementation-status`.

Hi sectored counts (``codice_l1a_hi_sectored.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.2.4. Accumulated into **12 spin-sector bins (30 deg)**,
summed over **16 spins (4 min)**, **x2-spaced** energy-per-nucleon bins.
Dimensions :math:`C_j(i, n, k)` = (energy bins, 12 spin sectors, 12 SSD IDs).
Species H, He3He4, CNO, Fe.

Hi instrument counters
^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 10.2.1. Both packets are packed identically but decoded
differently. **Aggregated**: all spin sectors and SSDs summed into a single
value for DCR, STO, SPO, MST and ASIC 1/2 invalid flag events. **Singles**: TCR,
SSDO and STSSD summed over spin sectors but stored **per SSD** (array of 12).
Both summed over 16 spins.

Rate-type map (**[DOC]**): 0 = TCR (per SSD), 1 = DCR, 2 = STO, 3 = SPO,
4 = SSDO (per SSD), 6 = STSSD (per SSD), 7 = MST, 12 = Low TOF Cutoff,
15 = ASIC 1/2 flag- and channel-invalid counts. Types 5, 8-11, 13-14 reserved.

Direct events (``codice_l1a_de.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Sections 10.2.2 (Hi) and 10.3.2 (Lo). Bit layouts:

.. list-table::
   :header-rows: 1
   :widths: 26 12 26 12 24

   * - Hi field
     - Bits
     - Lo field
     - Bits
     - Note
   * - SSD Energy
     - 11
     - APD Gain
     - 1
     -
   * - TOF
     - 10
     - APD ID
     - 5
     -
   * - SSD ID
     - 4
     - Position
     - 5
     - Lo direction comes from ``position``, not ``apd_id``
   * - Energy Range (gain)
     - 2
     - APD Energy
     - 9
     -
   * - Multi-Flag
     - 1
     - TOF
     - 10
     -
   * - PHA Type
     - 2
     - Multi-Flag
     - 1
     -
   * - Spin Sector
     - 5
     - PHA Type
     - 2
     - Hi spin sector 0-23
   * - Spin Number
     - 4
     - Spin Sector
     - 5
     -
   * - Priority
     - 3
     - ESA Step
     - 7
     -
   * - Spare
     - 22
     - Priority
     - 3
     -
   * -
     -
     - Spare
     - 16
     -

**[DOC]** Hi gain (``Energy Range``) mapping: 0 = no energy, 1 = low gain,
2 = mid gain, 3 = high gain. **[CODE]** ``GAIN_ID_TO_STR = {1: "LG", 2: "MG",
3: "HG"}``. Per the section 9.2 caveats, LG is currently unusable and only the
upper part of MG is usable; see :ref:`codice-data-caveats`.

**[DOC]** Three PHA event types: **TCR** (start + stop + SSD), **DCR** (start +
stop, no SSD) and **SSD** (SSD only). Each type populates a different subset of
the fields at different resolutions - a DCR event has no energy, no E range and
no SSD ID.

**[CODE]** ``DE_DATA_PRODUCT_CONFIGURATIONS`` gives, per APID, ``num_priorities``
(Hi 6, Lo 8) and the bit structure with ``dtype`` and ``fillval`` per field.
``MAX_DE_EVENTS_PER_PACKET = 10000``. Segmented packets are reassembled using
``SegmentedPacketOrder`` (3 = unsegmented, 1 = first, 0 = continuation,
2 = last), then decompressed with LZMA when the ``compressed`` flag is set, then
bit-unpacked into per-priority arrays with dims ``(epoch, priority, event_num)``.

**[DOC]** Each priority is written to its own CDF variable with the number of
events and the ``Suspect`` flag stored per priority.
