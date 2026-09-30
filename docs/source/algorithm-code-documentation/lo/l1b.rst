.. _lo-l1b:

Level 1B - Annotation, Rates, and Time Selection
================================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Module:** ``imap_processing/lo/l1b/lo_l1b.py`` (~2900 lines, the largest
single file in the instrument).

L1B is where the physics starts. It takes L1A's raw fields and produces:

* per-event annotation (real time, sky direction, species, coincidence class),
* count rates with honest exposure times,
* the time-selection bookkeeping (badtimes, goodtimes) that everything
  downstream filters on,
* background rate estimates.

``lo_l1b(sci_dependencies, anc_dependencies, descriptor)`` is a router; the
descriptor picks one of six branches. See :ref:`lo-data-products` for the
descriptor-to-product table.

Foundational concepts
---------------------

Aggregated Science Cycle (ASC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** The ASC is the unit of aggregation: **28 spins**, over which the ESA
sweeps its 7 steps twice. Building an ASC means gathering the ``ILO_SCI_DE``,
``ILO_SCI_CNT`` and ``ILO_SPIN`` packets of one cycle, sorting them by MET, and
tagging the cycle with its badtime overlap and pivot platform position.

**[CODE]** ``set_spin_cycle``, ``set_spin_cycle_from_spin_data``,
``match_science_to_spin_asc``, ``find_valid_asc``, ``_check_sufficient_spins``.

.. warning::

   ``set_spin_cycle`` still assumes exactly 28 spins per ASC. There is an
   explicit TODO saying this must be updated for direct events, because the
   assumption does not hold in general. Treat any DE result that depends on
   spin indexing as provisional. **[CODE]**

The resweep
^^^^^^^^^^^

**[DOC]** The on-board sweep table may repeat energy steps, e.g. the stepping
sequence ``[1, 1, 3, 4, 5, 6, 7]`` measures level 1 twice and never measures
level 2. Products are reported per **true ESA level**, so measured step indices
must be histogrammed into levels:

.. code-block:: text

   for i in 1..7:
       iesa[i] = sweeptable_level_for_step(i)
   for i in 1..7:
       CNT_RSWP[iesa[i]]  += CNT[i]
       EXPO_RSWP[iesa[i]] += EXPO[i]
   RATE_RSWP[i] = CNT_RSWP[i] / EXPO_RSWP[i]  if EXPO_RSWP[i] > 0 else 0

A level with zero exposure was never measured in that cycle. A zero rate is
therefore a *signal that resweeping happened*, not a measurement of zero.

**[CODE]** ``resweep_histogram_data`` implements this with ``np.add.at`` over
an ``energy_mapping`` obtained from ``_get_esa_level_indices``, which reads the
sweep-table and ESA-mode-LUT ancillaries. It simultaneously builds an
**exposure factor** array counting how many ``N_SPINS_PER_ESA_LEVEL`` blocks
landed in each level, which is what the rate divisor is scaled by.

Exposure time
^^^^^^^^^^^^^

**[DOC]** For an ASC, with the mean spin duration taken from real spin data:

.. math::

   \langle \text{spin duration} \rangle
     = \frac{1}{28}\sum_{k=\text{ASC start}}^{\text{ASC start}+27}
       \bigl(\text{spin end}_k - \text{spin start}_k\bigr)

.. math::

   T_{6^\circ} = \frac{4 \, \langle \text{spin duration}\rangle}{60},
   \qquad
   T_{60^\circ} = \frac{4 \, \langle \text{spin duration}\rangle}{6}

The factor 4 is the number of spins each ESA level is held for within one cycle
(2 spins per step, twice per cycle). The denominator is the number of spin bins
of that angular width.

**[CODE]** ``calculate_histogram_rates`` computes
``exposure_time_6deg = spin_durations / 60`` and
``exposure_time_60deg = spin_durations / 6``, then multiplies by the per-level
``exposure_factors`` from the resweep, which carry the factor of
``N_SPINS_PER_ESA_LEVEL = 4``. Same result, factored differently.

Rates are then simply

.. math::

   R = \frac{C}{T_{\text{effective}}}, \qquad
   \delta R = \frac{\sqrt{C}}{T_{\text{effective}}}

with zero returned wherever the effective exposure is zero.

Product: Annotated Direct Events (``descriptor="de"``)
------------------------------------------------------

``imap_lo_l1b_de``. The densest algorithm in the instrument. Entry points:
``l1b_de`` and ``initialize_l1b_de``.

Processing steps, in order
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **ESA mode** - ``set_esa_mode`` reads the ``sweep-table`` ancillary CSV,
   finds the row covering the data's date and returns ``0`` for ``HiRes`` or
   ``1`` for ``HiThr``.

2. **Spin cycle assignment** - ``set_spin_cycle_from_spin_data`` matches each
   science packet to its spin packet, rejecting ASCs with insufficient or
   invalid spins.

3. **Event time** - ``set_event_met`` / ``set_each_event_epoch``.
   **[DOC]** the algorithm is:

   .. code-block:: text

      CLOCK_RESOLUTION = 1 << 12
      SECONDS_PER_TICK = SPIN_DURATION / CLOCK_RESOLUTION
      last_de_time     = CLOCK_RESOLUTION << 1

      for de in science_cycle.direct_events:
          if de.time < last_de_time:        # tick counter wrapped
              current_spin_index += 1
              current_spin_time = spin_times[current_spin_index].time
          de.full_time = current_spin_time + SECONDS_PER_TICK * de.time
          last_de_time = de.time

   ``SPIN_DURATION`` must be the measured spin duration, not the nominal 15 s.

4. **Golden triple TOF1 recovery** - ``calculate_tof1_for_golden_triples``
   reconstructs the untransmitted TOF from the checksum relation.
   **[CODE]** the left checksum boundary is hardcoded to ``-21`` pending a LUT.

5. **Coincidence type** - ``set_coincidence_type`` turns the 4-bit
   ``ABSENT`` code into a labeled class (triple / double / golden).

6. **TOF engineering units** - ``convert_tofs_to_eu``:

   .. math::

      \mathrm{TOF_{EU}} = C_0 + C_1 \cdot \mathrm{TOF_{DN}}

   with coefficients in ``lo/l1b/tof_conversions.py``:

   .. list-table::
      :header-rows: 1
      :widths: 20 40 40

      * - Channel
        - :math:`C_0`
        - :math:`C_1`
      * - TOF0
        - 5.52524e-01
        - 1.68374e-01
      * - TOF1
        - -7.20181e-01
        - 1.65124e-01
      * - TOF2
        - 3.74422e-01
        - 1.66409e-01
      * - TOF3
        - 4.6726e-01
        - 1.7144e-01

   These came from "Lo's TOF Conversion_annotated.docx" and an email; TOF3 was
   updated in March 2026. They are **not in the algorithm document**, which is
   an open action item. **[CODE]**

7. **Species identification** - ``identify_species``. **[DOC + CODE agree]**
   Classification is by **TOF2** alone, valid for PAC voltages of 7-12 kV:

   .. list-table::
      :header-rows: 1
      :widths: 20 30 50

      * - Species
        - TOF2 range [ns]
        - Label
      * - Hydrogen
        - 13 to 40
        - ``"H"``
      * - Oxygen
        - 75 to 200
        - ``"O"``
      * - anything else
        - -
        - ``"U"`` (unknown)

   .. note::

      **[DOC]** The original design used a PAC-scaled quantity
      :math:`\mathrm{TOF0}_s = \mathrm{TOF0} + \mathrm{MCP\_TOF\_V}/\mathrm{PAC\_VSET}`,
      but flight software cannot supply it for all events, so only TOF2 is
      used. The document also says species ID applies only to triple
      coincidences; **the code applies the TOF2 test to every event that has a
      TOF2**, without restricting to triples. Confirm before relying on it.

8. **Look direction** - ``set_pointing_direction``. Calls
   ``lo_instrument_pointing(et, pivot_angle, SpiceFrame.IMAP_HAE,
   cartesian=True)`` and stores ``hae_x``, ``hae_y``, ``hae_z``.

9. **Pointing bins** - ``set_pointing_bin``:

   * Transform HAE cartesian to the despun ``IMAP_DPS`` frame.
   * Convert to latitudinal; longitude wrapped to [0, 360).
   * Latitude is made relative to the boresight:
     ``lat -= (90 - pivot_angle)``. Values outside +/- 2 degrees log a warning.
   * Bin: ``lon_bins = linspace(0, 360, 3601)`` giving 3600 spin bins of 0.1
     degrees; ``lat_bins = linspace(-2, 2, 41)`` giving 40 off-angle bins of
     0.1 degrees. Results stored as ``spin_bin`` and ``off_angle_bin``.

10. **Badtime flag** - each event inherits its ASC's badtime status.

Output variables (document naming)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** ``TIME``, ``Spin_cycle``, ``TOF0``-``TOF3``, ``POSITION``,
``ENERGY_LEVEL`` (reswept), ``ESA_MODE``, ``MODE_BIT``, ``ABSENT``,
``COINCIDENCE``, ``SPECIES``, ``BADTIME`` (1 = bad, 2 = good), ``SPINBIN``
(0-59), ``POINTINGBIN`` (0-3599), ``DIRECTION`` (unit 3-vector). The code's
variable names differ; several are still missing CDF attributes (see the TODO
list in :ref:`lo-implementation-status`).

Product: Histogram and Monitor Rates (``descriptor="all-rates"``)
-----------------------------------------------------------------

``imap_lo_l1b_histrates`` and ``imap_lo_l1b_monitorrates``. Entry point
``l1b_allrates``, then ``resweep_histogram_data`` and
``calculate_histogram_rates``, split by ``split_rate_dataset``.

* **Histogram rates** carry the on-board ``hydrogen`` and ``oxygen``
  histograms, shape ``(epoch, 7 esa_step, 60 spin_bin_6)``, plus
  ``exposure_time_6deg``.
* **Monitor rates** carry the singles (``start_a``, ``start_c``, ``stop_b0``,
  ``stop_b3``), the TOF doubles, the discarded counters and the position
  counters at ``(epoch, 7, 6)`` with ``exposure_time_60deg``, and the triples
  (``tof0_tof1``, ``tof0_tof2``, ``tof1_tof2``, ``silver``) at
  ``(epoch, 7, 60)``.

Product: Direct Event Rates (``descriptor="derates"``)
------------------------------------------------------

``imap_lo_l1b_derates``, from ``calculate_de_rates``.

**[DOC]** This is a **diagnostic**: it re-histograms the annotated direct
events onto the same grid as the on-board histograms so the two can be
compared. A disagreement indicates instrument saturation or event loss in
telemetry. It also gives per-coincidence-class counts, which the on-board
histograms do not.

Binning is by ``(ASC, SPINBIN, ENERGY_LEVEL)`` with separate accumulators for
``H_CNTS``, ``O_CNTS``, ``TRP_CNTS`` (triples) and ``DBL_CNTS`` (doubles), each
divided by the 6-degree exposure time to give rates.

**[DOC]** Coincidence class from the ``ABSENT`` code:

* Triples: 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xC
* Doubles: 0x6, 0x7, 0xA, 0xB, 0xD

Product: Badtimes (``descriptor="badtimes"``)
---------------------------------------------

``imap_lo_l1b_badtimes``, from ``create_badtimes_dataset``.

**[DOC]** A 7-minute window (one histogram accumulation, two ESA cycles) is
flagged bad in its entirety if *any* of these occur inside it:

1. No histogram and/or direct event data.
2. Off-nominal instrument state (gain or threshold testing, off-nominal
   housekeeping).
3. Spacecraft maneuver or pivot platform motion.
4. Loss of pointing accuracy: star-sensor pointing disagrees with the
   spacecraft star tracker, or quaternions do not show regular spin motion.
5. A spin packet not containing 28 spins, i.e. an incomplete science cycle.

Output columns: ``YYYYDDD``, ``START``, ``END``, ``BIN_START``, ``BIN_END``,
``LO``, ``BADTIME_FLAG`` (per ESA step, ``[N, 7]``), ``COMMENTS``.

**[CODE]** The implementation currently covers **only criterion 3, partially**:
it reads spin data and flags spins with ``thruster_firing``. If no spin data is
available it returns a correctly-shaped empty dataset. There is a bare
``# TODO: Add badtimes``. Criteria 1, 2, 4 and 5 are not implemented.

Product: Goodtimes and background rates (``descriptor="goodtimes"``)
--------------------------------------------------------------------

``imap_lo_l1b_goodtimes`` and ``imap_lo_l1b_bgrates``, from
``l1b_bgrates_and_goodtimes``.

.. important::

   The algorithm document says the goodtimes algorithm is **TBD**, done
   manually for IBEX-Lo, to be developed from IBEX data and refined after
   launch. The code has a **complete automated implementation** that is not
   described in the document at all. This page is the only description of it.
   **[CODE]**

Pivot angle determination
^^^^^^^^^^^^^^^^^^^^^^^^^

Taken as the **median** of ``pcc_coarse_pot_pri`` from ``imap_lo_l1b_nhk`` over
hours 0.5 to 22.5 of the day (``LoConstants.PIVOT_HK_HOUR_RANGE``), avoiding
the repointing maneuver at each end. Defaults to 90 degrees if unavailable.

Background thresholds
^^^^^^^^^^^^^^^^^^^^^

The measured pivot angle is matched to a ``PivotAngleSpec`` in
``LoConstants.PIVOT_ANGLES`` by its ``[min, max]`` window, giving ram and
anti-ram background-rate thresholds. Only 75, 90 and 105 degrees have
pivot-specific values; everything else falls back to
``THRESHOLD_BG_RATE_RAM_DEFAULT = 0.028`` and
``THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT = 0.014`` counts/s, which are the
90-degree values.

An optional ancillary matching ``bg-rates-anti-ram-overrides`` can override the
anti-ram threshold for specific ``(year, doy)`` pairs, for anomalous days.

The ram / anti-ram split
^^^^^^^^^^^^^^^^^^^^^^^^

Hydrogen histogram counts are summed over spin-angle bins:

* **ram** - bins ``0:20`` and ``50:60``, restricted to
  ``RAM_ESA_LEVELS = (6, 7)``
* **anti-ram** - bins ``20:50``, all ESA levels

The anti-ram signal is used as a **background proxy**: it looks away from the
ENA source, so counts there are dominated by background.

Expected exposures:

.. code-block:: python

   exposure     = HISTOGRAM_CYCLE_EPOCHS * N_CYCLE_AVE * EXPOSURE_FACTOR
                # 420 * 7 * 0.5
   exposure_ram = exposure * len(RAM_ESA_LEVELS) / N_ESA_LEVELS
   exposure_sum = HISTOGRAM_CYCLE_EPOCHS * N_CYCLE_SUM * EXPOSURE_FACTOR

``EXPOSURE_FACTOR = 0.5`` is the fraction of a cycle that contributes real
exposure.

Interval detection
^^^^^^^^^^^^^^^^^^

Walking the histogram epochs one ``N_CYCLE_SUM`` block at a time:

1. If the gap to the next block exceeds ``interval + DELAY_MAX`` (100 s), or
   the gap from the previous epoch exceeds ``DELAY_MAX``, close the open
   interval. This is the missing-data rule.
2. Over a sliding window of ``N_CYCLE_AVE = 7`` cycles centered on the current
   epoch, compute ``ram_rate`` and ``anti_ram_rate`` from hydrogen counts.
3. **The interval is good while both rates are below their thresholds.** The
   first good block opens an interval; the first bad block closes it.
4. While good, accumulate a *synthetic floor* (``BG_RATES[elem] * exposure``,
   the modeled background) and a *proxy floor* (measured anti-ram counts) per
   species, plus the exposures.

Background rate output
^^^^^^^^^^^^^^^^^^^^^^

Per species:

.. math::

   R_{bg} = \frac{\text{synthetic floor}}{T_{\text{goodtime,avg}}},
   \qquad
   \sigma_{R_{bg}} = \frac{\sqrt{\text{synthetic floor}}}{T_{\text{goodtime,avg}}}

with fallbacks:

* no goodtime exposure at all: ``R_bg = anti_ram_threshold *
  BG_RATE_FALLBACK_SCALE[elem]`` (H: 1.0, O: 0.3)
* a computed rate of exactly zero: ``R_bg = anti_ram_threshold /
  BG_RATE_FLOOR_DIVISOR[elem]`` (H: 50, O: 150)

Nominal rates are ``BG_RATES = {"H": 0.0014925, "O": 0.000136635}`` counts/s.

Goodtime intervals are padded by ``GOODTIME_PADDING = 2.0`` s at each edge so
whole cycles are covered.

Product: Processed star sensor (``descriptor="prostar"``)
----------------------------------------------------------

``imap_lo_l1b_prostar``, from ``l1b_star``.

**[DOC]** "Algorithms TBD based on calibration and on-orbit analysis." The
document only specifies passing through ``TIME``, ``Spin_number``, ``COUNT``,
and ``DATA[N, 720]``.

**[CODE]** The code is well ahead of the document:

* ``filter_valid_star_records`` drops records whose ``COUNT`` is below
  ``STAR_MIN_COUNT_THRESHOLD = 700``.
* ``get_star_bin_offset`` chooses a fractional bin shift from the IFB star-sync
  housekeeping state: ``"DS"`` (sync disabled, pre-FSW 4.8) -> ``0.5``
  (bin center); ``"EN"`` (FSW 4.8+) -> ``0.0`` (left edge).
* ``STAR_END_BINS_TO_EXCLUDE = 2`` end bins are dropped from each profile
  average.
* ``get_sampling_cadence_from_nhk`` derives the sample cadence,
  ``calculate_star_sensor_profile_for_group`` and
  ``calculate_star_sensor_profiles_by_group`` build the time-vs-spin-angle
  profiles.

Spacecraft state and SPICE
--------------------------

**[DOC]** L1B needs 10 Hz spacecraft position and orientation in SPICE format,
and must efficiently compute the boresight direction in the spin frame. Fast
linear or quadratic interpolation of SPICE-derived pointing is suggested.

**[CODE]** Everything goes through ``imap_processing.spice``:
``geometry.SpiceFrame``, ``frame_transform``, ``frame_transform_az_el``,
``cartesian_to_latitudinal``, ``spin.get_spin_number``,
``repoint.get_pointing_times_from_id``, and the ``time`` helpers
``met_to_ttj2000ns`` / ``ttj2000ns_to_et`` / ``ttj2000ns_to_met``.
