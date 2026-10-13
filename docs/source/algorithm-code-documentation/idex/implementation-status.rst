.. _idex-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/idex`` against the
IDEX Algorithms Document dated 1 June 2026. If you change something material,
update this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Level
     - State
     - Notes
   * - L0
     - **Complete**
     - 54 lines, one function. Science and housekeeping decommutated
       separately because the science XTCE branches.
   * - L1A science
     - **Mature**
     - Fragment assembly, retransmit handling, both waveform decode paths,
       time-axis reconstruction and incomplete-event filtering are all done and
       tested. The event-flag classification is an undocumented addition.
   * - L1A messages / catlst
     - **Complete, partly undocumented**
     - Template rendering works and degrades gracefully. The ``catlst``
       products are not in the document at all, and an ``l1b_catlst`` is
       emitted by the *L1A* job.
   * - L1B science
     - **Complete, diverges from the document**
     - All six operations from section 4.7.1 are implemented. The TOF
       conversion factors and their **units** differ from Table 4.2; the code
       is newer.
   * - L1B messages
     - **Complete, diverges from the document**
     - ``pulser_on`` uses paired-transition logic the document does not
       describe. Extremely brittle exact-string matching.
   * - L2A
     - **Written, largely withheld**
     - Target and ion-grid fits, velocity and mass are published. The entire
       TOF mass-spectrum path is computed and then NaN-filled. Ion-grid
       velocity/mass is implemented despite being listed as future work.
   * - L2B / L2C
     - **Written, partly withheld; undocumented**
     - Dust-hit counts and uptime-corrected rates are published; everything
       mass- or charge-resolved is fill-valued. ~840 lines that the document
       covers with two flowchart boxes.
   * - L3
     - **Does not exist**
     - Not a gap. See :ref:`idex-l3-scope`.
   * - Quicklook
     - **Not started**
     - No IDEX quicklook code, and the document does not specify any.

Where the code has moved past the document
------------------------------------------

The 1 June 2026 document was written against a snapshot of this repository and
is unusually faithful to it. These are the places it has since fallen behind.
**In all of them the code is authoritative.**

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Topic
     - Divergence
   * - **Event and saturation flags**
     - ``idex_event_flags.py`` (~480 lines, ten flags) appears nowhere in the
       document. It gates L2A's derived estimates and is the sole filter on
       L2B/L2C counts. See :ref:`idex-event-classification`.
   * - **TOF conversion factors**
     - Document Table 4.2: TOF High/Mid/Low = 2.89e-4 / 1.13e-2 / 5.14e-1
       **pC/DN**. Code ``ConversionFactors``: 7.50e-5 / 2.93e-3 / 1.34e-1
       **mA/DN**. The three low-rate channels agree exactly. The L1B variable
       attrs say ``mA``, and ``test_idex_l1b.py`` rescales the IDEX team's own
       validation file from "the legacy pC factors" before comparing.
   * - **Ion-grid velocity and mass**
     - Document section 4.7.9 lists it as future work and section 4.7.7 lists
       ``ion_grid_velocity_estimate`` and ``ion_grid_dust_mass_estimate`` in
       the NaN block. Both are **implemented and published**, masked only by
       saturation and science-event state.
   * - **L2B / L2C**
     - The document describes them in one sentence each plus a flowchart box.
       ``idex_l2b.py`` implements daily binning, four bin schemes, an uptime
       model derived from the message log, rate quality flags, and a
       rectangular sky map.
   * - **``pulser_on`` logic**
     - Document section 4.7.2 describes unconditional exact-match assignment.
       Code requires a ``PULSER_ON`` message **immediately followed** by a
       ``PULSER_OFF`` message.
   * - **Event message strings**
     - The document quotes rendered strings without the dictionary wrapper
       (``SCI state change: ACQSETUP ==> ACQ``). The code's renderer emits
       ``sciState16Dictionary(ACQSETUP)``, and ``EventMessage`` matches the
       wrapped form.
   * - **Catalog list products**
     - Section 4.6.2 states L1A produces only ``sci`` and ``msg``. The code
       also produces ``l1a_catlst-10days`` and ``l1b_catlst-10days``.
   * - **Trigger origin labels**
     - Document: "software trigger". Code: ``"SW trigger"``. Cosmetic, but the
       code's spelling is what appears in the CDF.

Deliberately withheld products
------------------------------

Not bugs. The variables are created so the CDF schema stays stable and then
overwritten so unvalidated numbers are not mistaken for science. The test suite
asserts this behavior.

**L2A** - filled with ``np.nan`` unconditionally:

``tof_peak_area_under_fit``, ``tof_peak_chi_square``,
``tof_peak_fit_parameters``, ``tof_peak_kappa``,
``tof_peak_reduced_chi_square``, ``tof_snr``, ``mass``, ``mass_scale``.

**L2B** - filled with ``np.iinfo(np.int64).min`` (counts) or ``np.nan`` (rates):

``counts_by_mass``, ``counts_by_charge``, ``rate_by_mass``, ``rate_by_charge``.

**L2C** - same:

``counts_by_mass_map``, ``counts_by_charge_map``, ``rate_by_mass_map``,
``rate_by_charge_map``.

What a consumer can actually use today: per-event target and ion-grid fits,
impact charges, velocity and mass estimates; the per-event SPICE context; and
monthly dust-hit counts and uptime-corrected rates by spin quadrant and by 6°
sky pixel. **No composition, no mass spectrum, no mass- or charge-resolved
rates.**

Suspected defects
-----------------

Each of these was found by reading the code against the document. None is
confirmed by the IDEX team. Confirm before changing behavior.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Where
     - Issue
   * - ``idex_l2b.compute_counts_by_charge_and_mass``
     - **Unit mismatch on dust mass.** Applies ``FG_TO_KG`` (×1e-15) to
       ``target_low_dust_mass_estimate``, which L2A computes as C ÷ (C/kg) =
       **kilograms** and declares as ``UNITS: Kg``. Against mass bin edges of
       6.31e-17 to 1.00e-14 kg this drives every value below the lowest edge,
       where the clip puts it in the bottom bin. Currently invisible because
       the mass-binned outputs are fill-valued. **Must be resolved before those
       products are released.** Either L2A should report femtograms (and its
       attrs updated) or L2B should drop the conversion.
   * - ``idex_l2a``
     - **Dimension name mismatch on the TOF peak fits.**
       ``tof_peak_fit_parameters`` is produced with core dims
       ``["mass_index", "peak_fit_parameters_index"]`` (plural), but the index
       and label variables, and the ``DEPEND_1`` / ``DEPEND_2`` entries in
       ``imap_idex_l2a_variable_attrs.yaml``, all use
       ``peak_fit_parameter_index`` (singular). The labels therefore do not
       attach to the array's third dimension.
   * - ``idex_l2b._get_dust_hit_indices``
     - **Silent empty result.** If ``dust_hit_flag`` is absent from the L2A
       input, returns an empty index array rather than raising. Every count and
       rate in the month becomes zero while
       ``rate_calculation_quality_flags`` still reads 1, because that flag only
       reports uptime problems. A month built from pre-flag L2A files would
       look like a month with no dust.
   * - ``idex_l1a.PacketParser._create_science_dataset``
     - **No guard for an all-bad file.** If every event is skipped
       (conflicting fragments or wrong waveform lengths),
       ``xr.concat(processed_dust_impact_list, dim="epoch")`` is called on an
       empty list and raises.
   * - ``idex_l1a._create_science_dataset``
     - **Fatal on out-of-order packets.** A waveform packet arriving before its
       header raises ``KeyError`` and aborts the entire file. Everywhere else
       L1A is defensive and skips the affected event; this one case is not.
   * - ``idex_l1a`` waveform decoders
     - **Unasserted length agreement.** The uncompressed path drops the last 4
       samples, the compressed path drops the last 3. Both are expected to land
       on 8189 high-rate samples so that a window containing both kinds of
       event concatenates cleanly. Nothing checks it.
   * - ``idex_l2a.calculate_kappa``
     - **Signed mean where the docstring says magnitude.** Computes
       ``mean(mass[peaks] - round(mass[peaks]))``, which ranges over (−0.5,
       0.5] and cancels, so a spectrum with peaks equally split above and below
       integer mass scores near-perfect. The docstring claims a 0-1 range where
       "closer to zero indicates better accuracy". An absolute value or RMS
       would match the stated intent.
   * - ``idex_l2a.time_to_mass``
     - **Hardcoded, unvalidated search window.** Stretch factors are
       ``linspace(1400, 1500, 10)``. Nothing checks whether the winning stretch
       landed on an edge of that bracket, and no diagnostic is emitted if it
       does.
   * - ``idex/atomic_masses.csv``
     - **Apparent off-by-one between mass and isotope name**: ``22,Na``,
       ``23,Mg24``, ``27,Si28``, ``39,Ca40``, ``53,Fe54``, ``196,Au197``.
       Every ``time_to_mass`` stretch factor is fitted against these numbers.
   * - ``idex_l2a.load_calibration_files``
     - **No validation.** ``.values.flatten()[:8]`` on a CSV read with
       ``skiprows=1, header=None``. A reordered or differently-shaped
       calibration file yields eight wrong numbers rather than an error.
   * - ``idex_l2b.compute_rates_*``
     - **Inconsistent no-data sentinel.** Charge/mass-binned rate arrays are
       initialised to ``-1.0``; the agnostic rate arrays are initialised to
       ``np.nan``. Both mean "no uptime data".
   * - ``idex_l2b``
     - **DOY is the daily key.** ``epoch_to_doy`` groups by day of year, so a
       product spanning more than one calendar year would collide days from
       different years. Safe at a monthly cadence; unsafe if the cadence grows.

Dead code
---------

``idex_l2a.remove_signal_noise()``, ``sine_fit()`` and
``butter_lowpass_filter()`` implement a three-stage noise filter (linear
detrend, sine-wave background subtraction at
``TARGET_NOISE_FREQUENCY = 7000``, then a second-order Butterworth low-pass at
``TARGET_HIGH_FREQUENCY_CUTOFF = 100``). **Nothing in the pipeline calls them.**
``estimate_dust_mass()`` has a ``remove_noise: bool = False`` parameter that
only logs "remove_noise is ignored for this fit path" when set - the fit always
runs on the raw low-rate waveform.

They are exercised by ``test_idex_l2a.py`` and are presumably retained for a
future filtered-fit path. Do not delete them without asking the IDEX team, and
do not assume the fit is filtered when reading the L2A output.

Not implemented at all
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Status
   * - Combined target-channel fit
     - **[DOC]** 4.7.8. Not started. The saturation flags it needs already
       exist.
   * - Combined TOF-channel fit
     - **[DOC]** 4.7.10. Not started. Would also supply the fit-validity flag
       both future-work sections promise.
   * - Dead-time correction to rates
     - ``dead_time`` is computed at L1B and never consumed. L2B corrects for
       acquisition uptime only.
   * - Pulser-based conversion monitoring
     - **[DOC]** 4.8 describes periodic **manual** review of pulser
       injections. The pipeline flags pulser events but derives nothing from
       them. Arguably correct as-is.
   * - Decontamination-cycle handling
     - **[DOC]** chapter 2 describes monthly 8-hour 120 °C bakeouts. Visible
       only as event-message log entries; nothing models or flags them.
   * - CRC / checksum verification
     - The science XTCE defines ``CHECKSUM``. Nothing verifies it.
   * - ``l2c`` CLI branch
     - ``"l2c"`` is in ``PROCESSING_LEVELS`` but ``Idex.do_processing`` has no
       branch for it - it raises ``NotImplementedError``. L2C is produced by
       the ``l2b`` job. Harmless, but confusing.

Hard failures in the code
-------------------------

Explicit exceptions, so you know what a bad input looks like:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Location
     - Condition
   * - ``cli.py`` ``Idex.do_processing``
     - ``NotImplementedError`` for any level other than l1a/l1b/l2a/l2b.
   * - ``cli.py`` ``Idex.do_processing``
     - ``ValueError`` if dependency counts are wrong: >2 (l1a), ≠3 for
       ``sci-10days`` or ≠1 otherwise (l1b), ≠3 (l2a), <3 or >4 (l2b).
   * - ``cli.py`` ``Idex.do_processing``
     - ``ValueError`` if no ``source="idex"`` science file is found for L1B, or
       if none has ``self.start_date`` in its filename.
   * - ``idex_utils.get_10_day_window_end_date``
     - ``ValueError`` if the start date is not a row in
       ``idex_10_day_CDF_names.csv``, or if more than one row matches.
   * - ``idex_l1a._create_science_dataset``
     - ``KeyError`` if a waveform packet arrives before its header packet.
   * - ``idex_l1b.idex_l1b``
     - ``ValueError`` for any descriptor not starting with ``sci-10days`` or
       ``msg-10days``.
   * - ``idex_l2a._mask_saturated_derived_estimates``
     - ``KeyError`` if any of the three low-rate saturation flags is missing
       from the L2A dataset.

Soft failures worth knowing
---------------------------

These return ``None`` or NaN instead of raising, which means they show up as
missing data rather than a failed job:

* ``idex_l1a()`` skips (with a warning) any product type with no events inside
  the 10-day window, and can return an empty list.
* ``RawDustEvent.process()`` returns ``None`` for conflicting fragments or a
  waveform-length mismatch.
* ``idex_l1b_msg()`` returns ``None`` if no science or pulser transition is
  present - so **no L1B message file is written**, and L2B then has no uptime
  information for that period and falls back to ``rate_calculation_quality_flags
  = 0``.
* ``estimate_dust_mass()`` returns all-NaN on a ``curve_fit`` ``RuntimeError``.
* ``invert_rise_time_to_velocity()`` returns NaN for a non-finite or
  non-positive rise time, or if no root exists in 0.1-100 km/s.
* ``calculate_snr()`` returns all-NaN if the −7 to −5 µs baseline window is
  empty.

Testing
-------

**[DOC]** Section 4.9 sets a project-level requirement of **at least 90 %
automated test coverage** for the IDEX pipeline, consistent with the repo-wide
Codecov patch-coverage gate.

**[CODE]** ``imap_processing/tests/idex/``, ~3000 lines across six test modules
plus a 216-line ``conftest.py``.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Module
     - Covers
   * - ``test_idex_l0.py``
     - Decommutation, packet counts.
   * - ``test_idex_l1a.py``
     - Event assembly, duplicate and conflicting fragments, incomplete events,
       compressed vs uncompressed decode consistency, event counts, comparison
       against the IDEX team's validation HDF5.
   * - ``test_idex_l1b.py``
     - EU conversion, waveform units, setting unpacking, trigger decode, dead
       time, SPICE attachment (mocked), message-state reduction, CDF writing.
   * - ``test_idex_l2a.py``
     - Calibration loading, time-to-mass, SNR, peak fits, smooth power law,
       rise-time inversion, waveform fits, mass estimation, and explicit
       assertions that the NaN block is filled.
   * - ``test_idex_l2b.py``
     - Counts, rates, uptime percentage, spin binning, map binning, fill block.
   * - ``test_idex_event_flags.py``
     - Classification branches and saturation thresholds, synthetic waveforms
       only.

Fixtures worth knowing about (``conftest.py``):

* Three L0 ``.pkts`` files in ``test_data/`` - one science (2023-12-18), one
  event-message (2025-01-08), one catalog-list (2024-12-06) - plus a matched
  compressed / non-compressed pair from 2023 day 102.
* ``l1a_example_data`` and ``l1b_example_data`` load **IDEX-team-produced HDF5
  validation files** via ``xr.open_datatree``, one group per event.
* ``ancillary_files`` points at the two calibration CSVs kept in ``test_data/``.
* ``get_spice_data`` is **mocked wholesale** - ones for ephemeris, uniform
  random for spin phase, longitude and latitude.

.. important::

   **Some IDEX tests need the SDC test-data download.** The repository's default
   selection is ``-m "not external_kernel and not external_test_data"``. Seven
   IDEX tests carry an explicit ``@pytest.mark.external_test_data`` decorator -
   and they are the ones that compare against the IDEX team's validation HDF5
   files, i.e. the only tests that check the numbers rather than the plumbing.
   To run them::

       poetry run pytest imap_processing/tests/idex -vvv -m "external_test_data"

   ``imap_processing/tests/idex/conftest.py`` additionally sets
   ``pytestmark = pytest.mark.external_test_data`` at module scope. It is the
   only conftest in the repository that does so, and whether pytest propagates a
   conftest-level ``pytestmark`` to collected tests is version-dependent - so
   the effective count of skipped-by-default IDEX tests may be 7 or may be all
   100. Check with ``pytest --collect-only -m external_test_data`` before
   relying on a green local run. Either way, the ``l1a_example_data`` and
   ``l1b_example_data`` fixtures request ``_download_test_data`` directly, so
   they attempt a network fetch regardless of markers.

   Separately: **no IDEX test exercises real SPICE geometry.**
   ``test_get_spice_data`` mocks the SPICE functions and uses a fake spin table
   (the ``furnish_kernels`` call is commented out), so it verifies array shapes
   and names but not values; the ``l1b_dataset`` fixture mocks
   ``get_spice_data`` wholesale. Ephemeris, boresight pointing, solar longitude
   and spin phase are structurally tested and numerically untested.
