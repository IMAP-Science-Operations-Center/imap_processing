.. _glows-ancillary:

Ancillary and Settings Files
============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

GLOWS is unusually dependent on instrument-team-supplied files. Almost every
decision the pipeline makes about *which data to trust* comes out of one of
these, and none of the science-relevant numbers are hard-coded in the repository.

**[DOC §3.12]** defines nine files. Seven are used at or below L2; two are
L3-only and are listed here only so you recognise them.

The canonical list
------------------

Names follow ``imap_glows_<descriptor>_YYYYMMDD_vXXX.<ext>``.

.. list-table::
   :header-rows: 1
   :widths: 5 34 10 51

   * - #
     - File / descriptor
     - Level
     - Purpose
   * - 1
     - ``l1b-map-of-excluded-regions`` (``.dat``)
     - L1B
     - Points in the sky (ecliptic J2000) that densely cover regions to be
       excluded - principally the neighbourhood of the galactic plane. Drives
       ``is_inside_excluded_region``.
   * - 2
     - ``l1b-conversion-table-for-anc-data`` (``.json``)
     - L1B
     - ``min``/``max``/``n_bits`` (and unused ``p01``-``p04``) for decoding
       filter temperature, HV voltage, spin period, spin phase and pulse length
       from integers to physical units.
   * - 3
     - ``l1b-map-of-uv-sources`` (``.dat``)
     - L1B
     - Catalogue of bright UV point sources with a **per-source masking
       radius**. Drives ``is_close_to_uv_source``.
   * - 4
     - ``l1b-exclusions-by-instr-team`` (``.dat``)
     - L1B
     - ``unique_block_identifier`` + a 0/1 mask string per block, for anything
       the GLOWS team wants excluded by hand. Drives
       ``is_excluded_by_instr_team``.
   * - 5
     - ``l1b-suspected-transients`` (``.dat``)
     - L1B
     - Same format; for bins likely to carry transient signal (comets, etc.).
       Drives ``is_suspected_transient``.
   * - 6
     - ``l2-calibration`` (``.dat``)
     - L2
     - ``start_time_UTC cps_per_R``. Each value applies from its start time
       until the next; the last row is open ended.
   * - 7
     - ``l3a-map-of-extra-helio-bckgrd`` (``.dat``)
     - L3A
     - All-sky HEALPix map of the time-independent extra-heliospheric
       background. **Not used in this repository.**
   * - 8
     - ``l3a-time-dep-bckgrd`` (``.dat``)
     - L3A
     - Per-pointing background corrections on a 3600-bin grid, to be linearly
       interpolated onto the L3A grid. **Not used in this repository.**
   * - 9
     - ``pipeline-settings`` (``.json``)
     - L1B, L2
     - Thresholds, the two active-flag masks, day/night offsets, the number of
       L3A bins. The control panel for the whole pipeline.

Document §4.15 additionally lists L3A-to-L3E inputs -
``imap_glows_bad_days_list``, ``imap_glows_WawHelioIonMP``,
``imap_glows_uv-anisotropy-*``, ``imap_glows_p-density-*``,
``imap_glows_sw-speed-*``, ``imap_glows_lya-*``, ``imap_glows_phion-*``,
``imap_glows_e-density-*``. None of these are relevant here.

How they are loaded
-------------------

**[CODE]** ``GlowsAncillaryCombiner`` in
``imap_processing/ancillary/ancillary_dataset_combiner.py``, a subclass of the
generic ``AncillaryCombiner``. The CLI constructs one per descriptor with a
3-day end-date buffer, and passes ``.combined_dataset`` into the processing
function. The combiner handles time-ranged, versioned ancillary files: multiple
deliveries are merged into one dataset with an ``epoch`` dimension representing
validity, and the consumer selects with ``.sel(epoch=day, method="nearest")``.

``convert_file_to_dataset`` dispatches on the **filename substring**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Substring
     - Resulting variables
   * - ``excluded-regions``
     - ``ecliptic_longitude_deg``, ``ecliptic_latitude_deg`` on dim ``region``.
       Handles the empty-file case explicitly.
   * - ``uv-sources``
     - ``object_name``, ``ecliptic_longitude_deg``, ``ecliptic_latitude_deg``,
       ``angular_radius_for_masking`` on dim ``source``.
   * - ``suspected-transients``
     - ``l1b_unique_block_identifier``, ``histogram_mask_array`` on dim
       ``time_block``.
   * - ``exclusions-by-instr-team``
     - Identical to the above.
   * - ``l2-calibration``
     - ``start_time_utc``, ``cps_per_r`` on dim ``time_block``.
   * - ``*.json``
     - Generic ``convert_json_to_dataset``, which **flattens** nested JSON into
       dotted-ish variable names.
   * - anything else
     - ``ValueError: Unknown GLOWS ancillary file type``.

.. warning::

   Dispatch is by substring on the file name. Rename an ancillary file and the
   pipeline will raise ``ValueError`` rather than mis-parse - but a name that
   happens to contain two of these substrings would match whichever branch comes
   first.

The conversion table is the one exception: it is **not** run through the
combiner. The CLI opens it with ``json.load`` and passes the raw ``dict`` to
``AncillaryParameters``.

Bundled example copies
----------------------

**[CODE]** ``imap_processing/glows/ancillary/`` ships example copies of the
instrument-team files. These are for development and tests; in production the
SDC supplies them through ``ProcessingInputCollection``.

.. list-table::
   :header-rows: 1
   :widths: 56 44

   * - File
     - Notes
   * - ``imap_glows_pipeline-settings_20250923_v002.json``
     - Version "0.1", created 2023-05-27. **No** ``sunrise_offset``,
       ``sunset_offset`` or ``spin_offset_correction``.
   * - ``imap_glows_pipeline_settings_v001.json``
     - Older, underscore-separated name. Would **not** match the SDC descriptor
       convention.
   * - ``l1b_conversion_table_v001.json``
     - Also non-conforming name (underscores, no ``imap_glows_`` prefix).
   * - ``imap_glows_map-of-uv-sources_20250923_v002.dat``
     - Version 0.2. Header records
       ``star_background_max: 2 cts/s, min_masking_angle: 0.5 deg``.
   * - ``imap_glows_map-of-excluded-regions_20250923_v002.dat``
     - Version 0.2, generated by the team's
       ``generate_map_of_excluded_regions.py``.
   * - ``imap_glows_exclusions-by-instr-team_20250923_v002.dat``
     - Header only - no exclusions defined yet.
   * - ``imap_glows_suspected-transients_20250923_v002.dat``
     - Header only.

File formats
------------

``map-of-uv-sources``
^^^^^^^^^^^^^^^^^^^^^

Whitespace-separated, ``#``-comment header, four columns:

.. code-block:: text

   # columns: object_name, ecliptic_longitude_deg, ecliptic_latitude_deg, angular_radius_for_masking
   **HDO221A  220.5517651786544  -45.67397641472249  1.6606970217308294
   HD100600   167.4557237863621   12.89523400723998  1.6077588256414592
   HD10144    345.2999797928628  -59.3822555493876   3.949782859665976
   HD10205     38.90283578666903  27.9330576386139   0.5

**[DOC §12.7.6]** The radius depends on the source's brightness: up to ~4° for
the brightest, with a floor of 0.2-0.6° (comfortably larger than the 3σ
nutation). The catalogue is expected to hold "probably hundreds" of sources and
to be updated during the mission. Read with ``np.loadtxt(dtype=str)``, so the
name column must not contain spaces.

``map-of-excluded-regions``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two columns, longitude and latitude in ecliptic J2000 degrees. **No radius
column** - the region is defined by dense point coverage.

.. code-block:: text

   # columns: ecliptic_longitude_deg, ecliptic_latitude_deg
   8.437499999999998579e+01 2.074237995448714500e+01
   8.718749999999998579e+01 2.074237995448714500e+01

``exclusions-by-instr-team`` / ``suspected-transients``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   # Value 1 in mask_array means that a given bin is to be excluded
   # columns: l1b_unique_block_identifier, histogram_mask_array
   2026-04-15T03:24:36 0000000011110000....0000

Split on the **first space**: everything before is the block identifier,
everything after is the mask string of ``n_bin`` characters. Blocks not listed
default to all zeros.

``l2-calibration``
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   # columns: start_time_UTC, cps_per_R
   2025-09-01T00:00:00 3.37

Same first-space split; the second field is parsed as ``float``.

``pipeline-settings``
^^^^^^^^^^^^^^^^^^^^^

The bundled file, annotated with what the code does with each key:

.. list-table::
   :header-rows: 1
   :widths: 46 14 40

   * - Key
     - Bundled value
     - Used?
   * - ``filter_based_on_daily_statistical_error.n_sigma_threshold_lower`` /
       ``_upper``
     - 3.0 / 3.0
     - **No.** ``is_beyond_daily_statistical_error`` is a stub.
   * - ``filter_based_on_comparison_of_spin_periods.relative_difference_threshold``
     - 7.0e-5
     - **No.** ``is_spin_period_difference_beyond_threshold`` is a stub.
   * - ``filter_based_on_temperature_std_dev.std_dev_threshold__celsius_deg``
     - 2.03
     - Yes - flag 12.
   * - ``filter_based_on_hv_voltage_std_dev.std_dev_threshold__volt``
     - 50.0
     - Yes - flag 13.
   * - ``filter_based_on_spin_period_std_dev.std_dev_threshold__sec``
     - 0.033333
     - Yes - flag 14.
   * - ``filter_based_on_pulse_length_std_dev.std_dev_threshold__usec``
     - 1.0
     - Yes - flag 15.
   * - ``filter_based_on_maps.angular_radius_for_excl_regions__deg``
     - 2.0
     - **No.** L1B hard-codes 0.05°.
   * - ``active_bad_time_flags`` (17 booleans)
     - see below
     - Yes - the L2 good-time mask.
   * - ``active_bad_angle_flags`` (4 booleans)
     - all ``true``
     - Parsed into ``PipelineSettings`` but **never applied**.
   * - ``number_of_good_histograms_at_night``
     - 3
     - **No.**
   * - ``l3a_nominal_number_of_bins``
     - 90
     - **No** - L3A is not produced here.
   * - ``sunrise_offset`` / ``sunset_offset``
     - *absent*
     - Default 0.0 when absent. Used by ``apply_is_night_offsets``.
   * - ``spin_offset_correction``
     - *absent*
     - Default 0.0 when absent. Added to the position-angle offset at both L1B
       and L2.

``active_bad_time_flags`` in the bundled file has ``is_night: false`` and
``is_spin_period_difference_beyond_threshold: false``, all others ``true``. The
validation file used by the tests differs: ``is_overexposed``,
``is_hv_test_in_progress`` and ``is_beyond_daily_statistical_error`` are
``false``, ``is_night`` is ``true``, and both offsets plus
``spin_offset_correction: 1.047`` are present.

.. important::

   **The pipeline settings file materially changes which data survives to L2.**
   When investigating "why did this pointing produce no output", check which
   settings file version was served **before** looking at the code.

``PipelineSettings`` parsing
----------------------------

**[CODE]** ``PipelineSettings`` in ``glows/l1b/glows_l1b_data.py`` handles two
shapes, because ``convert_json_to_dataset`` may or may not have flattened the
nested objects:

* ``active_bad_angle_flags`` as a single array variable, **or** four separate
  ``active_bad_angle_flags_<name>`` variables, **or** absent → default
  ``[True] * 4``.
* ``active_bad_time_flags`` likewise, keyed on ``BAD_TIME_FLAG_NAMES``, → default
  ``[True] * 17``.
* ``sunrise_offset``, ``sunset_offset``, ``spin_offset_correction`` via
  ``pipeline_dataset.get(name, 0.0)``.
* ``processing_thresholds``: every variable whose name contains ``"threshold"``
  or ``"limit"``, looked up later by ``get_threshold(suffix)`` using
  ``str.endswith``.

.. warning::

   Every default is permissive: a settings file that fails to parse, or is
   missing a section, silently produces "all flags active" rather than an error.
   That is safe for the flag masks (more data rejected) but it means a
   mis-delivered file will not announce itself. The one exception is a missing
   *threshold*, which crashes with ``TypeError`` when compared against
   ``None``.

What the SDC provides versus what GLOWS provides
------------------------------------------------

**[DOC §3.2]** splits the inputs three ways:

1. **GLOWS instrument telemetry** - the L0 packets.
2. **Ancillary data from the SDC/POC/MOC** - spin period at high cadence,
   position-angle offset for GLOWS, spin-axis orientation, spacecraft state
   vectors. **[CODE]** All of these arrive through **SPICE kernels and the spin
   table**, not through GLOWS-specific ancillary files. See
   :ref:`glows-l1b`.
3. **Input provided by the GLOWS team** - the nine files above.

The document also notes an IMAP-level dependency that is *not* satisfied by
anything here: a **thruster operation flag**, so GLOWS can know whether the
thrusters fired during a given second. Document §12.1.2 item 4 lists it as TBD.

Finally, §3.14 item 4 notes that data from **SWE** (electron fluxes) and **HIT**
(high-energy cosmic rays) would be useful for identifying periods of elevated
particle background - but explicitly states that other-instrument data are **not
anticipated in the L0-to-L2 pipeline**. They belong at L3. Do not add them here.
