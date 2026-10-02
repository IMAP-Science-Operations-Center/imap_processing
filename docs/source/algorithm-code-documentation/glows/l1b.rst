.. _glows-l1b:

Level 1B - Physical Units, Flags and Geometry
=============================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Goal:** decode onboard integer encodings into physical units, attach
everything the spacecraft knows (position, velocity, spin axis, spin period),
and compute the two flag systems that decide what L2 is allowed to use.

Nothing is calibrated here. **The histogram counts pass through untouched.**

Entry points in ``glows/l1b/glows_l1b.py``:

* ``glows_l1b(input_dataset, excluded_regions, uv_sources, suspected_transients,
  exclusions_by_instr_team, pipeline_settings_dataset, conversion_table_dict)``
  → one histogram ``xr.Dataset``
* ``glows_l1b_de(input_dataset, conversion_table_dict)`` → one direct-event
  ``xr.Dataset``

Both are driven by ``xr.apply_ufunc(..., vectorize=True)``, which constructs one
``HistogramL1B`` / ``DirectEventL1B`` dataclass **per epoch** and unpacks its
fields back into aligned ``DataArray``\ s. The ordering of the dataclass fields
is therefore load-bearing:

.. warning::

   ``HistogramL1B``'s docstring says it outright: *"IMPORTANT: The order of the
   fields inherited from L1A must match the order of the fields in the DataSet
   created in decom_glows.py."* Reordering a field, or inserting one in the
   middle, silently mis-assigns every subsequent variable. Add new fields at the
   end and check ``output_dimension_mapping`` in ``glows_l1b.py``.

Integer decoding
----------------

**[DOC §11.4]** Every averaged ancillary quantity is encoded onboard into
``n_T`` bits over a fixed range:

.. math::

   \tau = \left\lfloor \frac{T - T_{min}}{T_{max} - T_{min}}(2^{n_T} - 1)
   \right\rfloor = A\,T + B

.. math::

   A = \frac{2^{n_T} - 1}{T_{max} - T_{min}}, \qquad B = -T_{min} A

Decoding (Eq. 39/40):

.. math::

   T_d = (\tau - B) / A

Spreads are downlinked as the **variance**, encoded with the same ``A`` but over
``2 n_T`` bits, because ``⟨τ²⟩ - ⟨τ⟩² = A²(⟨T²⟩ - ⟨T⟩²)`` (Eq. 43). So (Eq. 44):

.. math::

   \Delta T_d^2 = \Delta\tau^2 / A^2, \qquad \sigma = \sqrt{\Delta T_d^2}

.. important::

   **At L0/L1A the spread measure is a variance. At L1B and above it is a
   standard deviation.** The variable names change accordingly
   (``*_variance`` → ``*_std_dev``).

**[CODE]** ``AncillaryParameters`` in ``glows_l1b_data.py`` implements exactly
this:

.. code-block:: python

   def decode(self, param_key, encoded_value):
       params  = getattr(self, param_key)
       param_a = (2 ** params["n_bits"] - 1) / (params["max"] - params["min"])
       param_b = -params["min"] * param_a
       return np.double((encoded_value - param_b) / param_a)

   def decode_std_dev(self, param_key, encoded_value):
       params  = getattr(self, param_key)
       param_a = (2 ** params["n_bits"] - 1) / (params["max"] - params["min"])
       return np.double(np.sqrt(encoded_value / (param_a ** 2)))

The parameters come from the ``l1b-conversion-table-for-anc-data`` JSON. The
bundled copy (``glows/ancillary/l1b_conversion_table_v001.json``):

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 10 40

   * - Parameter
     - min
     - max
     - n_bits
     - Physical unit
   * - ``filter_temperature``
     - -30.0
     - 80.0
     - 8
     - °C
   * - ``hv_voltage``
     - 0.0
     - 3500.0
     - 12
     - V
   * - ``spin_period``
     - 0.0
     - 20.9712
     - 16
     - s
   * - ``spin_phase``
     - 0.0
     - 360.0
     - 16
     - deg
   * - ``pulse_length``
     - 0.0
     - 255.0
     - 8
     - µs

``AncillaryParameters.__init__`` validates the key sets and raises ``KeyError``
if the file does not conform. ``filter_temperature``, ``hv_voltage`` and
``pulse_length`` may additionally carry ``p01``-``p04`` polynomial coefficients;
they are all ``0.0`` in the bundled file and **the code never uses them**.

.. note::

   **[DOC Table 11.1 note 1]** The filter temperature relation is expected to be
   *nonlinear* in reality (12-bit ADC values squeezed into 8 bits), and may
   eventually need a lookup table. The ``p01``-``p04`` slots are presumably where
   that would live. Also note the document quotes ``hv_voltage`` max as
   ``56012.82 V`` for a 16-bit field, while the delivered table uses 3500 V with
   12 bits - **use the delivered table**, which is what the code does.

Histogram L1B
-------------

Processing steps
^^^^^^^^^^^^^^^^

**[DOC §3.7.1]** lists eight steps. **[CODE]** ``HistogramL1B.__post_init__``:

.. list-table::
   :header-rows: 1
   :widths: 6 46 48

   * - #
     - Document step
     - Code
   * - 1
     - Times from int sec/subsec to float
     - Already floats out of L1A; passed straight through.
   * - 2
     - Decode onboard flag word to a readable structure
     - ``deserialize_flags`` → 10 booleans, then inverted (see below).
   * - 3
     - Copy ``is_generated_on_ground``
     - Copied and inverted into flag slot 11.
   * - 4
     - Bad-time masking - ground-computed flags
     - ``compute_flags`` adds 7 more flags.
   * - 5
     - Bad-angle masking - per-bin flag array
     - ``_compute_histogram_flag_array`` → shape ``(4, n_bin)``.
   * - 6
     - Decode ancillary values and spreads
     - ``AncillaryParameters.decode`` / ``decode_std_dev``.
   * - 7
     - Add SDC-supplied ancillary (spin period, spin axis, ephemeris,
       position-angle offset)
     - ``update_spice_parameters``.
   * - 8
     - Generate the IMAP spin-angle grid for bin centres
     - ``imap_spin_angle_bin_cntr``.

Bin centres
^^^^^^^^^^^

**[DOC §3.2]** ``ψ_i = (360°/n_bin)(i - 1/2) = 0.1°(i - 0.5)``, so the **left
edge** of bin 1 is at ψ = 0, not its centre.

**[CODE]**

.. code-block:: python

   n_bins = len(self.histogram)
   phi = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
   self.imap_spin_angle_bin_cntr = phi * 360.0

Same thing, zero-indexed. Note that ``n_bins`` comes from the **length of the
histogram array**, which after L1A is always 3600 including fill bins -
``number_of_bins_per_histogram`` is carried separately.

Unique block identifier
^^^^^^^^^^^^^^^^^^^^^^^

**[DOC Table 3.8 item 2]** ``YYYY-MM-DDThh:mm:ss`` from the IMAP UTC time. This
string is the join key for two of the ancillary mask files.

**[CODE]**

.. code-block:: python

   datetime64_time = met_to_datetime64(self.imap_start_time)
   self.unique_block_identifier = np.datetime_as_string(datetime64_time, "s")

The 17 bad-time flags
^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``BAD_TIME_FLAG_NAMES`` in ``glows/__init__.py`` fixes both the names
and the array order. ``FLAG_LENGTH = 17``.

.. list-table::
   :header-rows: 1
   :widths: 5 44 12 39

   * - Idx
     - Name
     - Source
     - How it is set **[CODE]**
   * - 0
     - ``is_pps_missing``
     - onboard
     - Bit 0 of ``flags_set_onboard``.
   * - 1
     - ``is_time_status_missing``
     - onboard
     - Bit 1.
   * - 2
     - ``is_phase_missing``
     - onboard
     - Bit 2.
   * - 3
     - ``is_spin_period_missing``
     - onboard
     - Bit 3.
   * - 4
     - ``is_overexposed``
     - onboard
     - Bit 4. At least one bin overexposed.
   * - 5
     - ``is_direct_event_non_monotonic``
     - onboard
     - Bit 5.
   * - 6
     - ``is_night``
     - onboard
     - Bit 6. **Index referenced by
       ``GlowsConstants.IS_NIGHT_FLAG_IDX = 6``.**
   * - 7
     - ``is_hv_test_in_progress``
     - onboard
     - Bit 7. Monthly gain test.
   * - 8
     - ``is_test_pulse_in_progress``
     - onboard
     - Bit 8.
   * - 9
     - ``is_memory_error_detected``
     - onboard
     - Bit 9.
   * - 10
     - ``is_generated_on_ground``
     - ground
     - ``1 - is_generated_on_ground`` from L1A. Always 1 today.
   * - 11
     - ``is_beyond_daily_statistical_error``
     - ground
     - **Hard-coded to 1 (good).** Placeholder.
   * - 12
     - ``is_temperature_std_dev_beyond_threshold``
     - ground
     - ``filter_temperature_std_dev <= std_dev_threshold__celsius_deg``.
   * - 13
     - ``is_hv_voltage_std_dev_beyond_threshold``
     - ground
     - ``hv_voltage_std_dev <= std_dev_threshold__volt``.
   * - 14
     - ``is_spin_period_std_dev_beyond_threshold``
     - ground
     - ``spin_period_std_dev <= std_dev_threshold__sec``.
   * - 15
     - ``is_pulse_length_std_dev_beyond_threshold``
     - ground
     - ``pulse_length_std_dev <= std_dev_threshold__usec``.
   * - 16
     - ``is_spin_period_difference_beyond_threshold``
     - ground
     - **Hard-coded to 1 (good).** The code comment calls this slot
       ``is_beyond_background_error``, which is a *different* condition from the
       document's flag 30.17.

.. danger::

   **Polarity is inverted relative to the document.**

   **[DOC Table 3.10]**: *"GLOWS uses a convention for bad-time flags, where
   false value corresponds to normal conditions and true value indicates a
   problem."*

   **[CODE]** ``compute_flags`` returns ``1 = good, 0 = bad``:

   .. code-block:: python

      onboard_flags = (1 - self.deserialize_flags(int(self.flags_set_onboard))).astype(np.uint8)
      ...
      is_temp_ok = np.uint8(self.filter_temperature_std_dev <= temp_threshold)

   and L2 selects blocks where **all active flags equal 1**. When comparing
   against the GLOWS team's JSON validation output, or reading the document,
   remember to invert. This is the most common source of confusion in GLOWS
   code review.

Threshold lookup
^^^^^^^^^^^^^^^^

``PipelineSettings.processing_thresholds`` collects every pipeline-settings
variable whose name contains ``"threshold"`` or ``"limit"``, and
``get_threshold(suffix)`` returns the first whose key **ends with** the given
suffix. The suffixes used are ``std_dev_threshold__celsius_deg``,
``std_dev_threshold__volt``, ``std_dev_threshold__sec`` and
``std_dev_threshold__usec``. This suffix matching exists because
``convert_json_to_dataset`` flattens nested JSON into names like
``filter_based_on_temperature_std_dev_std_dev_threshold__celsius_deg``.

.. warning::

   ``get_threshold`` returns ``None`` when nothing matches, and the comparison
   ``value <= None`` then raises ``TypeError``. A pipeline-settings file missing
   a threshold key will crash L1B rather than degrade gracefully.

The 4 bad-angle flags
^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``GLOWSL1bFlags`` in ``imap_processing/quality_flags.py``:

.. code-block:: python

   IS_CLOSE_TO_UV_SOURCE     = 2**0
   IS_INSIDE_EXCLUDED_REGION = 2**1
   IS_EXCLUDED_BY_INSTR_TEAM = 2**2
   IS_SUSPECTED_TRANSIENT    = 2**3

``_compute_histogram_flag_array`` returns a ``(4, n_bin)`` ``uint8`` array. Row
``k`` holds the bit value ``2**k`` where set, ``0`` elsewhere - so each row is
effectively a scaled boolean, and the four rows can be OR-ed into a single
bitmask (which is what L2 does).

Sky masking geometry
""""""""""""""""""""

**[DOC §3.7.1, §12.7.6]** For each bin, compute where on the sky the boresight
was pointing, and compare to the UV-source catalogue and the excluded-region
point set. The GLOWS team's own implementation uses
``astropy.coordinates.search_around_sky`` to compare 3600 bin positions against
thousands of mask points quickly.

**[CODE]** ``flag_uv_and_excluded`` does it with dot products instead:

.. code-block:: python

   # 1. Bin look directions in the despun frame
   azimuth = (imap_spin_angle_bin_cntr - position_angle_offset_average + 360.0) % 360.0
   elevation = get_instrument_mounting_az_el(SpiceFrame.IMAP_GLOWS)[1]
   look_vecs_dps = spherical_to_cartesian(
       np.stack([np.ones_like(azimuth), azimuth,
                 np.full_like(azimuth, elevation)], axis=-1))

   # 2. Rotate to the ecliptic frame at the block start time
   look_vecs_ecl = frame_transform(
       data_start_time_et, look_vecs_dps,
       SpiceFrame.IMAP_DPS, SpiceFrame.ECLIPJ2000,
       allow_spice_noframeconnect=True)

   # 3. cos(separation) = dot product of unit vectors
   uv_cos_sep = look_vecs_ecl @ uv_vecs.T                  # (nbin, n_src)
   close_to_uv_source = np.any(uv_cos_sep >= np.cos(uv_radius)[None, :], axis=1)

   region_cos_sep = look_vecs_ecl @ region_vecs.T          # (nbin, n_region)
   half_bin_rad = np.deg2rad(0.1 / 2)
   inside_excluded_region = np.any(region_cos_sep >= np.cos(half_bin_rad), axis=1)

Points worth noting:

* Each UV source carries **its own masking radius**, from the catalogue's fourth
  column. Bright sources get up to ~4°; the floor is 0.2-0.6° (larger than the
  3σ nutation).
* Excluded regions have **no per-point radius**. The code uses a hard-coded
  half-bin-width of **0.05°**, on the reasoning that the region point set
  densely covers the area. **[DOC §12.7.6]** expects
  ``angular_radius_for_excl_regions__deg`` from pipeline settings, "probably set
  to half the nominal radius of the GLOWS FOV". The bundled settings file
  supplies ``2.0``; the validation settings file supplies ``0.5``. **The code
  ignores both.** See :ref:`glows-implementation-status`.
* ``allow_spice_noframeconnect=True`` is deliberate: the DPS CK intentionally
  excludes the ~2 minute repointing transition, and a histogram block can land
  in that gap.
* The transform uses the **block start time only**, not a time range - a block
  spans ~2 minutes so this is a small approximation.

Instrument-team masks
"""""""""""""""""""""

``flag_from_mask_dataset`` looks up ``unique_block_identifier`` in the
``l1b-exclusions-by-instr-team`` / ``l1b-suspected-transients`` dataset and
parses the matching ``"0"``/``"1"`` character string into a boolean array. **No
match → all zeros**, which is the document's specified default.

What SPICE contributes
^^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``HistogramL1B.update_spice_parameters``. The time range used is
``np.arange(start_et, end_et)`` - i.e. **one sample per second** across the
block.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Output
     - How
   * - ``spin_period_ground_average``, ``spin_period_ground_std_dev``
     - ``get_spin_data()``, filtered to
       ``data_start_met <= spin_start_met <= data_end_met``, then
       ``np.average`` / ``np.std`` of ``spin_period_sec``.
   * - ``position_angle_offset_average``
     - ``360 - get_spin_angle(get_instrument_spin_phase(imap_start_time,
       instrument=SpiceFrame.IMAP_GLOWS), degrees=True)`` plus
       ``spin_offset_correction`` from pipeline settings.
   * - ``position_angle_offset_std_dev``
     - Hard-coded ``0.0`` per document §10.6.
   * - ``spin_axis_orientation_average`` / ``_std_dev``
     - Transform ``[0, 0, 1]`` from ``IMAP_SPACECRAFT`` to ``ECLIPJ2000`` at
       every second, convert to spherical, then **circular** mean/std
       (``scipy.stats.circmean`` / ``circstd``) for both longitude and latitude.
       Output in degrees, ``[lon, lat]``.
   * - ``spacecraft_location_average`` / ``_std_dev``
     - ``geometry.imap_state(et=time_range, ref_frame=ECLIPJ2000,
       observer=SpiceBody.SUN)`` columns 0-2. km.
   * - ``spacecraft_velocity_average`` / ``_std_dev``
     - Same call, columns 3-5. km/s.

**[DOC §12.6.1]** asks for exactly these ten quantities. The document also notes
that L1B carries only averages and spreads, and muses that a companion product
with the **full time series** might be useful. That does not exist.

.. note::

   **Two spin periods coexist from L1B onward.** ``spin_period_average`` is what
   the instrument used onboard for histogramming (from the star tracker, via the
   1 PPS messages). ``spin_period_ground_average`` is recomputed on the ground
   from the SDC spin table, which is more accurate because it can assume
   constancy over long timescales. **[DOC Table 3.13 footnote]** L2 is expected
   to report the **onboard** value provided the two agree to better than about
   ``(0.2/360)·⟨P⟩`` - and flag 16 exists to detect when they do not.

Direct event L1B
----------------

**[DOC §3.7.2]** Four steps: convert times to floats, group the onboard flags
into one substructure, decode the encoded ancillary values, and split the L1A
direct-event array into times and pulse lengths.

**[CODE]** ``DirectEventL1B``:

.. code-block:: python

   self.direct_event_glows_times, self.direct_event_pulse_lengths = \
       self.process_direct_events(direct_events)          # de[0],de[1] -> secs; de[2]

   self.glows_time_last_pps = TimeTuple(int(self.glows_time_last_pps),
                                        glows_ssclk_last_pps).to_seconds()

   self.filter_temperature      = anc.decode("filter_temperature", ...)
   self.hv_voltage              = anc.decode("hv_voltage", ...)
   self.spin_period             = anc.decode("spin_period", ...)
   self.spin_phase_at_next_pps  = anc.decode("spin_phase", ...)

   self.de_flags = np.array([catbed_heater_active, spin_period_valid,
                             spin_phase_at_next_pps_valid, spin_period_source,
                             glows_time_on_pps_valid, time_status_valid,
                             housekeeping_valid, is_pps_autogenerated,
                             hv_test_in_progress, pulse_test_in_progress,
                             memory_error_detected])

The 11 DE flags match **[DOC Table 3.12 item 11]** in name and order. They are
copied through **without polarity inversion**, unlike the histogram flags.

Two things the document asks for that are **not** produced:

* ``unique_identifier`` (Table 3.12 item 2) - the code has it commented out with
  a note that strings cannot live in the data section and should be an
  attribute.
* ``direct_event_pulse_lengths`` in **µs** - the code copies the raw encoded
  value (``de[2]``) with no conversion. The document says µs (and marks it TBC).

The ``multi_event`` element of the L1A direct-event tuple is dropped; a ``TODO``
asks where it should go.

L1B output datasets
-------------------

``imap_glows_l1b_hist``
^^^^^^^^^^^^^^^^^^^^^^^

Before processing, histograms with ``imap_start_time == 0.0`` are dropped with a
warning (a second line of defence after the L1A filter).

Coordinates created by ``create_l1b_hist_output``: ``epoch``, ``bins``,
``bins_label``, ``bad_angle_flags`` (0-3), ``bad_time_flags`` (0-16),
``ecliptic`` (0-2), ``latitudinal`` (0-1).

Variables, in the order of the ``HistogramL1B`` dataclass fields:

.. list-table::
   :header-rows: 1
   :widths: 40 24 36

   * - Variable
     - Dims
     - Notes
   * - ``histogram``
     - ``(epoch, bins)``
     - Raw counts, unchanged from L1A, fill 65535.
   * - ``seq_count_in_pkts_file``, ``first_spin_id``, ``last_spin_id``
     - ``(epoch,)``
     - Passthrough.
   * - ``flags_set_onboard``
     - ``(epoch,)``
     - The raw 16-bit word, still carried. A ``TODO`` says it should be
       renamed at L1B.
   * - ``is_generated_on_ground``
     - ``(epoch,)``
     - Passthrough.
   * - ``number_of_spins_per_block``, ``number_of_bins_per_histogram``,
       ``number_of_events``
     - ``(epoch,)``
     - Passthrough.
   * - ``filter_temperature_average`` / ``_std_dev``
     - ``(epoch,)``
     - °C.
   * - ``hv_voltage_average`` / ``_std_dev``
     - ``(epoch,)``
     - V.
   * - ``spin_period_average`` / ``_std_dev``
     - ``(epoch,)``
     - s, onboard value.
   * - ``pulse_length_average`` / ``_std_dev``
     - ``(epoch,)``
     - µs.
   * - ``imap_start_time``, ``imap_time_offset``, ``glows_start_time``,
       ``glows_time_offset``
     - ``(epoch,)``
     - Floats.
   * - ``unique_block_identifier``
     - ``(epoch,)``
     - ISO-8601 string.
   * - ``imap_spin_angle_bin_cntr``
     - ``(epoch, bins)``
     - ψ in degrees. **Not** ψ\ :sub:`PA`.
   * - ``histogram_flag_array``
     - ``(epoch, bad_angle_flags, bins)``
     - The ``(4, n_bin)`` bad-angle array.
   * - ``spin_period_ground_average`` / ``_std_dev``
     - ``(epoch,)``
     - s, from the SDC spin table.
   * - ``position_angle_offset_average`` / ``_std_dev``
     - ``(epoch,)``
     - deg; the std dev is always 0.
   * - ``spin_axis_orientation_average`` / ``_std_dev``
     - ``(epoch, latitudinal)``
     - ``[lon, lat]`` in degrees.
   * - ``spacecraft_location_average`` / ``_std_dev``
     - ``(epoch, ecliptic)``
     - ``[X, Y, Z]`` km, ecliptic, Sun-centred.
   * - ``spacecraft_velocity_average`` / ``_std_dev``
     - ``(epoch, ecliptic)``
     - ``[Vx, Vy, Vz]`` km/s.
   * - ``flags``
     - ``(epoch, flag_dim)``
     - The 17 bad-time flags, ``1 = good``.

Global attributes: ``flight_software_version`` (from L1A) and
``pkts_file_name`` (the ``.pkts`` entries of the input ``Parents`` attribute).

.. warning::

   ``flags`` is emitted on a dimension named ``flag_dim`` (from
   ``output_dimension_mapping``) while the dataset declares a coordinate named
   ``bad_time_flags``. Both have length 17. This looks like an oversight - the
   flags variable ends up on an unlabelled dimension. Same pattern appears at L2
   with ``latitudinal``. Worth confirming against a written CDF before relying
   on either name.

``imap_glows_l1b_de``
^^^^^^^^^^^^^^^^^^^^^

Coordinates: ``epoch``, ``within_the_second``, ``within_the_second_label``,
``flags`` (0-10, i.e. **11** DE flags - not the 17 histogram flags).

Variables follow the ``DirectEventL1B`` field order:
``seq_count_in_pkts_file``, ``number_of_de_packets``, ``imap_time_last_pps``,
``glows_time_last_pps``, ``imap_time_next_pps``, ``spin_period``,
``spin_phase_at_next_pps``, ``number_of_completed_spins``,
``filter_temperature``, ``hv_voltage``, ``de_flags``,
``direct_event_glows_times``, ``direct_event_pulse_lengths``.

Global attribute: ``missing_packets_sequence``, propagated from L1A.

Testing
-------

* ``test_glows_l1b_data.py`` - ``AncillaryParameters`` validation,
  ``deserialize_flags`` parametrised over flag words, ``PipelineSettings``
  construction from the flattened JSON form, ``get_threshold``, spin-axis
  circular statistics near the 0/360 wrap, and number-for-number comparison
  against ``imap_glows_l1b_hist_full_output.json`` and
  ``imap_glows_l1b_de_output.json`` from the GLOWS team's bundle.
* ``test_glows_l1b.py`` - the ``apply_ufunc`` dimension mapping, the
  zero-start-time filter, the case where the histogram length differs from
  ``NBINS``, and the look-vector azimuth formula. The SPICE-dependent test is
  marked ``@pytest.mark.external_kernel`` and uses the
  ``use_fake_spin_data_for_time`` fixture.
