.. _idex-event-classification:

Event Classification and Saturation Flags
=========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. important::

   **Everything on this page is [CODE].** ``imap_processing/idex/idex_event_flags.py``
   (~480 lines) has no counterpart anywhere in the 1 June 2026 algorithm
   document - not in the L1A section, not in the product description, not in
   the flowchart. It was added after the document snapshot.

   It matters more than its absence from the document suggests: these flags are
   what decide whether an L2A velocity and mass estimate is published at all,
   and they are the sole filter on which events reach the L2B and L2C count
   products.

Why it exists
-------------

IDEX triggers on things that are not dust. The onboard pulser injects known
charges to verify the DN-to-engineering-unit conversions; software and external
triggers capture noise baselines on demand; and a real impact can drive a
high-gain channel past its ADC ceiling, making the fitted charge meaningless.

The flags answer three questions per event:

1. **What kind of event is this?** science, pulser, or noise capture - mutually
   exclusive.
2. **Does the TOF waveform actually look like a dust impact?** the dust-hit
   flag.
3. **Which of the six channels are saturated?** one flag per channel.

The ten flags
-------------

All ten are ``uint8``, event-indexed, valued 0 or 1. They are created at
**L1A** (``RawDustEvent.process()`` calls ``classify_event_flags()``), copied
forward unchanged at **L1B** (``idex_l1b_science()``), and copied forward again
at **L2A**.

``idex_event_flags.EVENT_FLAG_NAMES`` - mutually exclusive event type, plus the
dust-hit qualifier:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Set when
   * - ``noise_capture_flag``
     - No trigger channel is active at all, **or** a software/external trigger
       is present and the only active channel is TOF High.
   * - ``pulser_flag``
     - TOF High is the *only* active channel, its trigger mode is ``1``
       (Threshold), and its trigger threshold is exactly **1000 DN**
       (``_PULSER_THRESHOLD_DN``).
   * - ``science_event_flag``
     - Everything else. This is the default, not a positive test.
   * - ``dust_hit_flag``
     - Only ever set on a science event, and only when the TOF waveform passes
       the peak test described below.

``idex_event_flags.SATURATION_FLAG_NAMES`` - one per waveform channel:

``tof_high_saturation_flag``, ``tof_mid_saturation_flag``,
``tof_low_saturation_flag``, ``target_high_saturation_flag``,
``target_low_saturation_flag``, ``ion_grid_saturation_flag``.

``ALL_FLAG_NAMES`` is the concatenation, and is what the L1B and L2A code
iterate over.

Event type classification
-------------------------

The inputs are raw header telemetry, not waveforms:

.. code-block:: python

   trigger_id  = telemetry["idx__txhdrtrigid"]
   active      = {channel for bit, channel in _TRIGGER_CHANNELS.items()
                  if trigger_id & (1 << bit)}
   # plus any channel whose idx__txhdr{hg,mg,lg}trigmode != 0
   sw_or_ext   = bool(trigger_id & ((1 << 4) | (1 << 5)))
   hg_mode     = telemetry["idx__txhdrhgtrigmode"]
   hg_threshold= (telemetry["idx__txhdrhgtrigctrl1"] >> 22) & 0x3FF

``_TRIGGER_CHANNELS`` maps bits 0-3 to ``"TOF H"``, ``"TOF L"``, ``"TOF M"``,
``"Target H"`` - the same bit assignment L1B uses for ``trigger_origin``, but
here bits 4 and 5 (software and external) are handled separately rather than
becoming channels.

Note that the active-channel set is the **union** of two sources: the trigger-ID
bits, *and* any gain channel with a non-zero trigger mode. An armed-but-not-
firing channel therefore still counts as active, which is what keeps a
genuinely multi-channel event from being mistaken for a pulser event.

The decision, in order:

.. code-block:: text

   if not active  or  (sw_or_ext and active ⊆ {"TOF H"}):
       -> noise_capture
   elif active == {"TOF H"} and hg_mode == 1 and hg_threshold == 1000:
       -> pulser
   else:
       -> science_event

The pulser test is deliberately narrow - it recognises the *specific*
configuration the onboard stimulus sequence uses. A pulser run at any other
threshold would be classified as a science event.

Dust-hit detection
------------------

``_has_dust_hit()``. A science event is only a dust hit if the **raw** TOF
waveform contains at least **two peaks** that each clear **7 baseline sigma**
and have a FWHM of at least **20 ns**.

The parameters:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Constant
     - Value
     - Role
   * - ``_BASELINE_WINDOW_US``
     - 3.0
     - Baseline is estimated from the first 3 µs of the record.
   * - ``_PEAK_THRESHOLD_SIGMA``
     - 7.0
     - Minimum peak height above baseline, in sigma.
   * - ``_MIN_PEAK_WIDTH_US``
     - 0.020
     - Minimum FWHM, i.e. 20 ns.
   * - ``_MIN_PEAK_COUNT``
     - 2
     - Peaks required.
   * - ``_MIN_PEAK_DISTANCE_US``
     - 0.030
     - Minimum separation between peaks, 30 ns, converted to samples via the
       median sample spacing.

Baseline and noise (``_baseline_corrected``) use **robust** statistics, not the
mean and standard deviation:

.. math::

   \mathrm{baseline} = \mathrm{median}(x_{\text{first }3\mu s}),
   \qquad
   \sigma = 1.4826 \times \mathrm{MAD}

with a fallback to ``nanstd`` if the MAD is zero or non-finite. The 1.4826
factor is the standard MAD-to-sigma conversion for Gaussian noise; it is used
because a real TOF record has large outliers by construction and a plain
standard deviation would be dominated by the signal it is trying to detect.

Peak candidates are always found on **TOF High**, because it has the best
sensitivity. But a saturated peak has a flat top and no meaningful FWHM. So
``_saturation_aware_width()`` does the following per peak:

1. If the TOF High sample at the peak is not saturated, measure the FWHM on
   TOF High.
2. Otherwise, walk down the gain stages - **Mid, then Low** - find the sample
   nearest in *time* to the peak, and if that sample is not saturated, measure
   the FWHM on that channel instead.
3. If all three are saturated, return NaN and the peak does not qualify.

``_fwhm()`` finds the half-maximum crossings by walking outward from the peak
and then **linearly interpolating** the crossing time between the two bracketing
samples (``_crossing_time``). It requires both sides to be properly bracketed -
a peak that runs off the end of the record returns NaN rather than a truncated
width.

The result is a detector that finds two distinct ion arrivals - which is what a
mass spectrum looks like and what electronic noise generally does not.

Saturation flags
----------------

``classify_saturation_flags()``. A channel is saturated if **any finite sample**
reaches 95 % of the channel's full scale:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Channels
     - Full scale
     - Threshold at 95 %
   * - The three TOF (10-bit)
     - ``_TOF_MAX_DN`` = 1023
     - 971.85 DN
   * - The three low-rate (12-bit)
     - ``_LOW_RATE_MAX_DN`` = 4095
     - 3890.25 DN

``_SATURATION_FRACTION = 0.95``. The comparison is on the **raw DN** waveform at
L1A, before any engineering-unit conversion - which is the only place the ADC
ceiling is meaningful.

The three low-rate waveforms are optional arguments (defaulting to ``None``) so
that the classification API can be called with TOF waveforms alone; a ``None``
channel gets a flag of 0, not a fill value.

What downstream code does with them
-----------------------------------

This is the part worth memorising, because it is where the flags become
scientifically load-bearing.

**L2A - ``_mask_saturated_derived_estimates()``.** For each of ``target_low``,
``target_high``, ``ion_grid``, if the channel's saturation flag is 1, then
``{channel}_impact_charge``, ``{channel}_velocity_estimate`` and
``{channel}_dust_mass_estimate`` are set to NaN. The **fit parameters and fit
results survive** - they stay available for diagnostics. This function raises
``KeyError`` if any of the three saturation flags is missing from the dataset,
so L2A cannot run on an L1B file produced before the flags existed.

**L2A - ``_mask_non_science_derived_estimates()``.** If
``science_event_flag != 1``, the six velocity and mass estimates are set to NaN.
Again the fits survive: the code comment states that "fits and fitted charges
remain available for diagnostics in all instrument modes." Unlike the saturation
masking, this one is tolerant - a missing ``science_event_flag`` logs a debug
message and skips.

**L2A - ``calculate_ion_grid_velocity_and_mass()``.** Consumes three saturation
flags directly to pick which target channel to use as the denominator of the
charge ratio: Target High if unsaturated, else Target Low if unsaturated, else
give up. Returns ``(NaN, NaN)`` immediately if Ion Grid itself is saturated.

**L2B / L2C - ``_get_dust_hit_indices()``.** Every count in every L2B and L2C
product is filtered to ``dust_hit_flag == 1``:

.. code-block:: python

   if "dust_hit_flag" not in l2a_dataset:
       return np.array([], dtype=int)
   dust_hit = l2a_dataset["dust_hit_flag"].data[current_day_indices] == 1
   return current_day_indices[dust_hit]

.. warning::

   Note the failure mode: if ``dust_hit_flag`` is absent from the L2A input,
   this returns an **empty index array** rather than raising. Every count in
   the monthly product would then be zero, every rate would be zero, and
   ``rate_calculation_quality_flags`` would still read 1 (it only reports
   uptime problems). An L2B product built from pre-flag L2A files would look
   like a month in which IDEX detected no dust at all.

Testing
-------

``imap_processing/tests/idex/test_idex_event_flags.py`` (175 lines) covers the
classification branches and the saturation thresholds with synthetic waveforms,
and does not require external test data.
