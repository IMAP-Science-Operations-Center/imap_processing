.. _mag-l1c:

L1C - Gap Filling from Burst Mode
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/mag/l1c/mag_l1c.py``,
``imap_processing/mag/l1c/interpolation_methods.py``.

**Document:** section 7.3.4.

Why L1C exists
--------------

MAG transmits either normal-mode or burst-mode telemetry, never both. So
whenever the instrument is in burst mode (nominally ~1 hour a day), the
normal-mode stream has a **hole**. The MAG science team wants a **continuous
normal-mode L2 product**, so the ground software synthesises the missing
normal-mode samples by filtering and interpolating burst data.

L1C is therefore **normal mode only**. Burst data is an input, not an output.
The steps below are applied independently to MAGo and MAGi.

Inputs
------

* L1B normal-mode data for one sensor (may be absent).
* L1B burst-mode data for the same sensor (may be absent).
* **[CODE]** Optionally, **the previous day's L1C file** for the same sensor.
* SDC configuration: ``L1C_INTERPOLATION_METHOD``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Available inputs
     - Behaviour **[CODE]**
   * - norm + burst
     - Full ``process_mag_l1c`` path.
   * - norm only
     - ``fill_normal_data`` - pass through, no interpolation.
   * - burst only
     - Full path with a synthetic empty normal timeline. A trailing-edge
       correction shortens the usable burst range by one output cadence, because
       CIC delay compensation eats the end of the filtered series.
   * - neither
     - ``ValueError``.

If a second dataset is supplied it must be the opposite mode for the same
sensor, otherwise ``select_datasets`` raises ``RuntimeError``.

The previous-day input **[CODE]**
---------------------------------

Not in the algorithm document, but the document does ask for timestamps that are
"regular and consistent, across boundaries between days". The code implements
this by accepting the previous day's L1C file, delivered by sds-data-manager
orchestration.

``_get_last_timestamp_and_rate_from_previous_day_in_ns`` takes the last sample
before midnight and the spacing that precedes it, matches that spacing against
the known ``VecSec`` rates, and uses it to seed the phase of any gap that opens
the current day. Without it, a leading gap is simply counted from the window
boundary.

``_validated_previous_day`` **raises** if the file is not L1C for the same
sensor or has no epochs - a wrong file is treated as an orchestration error, not
something to work around.

Configuration
-------------

**[CODE]** ``imap_processing/mag/imap_mag_sdc_configuration_v001.py``:

.. code-block:: python

   L1C_INTERPOLATION_METHOD = "linear_filtered"
   ALWAYS_OUTPUT_MAGO = True

The document is internally inconsistent here: its input list for section 7.3.4
says ``L1C_INTERPOLATION_METHOD`` defaults to ``LINEAR``, while step 4 of the
same section says "with Linear Filtered being the default". **The code uses
``linear_filtered``**, which matches the later statement and is the physically
correct choice, since the flight software uses a CIC filter to produce the
telemetered cadence in the first place.

The document is also explicit that the method is a **software configuration
setting**, not something the MAG team delivers in a file. That is why it lives
in a Python module in this repository rather than in an ancillary CDF.

Processing steps
----------------

1. Mark measured samples
^^^^^^^^^^^^^^^^^^^^^^^^

Every sample carries a **generated flag**. ``ModeFlags`` in ``mag/constants.py``:

.. code-block:: text

   ModeFlags.NORM    =  0    directly measured normal-mode sample
   ModeFlags.BURST   =  1    synthesised from interpolated burst data
   ModeFlags.MISSING = -1    placeholder; removed before output

Internally the timeline is an ``(n, 8)`` float array:

.. code-block:: text

   column 0    epoch (TTJ2000 ns)
   columns 1-4 vector x, y, z, range
   column 5    generated flag
   columns 6-7 compression flags (is_compressed, compression_width)

2. Find gaps
^^^^^^^^^^^^

``find_all_gaps`` / ``find_gaps``. A gap is a spacing that exceeds the expected
cadence by more than a tolerance:

.. code-block:: python

   expected_gap = 1 / vectors_per_second * 1e9              # ns
   is_gap = (diff - expected_gap) > expected_gap * L1C_TIMESTAMP_GAP_TOLERANCE

``constants.L1C_TIMESTAMP_GAP_TOLERANCE = 0.075`` (7.5%), which allows for clock
drift: 75, 37.5, 18.75 or 9.375 ms at 1, 2, 4 or 8 Hz respectively.

The expected cadence comes from the ``vectors_per_second`` global attribute.
``_find_rate_segments`` splits the day into contiguous rate segments, walking
each declared transition **backward** while the observed cadence already matches
the new rate. This stops a delayed Config-mode boundary from producing a
spurious one-sample micro-gap.

If ``day_to_process`` is supplied, gaps are also added from the start of the
25-hour window to the first sample and from the last sample to the end of the
window.

Gaps are returned as ``(start, end, vectors_per_second)`` where ``start`` and
``end`` are both real timestamps.

3. Generate a new timeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

``generate_timeline`` / ``generate_missing_timestamps``. For each gap, timestamps
are generated with ``np.arange(start, end, 1e9 // rate)`` at the rate declared
for that gap, in integer nanoseconds. Timestamps that already exist are removed.

.. important::

   **This is a deliberate deviation from the document.**

   **[DOC]** section 7.3.4 step 3 describes: find the burst time ``tC`` nearest
   to the last pre-gap normal time ``tA``; decimate the burst *timestamps* by
   ``M`` to the normal cadence such that ``tC`` is included; then subtract
   ``tA - tC`` from every decimated time, so the bridged series has **zero
   jitter** relative to the real normal-mode samples.

   **[CODE]** generates a regular grid from the gap start at the declared
   cadence and interpolates burst data onto it. The intent (a regular,
   jitter-free bridging series) is the same, and the previous-day mechanism
   handles phase continuity across day boundaries, but the construction is not
   the document's.

   ``generate_missing_timestamps`` raises if gap bounds are not integers -
   float64 cannot represent TTJ2000 nanoseconds exactly.

4. Interpolate burst data into the gaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``interpolate_gaps``. Per gap:

* Determine the burst rate at the gap start from the burst file's
  ``vectors_per_second`` attribute, and the normal rate from the gap tuple.
* Take a **buffer** of extra burst samples on each side:
  ``burst_buffer = (2 / norm_rate) * burst_rate`` samples. The CIC filter needs
  roughly two normal-mode cadences of extra data because it destroys the
  beginning and end of its output.
* Clip the gap timeline to the span where burst data actually exists.
* Call the configured interpolation function with the burst vectors (x, y, z
  only), burst epochs, target timestamps, and both rates.
* Write the interpolated vectors, set the generated flag to ``BURST``, and copy
  the range and compression flags from burst.

5. Identify remaining gaps and export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``remove_missing_data`` drops every row still flagged ``MISSING``.

**[DOC]** step 5 additionally requires: walk the completed vector list, treat any
spacing greater than **1.1 seconds** as a gap, and write the list of gap start
and end times as a header in the output file.

**[CODE]** this is **not implemented**. ``global_attributes["missing_sequences"]``
is set to ``""``, with a ``# TODO merge missing sequences? replace?`` at
``mag_l1c.py:137``, and the 1.1 second rule appears nowhere. See
:ref:`mag-implementation-status`.

Output variables:

* ``vectors`` - (x, y, z, range)
* ``vector_magnitude``
* ``compression_flags``

* ``generated_flag`` - the ``ModeFlags`` value per sample
* Global attribute ``interpolation_method``

.. _mag-interpolation:

Interpolation methods
---------------------

**[DOC]** six methods, all implemented in ``interpolation_methods.py`` and
exposed through the ``InterpolationFunction`` enum, which is callable:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Method
     - Implementation
   * - ``linear``
     - ``make_interp_spline(..., k=1)``. The document suggests ``numpy.interp``
       or ``InterpolatedUnivariateSpline(k=1)``.
   * - ``quadratic``
     - ``k=2``
   * - ``cubic``
     - ``k=3``
   * - ``linear_filtered``
     - CIC filter, then ``linear``. **Default.**
   * - ``quadratic_filtered``
     - CIC filter, then ``quadratic``
   * - ``cubic_filtered``
     - CIC filter, then ``cubic``

``remove_invalid_output_timestamps`` clips output timestamps to the span of the
input timestamps unless ``extrapolate=True``, so the pipeline never invents
science data outside the burst timeline.

The CIC filter
^^^^^^^^^^^^^^

**[DOC]** A cascaded integrator-comb filter is what the MAG flight software uses
to decimate 1920 raw samples/second down to the telemetered rate, so replicating
it on the ground makes a synthesised normal-mode sample match what the
instrument would have produced. A CIC filter depends on the ratio of input to
output sample rate and introduces a **constant time delay**.

The document's reference implementation:

.. code-block:: python

   decimation_factor = INPUT_SAMPLES_PER_SECONDS / 2
   CIC1 = ones(decimation_factor) / decimation_factor
   CIC2 = convolve(CIC1, CIC1)
   delay = (len(CIC2) - 1) // 2

   S_filtered = S[:-delay]
   A_filtered = lfilter(CIC2, 1, A, axis=0)[delay:]
   B = IUS(S_filtered, A_filtered, k=1)(T)

**[CODE]** ``cic_filter`` generalises this correctly:

.. code-block:: python

   decimation_factor = int(input_rate.value / output_rate.value)

The document's ``/ 2`` hardcodes a 2 Hz output, which is only right for the
default ``N_2_2`` normal mode. The code is right for ``N_4_1`` and ``N_4_4`` too.
``cic_filter`` raises ``ValueError`` if the burst input rate is not strictly
greater than the normal output rate.

``estimate_rate`` infers a rate from timestamp spacing by snapping to the nearest
value in ``POSSIBLE_RATES``, used only when a rate is not supplied.

Validation
----------

``test_mag_validation.py::test_mag_l1c_validation`` covers T013, T014, T015,
T016 and T024 for both sensors in
``imap_processing/tests/mag/validation/L1c/``. These are marked
``@pytest.mark.external_test_data``, so they are excluded from the default test
selection.
