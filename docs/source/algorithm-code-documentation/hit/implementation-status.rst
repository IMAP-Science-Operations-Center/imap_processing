.. _hit-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of a survey of ``imap_processing/hit`` and
``imap_processing/ialirt/l0/process_hit.py`` against algorithm document
*IMAP/HIT Science Algorithms* version 1.11.00, 2 June 2026 (Draft). If you
change something material, update this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Level
     - State
     - Notes
   * - L0 / L1A counts
     - **Complete and exact**
     - The frame byte map matches document Tables 11-26 byte for byte, and the
       decompression matches the heritage C exactly. Validated against the
       sample CSV exports in ``tests/hit/validation_data/``.
   * - L1A direct events
     - **Stub**
     - The product exists and carries the raw binary. **Nothing decodes it.**
       The single largest gap in HIT.
   * - L1A sectored counts
     - **Complete**
     - The mod-10 subcommutation, complete-set detection and 10-minute
       livetime shift all work, with the caveat that the shift is by array
       position rather than by time.
   * - L1A / L1B housekeeping
     - **Complete**
     - No Python algorithm - the same packet decommutated raw and derived, with
       all conversions (including the piecewise thermistors) in the XTCE.
   * - L1B science
     - **Complete**
     - Livetime fraction, rates, summed rates and sectored rates all
       implemented and tested. Uncertainty summing is linear, as the document
       specifies.
   * - L2
     - **Complete for what it claims**
     - All three intensity products work. Real gaps: no spin-rate correction,
       the ``b`` subtraction is in the wrong place if ``b`` ever becomes
       non-zero, and the L4 rates never become intensities.
   * - I-ALiRT
     - **Working, labels stale**
     - Produces all 12 public products correctly. Three blocks of slow-rate
       slot names do not match Table 38 as revised in v1.11.00.
   * - L3
     - **Out of scope, correctly**
     - Separate repository. See :ref:`hit-l3-scope`.
   * - Quicklook
     - **Not started**
     - No HIT quicklook code exists here, and the document does not specify
       any.

Ranked list of things to fix
----------------------------

Highest value first, in the judgement of whoever last surveyed this. Each has a
detailed entry below.

#. :ref:`hit-gap-events` - PHA event records are never decoded. Blocks all of
   L3 section 9.1.
#. :ref:`hit-gap-spinrate` - the 15th inclination bin is never corrected for
   spin-rate deviation, which the document explicitly assigns to the ground.
#. :ref:`hit-gap-geometric-mean` - arithmetic mean used where the document
   specifies the geometric mean, in every product above L1A.
#. :ref:`hit-gap-livetime-shift` - the sectored livetime shift is by array
   position, not by time; a dropped frame silently mispairs the data.
#. :ref:`hit-gap-background` - the background term ``b`` is subtracted after
   the division rather than from the counts.
#. :ref:`hit-gap-incomplete-frames` - incomplete science frames are silently
   discarded with no quality flag.
#. :ref:`hit-gap-ialirt-slots` - three blocks of I-ALiRT slow-rate slot names
   disagree with the revised Table 38.
#. :ref:`hit-gap-livetime-range` - no detection of out-of-range livetime, which
   the document asks for.
#. :ref:`hit-gap-l4rates` - ``l4fgrates`` / ``l4bgrates`` and ``ialirtrates``
   are carried to L1B and then dropped.
#. :ref:`hit-gap-sector-order` - the sectored ancillary table relies on row
   order for the declination assignment.
#. :ref:`hit-gap-systematics` - the chance-coincidence and livetime systematic
   corrections are defined but unimplemented (and unquantified by the
   instrument team).
#. :ref:`hit-gap-dead-code` - a handful of unused constants and stale
   comments.

Hard failures in the code
-------------------------

Explicit exceptions, so you know what a bad input looks like:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Location
     - Condition
   * - ``cli.py`` ``Hit.do_processing``
     - ``NotImplementedError`` for any data level other than l1a/l1b/l2.
   * - ``cli.py`` ``Hit.do_processing``
     - ``ValueError`` if more than 2 dependencies for l1a, more than one L1A
       file for l1b, not exactly 5 dependencies for l2, more than one science
       file for l2, or not exactly 4 ancillary files for l2.
   * - ``hit_l1a.hit_l1a``
     - ``ValueError`` if ``packet_date`` is missing.
   * - ``hit_l1a.subset_sectored_counts``
     - ``ValueError`` if no complete mod-10 set is found.
   * - ``hit_l1a.subset_livetime``
     - ``ValueError`` if the epoch array is empty, or if the first complete
       set starts fewer than 10 frames into the file.
   * - ``hit_l1b.hit_l1b``
     - ``ValueError`` for a descriptor other than ``hk``, ``standard-rates``,
       ``summed-rates`` or ``sectored-rates``.
   * - ``hit_l2.load_ancillary_data``
     - Bare ``StopIteration`` if no ancillary file matches
       ``dt<N>-factors`` for a state present in the data. **Not a friendly
       error.**
   * - ``decom_hit.assemble_science_frames``
     - ``IndexError`` on ``starting_indices[0]`` if no valid science frame is
       found in the file.

Silent behaviours worth knowing
-------------------------------

* ``hit_l2.hit_l2`` returns ``None`` if the input's ``Logical_source`` matches
  none of the three expected L1B products.
* ``hit_l1b.process_science_data`` returns ``None`` for an unrecognised
  descriptor (unreachable - ``hit_l1b`` validates first).
* Incomplete science frames are dropped with a ``print``, not a log or a flag.
* Sectored frames outside a complete 10-frame run are dropped entirely.
* ``add_cdf_attributes`` at L1A warns and continues for a variable missing from
  the attribute YAML; at L2 it logs an error and continues.

Detailed entries
----------------

.. _hit-gap-events:

1. PHA event records are never decoded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: high. The single biggest missing piece in HIT.**

**[CODE]** ``imap_hit_l1a_direct-events`` contains exactly one data variable,
``pha_raw``: the concatenated binary of packets 6-19 of each science frame, as
a string. Nothing parses it.

**[DOC]** The format is fully specified across sections 4.2.5-4.2.8 (PDF pages
25-33):

* **Event Buffer header** - 2 bytes at frame bytes 1572-1573, the number of
  event records. Unused buffer space is filled with ``0x00``; empty packets are
  not transmitted.
* **Event Record Header (Table 4)** - 32 bits: Particle ID (bits 0-7),
  priority buffer number (8-12), STIM tag (13), HAZ tag (14), time tag /
  latency bits (15-18), A/B tag (19), unread-ADCs flag (20), long event flag
  (21), culling flag (22), spare (23).
* **ADC fields (Table 5, Figure 9)** - 20 bits each: 12-bit signal (11 bits
  plus overflow), 6-bit detector ID, 1 gain bit (0 = high, 1 = low), 1
  End-of-Record bit. The **EOR bit**, not a count, marks the end of the record.
* **Padding (Table 6)** - records with an odd number of ADC fields are padded
  with 4 bits to a byte boundary. Minimum record is 2 ADC fields = 72 bits = 9
  bytes. Maximum is 54 ADC hits.
* **Extended Header Block** - 3 bytes appended when bit 21 is set: detector
  group flags (Table 7), ``DEINDEX`` (9 bits, 0-399) and ``EPINDEX`` (7 bits,
  0-127), the onboard matrix indices.
* **STIM Information Block** - 3 more bytes when bits 13 **and** 21 are both
  set: a seconds-within-minute counter, ``DACLEVEL`` (0-31) and
  ``DACCONFIGURATION`` (0-14).
* **Detector address table (Table 8)** - the 6-bit detector ID to name
  mapping, 0-63.

Two design notes from the document worth carrying into any implementation:

* **A single bit error in an EOR bit can make every subsequent event in the
  buffer unreconstructible.** The document acknowledges this and floats
  possible mitigations. A decoder should fail the rest of the buffer
  gracefully rather than produce garbage.
* **Latency bits do not necessarily correspond to the frame's timestamp.** An
  event may sit in a priority buffer for more than a minute before being
  telemetered. The 4 latency bits duplicate the low 4 bits of the onboard
  minute counter; the document says explicitly that *"it is up to the user of
  the data to determine what value of the latency bits precisely corresponds to
  times associated with given Science Frames."*

**Consequence:** all of L3 section 9.1 (incident ion energy, ADC-MeV
conversion, cosine correction, charge calculation) is blocked, and the early-
mission plan to characterise ion contamination in the I-ALiRT electron channels
from PHA data is blocked with it.

.. _hit-gap-spinrate:

2. No spin-rate correction to the inclination bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: high for anisotropy science. Affects the macropixel product.**

**[DOC]** Section 4.2.2:

   *"The 15 inclination 24 degree bins (labeled 0-14) are determined from
   timing (incrementing every second), with the assumption that the spin rate
   is the nominal 4 rpm. Deviation of the spin rate from 4 rpm will change the
   angular width of the fifteenth bin (smaller than 24 degrees for spin <= 4
   rpm, and larger for spin >= 4 rpm). This needs to be corrected on the
   ground. No onboard correction is planned. A new spin phase zero resets the
   inclination bin index."*

**[CODE]** ``AZIMUTH_ANGLES`` is a fixed array of 15 bin centres at 12, 36,
..., 348 degrees. Nothing reads the actual spin period or spin phase; no HIT
code touches SPICE geometry at all.

**Consequence:** the 15th inclination bin (``azimuth = 348``) has an incorrect
angular width, and therefore an incorrect effective geometry factor, whenever
the spin rate is not exactly 4 rpm - which is always. The document does not
specify the correction, only that it is required, so implementing this needs
the HIT team to define what "corrected" means (rebinning? a per-bin width
array? an effective geometry factor scaling?).

.. _hit-gap-geometric-mean:

3. Arithmetic mean used instead of geometric mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium. Affects every energy coordinate above L1A.**

**[DOC]** Section 6.1 specifies the geometric mean
:math:`\sqrt{E_{min} E_{max}}` as the characteristic energy, with
``delta_plus`` and ``delta_minus`` measured from it. The document explains the
reasoning: the true mean energy of particles in a bin depends on the spectrum,
so the label is only characteristic and the **edges** are the real quantity.

**[CODE]** ``hit_utils.add_energy_variables``:

.. code-block:: python

   energy_mean = np.round(
       np.mean(np.array([energy_min_values, energy_max_values]), axis=0), 3
   ).astype(np.float32)

**Consequence:** ``<species>_energy_mean``, ``_energy_delta_plus`` and
``_energy_delta_minus`` are wrong in ``imap_hit_l1a_counts-sectored``,
``imap_hit_l1b_summed-rates``, ``imap_hit_l1b_sectored-rates``, and all three
L2 products. The error is largest for the widest bins. For the sectored Fe
4.0-12.0 MeV/nuc bin the arithmetic mean is 8.00 and the geometric mean is
6.93 - a 15% shift in the plotted energy. For narrow standard bins such as
H 3.2-3.6 MeV/nuc the difference is under 0.2%.

The **bin edges are recoverable** in either convention
(``mean - delta_minus`` and ``mean + delta_plus``), so no information is lost -
but any consumer plotting against ``energy_mean``, or comparing to another
instrument that uses the geometric convention, will be off.

**Fix:** one line in ``add_energy_variables``. Check the CDF variable
attributes and any validation CSVs at the same time.

.. _hit-gap-livetime-shift:

4. The sectored livetime shift is positional, not temporal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium. Silent wrong answer in the presence of data gaps.**

**[CODE]** ``hit_l1a.subset_livetime`` locates the current block's first and
last epochs in ``epoch_livetime`` and then slices **10 array positions
earlier**:

.. code-block:: python

   start_trimmed = max(start_idx - 10, 0)
   end_trimmed = max(end_idx - 10, 0)

Similarly ``hit_l1b.sum_livetime_10min`` sums in fixed 10-element windows, and
``hit_l1a.find_complete_mod10_sets`` assumes consecutive array entries are
consecutive minutes.

**[DOC]** Section 6.2 and Figure 14 define the relationship in **time**: block
*n*'s counts pair with the livetimes accumulated during the preceding 10
minutes.

**Consequence:** these coincide only when there are no missing science frames.
A single dropped frame anywhere in or before a sectored block shifts the
pairing by one minute, silently, with no flag. Since incomplete frames are
already discarded without a trace (:ref:`hit-gap-incomplete-frames`), this is
reachable in real data.

**Fix:** index the livetime by epoch difference rather than array position, and
raise or flag when the expected 10 minutes are not all present.

.. _hit-gap-background:

5. The background term is subtracted in the wrong place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low today, high the day ``b`` becomes non-zero.**

**[DOC]** Equations 12, 14 and 16 subtract background **counts** from the
numerator before dividing:

.. math::

   j_i \propto \frac{\sum_j q_{ij} - b_{ij}}{\sum_j \sigma_{ij}\lambda_{ij}}

**[CODE]** ``hit_l2.calculate_intensities`` subtracts ``b`` from the finished
intensity:

.. code-block:: python

   intensity = (rates / (delta_time * delta_e * geometry_factor * efficiency)) - b

These agree only if ``b`` is delivered in intensity units rather than counts.
**Every ``b`` in the current ancillary CSVs is 0**, so there is no numerical
difference today.

There is a second issue in the same place: the uncertainty arrays go through
the identical call, so a non-zero ``b`` would also be subtracted from the
uncertainties, which is meaningless.

**Fix when needed:** establish the units of ``b`` with the HIT team, then
either move the subtraction into the numerator or document that the CSV column
is an intensity - and give the uncertainty arrays their own path.

.. _hit-gap-incomplete-frames:

6. Incomplete science frames vanish silently
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium.**

**[CODE]** ``decom_hit.assemble_science_frames`` keeps only runs of 20 packets
whose grouping flags match ``FLAG_PATTERN`` **and** whose sequence counters are
consecutive. Everything else is dropped. There is an explicit ``TODO``:

   *"The code currently skips all incomplete science frames. Only discard
   incomplete science frames in the middle of the CCSDS file or use fill
   values?"*

Notification of dropped packets at the file boundaries is done with ``print``,
not ``logger``, so it does not reach the batch job's logs in a structured way.
There is no quality flag anywhere in the HIT products (contrast SWE, which has
``SweL1bFlags`` in ``imap_processing/quality_flags.py`` - HIT has no entry
there).

**Consequence:** a user cannot tell a quiet minute from a lost minute, and the
positional livetime shift above turns lost minutes into wrong answers.

.. _hit-gap-ialirt-slots:

7. I-ALiRT slow-rate slot names are stale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low today (no product reads the affected slots), medium for anyone
extending the product.**

See the warning in :ref:`hit-ialirt` for the slot-by-slot comparison. Slots
8-10, 38-39 and 47-54 in ``HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"]`` disagree with
Table 38 as revised in document version 1.11.00. The code comment also cites
"Table 37", which was the table number in an earlier revision.

Note that Table 38 itself repeats a block of four names at slots 47-50 and
51-54, which looks like an error in the document. **Confirm with the HIT team
before changing anything.**

.. _hit-gap-livetime-range:

8. No detection of out-of-range livetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low.**

**[DOC]** Section 6.2: the piecewise livetime conversion *"will be incorrect
for values of live time below 0.001667% (count 16,000). This situation should
be detectable via other mnemonics, such as NUMTRIG (Table 12) or STIM event
counts (PBUFRATES #29 and #30, Table 16)."*

**[CODE]** ``livetime_fraction_calculation`` applies the piecewise function
unconditionally. Nothing cross-checks ``num_trig`` or ``pbufrates[29]`` /
``pbufrates[30]``, and there is no flag on the output.

The document does not say what the check should be, so this needs HIT team
input before it can be implemented.

.. _hit-gap-l4rates:

9. The L4 and I-ALiRT rate arrays stop at L1B
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low. Possibly correct by design.**

**[CODE]** ``l4fgrates`` (48), ``l4bgrates`` (24) and ``ialirtrates`` (20) are
decommutated at L1A, livetime-corrected at L1B, and then **never referenced
again**. ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING`` covers only R2, R3 and R4.

**[DOC]** Section 4.2.1 introduces ``RNG2I``/``RNG3I``/``RNG4I`` - ions that
entered through an I-ALiRT aperture, lost energy in L4, and therefore sit at
higher incident energy - as first-class new-for-HIT ranges. Tables 25 and 26
give the 48 foreground and 24 background rates with their own Particle IDs.
But **Table 31 (the L2 standard products) does not include them**, and none of
the geometry-factor tables (32-37) has entries for them.

So the current behaviour matches the document. The open question is whether
the instrument team intends them to become an L2 product later. Worth asking
before anyone "cleans up" the unused arrays.

.. _hit-gap-sector-order:

10. Sectored ancillary data depends on CSV row order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low, but it is a silent failure mode.**

**[CODE]** ``hit_l2.get_species_ancillary_data`` groups the ancillary
DataFrame by ``lower energy (mev)`` and collects each column with
``.apply(list)``. For the sectored family, the resulting inner list is assumed
to be sectors 0-7 **in file order**. The ``Sector`` column is present in the
CSV and is never read.

**Consequence:** a redelivered ancillary file with the rows sorted differently
would scramble the declination assignment with no error.

**Fix:** sort by, or index on, the ``sector`` column.

.. _hit-gap-systematics:

11. Systematic uncertainty corrections are unimplemented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: none yet - the instrument team has not quantified them.**

**[DOC]** Section 4.3.2 names two sources:

* **Chance coincidences.** Mitigated by detector segmentation; *"further
  consistency checks will be employed during ground processing"* - unspecified.
* **Livetime errors at high rates.** The counter does not include the
  coincidence window opened by each trigger. The proposed correction is
  :math:`\mathrm{livetime_{corr}} = \mathrm{livetime} + (\Delta t \times N_{trig})`,
  with :math:`\Delta t` *"to be explored using high intensity runs at the
  accelerators during HIT calibrations"*.

**[CODE]** Neither correction exists. ``add_systematic_uncertainties`` writes
zeros and ``add_total_uncertainties`` does the quadrature sum - which is
exactly what the document prescribes for launch. The plumbing is in place; only
the numbers are missing.

Note that ``num_trig`` (NUMTRIG) **is** available at L1A and L1B, so when
:math:`\Delta t` arrives the correction is a small change in
``livetime_fraction_calculation``. Remember the document's caveat that NUMTRIG
is accumulated **only during even-numbered seconds**, so it represents half the
triggers in the minute; the ``.HAZ`` counters cover the odd seconds.

.. _hit-gap-dead-code:

12. Dead constants and stale comments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: cosmetic.**

* ``l1b/constants.py`` ``LIVESTIM_PULSES = 270`` - documented as being used to
  calculate the fractional livetime; never referenced.
* ``l0/constants.py`` ``MOD_10_PATTERN`` - superseded by
  ``np.arange(10)`` inside ``find_complete_mod10_sets``; never referenced.
* ``ialirt/l0/process_hit.py`` cites "Table 37"; the correct reference in
  v1.11.00 is Table 38.
* ``hit_l1a.hit_l1a`` docstring says the L0 file has "a 20-minute buffer before
  and after the processing day"; the document specifies 5 minutes before and
  15 after.
* ``hit_l1b.process_standard_rates_data`` omits ``l4fgrates_index`` and
  ``l4bgrates_index`` from its explicit coordinate list (they get created
  implicitly).
* ``hit_utils.initialize_particle_data_arrays`` creates
  ``<species>_energy_mean`` as an ``int8`` zero array, which
  ``add_energy_variables`` immediately overwrites with ``float32``.

Test coverage
-------------

**[CODE]** ``imap_processing/tests/hit/`` is reasonably thorough for what is
implemented:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Covers
   * - ``test_decom_hit.py``
     - Bit parsing, frame flag matching, sequence checks, decompression, epoch
       mean, full ``decom_hit``.
   * - ``test_hit_l1a.py``
     - Sectored subcommutation, livetime coordinate handling, complete-set
       detection, uncertainties, CDF attributes, processing-day filtering, plus
       **validation against** ``hskp_sample_raw.csv`` and ``sci_sample_raw.csv``.
   * - ``test_hit_l1b.py``
     - Rates, 10-minute livetime summing, all three science products,
       housekeeping, the livetime piecewise fit, plus **validation against**
       ``hskp_sample_eu_3_6_2025.csv`` and
       ``hit_l1b_standard_sample2_nsrl_v4_3decimals.csv``.
   * - ``test_hit_l2.py``
     - Ancillary loading and reshaping, intensity calculation, systematic and
       total uncertainties, all three L2 products, the 10-minute regrouping.
   * - ``test_hit_utils.py``
     - APID lookup, leak-variable concatenation, housekeeping, energy
       variables, cross-range summing.
   * - ``tests/ialirt/unit/test_process_hit.py``
     - The I-ALiRT grouping and the 12 products, against
       ``hit_ialirt_sample.ccsds`` / ``.csv``.

Notable coverage gaps: nothing exercises ``pha_raw`` beyond its presence,
nothing tests a science file with missing or corrupt frames, and there is no
end-to-end L1A-to-L2 validation against instrument-team-supplied L2 values.
