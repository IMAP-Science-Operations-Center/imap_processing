.. _mag-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document (SW-009) and, where it overlaps, the public CMAD (see
:ref:`mag-cmad`). **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/mag`` and
``imap_processing/ialirt/l0/parse_mag.py``. If you change something material,
update this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Level
     - State
     - Notes
   * - L0 / L1A
     - **Mature**
     - Both APIDs, uncompressed and compressed vector paths, all eight
       validation cases pass. Sequence-gap bookkeeping is incomplete.
   * - L1B
     - **Complete but simplified**
     - All three algorithmic steps implemented and validated. Only one
       calibration is applied per day where the document allows several.
   * - L1C
     - **Complete, different construction**
     - Gap filling and all six interpolation methods work and are validated.
       Timeline construction takes a different (defensible) approach to the
       document, and the output gap header is missing.
   * - L1D
     - **Substantially complete**
     - All major steps implemented. Missing the per-sensor spin-offset split,
       the gradiometer quality flag propagation, and provenance headers.
   * - L2
     - **Complete for the nominal path**
     - Everything the document asks for except metadata pass-through, the
       sensor-attribute cross-check, and provenance headers. Adds GSM.
   * - L3
     - **Out of scope**
     - MAG has no L3. L2 is the final released product.
   * - I-ALiRT
     - **Works end to end**
     - Produces GSE/GSM/RTN vectors, magnitude and clock angles. No raw product,
       simplified validity logic, whole groups dropped on packet loss.

There are **no** ``NotImplementedError`` raises anywhere in the MAG code, apart
from the generic unknown-data-level branch in ``Mag.do_processing``.

Suspected bugs
--------------

These need confirmation with the MAG team or a test before being changed. They
are ordered by how likely they are to affect released data.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Location
     - Issue
   * - ``mag_l1c.py``, ``vector_magnitude``
     - ``np.linalg.norm(x[:4])`` is applied to the ``direction`` core dimension,
       which has **length 4 (x, y, z, range)**. Slicing ``[:4]`` keeps all four,
       so **the sensor range is included in the magnitude**. It almost certainly
       should be ``x[:3]``. Nothing downstream reads L1C ``vector_magnitude``
       (L1D and L2 recompute it from three components), so the impact is limited
       to the L1C product itself.
   * - ``mag_l1d_data.py``, ``apply_spin_offsets``
     - The loop is ``for index in range(n_chunks - 1)``. With **exactly one**
       spin-offset chunk the loop body never runs, so the x and y components are
       left at ``FILLVAL`` while z is copied through. This is reachable on short
       or heavily gapped days (fewer than ``2 * number_of_spins`` valid spins).
   * - ``mag_l1a_data.py``, ``append_vectors``
     - ``missing_sequences`` is built with
       ``range(most_recent + 1, vector_sequence)``, which produces an **empty
       range** when the 14-bit CCSDS source sequence counter wraps 16383 -> 0.
       Real gaps across a rollover are silently lost. The document explicitly
       warns the counter is a rolling 14-bit unsigned integer.
   * - ``mag_l1d_data.py``, ``generate_dataset``
     - When ``ALWAYS_OUTPUT_MAGO`` is ``False``, ``vectors``, ``epoch`` and
       ``range`` are swapped to MAGi but ``magnitude``, ``quality_flags`` and
       ``quality_bitmask`` are not. Since MAGi generally has a different sample
       count, this will raise a shape error rather than silently mislabel data -
       but it means the MAGi path is untested.
   * - ``mag_l1d.py``, gradiometry gate
     - ``if not input_mago_norm.attrs.get("all_vectors_primary", 1)``. L1A writes
       ``is_mago``/``is_active`` as the **strings** ``"True"``/``"False"`` but
       ``all_vectors_primary`` as a **bool**. If that attribute round-trips
       through CDF as the string ``"False"``, ``not "False"`` is ``False`` and
       gradiometry stays enabled when it should be disabled. Worth pinning down
       with a round-trip test.
   * - ``mag_l1d_data.py``, ``apply_gradiometry_offsets``
     - ``np.apply_along_axis(np.dot, 1, offsets, K)`` computes
       :math:`\mathbf{o}^{T}\mathrm{K}`, i.e. :math:`\mathrm{K}^{T}\mathbf{o}`,
       not :math:`\mathrm{K}\mathbf{o}`. Harmless for a symmetric or diagonal
       kappa, wrong for a general one. Confirm the intended convention before
       any off-diagonal kappa is delivered.
   * - ``mag_l1c.py``, ``interpolate_gaps``
     - Range and compression flags for an interpolated sample are taken from
       ``burst_vectors[burst_gap_start + index]``, where ``index`` counts
       positions in the **output** timeline, not the burst timeline. The two
       only coincide when the burst and normal rates are equal. The document
       says "all sample properties (such as sensor range) should be interpolated
       and output".

Deviations from the algorithm document
--------------------------------------

These are design decisions, not bugs, but they will surprise anyone reading the
document first.

L1C timeline construction
^^^^^^^^^^^^^^^^^^^^^^^^^

**Document** (7.3.4 step 3): find the burst time ``tC`` closest to the last
pre-gap normal time ``tA``, decimate the burst *timestamps* to the normal
cadence such that ``tC`` is included, then subtract ``tA - tC`` from every
decimated time so the bridging series has zero jitter relative to the real
normal-mode samples.

**Code:** generates a regular grid with ``np.arange(gap_start, gap_end,
1e9 // rate)`` and interpolates burst data onto it. Cross-day phase continuity
is handled instead by the previous-day L1C input.

The intent is the same and the code's approach arguably produces a cleaner
timeline, but it is not the document's algorithm and the two will not produce
identical timestamps.

The CIC decimation factor
^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** ``decimation_factor = INPUT_SAMPLES_PER_SECONDS / 2``.

**Code:** ``decimation_factor = input_rate / output_rate``.

The document hardcodes a 2 Hz output, correct only for the default ``N_2_2``
normal mode. **The code is right**; the document's template is a simplification.

Product naming
^^^^^^^^^^^^^^

**Document:** ``imap_mag_l1a_raw_normal_[date]_[version].cdf``.

**Code:** ``imap_mag_l1a_norm-raw``, following the IMAP filename convention where
the descriptor is one hyphenated token. All code strings are the authority.

GSM at L2
^^^^^^^^^

The document lists DSRF, SRF, RTN and GSE for L1D and L2, and GSM only for
I-ALiRT. The code additionally produces ``imap_mag_l2_{norm,burst}-gsm``, with
matching global attributes. This is an intentional extension. L1D does **not**
produce GSM. The public CMAD (section 4.1.1) lists the ``gsm`` L2 products
alongside the other frames, so the project documentation now agrees with the
code.

Fine time divisor
^^^^^^^^^^^^^^^^^

``TimeTuple`` uses ``MAX_FINE_TIME = 65536``. Section 7.4.2 of the document says
"One second is equal to **65535** fine time units" for I-ALiRT. The main science
sections do not state a divisor. The difference is ~15 microseconds - well below
the tens-of-milliseconds time shift the pipeline already corrects for, but it is
a real discrepancy that should be settled with the MAG team.

Not implemented
---------------

Sequence-counter provenance headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The document asks for "first and last sequence counter in the data file" and a
list of sequence gaps at **L1A raw, L1A per-sensor, L1D normal mode, and L2
normal mode**.

* **L1A:** only the gap list, as the ``missing_sequences`` global attribute, and
  it does not handle rollover. First/last counters are absent.
* **L1B:** propagates ``missing_sequences``.
* **L1C:** sets ``missing_sequences`` to ``""``. ``# TODO merge missing
  sequences? replace?`` at ``mag_l1c.py:137``.
* **L1D and L2:** pass ``global_attributes={}``, so **no** sequence provenance
  reaches either product.

L1C gap header (the 1.1 second rule)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.3.4 step 5: walk the completed vector list, treat any spacing greater
than **1.1 seconds** as a gap, and write the gap start/end times as a header in
the output file. Not implemented anywhere. The code's gap detection is a
*relative* 7.5% tolerance used for *input* gap finding, which is a different
thing for a different purpose.

Multiple ENG calibrations within one day
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.3.3 step 1: several ENG calibration files may be valid for
non-overlapping ranges within the processing window, and the latest generated
file wins per vector. The code applies one matrix and one time shift for the
whole day. ``# TODO: Check validity of time range for calibration`` at
``mag_l1b.py:73``.

Per-sensor spin-average offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.3.5 step 6: "Process each sensor's data independently", and the
outputs list "2 x CDF files for Spin Average offsets in NM for MAGo and MAGi".

The code computes spin offsets from **MAGo only** (``self.vectors`` inside
``MagL1d``) and applies the same offsets to both sensors, emitting a single
``imap_mag_l1d_spin-offsets`` file.

Gradiometer quality flag and magnitude at L1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.3.5 step 11: set a quality flag when the gradiometer offset magnitude
exceeds ``quality_flag_threshold``, and **add the gradiometer offset magnitude
as a data point per vector** in the science product.

The code computes both inside ``calculate_gradiometry_offsets`` and writes them
to the ``gradiometry-offsets`` **ancillary** dataset only. The L1D science
product's ``quality_flags`` remains hardcoded ``np.zeros`` and there is no
per-vector gradiometer magnitude variable.

L2 metadata pass-through
^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.2: the offsets file may carry additional metadata (for instance,
periods when interference was occurring and was removed) and "this metadata
should be copied into the final science file allowing the MAG team to apply
additional metadata based on currently unknown calibration steps".

The code copies only ``quality_flag`` and ``quality_bitmask``. Any other
metadata in the offsets file is dropped.

L2 sensor-attribute cross-check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Document 7.3.6 step 3: "Ensure that the sensor attributes in the offset and
calibration file match MAGo (or vice versa) and fail if they do not match." Not
implemented. There is a ``# TODO Check that the input file matches the offsets
file`` at ``mag_l2.py:97``. Epoch equality **is** enforced, which catches the
most likely mismatch.

Quality bitmask definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Bit assignment: resolved.** SW-009 section 7.2 and the ``VAR_NOTES`` on
``qf_bitmask`` in ``imap_mag_l2_variable_attrs.yaml`` used to disagree: SW-009
lists eight named bits including ``SCTONES`` and ``PIVOTPLATFORMINTERFERENCE``,
with ``SEC_SENS`` last. The public CMAD (section 5.4.5) now defines the bitmask
explicitly, and it **matches the YAML**:

* Bit 0: data sourced from the secondary sensor
* Bit 1: thruster firing signals removed
* Bit 2: spacecraft interference (TCMs, IMAP-Lo pivot platform motion)
* Bit 3: instrument signals removed
* Bits 4-7: reserved for in-flight calibration

Treat SW-009's list as superseded. See :ref:`mag-cmad-quality`.

**Still not done:** MAG does not use ``imap_processing/quality_flags.py``; there
is no ``MagQualityFlags`` enum, and the bitmask is copied through opaquely from
the offsets file. The YAML ``VAR_NOTES`` is the only in-repo definition.

L2 provenance of Imperial calibration inputs (unconfirmed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMAD ICL-017 section 2 says the Imperial calibration input files (matrix
version, spin-plane and spin-axis offset CSVs, listed in
``calibration_input_release1_v001.json``) "are captured in the parent metadata
field of the released L2 science data files".

**[CODE]** ``Mag.do_processing`` in ``cli.py`` sets L2 ``Parents`` to the SDC
dependency file names (calibration and offsets files) plus the one L1B/L1C file
actually used. The offsets file's own ``Parents`` is used only to **locate** that
L1 file: ``retrieve_mag_l1_inputs_from_l2_offsets`` downloads every entry and L2
uses the first. It is never copied to the output.

Whether this matters depends on what Imperial puts in the offsets file, which
has not been checked against a real delivered file. If the offsets ``Parents``
lists anything besides the L1 file (such as the calibration-input JSON):

* those entries do not reach L2 ``Parents``, contrary to the CMAD statement;
* ``retrieve_mag_l1_inputs_from_l2_offsets`` would try to ``download()`` them
  from the SDC, and would fail if they are not SDC-hosted files.

If instead the calibration-input names live only inside the offsets file, the
CMAD statement is satisfied indirectly, because the offsets file is itself a
parent of the L2 product.

I-ALiRT gaps
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Document requirement (7.4)
     - Status
   * - Step 2: output a raw I-ALiRT data product named "MAG I-ALiRT RAW"
     - **Not implemented.** The pipeline goes straight to the L1D-equivalent
       dict.
   * - Step 2: use ``PHSEQCNT`` to identify non-sequential packets and record
       gap start/end times as headers
     - **Not implemented.**
   * - Step 3: if any one of the four packets is missing, NaN only the
       respective fields
     - **Not implemented.** The whole group is dropped and logged.
   * - Step 6: ``PRI_ISVALID`` requires the validity bit in **both** packets 0
       and 1; ``SEC_ISVALID`` in both packets 2 and 3
     - **Not implemented.** ``pri_isvalid`` comes from packet 1 only,
       ``sec_isvalid`` from packet 3 only.
   * - Step 8c: RTN output "maybe, TBC based on discussion with space weather
       users"
     - **Implemented** - RTN is produced.

The I-ALiRT path also uses a **single** engineering calibration dataset via
``retrieve_matrix_from_single_l1b_calibration``, with no day selection or
multi-file combining.

Outstanding TODOs in the code
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Location
     - TODO
   * - ``l0/decom_mag.py:84``
     - "Correct CDF attributes from email"
   * - ``l0/decom_mag.py:137``
     - Epoch on the raw product is the **packet** start time; confirm this is
       what MAG expects and fix ``CATDESC`` if so.
   * - ``l0/decom_mag.py:145``
     - Units for ``raw_vectors`` are undefined.
   * - ``l1b/mag_l1b.py:73``
     - Check validity of the time range for the calibration.
   * - ``l1c/mag_l1c.py:57``
     - Find missing sequences and output them; handle a missing burst file by
       passing the norm file through.
   * - ``l1c/mag_l1c.py:572``
     - ``generate_empty_norm_array`` fills with ``np.zeros`` instead of
       ``FILLVAL``. Rows that are never filled are dropped by
       ``remove_missing_data``, so this is currently benign.
   * - ``l1c/mag_l1c.py:665``
     - "we need extra data at the beginning and end of the gap" - the CIC buffer
       heuristic may be insufficient at gap edges.
   * - ``l1c/mag_l1c.py:842``
     - When falling back to the previous file, also retrieve the expected
       vectors per second.
   * - ``l1d/mag_l1d.py:137``
     - Frame-specific CDF attributes may be required for L1D.
   * - ``l1d/mag_l1d_data.py:725``
     - Should gradiometry extrapolate, or should non-overlapping data be
       removed?
   * - ``l2/mag_l2.py:63, 97``
     - Retrieve the input file from the offsets dataset in ``cli.py`` (this is
       now done); check that the input file matches the offsets file.

Testing notes
-------------

* Unit tests: ``imap_processing/tests/mag/test_mag_decom.py``,
  ``test_mag_l1a.py``, ``test_mag_l1b.py``, ``test_mag_l1c.py``,
  ``test_mag_l1d.py``, ``test_mag_l2.py``.
* Validation tests against MAG-team reference outputs live in
  ``test_mag_validation.py`` with data in
  ``imap_processing/tests/mag/validation/``:

  .. list-table::
     :header-rows: 1
     :widths: 18 22 60

     * - Level
       - Cases
       - Marked ``external_test_data``?
     * - L1A
       - T001 - T008
       - No - these run by default.
     * - L1B
       - T009 - T012
       - No.
     * - L1C
       - T013 - T016, T024
       - **Yes** - excluded from the default selection.
     * - L2
       - T021 (burst), T022 (norm)
       - **Yes.**

* There are **no validation cases for L1D** and none for I-ALiRT MAG in the MAG
  validation harness.
* L1D and L2 tests need SPICE. Fixtures come from
  ``imap_processing/tests/conftest.py`` plus
  ``imap_processing/tests/mag/validation/calibration/spice/fake_mag_spin_data.csv``.

Where to start if you are picking up work
------------------------------------------

Roughly in order of value per unit effort:

1. **Confirm the L1C ``vector_magnitude`` slice** and the ``apply_spin_offsets``
   single-chunk case. Both are small, both are testable, both affect data.
2. **Fix the 14-bit sequence-counter rollover** and add the first/last sequence
   counter attributes. This is the most-repeated missing requirement in the
   document and it is provenance data that cannot be reconstructed later.
3. **Put the quality bitmask in** ``quality_flags.py`` as a real enum and make
   the YAML derive from it. The bit assignment is now settled by CMAD section
   5.4.5 (see :ref:`mag-cmad-quality`). While there, check what a real delivered
   offsets file carries in ``Parents`` (see the provenance note above).
4. **Propagate the gradiometer quality flag and magnitude into the L1D science
   product.** The values are already computed.
5. **Split spin-average offsets per sensor.** Currently MAGo offsets are applied
   to MAGi.
6. **Add L1D validation cases.** L1D is the second-largest body of MAG
   algorithm code and has no reference-output tests.
