.. _swe-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of a survey of ``imap_processing/swe`` and
``imap_processing/ialirt/l0/process_swe.py`` against algorithm document
CN102D-D0001, Issue Draft, 15 June 2026. If you change something material,
update this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Level
     - State
     - Notes
   * - L0 / L1A
     - **Complete**
     - Decommutation and decompression are done and validated against GSEOS
       exports. No rejection criteria exist in the document, so none are
       missing.
   * - L1A / L1B housekeeping
     - **Complete**
     - No algorithm - the same packet decommutated raw and derived. The
       ``HVPS_ESA_DAC`` dual-range conversion is the one document requirement
       not met.
   * - L1B science
     - **Mature**
     - The checkerboard, timing, deadtime and gain calibration all work and are
       tested. The active ESA table number is hard-wired to 0 and the
       uncertainty is in the wrong units.
   * - L2
     - **Complete for what it claims, with real gaps**
     - Phase space density, flux, spin angle binning all work. **The
       end-detector correction from the heritage code is missing**, the frame
       is instrument rather than despun spacecraft, and the uncertainty path
       has a double-binning bug.
   * - I-ALiRT
     - **Working, a few undocumented choices**
     - Produces records; the ``Cmin`` definition, the unconditional polar
       search and the hard-coded 80 ms all differ from or extend the document.
   * - L3
     - **Out of scope, correctly**
     - Separate repository. See :ref:`swe-l3-scope`.
   * - Quicklook
     - **Not started**
     - No SWE quicklook code exists here.

Ranked list of things to fix
----------------------------

Highest value first, in the judgement of whoever last surveyed this. Each has a
detailed entry below.

#. :ref:`swe-gap-end-detector` - missing factor of 2 on CEMs 1 and 7 at L2.
#. :ref:`swe-gap-uncert-units` - uncertainty is in counts, data in counts/s.
#. :ref:`swe-gap-double-bin` - ``flux_stat_uncert`` binned twice.
#. :ref:`swe-gap-esa-table` - the active ESA table number is never read.
#. :ref:`swe-gap-frame` - instrument frame vs despun spacecraft at L2.
#. :ref:`swe-gap-none-dataset` - ``swe_l1b`` appends ``None`` when no full cycle
   is found.
#. :ref:`swe-gap-dims` - mislabelled dimensions on three L2 variables.
#. :ref:`swe-gap-esa-dac` - ``HVPS_ESA_DAC`` dual-range conversion missing.
#. :ref:`swe-gap-calfile` - ``k`` and the geometric factors are hard-coded.
#. :ref:`swe-gap-dead-code` - three dead functions, one of which cannot run.

Hard failures in the code
-------------------------

Explicit exceptions, so you know what a bad input looks like:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Location
     - Condition
   * - ``cli.py`` ``Swe.do_processing``
     - ``NotImplementedError`` for any data level other than l1a/l1b/l2.
   * - ``cli.py`` ``Swe.do_processing``
     - ``ValueError`` if the dependency count is not exactly 2 (l1a), 5
       (l1b sci), 2 (l1b hk) or 2 (l2).
   * - ``cli.py`` ``Swe.do_processing``
     - ``ValueError`` if more than one science file is supplied for l1b or l2.
   * - ``swe_l1b.calculate_calibration_factor``
     - ``ValueError`` if any acquisition time falls outside the in-flight
       calibration time range. **Deliberate** - SWE does not want
       extrapolation, matching the heritage ``electron_cal()`` which exits.
   * - ``swe_l1b.get_esa_dataframe``
     - ``ValueError`` for an ESA table number outside ``[0, 1]``. Unreachable -
       the function is never called.
   * - ``swe_l2.find_angle_bin_indices``
     - ``ValueError`` if any spin angle is outside ``[0, 360)``.

Detailed entries
----------------

.. _swe-gap-end-detector:

1. The end-detector correction is missing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: high. A factor of 2 on two of the seven detectors.**

**[DOC]** The heritage ``fspace()`` routine reproduced in section 3.4.4 does not
end after the phase space density loop. It has a second loop:

.. code-block:: c

   for (i=0; i<N_ENERGIES; i++){
      for (k=0; k<N_PHI; k++){
         if (k % 2 == 0) {
            fv[i][0][k] = 0.5*fv[i][0][k+1];
            fv[i][6][k] = 0.5*fv[i][6][k+1];
         }
         else {
            fv[i][0][k] = 0.5*fv[i][0][k];
            fv[i][6][k] = 0.5*fv[i][6][k];
         }
      }
   }

Indices 0 and 6 are **CEM 1 and CEM 7**, the two outermost detectors (polar
angle ±63 degrees). Every value from those two detectors is halved; for even
azimuth indices the value is additionally taken from the neighbouring odd
index first.

**[CODE]** ``swe_l2.calculate_phase_space_density()`` implements only the first
loop. Nothing anywhere applies the end-detector factor.

**Before implementing this**, note two things. First, the halving is stated
without explanation in the document - it is heritage ACE code, and the
``k+1`` copy in particular reflects the **ACE** 32-azimuth-bin layout, not
SWE's 30. Second, the SWE geometric factors for CEMs 1 and 7 (424.4e-6 and
425.2e-6) are already the smallest of the seven, so it is possible the SWE
geometric factors already absorb whatever the ACE code was compensating for.
**This needs a question to Ruth Skoug, not a patch.** But it is the single
largest unexplained difference between the document and the code.

.. _swe-gap-uncert-units:

2. Count uncertainty is in the wrong units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: high. A factor of ~12.5 on every L2 uncertainty.**

**[CODE]** ``swe_l1b_science()``:

.. code-block:: python

   counts_stat_uncert = np.sqrt(populated_data["science_data"])

``populated_data["science_data"]`` is the **decompressed counts** - not
deadtime-corrected, not gain-calibrated, and **not divided by the accumulation
time**. Meanwhile the L1B ``science_data`` output is a **count rate**.

At L2, ``calculate_phase_space_density()`` is applied to both with the same
formula, which assumes its input is ``C/tau``. The uncertainty therefore comes
out too small by a factor of ``tau``, i.e. **0.08 for the nominal 80 ms
accumulation - a factor of 12.5**.

The code carries ``# TODO: Update this if SWE like to include deadtime
correciton.``, which suggests the omission of the deadtime correction was
deliberate; it says nothing about the rate conversion. **Confirm with SWE
whether ``counts_stat_uncert`` is meant to be in counts or counts/s before
changing it** - the variable name says counts and the CDF attributes give it no
units, so this may be intended at L1B and simply mishandled at L2.

Also missing: the document's compression is lossy, and at high count rates the
**quantization uncertainty dominates Poisson**. A decompressed value of 24063
is only known to ±512, while ``sqrt(24063)`` is 155. Neither the document nor
the code addresses this.

.. _swe-gap-double-bin:

3. ``flux_stat_uncert`` is binned twice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: high, and unambiguous.**

**[CODE]** ``swe_l2()``, the uncertainty block:

.. code-block:: python

   phase_space_density_uncert = calculate_phase_space_density(...)
   phase_space_density_uncert = put_uncertainty_into_angle_bins(   # binned once
       phase_space_density_uncert, spin_angle_bins_indices
   )
   dataset["psd_stat_uncert"] = ...

   flux_uncert = calculate_flux(
       phase_space_density_uncert,          # <- already binned
       l1b_dataset["esa_energy"].data,      # <- NOT binned
   )
   flux_uncert = put_uncertainty_into_angle_bins(flux_uncert, spin_angle_bins_indices)

``flux_stat_uncert`` is quadrature-summed into angle bins a **second** time,
and the second call multiplies already-binned uncertainties by unbinned
energies. Compare the main flux path, which correctly computes flux from the
**unbinned** phase space density and bins once.

The fix is to compute ``flux_uncert`` from the unbinned
``phase_space_density_uncert`` and bin it once, mirroring the main path.

.. _swe-gap-esa-table:

4. The active ESA table number is never read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium. A latent silent-data-loss bug.**

**[CODE]** ``get_checker_board_pattern()`` and ``get_esa_energy_pattern()``
both take ``esa_table_num: int = 0`` and both are called **without** the
argument:

.. code-block:: python

   checkerboard_pattern = get_checker_board_pattern(esa_lut_files[0])
   esa_energies = get_esa_energy_pattern(esa_lut_files[0])

The packet's actual ``esa_table_num`` field is read only to **filter**:

.. code-block:: python

   science_data = l1a_data_copy["esa_table_num"].data == 0

**[DOC]** section 3.4.1 is explicit that eight tables exist and can be
selected, and that new tables can be uploaded: "while not the baseline plan,
SWE flight software does include the ability to select from 8 onboard look up
tables to define different energy stepping schemes... If another scheme is
used, processing code modifications would consist of minor adjustments in
sorting the measurements by angle and energy step".

Today, if SWE commands table 2 for science, **every packet is dropped at L1B
with only a log line**. The LUT already has all eight tables, and both
functions already accept the parameter - the change is to thread
``esa_table_num`` through, and to replace the ``== 0`` filter with a check
against whichever tables are known to be calibration tables.

There is also a three-way disagreement about which table numbers are valid:
``swe_l1b_science()`` accepts only 0, ``get_esa_dataframe()`` accepts 0 and 1,
and the LUT contains 0-7.

.. _swe-gap-frame:

5. L2 spin angles are in the instrument frame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium. A deliverable-definition question, not a bug.**

**[DOC]** section 3.4.4: "To facilitate comparison with other IMAP instruments,
required for Level 3 processing, the spin phase angles will be calculated in
**despun spacecraft coordinates**." Section 3.4.4 again for the bins: "these
bins will be defined to be 12 degrees wide, **in despun spacecraft
coordinates**." Section 3.4.6: L3 works in DSC because both SWE and MAG are
available there.

**[CODE]** ``swe_l2()`` calls
``get_instrument_spin_phase(..., SpiceFrame.IMAP_SWE)``, which returns
``(spacecraft_spin_phase + instrument_spin_offset) % 1`` - the **instrument**
spin phase. The L2 coordinate is named ``inst_az``, and the CEM angle
coordinate ``inst_el``, so the code is internally consistent and honest about
what it produces.

The two differ by a fixed mounting offset that SPICE already knows. Resolve
with the SWE team which frame the L2 deliverable should be in; do not change it
unilaterally, since it would shift every ``inst_az`` value and rebin the
primary product.

.. _swe-gap-none-dataset:

6. ``swe_l1b`` appends ``None`` when no full cycle is found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: medium. A crash with a confusing message.**

**[CODE]** ``swe_l1b_science()`` returns ``None`` if
``full_cycle_data_indices.size == 0``, logging "No full cycle data found.
Skipping." But the caller does not check:

.. code-block:: python

   if has_science_data:
       science_dataset = swe_l1b_science(dependencies)
       processed_datasets.append(science_dataset)

so ``None`` reaches the CLI, which passes it to ``write_cdf()``. A day with a
short or badly-gapped science file fails with an ``AttributeError`` deep in the
CDF writer rather than the intended "skipping" behavior. Guard the append.

.. _swe-gap-dims:

7. Mislabelled dimensions on three L2 variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Severity: low to medium. Nothing raises, because the sizes coincide.**

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Variable
     - Declared dims
     - What it actually holds
   * - ``inst_az_spin_sector``
     - ``epoch, energy, inst_az``
     - ``(epoch, esa_step, spin_sector)`` - the **unbinned** spin angle of each
       measurement.
   * - ``psd_stat_uncert``
     - ``epoch, esa_step, spin_sector, cem_id``
     - **Binned** data, so ``epoch, energy, inst_az, inst_el``.
   * - ``flux_stat_uncert``
     - ``epoch, esa_step, spin_sector, cem_id``
     - Same.

This passes silently because ``N_ESA_STEPS == 24`` matches the ``energy``
coordinate length and ``N_ANGLE_SECTORS == N_ANGLE_BINS == 30``.

It matters for two reasons. First, a user slicing by ``inst_az`` on
``inst_az_spin_sector`` gets the wrong thing. Second, the CDF attributes make
``psd_stat_uncert`` the ``DELTA_MINUS_VAR``/``DELTA_PLUS_VAR`` of **both**
``phase_space_density`` (binned, ``DEPEND_1: energy``) and
``phase_space_density_spin_sector`` (unbinned, ``DEPEND_1: esa_step``). Only
one of those pairings can be ISTP-correct. The clean fix is a second pair of
uncertainty variables, one binned and one not.

.. _swe-gap-esa-dac:

8. ``HVPS_ESA_DAC`` dual-range conversion is missing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Section 3.4.2.3 and Appendix A: "Please note for ``HVPS_ESA_DAC``,
there will be two different conversion based on whether we are in high range or
low range of the ESA voltage." Low range ``0.007852613 * x``, high range
``0.384617788 * x`` - a factor of 49 apart.

**[CODE]** ``convert_raw_to_eu()`` keys on mnemonic only and has no mechanism
for a state-dependent conversion. The fixture EU CSV has no ``HVPS_ESA_DAC``
row at all (it does have the separate ``HVPS_VESA`` and
``HVPS_VESA_LOW_RANGE`` monitors). Housekeeping only, so no science product is
affected, but an HK plot of ESA DAC will be wrong by 49x whenever the range
does not match whichever single conversion is eventually added.

Also unresolved: the fixture's ``HVPS_ICEM`` coefficient (``0.000064103``)
differs from Appendix A (``0.064103``) by 1000. Probably amps versus
milliamps; ask rather than patch.

.. _swe-gap-calfile:

9. Calibration constants are hard-coded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Both the analyzer constant ``k`` and the geometric factors are said to
be things that "will be"/"can also be" stored in a calibration data file.

**[CODE]** Both live in ``swe_constants.py`` under a shared
``# TODO: add these to instrument status summary``:
``ENERGY_CONVERSION_FACTOR = 4.75`` and ``GEOMETRIC_FACTORS``.

The document flags the geometric factors as "as of October 2025, and may be
refined by further analysis", so an update is likely. Promoting both to an
ancillary file is straightforward and would remove a code release from the loop
whenever calibration is refined. A geometric factor change requires L2
reprocessing only, not L1B.

.. _swe-gap-dead-code:

10. Dead code
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symbol
     - State
   * - ``swe_utils.read_lookup_table()``
     - Reads ``imap_processing/swe/l1b/swe_esa_lookup_table.csv``. **That file
       does not exist in the repository.** Calling it raises
       ``FileNotFoundError``. Superseded by the ``esa-lut`` ancillary file.
   * - ``swe_l1b.get_esa_dataframe()``
     - Its only caller is itself; calls ``read_lookup_table()``, so it cannot
       run either.
   * - ``swe_l1b.filter_full_cycle_data()``
     - Never called. Superseded by ``isel`` on the full-cycle indices.
   * - ``swe_constants.ELECTRON_MASS``
     - Never referenced. The electron mass is already baked into
       ``VELOCITY_CONVERSION_FACTOR`` and ``FLUX_CONVERSION_FACTOR``.
   * - ``process_swe.get_ialirt_energies()``
     - Never called. The I-ALiRT energies are instead listed as
       ``ialirt/utils/constants.py::swe_energy``, and *also* flagged in the ESA
       LUT ``ialirt`` column - three encodings of the same fact. Note the name
       is misleading: it returns ESA **voltages**, not energies.

Smaller deviations, deliberate or benign
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Item
     - Note
   * - Deadtime constant
     - **[DOC]** 1.5e-6 s (with ``dead time = XXXXX sec`` in the C comment and
       a statement that the real value comes from SWE calibration).
       **[CODE]** 360e-9 s. Intended substitution of the measured value. The
       0.1 "arbitrary x10 cutoff" floor is preserved.
   * - Geometric factors for I-ALiRT
     - **[DOC]** The I-ALiRT fragment assigns a different set than its own
       comment header and than ``fspace()``. **[CODE]** uses one shared set.
       The document is internally inconsistent; worth confirming.
   * - ``Cmin`` in the I-ALiRT azimuthal search
     - **[DOC]** ``min(C180, C90)`` where ``C90`` averages all four bins at
       ±90 degrees. **[CODE]** ``min(C180, C_+90, C_-90)`` with the two sides
       kept separate. Slightly more permissive.
   * - Polar search gating
     - **[DOC]** run it "if counterstreaming is not observed" azimuthally.
       **[CODE]** always runs it and takes ``max()``. Logically equivalent.
   * - I-ALiRT accumulation time
     - **[CODE]** hard-codes 80 ms because ``ACQ_DURATION`` is not in the
       I-ALiRT packet. Correct given the telemetry.
   * - I-ALiRT quarter-cycle vs half-cycle
     - **[DOC]** analyse each quarter cycle separately, then combine two for
       the 30 s cadence. **[CODE]** builds one ``(8, 7, 30)`` array per half
       cycle. Equivalent in practice, because each energy is populated from
       exactly one quarter cycle and the peak offsets preserve bin parity.
   * - Negative-count guard
     - **[DOC]** heritage ``fspace()`` sets ``fv = 0`` for negative counts.
       **[CODE]** absent at L2; present in the I-ALiRT
       ``normalize_counts()``.
   * - Last azimuthal step
     - **[DOC]** "it may be necessary to leave out the last azimuthal step of
       each quarter cycle" if the quarter cycle runs long relative to the spin.
       **[CODE]** does not, and emits no diagnostic that would reveal the need.
       Explicitly deferred to flight in the document.
   * - Operating mode
     - Nothing checks HVSCI. Calibration data is excluded by
       ``esa_table_num``, which is a proxy.
   * - Other packets
     - **[DOC]** section 3.4.2.5 says autonomy, static HK, event message and
       memory dump packets should also be decommutated. None are in the XTCE
       or the APID enum.
   * - ``SWE_CEM_RAW`` field count
     - **[DOC]** Appendix describes 7 four-byte count fields. **[CODE]** XTCE
       has 14 - latched and live. Trust the XTCE; Appendix B warns the
       definitions were still moving.
   * - ``n_cycles`` naming
     - In ``swe_l1b_science()``, ``n_cycles`` is the number of **quarter**
       cycles; the code divides by 4 where it means full cycles. Correct but
       easy to misread.

Quality flags
-------------

``SweL1bFlags`` has exactly one SWE-specific bit:

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Flag
     - Bit
     - Set when
   * - ``NONE``
     - -
     - Default.
   * - ``INF``
     - 0
     - Inherited from ``CommonFlags``. **Never set by SWE code.**
   * - ``NEG``
     - 1
     - Inherited from ``CommonFlags``. **Never set by SWE code.**
   * - ``LAST_CAL_INTERVAL``
     - 2
     - Any acquisition time in the full cycle falls after the second-to-last
       in-flight calibration entry, i.e. the factors are the last measured ones
       held constant rather than a true interpolation. **[CODE]** only; not in
       the document.

Candidates for additional flags, all currently unused telemetry:
``SPIN_PERIOD_VALIDITY``, ``SPIN_PHASE_VALIDITY``, ``SPIN_PERIOD_SOURCE``,
``CEM_NOMINAL_ONLY``, ``HIGH_COUNT``, ``REPOINT_WARNING``, ``STIM_ENABLED``,
and ``CKSUM``. All are parsed and carried through to L1B as metadata; **none
influences processing or sets a bit.** ``HIGH_COUNT`` and ``STIM_ENABLED`` in
particular look like they were meant to.

Not implemented at all
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Note
   * - Checksum verification
     - ``CKSUM`` is parsed and carried in every product; nothing checks it.
       Would need raw packet bytes, which ``packet_file_to_datasets`` does not
       return.
   * - Quicklook
     - No SWE quicklook code.
   * - Compression quantization uncertainty
     - See :ref:`swe-gap-uncert-units`.
   * - Energy uncertainty
     - No uncertainty is attached to ``esa_energy`` or the derived energies.
   * - In-flight calibration analysis
     - **[DOC]** explicitly a ground/science-team activity, not the SDC's. The
       SDC's only job is to keep it out of L1B+, which is done.
   * - L3
     - Correctly out of scope. See :ref:`swe-l3-scope`.
