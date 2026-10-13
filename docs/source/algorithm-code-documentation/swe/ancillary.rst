.. _swe-ancillary:

Ancillary Files, Calibration and External Dependencies
======================================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

What the pipeline actually reads
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 12 16 52

   * - Descriptor
     - Level
     - Format
     - Consumed by
   * - ``esa-lut``
     - L1B
     - CSV
     - ``get_checker_board_pattern()``, ``get_esa_energy_pattern()``
   * - ``l1b-in-flight-cal``
     - L1B, I-ALiRT
     - CSV
     - ``read_in_flight_cal_data()``
   * - ``eu-conversion``
     - L1B
     - CSV
     - ``convert_raw_to_eu()``
   * - SPICE kernels
     - L1A, L1B, L2
     - SPICE
     - ``met_to_ttj2000ns``, ``get_instrument_spin_phase``

Representative copies of all three CSVs live in
``imap_processing/tests/swe/lut/``. They are test fixtures, not the operational
files, but the **column names and shapes are the contract** and the code will
break on anything else.

.. note::

   SWE also delivers a fourth ancillary file under the ``config`` descriptor.
   It holds the tunable constants of the **L3** algorithms - geometric
   fractions, pitch angle, gyrophase and energy bin definitions,
   ``in_vs_out_energy_index``, ``core_halo_breakpoint_initial_guess`` and about
   a dozen similar values. **Nothing in this repository reads it**, because SWE
   L3 is produced elsewhere (:ref:`swe-l3-scope`). It is mentioned here only so
   that finding it in a dependency list or an archive manifest does not look
   like a missing L1B or L2 input.

.. _swe-esa-lut:

ESA lookup table (``esa-lut``)
------------------------------

Fixture: ``imap_swe_esa-lut_20250301_v000.csv``, 384 data rows.

This single file encodes the entire onboard ESA stepping scheme, and it is what
makes the checkerboard data-driven rather than hard-coded.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Column
     - Meaning
   * - ``table_idx``
     - Which onboard table, **0-7**. This matches the document's statement that
       "SWE flight software does include the ability to select from 8 onboard
       look up tables". 48 rows each.
   * - ``esa_step``
     - Step index within the full cycle. Only the **first 12 steps of each
       quarter cycle** are listed - 0-11, 180-191, 360-371, 540-551 - because
       the 12-step pattern repeats through the remaining 168 steps of each
       quarter cycle.
   * - ``esa_v``
     - ESA plate voltage in volts for that step.
   * - ``v_index``
     - Which of the 24 energy rows, **1-24** (the code subtracts 1).
   * - ``ialirt``
     - 1 if this step is one of the eight downlinked in real time.

What the tables contain in the fixture:

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - ``table_idx``
     - ``ialirt`` rows
     - Contents
   * - 0
     - 16
     - **The nominal science table.** All 24 voltages, 0.56 V to 1108.66 V, in
       the interleaved even/odd arrangement. 16 flagged rows = 8 energies,
       each appearing twice in the 48-row block.
   * - 1
     - 0
     - **A calibration (gain sweep) table.** All 48 rows hold the single
       voltage 5.64 V at ``v_index`` 8 - exactly the fixed-ESA configuration
       the weekly gain sweep needs.
   * - 2-7
     - 0
     - Additional tables. Not exercised by any test.

.. important::

   **[CODE]** The relationship "``esa_table_num == 0`` means science, anything
   else means calibration" is what L1B filters on, and it is only true because
   of how the operational LUT is populated. Two places disagree about how many
   tables are legitimate:

   * ``swe_l1b_science()`` keeps only ``esa_table_num == 0``.
   * ``get_esa_dataframe()`` raises ``ValueError`` for anything outside
     ``[0, 1]``.
   * ``get_checker_board_pattern()`` and ``get_esa_energy_pattern()`` default
     to ``esa_table_num=0`` and are **always called with the default** - the
     packet's actual ``esa_table_num`` is never passed through.

   If SWE ever commands a different science table, L1B will silently drop every
   packet. See :ref:`swe-implementation-status`.

In-flight calibration (``l1b-in-flight-cal``)
---------------------------------------------

Fixture: ``imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv``.

.. code-block:: text

   met_time,cem1,cem2,cem3,cem4,cem5,cem6,cem7
   453050308,1,1,1,1,1,1,1
   553051294,1,1,1,1,1,1,1
   1782864000,2,2,2,2,2,2,2

One row per weekly gain sweep: a MET timestamp followed by seven multiplicative
factors, one per CEM, applied to counts to correct for gain degradation.

**[DOC]** section 3.3.4 specifies:

* Filename ``imap_swe_l1b-in-flight-cal_YYYYMMDD_vXXX.csv`` where ``YYYYMMDD``
  is the date from which the file should first be used. Version nominally 001.
* All factors are **1.0 for times before commissioning**.
* When a new calibration point is added, the **filename date is set to the date
  of the previous calibration** - because linear interpolation only becomes
  possible back to that point once the new point exists. This is what drives
  reprocessing.
* The file carries a **trailing row with a far-future timestamp** repeating the
  most recent factors, so quicklook processing can run before the next
  calibration. (The fixture's third row, MET 1782864000, is that padding row.)
* **[DOC]** Section 3.4.3: factors come from comparing counts at the operating
  CEM level (step 2 of the gain run) to counts at the next higher level (step
  3), assuming the higher level is correct.

**[CODE]** ``read_in_flight_cal_data()`` accepts a list of files, concatenates,
drops rows with no MET, sorts, and de-duplicates on MET keeping the last. The
interpolation and the ``LAST_CAL_INTERVAL`` flag are described in
:ref:`swe-l1`.

.. note::

   **[DOC]** The same factors are used for I-ALiRT, but I-ALiRT "will
   necessarily use the most recent calibration factors available, as
   interpolation between data points is of course not possible for real time
   analysis". **[CODE]** ``process_swe.py`` does exactly that: a
   ``searchsorted(..., side="right") - 1`` to pick the last row at or before
   the group midpoint, with no interpolation.

EU conversion table (``eu-conversion``)
---------------------------------------

Fixture: ``imap_swe_eu-conversion_20240510_v000.csv``, 63 rows. Standard IMAP
format consumed by ``convert_raw_to_eu()``: ``packetName``, ``mnemonic``,
``convertAs`` (all ``UNSEGMENTED_POLY``), and coefficients ``c0``-``c7``.

**[DOC]** Appendix A gives the algebraic expressions. Science packet:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Mnemonic
     - Conversion
   * - ``SPIN_PHASE``
     - ``0.005493 * x``
   * - ``SPIN_PERIOD``
     - ``0.00032 * x``
   * - ``THRESHOLD_DAC``
     - ``0.001221 * x``

App housekeeping packet (selection - the full set is in the CSV):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Mnemonic
     - Conversion
   * - ``HVPS_CEM_DAC``
     - ``1.025641 * x``
   * - ``HVPS_VBULK``
     - ``0.51282 * x``
   * - ``HVPS_VCEM``
     - ``1.34616 * x``
   * - ``HVPS_VESA``
     - ``0.38462 * x``
   * - ``HVPS_VESA_LOW_RANGE``
     - ``0.0078526 * x``
   * - ``HVPS_ICEM``
     - ``0.064103 * x`` **[DOC]** / ``0.000064103 * x`` **[CODE]**
   * - ``FEE_TEMP``, ``SENSOR_TEMP``, ``HVPS_TEMP``, ``CDH_*_TEMP``
     - 6th- or 5th-order polynomials (see Appendix A / the CSV)
   * - ``LVPS_*_BOARD_TEMP``
     - ``-273.2 + 0.1444619083 * x``
   * - ``LVPS_*_VMON`` / ``IMON``, ``CDH_*_VMON``
     - Linear, some with negative slopes for the negative rails
   * - ``HVPS_ESA_DAC``
     - **Two conversions.** Low range ``0.007852613 * x``; high range
       ``0.384617788 * x``.

.. warning::

   **[DOC]** "Please note for ``HVPS_ESA_DAC``, there will be two different
   conversion based on whether we are in high range or low range of the ESA
   voltage."

   **[CODE]** ``convert_raw_to_eu()`` selects on ``mnemonic`` alone; there is
   no range-dependent branch and no ``HVPS_ESA_DAC`` row in the fixture CSV.
   The dual-range handling is not implemented. See
   :ref:`swe-implementation-status`.

   Separately, ``HVPS_ICEM`` in the fixture differs from the document by a
   factor of 1000 (amps versus milliamps, most likely). Worth a question to
   the SWE team rather than a unilateral edit.

Calibration constants that are **not** ancillary files
------------------------------------------------------

Two quantities the algorithm document says "will be stored in a calibration
data file" are instead hard-coded in
``imap_processing/swe/utils/swe_constants.py``, both under a shared
``# TODO: add these to instrument status summary``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constant
     - Status
   * - ``ENERGY_CONVERSION_FACTOR = 4.75``
     - The analyzer constant ``k = E/V``. **[DOC]** "determined from ground
       calibration. This value will be stored in a calibration data file, where
       it can be read by the processing code."
   * - ``GEOMETRIC_FACTORS``
     - ``[424.4, 564.5, 763.8, 916.9, 792.0, 667.7, 425.2] * 1e-6``
       cm^2 sr eV/eV. **[DOC]** "nominal SWE geometric factors ... as of
       October 2025, and may be refined by further analysis"; "geometric
       factors can also be stored in a calibration data file to be read by the
       processing routine."

Changing either one changes every L2 number, so promoting them to an ancillary
file is a real (and probably inevitable) piece of work. Note that a geometric
factor update would **not** require reprocessing L1B - the time-varying part of
the detector response is handled entirely by the in-flight calibration at L1B.

Constants worth knowing
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Name
     - Value
     - Meaning
   * - ``N_ESA_STEPS``
     - 24
     - Distinct ESA voltages / energies per full cycle.
   * - ``N_ANGLE_SECTORS`` / ``N_ANGLE_BINS``
     - 30
     - Spin sectors (L1B) and 12-degree angle bins (L2). Same number, different
       meanings.
   * - ``N_CEMS``
     - 7
     - Detectors.
   * - ``N_QUARTER_CYCLES``
     - 4
     - Packets per full cycle.
   * - ``N_QUARTER_CYCLE_STEPS``
     - 180
     - Measurements per packet.
   * - ``ENERGY_CONVERSION_FACTOR``
     - 4.75
     - Analyzer constant k.
   * - ``VELOCITY_CONVERSION_FACTOR``
     - 1.237e31
     - ``v^4 / E^2`` for electrons, cm/s and eV.
   * - ``FLUX_CONVERSION_FACTOR``
     - 6.187e30
     - ``j / (fv * E)``. Exactly half of the above.
   * - ``ELECTRON_MASS``
     - 9.10938356e-31 kg
     - Defined but **not referenced anywhere** - the two conversion factors
       above already have it baked in.
   * - ``CEM_DETECTORS_ANGLE``
     - -63 ... +63
     - Polar angle per CEM.
   * - deadtime
     - 360e-9 s
     - Local to ``deadtime_correction()``. **[DOC]** shows 1.5e-6 as a
       placeholder.
   * - BDE threshold
     - 1.75
     - Default in ``determine_streaming()``. **[DOC]** initial value from
       Genesis, to be refined in flight.
   * - BDE minimum steps
     - 3 of 8
     - Default in ``compute_bidirectional()``. **[DOC]** initial value.

External dependencies
---------------------

SPICE / spin
^^^^^^^^^^^^

The only hard external dependency below L3.

* ``imap_processing.spice.time.met_to_ttj2000ns`` - L1B epoch, I-ALiRT epoch.
* ``imap_processing.spice.spin.get_instrument_spin_phase`` and ``get_spin_angle``
  - L2 spin angles. The CLI declares "spin data" as the second L2 dependency.

MAG and SWAPI
^^^^^^^^^^^^^

**[DOC]** L3 needs the MAG field vector (pitch angle, and the field-aligned
temperature rotation) and SWAPI ion velocity, density and temperature (solar
wind frame transformation, spacecraft potential, and the assumption that the
bulk electron velocity equals the proton velocity).

**Neither is a dependency of anything in this repository.** See
:ref:`swe-l3-scope`.

Ultra deflector state
^^^^^^^^^^^^^^^^^^^^^

**[DOC]** An unusual cross-instrument coupling: the L3 break-point finding
algorithm "has been optimized for times when the Ultra deflector voltage is at
the nominal 3500 V level. The algorithm may fail at times when the Ultra
deflectors are turned off, and a flag is added to the L3 data to indicate these
times." L3's problem, but worth knowing the coupling exists.
