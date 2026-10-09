.. _swe-overview:

Instrument and Measurement Concepts
===================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is background. No code depends on it directly, but
almost every design decision in :ref:`swe-l1` and :ref:`swe-l2` only makes
sense once you have it.

What SWE measures
-----------------

**[DOC]** SWE measures the **3D distribution of solar wind thermal and
suprathermal electrons from 1 eV to 5 keV**. It is a heritage design, closely
following NASA's Ulysses/SWOOPS, ACE/SWEPAM and Genesis/GEM solar wind electron
instruments.

Electrons are the diagnostic of magnetic field topology in the solar wind.
Their pitch angle distribution tells you whether the local field line is
connected to the Sun at one end (unidirectional strahl), both ends
(counterstreaming - typically a coronal mass ejection), or disconnected. That
is why the two headline SWE science products are a **pitch angle distribution**
and a **bidirectional-electron flag**, and why SWE needs MAG data to be useful
at L3.

The sensor
----------

**[DOC]** The Sensor Head (SH) is a **spherical-section electrostatic analyzer
(ESA)** followed by **seven channel electron multiplier (CEM) detectors**.

* Electrons enter through an aperture oriented **normal to the spacecraft spin
  axis**.
* A **positive high voltage on the inner ESA plate** admits only electrons in a
  narrow band of energy and azimuthal angle. Stepping that voltage sweeps the
  energy range.
* Electrons arriving at different **polar** angles land on different CEMs,
  giving **21-degree polar resolution** across a fan-shaped field of view.
* As the spacecraft spins, the fan sweeps out **>95% of 4π steradians**,
  missing only small conical holes centered on the spin axis (parallel and
  antiparallel).

The consequence for the code: **polar angle is a property of the detector
index, not something you compute.** The central detector looks radially
outward, perpendicular to the spin axis, and the others are at nominally
±21, ±42 and ±63 degrees from it.

**[CODE]** ``swe_constants.CEM_DETECTORS_ANGLE = [-63, -42, -21, 0, 21, 42, 63]``,
which becomes the L2 ``inst_el`` coordinate. Note the ordering: index 0 is the
-63 degree detector and index 6 is +63. The algorithm document numbers the CEMs
1-7; the code indexes them 0-6 as ``cem_id``.

Electronics worth knowing about
-------------------------------

**[DOC]** SWE uses the IMAP Common Electronics (ICE) EBOX. Two things in it
matter for data processing:

* **HVPS.** Two independent supplies: the ESA supply (+1200 V, **dual range**)
  and the CEM bias supply (up to +4200 V, **commandable**, nominally +2800 V at
  start of mission). The ESA supply is stepped every ~83.333 ms.

  The dual-range ESA supply is why ``HVPS_ESA_DAC`` has **two different raw-to-
  engineering conversions** depending on whether the instrument is in low or
  high range (see :ref:`swe-ancillary`).

  The commandable CEM bias is why the **in-flight gain calibration** exists:
  as the CEMs age their gain drops, the weekly calibration sequence measures
  how far off nominal they are, and the bias is occasionally stepped up.

* **CDH.** Collects science data into memory, controls the HVPS, generates
  telemetry packets, and runs the ESA stepping tables from MRAM.

Spacecraft time
---------------

**[DOC]** SWE's FPGA has a **32-bit MET coarse counter** (whole seconds of
Mission Elapsed Time) and a **20-bit MET fine counter** (microseconds within
the second). The spacecraft delivers a time-and-status packet every second
carrying the SCLK of the next 1PPS; SWE loads it on the 1PPS edge and zeroes
the fine counter. If the packet does not arrive, the counters free-run.

Both counters tag science telemetry as ``ACQ_START_COARSE`` (seconds) and
``ACQ_START_FINE`` (microseconds). The 1PPS also paces FSW operations such as
telemetry generation.

Operating modes
---------------

**[DOC]** Six FSW modes, two in the boot FSW and four in the App FSW:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Mode
     - What it is
   * - Boot
     - Boot FSW. Always transitions to LVENG next.
   * - LVENG
     - Low voltage engineering. LV monitors and engineering HK; memory
       upload/dump/diagnostics allowed. **Safing from any mode lands here.**
   * - LVSCI
     - Low voltage science. End-to-end science data acquisition driven by a
       **stim pulse** into the FEE, rather than by real electrons.
   * - HVENG
     - High voltage engineering. High voltages set to fixed levels by ground
       command. Used heavily in ground test and during commissioning ramp-ups.
   * - HVSCI
     - **High voltage science. The only mode that produces science data.** ESA
       voltages are stepped by FSW; CEM counters accumulate at each step.
       Transitions out of HVSCI only happen after a complete 15-second quarter
       cycle.

Engineering modes (LVENG/HVENG) produce ``SWE_CEM_RAW`` packets: raw CEM counts
every 1 second, no compression, used for ground test and commissioning.

.. note::

   **[CODE]** Nothing in ``imap_processing/swe`` checks the operating mode. The
   L1B filter that separates science from calibration data keys on
   ``esa_table_num``, not on mode. See :ref:`swe-implementation-status`.

The measurement cycle
---------------------

This is the part to get right. Everything about the SWE data layout follows
from it.

**[DOC]** The ESA level is updated nominally every **83.333 ms**, which
corresponds to **2 degrees of spin** at a nominal 15-second spin period. The
83.333 ms is composed of:

.. code-block:: text

   SETTLE_DURATION  (nominally  3.333 ms)   high voltage settling, no counting
   ACQ_DURATION     (nominally 80.000 ms)   counters accumulate
   -------------------------------------
   step period      (nominally 83.333 ms)

Both are telemetered per packet **in microseconds** and are commandable in
flight.

**Spin-angle bin.** Six consecutive ESA steps take 0.5 seconds and cover
12 degrees of spin. Those six measurements are treated as belonging to one
**spin-angle bin**. A 15-second quarter cycle therefore contains 30 spin-angle
bins of 12 degrees each.

**Quarter cycle.** One quarter cycle is:

.. code-block:: text

   15 seconds x 12 ESA steps per second = 180 measurements
   180 measurements x 7 CEMs            = 1260 counter values
   1260 counter values, 8-bit compressed = 1260 bytes = the SCIENCE_DATA field

Nominally one quarter cycle is one spacecraft spin, but **SWE is not synced to
the spin**. Measurements are time-based, paced off the 1PPS. The quarter cycle
length is configurable and will be set **slightly longer than a spin period**
so that every energy step gets full angular coverage. IMAP's spin rate is
planned to be 3.9-4.1 RPM, i.e. a period of **14.6-15.4 seconds**.

.. note::

   The algorithm document writes this rate as "3.9 - 4.1 Hz". That is a slip -
   4 Hz would be a 0.25 s period, not the 14.6-15.4 s stated in the same
   sentence. Read it as RPM.

**Full cycle.** A SWE measurement covers **24 energies × 30 spin angles**. Only
a subset of those energy-angle bins is measured in each quarter cycle:

* In each quarter cycle, **6 of the 24 ESA levels** are measured at each spin
  angle.
* **Odd-numbered spin-angle bins get one set of 6 levels; even-numbered bins
  get a different set of 6.**
* Over four quarter cycles, all 24 ESA levels are measured at all 30 spin
  angles.

So the **full cycle** - four quarter cycles, nominally one minute - is the
smallest unit that is a complete measurement, and it is the L1B and L2 record.

**[DOC]** The 720 ESA steps used in a full cycle are precomputed on board
before acquisition begins, from a 24-element ESA table stored in CDH MRAM.
**Eight such tables** can be stored; one is selected before acquisition, and
new tables can be uploaded. The processing here assumes the nominal scheme.

.. _swe-checkerboard-concept:

The checkerboard
----------------

The odd/even alternation above is why the SWE code talks about a
**"checkerboard pattern"**. Picture the (24 energies × 30 spin angles) grid you
want to end up with. In a single quarter cycle you fill only a scattered
subset of its cells; the pattern of filled cells alternates between adjacent
columns, and four quarter cycles' patterns interlock to fill the grid exactly
once.

**[CODE]** ``swe_l1b.get_checker_board_pattern()`` builds a (24, 30) integer
array whose value at ``[energy_row, spin_column]`` is the index into the flat
720-element quarter-cycle-concatenated measurement array that belongs there.
``swe_l1b.populated_data_in_checkerboard_pattern()`` then applies it with fancy
indexing. The mapping is read from the **ESA LUT ancillary file**, not
hard-coded - see :ref:`swe-ancillary`.

The important property: **the checkerboard is a lossless permutation.** All 720
measurements of a full cycle land in the (24, 30) grid, exactly one per cell,
and nothing is averaged or dropped. That is why L1B can carry the original
per-measurement acquisition times alongside the counts.

Heritage, and what it implies
-----------------------------

**[DOC]** Section 3.4.1 is explicit that SWE processing is based on the
ACE/SWEPAM C codes (themselves descended from Ulysses/SWOOPS Fortran), because
SWEPAM is the only one of the three heritage instruments still operating.

Two differences from heritage that matter:

1. **Energy steps went from 20 to 24**, and the stepping scheme changed.
2. **Heritage instruments stepped contiguous energies per spin** (lowest *n*
   energies in spin 1, next *n* in spin 2, ...). **SWE covers the full energy
   range in every spin** at reduced energy and angle resolution.

Consequence: the SWEPAM code that *combines* measurements into a full
distribution had to be rewritten for SWE (that is the checkerboard), but the
code that *processes* a complete distribution - deadtime, gain calibration,
phase space density, moments - is taken essentially verbatim from heritage. The
algorithm document reproduces those heritage C fragments directly, which is why
they appear in these pages as C rather than as equations.

Vocabulary
----------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Term
     - Meaning
   * - **CEM**
     - Channel electron multiplier. Seven of them, at fixed polar angles.
       ``cem_id`` 0-6 in code; CEM 1-7 in the document.
   * - **ESA step**
     - One setting of the ESA plate voltage, held for
       ``SETTLE_DURATION + ACQ_DURATION``. 180 per quarter cycle, 720 per full
       cycle, drawn from 24 distinct voltages.
   * - **ESA level / ESA voltage**
     - One of the 24 distinct voltages in the active ESA table. Maps to a
       particle energy through the analyzer constant *k*.
   * - **Quarter cycle**
     - One ``SWE_SCIENCE`` packet. 15 s, 180 ESA steps, 1260 compressed bytes.
       ``QUARTER_CYCLE`` in telemetry counts 0-3.
   * - **Full cycle**
     - Four consecutive quarter cycles with ``QUARTER_CYCLE`` 0,1,2,3. ~1
       minute. The L1B and L2 record.
   * - **Spin sector**
     - In L1A, the index 0-179 of a measurement within a quarter cycle. In L1B
       and L2, the index 0-29 of a column of the checkerboard grid. **The same
       word is used for both; check the dimension size.**
   * - **Spin angle bin**
     - One of 30 fixed 12-degree-wide bins in *physical* spin angle, centered
       at 6, 18, ... 354 degrees. Produced at L2 by looking up each
       measurement's actual spin angle from SPICE. ``inst_az`` in the L2 CDF.
   * - **Checkerboard**
     - The (24, 30) reorganization of a full cycle's 720 measurements. See
       :ref:`swe-checkerboard-concept`.
   * - **Deadtime**
     - Detector recovery time after a count, during which arrivals are missed.
       Corrected with a non-paralyzable model.
   * - **In-flight calibration / gain sweep**
     - Weekly sequence that steps CEM bias around nominal to measure gain
       degradation. Produces the multiplicative per-CEM factors applied at
       L1B.
   * - **Stim pulse**
     - Electronic pulser into the FEE preamps, used to exercise the chain
       without real electrons. ``STIM_ENABLED`` / ``STIM_CFG_REG`` in
       telemetry.
   * - **BDE**
     - Bidirectional electrons. The I-ALiRT flag: 1 = counterstreaming,
       0 = nominal unidirectional flow.

Reference frames
----------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Frame
     - Where it appears
   * - **Instrument (IMAP_SWE)**
     - **[CODE]** What L2 actually produces. ``inst_az`` is the SWE spin angle
       from ``get_instrument_spin_phase(..., SpiceFrame.IMAP_SWE)``;
       ``inst_el`` is the fixed CEM polar angle.
   * - **Despun spacecraft (DSC)**
     - **[DOC]** What section 3.4.4 asks for at L2 - "to facilitate comparison
       with other IMAP instruments, required for Level 3 processing, the spin
       phase angles will be calculated in despun spacecraft coordinates" - and
       what L3 uses to combine SWE with MAG. See
       :ref:`swe-implementation-status` for the discrepancy.
   * - **RTN**
     - **[DOC]** L3 only. All moments are rotated to RTN.
   * - **Field-aligned**
     - **[DOC]** L3 only. Temperatures are additionally rotated to
       parallel/perpendicular to **B** from MAG.
