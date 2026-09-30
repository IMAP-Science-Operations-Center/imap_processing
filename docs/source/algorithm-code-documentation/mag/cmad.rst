.. _mag-cmad:

Upstream Calibration and Cleaning (CMAD)
========================================

This page summarises what the public **IMAP Calibration and Measurement
Algorithms Document (CMAD)** says about MAG, and how that relates to the
algorithm document the rest of these pages are built on.

Read it when you need to know **where the L2 offsets and calibration matrices
come from**, what the quality bitmask bits mean, or what artifacts remain in
released L2 data. None of the processing described here runs in this
repository; it runs at Imperial College London and its results reach the SDC as
the ``l2-calibration`` and ``l2-{norm,burst}-offsets`` ancillary files (see
:ref:`mag-ancillary`).

The algorithm document is not in the CMAD
-----------------------------------------

For SWAPI, CoDICE, HIT, SWE, IDEX and GLOWS, the CMAD embeds the instrument's
own SDC-facing algorithm document, sometimes at a newer revision. **MAG is
different.** IMAP-MAG-SW-009 (the *MAG Science Algorithm Document*) is not
embedded or cited anywhere in the CMAD. Its L0 to L1D and I-ALiRT procedures,
the compression appendix, and the calibration and offset file formats do not
appear in it at all. (IMAP-Lo is partly similar: the CMAD embeds only its
Mapping Algorithms appendix, not the main Data Product Algorithms document.)

Instead, CMAD section 4.4 (*Magnetic Fields*) embeds a different Imperial College
technical note, **IMAP-OPS-TN-ICL-013, *IMAP MAG Data Cleaning Processes***. Its
change log says Issue 3 was a "focused version to be used as input for CMAD" and
Issue 4 was "removal of non-relevant sections for CMAD", so the substitution was
deliberate on the MAG team's part.

The CMAD does not say why. The most likely reason is that the two documents
serve different readers:

* SW-009 exists "to completely recreate the data processing pipeline". It is an
  implementation specification for the SDC.
* The CMAD exists to "enable any user to fully understand the scientific data
  they are using". For MAG, what determines L2 science quality is the cleaning
  and offset determination done upstream at Imperial. The SDC's part is to apply
  the result.

The section 4.4 heading still reads "MAG data product definitions and processing
algorithms are embedded below", the boilerplate used for every instrument. The
CMAD is also marked *preliminary* and has open placeholders (for example "Include
details of I-ALiRT data here"). SW-009 may be added in a later version, but that
is speculation.

.. important::

   **These pages remain the only description of the SDC pipeline that is tied
   to the code.** The CMAD adds context from upstream of the pipeline; it
   neither replaces nor contradicts the L1A to L1D procedures in SW-009. Where
   the two documents disagree about L2, it is listed in
   :ref:`mag-cmad-vs-sw009` below.

Where MAG appears in the CMAD
-----------------------------

``IMAP_CMAD_20260722.pdf`` (version 1.1, preliminary). Printed page numbers are
one less than the PDF page index; PDF pages are given in parentheses.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - CMAD section
     - Printed (PDF) pages
     - Content
   * - 2.5 Instrument description
     - 6 (7)
     - One paragraph: two fluxgates on a 2.5 m boom for gradiometry, magnetic
       cleanliness programme. Defers to Horbury et al. [2025].
   * - 3.5.1 Strategy & Approach
     - 151 (152)
     - Ground calibration at Magnetsrode; in-flight calibration using the spin;
       cross-calibration with ACE, Wind, DSCOVR, Aditya-L1 and a co-launched
       mission.
   * - 3.5.2 Pre-Flight Instrument Calibration
     - 152-163 (153-164)
     - Embeds **IMAP-OPS-TN-ICL-017, *IMAP MAG Calibration Inputs
       Description***, Issue 2, 4 June 2026. Despite the section heading, this
       is entirely **in-flight** calibration. Section 3.5.3 (In-Flight) is an
       empty placeholder, and there is no pre-flight calibration report.
   * - 4.1.1 CDF data file contents
     - ~378-380, ~502-516
     - Variable listings for the ten L2 products,
       ``imap_mag_l2_{burst,norm}-{dsrf,gse,gsm,rtn,srf}``. No L1 products.
   * - 4.4 Magnetic Fields
     - 912-929 (913-930)
     - Embeds **IMAP-OPS-TN-ICL-013, *IMAP MAG Data Cleaning Processes***,
       Issue 4, 4 June 2026.
   * - 5.4.5 Data usage caveats, MAG
     - 1163-1166 (1164-1167)
     - **The authoritative quality flag and bitmask definitions**, plus
       descriptions of the remaining artifacts and their science impacts.

Both technical notes apply to **Imperial calibration code v2.2.0**
(``ImperialCollegeLondon/IMAP_MAG_Calibration`` on GitHub) and to **Data Release
1**, covering 1 January to 29 April 2026, released August 2026. Numbers quoted
below are for that release and **will change** in later ones.

.. _mag-cmad-chain:

The L2 offset production chain
------------------------------

**[CMAD]** ICL-013 section 2. The Imperial wrapper script
``calibrate_L2_offsets.m`` loads housekeeping and configuration, then runs, in
order:

1. **Apply the calibration matrices** (see :ref:`mag-cmad-matrices`).
2. **Remove the IMAP-Lo pivot platform** signal.
3. **Remove the Ultra decontamination heater and Hi-45/Hi-90** signals.
4. **Apply the spin-plane offset** correction (a baseline plus a
   spin-tone-reducing optimisation).
5. **Apply the spin-axis offset** correction (a solar-wind technique).
6. **Clean thruster and pre-thruster** signals.
7. **Output one offset per vector** of the input L1C (normal) or L1B (burst)
   file. This is the ``l2-{norm,burst}-offsets`` file that :ref:`mag-l2`
   consumes.

Steps 2, 3 and 6 are *cleaning*: they remove signals generated by the spacecraft
and other instruments. Steps 4 and 5 are what the MAG team calls *calibration*:
they estimate the sensor and static spacecraft offsets. Everything is folded
into the single per-vector offset, so the SDC cannot separate the contributions.

.. note::

   Because the offset is the sum of all of these corrections, a single L2
   ``FILLVAL`` vector can come from any of them. The one **known** source is a
   bug in burst-mode thruster cleaning (see :ref:`mag-cmad-thrusters`). The
   SDC's ``MagL2.apply_offsets`` is doing the right thing when it turns these
   vectors into ``FILLVAL``.

.. _mag-cmad-matrices:

Calibration matrices
--------------------

**[CMAD]** ICL-017 section 3.

Release 1 uses **CalibrationMatricesV9**, defined relative to the
``IMAP_MAG_BASE`` SPICE frame. That matches the design in :ref:`mag-overview`:
the calibration matrix takes each sensor into one idealised frame. The
parameterisation per sensor is three polar angles (Theta 1-3), three azimuths
(Phi 1-3) and two relative gains (Gain 1-2). This is presumably what arrives
here as ``URFTOORFO``/``URFTOORFI`` in ``l2-calibration``, but the CMAD does not
describe the delivered file.

How the angles were determined:

* The **19-20 January 2026 CME** provided a long high-field interval, which
  separates the angular contribution to spin tone from the offset contribution.
* **Theta 3 and Phi 3** (spin-axis alignment) minimise the spin tone in the DSRF
  Z axis after applying the matrix and despinning.
* **Theta 1 and Theta 2** minimise the spin tone in the DSRF spin plane, and are
  checked by recomputing the angles with the Kepko method (agreement within
  0.1 degrees overall, 0.01 degrees in the quietest window).
* **Phi 2** (the X-Y angle) minimises the **second harmonic** of the spin tone
  over 3-minute windows during the CME, as a weighted mean.
* MAGi angles are then further tuned to **co-align MAGi with MAGo** over the
  CME, so that gradiometry works.

.. note::

   The V9 gains are **not exactly 1** (they differ from unity by ~0.1%). SW-009
   and :ref:`mag-overview` say the in-flight gain and orthogonality corrections
   stay at the identity "until there is evidence to justify a change". That
   evidence has now arrived.

.. _mag-cmad-offsets:

Offsets
-------

**[CMAD]** ICL-017 sections 4-5. There are two techniques, one per axis group.
Both work in the sensor frame after the calibration matrix is applied, with Z
along the spin axis.

Spin-plane offsets (X, Y): Kepko method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Baseline:** rolling Kepko offsets over an **8-hour window**, with the angles
   fixed from the calibration matrix, computed **after** pivot-platform removal.
2. **Optimisation:** for each day, offsets are fitted on 3-hour intervals to
   **minimise spin tone in** :math:`|B|`, applied by linear interpolation, and
   checked hourly. Where spin tone is still too high, the fit is repeated on
   progressively shorter intervals, **down to 6 minutes**. This runs after
   pivot-platform removal, Hi/Ultra removal and baseline application.
3. **Alternative optimiser:** on a few days, minimising spin tone in :math:`|B|` removed
   real solar-wind power at the spin frequency and injected spin tone into the
   components. On those days the optimiser minimises **component-level** spin
   tone over longer intervals instead.

Inputs are version-controlled CSVs, one per day per sensor, listed in
``calibration_input_release1_v001.json``.

Spin-axis offset (Z): Leinweber method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Leinweber solar-wind technique produces **one offset per day** plus the
fraction of the day that was usable. Days with less than 5% usable data are
dropped, outliers beyond 1.5 standard deviations of a moving mean are
excluded, and the final value comes from a **linear fit** across the period.
Four CSVs per sensor cover Release 1.

.. important::

   SW-009 (and older text in these pages) says the offsets are determined "with
   the Leinweber method". In practice **Leinweber is used only for the spin
   axis**; the spin plane uses Kepko plus spin-tone optimisation.

Typical magnitudes
^^^^^^^^^^^^^^^^^^

These are useful as a sanity check when an offsets file looks wrong. They are
Release 1 means over 1 January to 29 April 2026 (ICL-017 tables 1-2); the standard
deviations are all 0.21 nT or less.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Sensor
     - X (spin plane)
     - Y (spin plane)
     - Z (spin axis)
   * - MAGo
     - +0.83 nT
     - -5.69 nT
     - +0.97 nT
   * - MAGi
     - +6.88 nT
     - -8.60 nT
     - +1.99 nT

The larger MAGi values are expected: MAGi is closer to the spacecraft.

Cleaning processes
------------------

**[CMAD]** ICL-013 section 3. All three use **housekeeping from the spacecraft
or other instruments** as the trigger, and most of them use **gradiometry**
(scaling the MAGi - MAGo difference and subtracting it from MAGo).

.. warning::

   The gradiometer factors below are **Imperial's L2 cleaning parameters**. They
   are unrelated to the kappa matrix in the ``l1d-calibration`` file that this
   repository applies at :ref:`mag-l1d`, even though both are called "kappa" or
   "gradiometer factor".

IMAP-Lo pivot platform
^^^^^^^^^^^^^^^^^^^^^^

The IMAP-Lo pivot platform had three set points (75, 90 and 105 degrees) and
moved up to once a day. Each position produces a different static field at both
sensors, larger at MAGi.

* **Cleaning:** per-axis, per-sensor deltas relative to the 90 degree position
  (zero by construction), derived from March-April 2026 movements, are applied
  according to IMAP-Lo housekeeping ``ILOGLOBAL.PPM_NHK_POT_PRI``. Between set
  points the delta is interpolated linearly. The deltas are up to ~0.25 nT at
  MAGo and ~0.6 nT at MAGi.
* **Not cleaned:** the oscillations *during* motion (about 6 minutes, generally
  near 10:35 UT). These are flagged instead (see :ref:`mag-cmad-quality`).

Ultra decontamination heaters and Hi-45/Hi-90
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added in Imperial code v2.1.0. A **superposed epoch analysis** over one month
of data (1-30 November) builds an average gradiometer profile of each signal, triggered by:

* ``U45_DECON_HTR_CURR`` / ``U90_DECON_HTR_CURR`` for the Ultra decontamination
  heaters, which switch on every 10 minutes;
* ``H45_CURR`` / ``H90_CURR`` for the Hi signal.

A scaled profile is then subtracted from MAGo at each trigger. The gradiometer
factors are **0.65** for the heaters (from a separate MAGo/MAGi epoch analysis,
possible because the signal is strong on the spin axis) and **1.5** for Hi (from
minimising spin-plane spin tone, because the Hi signal lies mostly in the spin
plane).

.. _mag-cmad-thrusters:

Thrusters and the pre-thruster signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The daily repointing produces a train of large spikes at thruster firing
(around 10:00 UT). About **one hour earlier** (around 09:00 UT) a short DC signal
of unknown origin appears, which the team calls the *pre-thruster signal*.

* **Trigger:** a jump of more than 0.05 mA between consecutive samples of
  ``TAC_THRUST_BUS_GRP_1_CURR`` or ``TAC_THRUST_BUS_GRP_2_CURR`` (spacecraft
  packet X285). The current rises an hour before firing, so the same trigger
  locates both signals.
* **Method:** select a short segment (about 2 minutes for the pre-thruster
  signal), transform to a despun frame, interpolate MAGi onto the MAGo timeline
  (more accurate when despun, because the field is nearly constant), and subtract
  a diagonal gradiometer factor times the difference from MAGo. This is applied
  wherever MAGi - MAGo exceeds 0.1 nT in any axis, plus about 2 seconds either
  side to avoid residual edge spikes.
* **Factors:** one diagonal matrix per signal, measured by hand per axis in the
  orthogonal MAG frame and rotated into the despun frame before use:
  pre-thruster diag(0.718, 0.963, 0.455), thruster diag(0.965, 0.845, 0.965).

.. warning::

   **Known bug (CMAD):** burst-mode thruster cleaning introduces ``FILLVAL``
   (``-1e31``) gaps of a few seconds, within about 2 minutes of the firing.
   Burst-mode cleaning also frequently leaves residual thruster spikes. Normal
   mode is cleaned effectively. The periods are flagged, and fixing this is a
   stated target for future Imperial releases.

Not cleaned
^^^^^^^^^^^

* **Trajectory correction manoeuvres (TCMs)**, every one to two months: long sequences of
  large pulses, sometimes with precursors outside the main sequence. Flagged for
  3-6 hours around each TCM.
* **Pivot platform motion** (above).

.. _mag-cmad-quality:

Quality flag and bitmask
------------------------

**[CMAD]** section 5.4.5. **This supersedes the list in SW-009 section 7.2** and
matches the ``VAR_NOTES`` on ``qf_bitmask`` in
``imap_processing/cdf/config/imap_mag_l2_variable_attrs.yaml``.

``quality_flags``: ``0`` good data, ``1`` bad data (not suitable for science).
A raised flag is **always** accompanied by a non-zero bitmask.

``quality_bitmask`` (bit 0 is the least significant):

.. list-table::
   :header-rows: 1
   :widths: 10 34 56

   * - Bit
     - Meaning
     - Raised for
   * - 0
     - Data sourced from the secondary sensor
     - Output is MAGi rather than MAGo.
   * - 1
     - Thruster firing signals have been removed
     - The daily thruster firing (around 10:00 UT) and the pre-thruster activity
       (around 09:00 UT). Cleaning may be imperfect, especially in burst mode.
   * - 2
     - Spacecraft interference impacts these data
     - Uncleaned spacecraft signals: 3-6 hours around TCMs, and IMAP-Lo pivot
       platform motion (around 10:35 UT, about 6 minutes).
   * - 3
     - Instrument signals have been removed
     - Signals from other instruments removed from the data.
   * - 4-7
     - Reserved for in-flight calibration
     - Currently unused.

SW-009's ``SCTONES`` and ``PIVOTPLATFORMINTERFERENCE`` bits no longer exist.
Pivot platform motion is now reported through bit 2. Spin tone has no bit; it
is handled by the spin-plane offset optimisation instead.

.. note::

   **[CODE]** Nothing changes in this repository: ``quality_flags`` and
   ``quality_bitmask`` are still copied through opaquely from the offsets file.
   MAG still has no enum in ``imap_processing/quality_flags.py``; see
   :ref:`mag-implementation-status`.

Remaining artifacts in L2
-------------------------

**[CMAD]** section 5.4.5 tells users to expect the following in released L2 data.
They matter here mainly for triaging "is this a pipeline bug?" reports.

* **Burst-mode thruster residuals** that can mimic solitons, mirror modes or
  other ion-scale structures.
* **Spin-axis offset uncertainty** of ~0.1 nT, at times up to ~0.5 nT. The spin
  axis is roughly GSE/GSM X and RTN R. The error shows up as a near-DC shift in
  that component, which biases :math:`|B|` and can shift field direction by up to ~10
  degrees for typical IMF strengths.
* **Spin tone** at the ~15 s spin period, in the spin-plane components (roughly
  GSE/GSM Y and Z, RTN T and N) and in :math:`|B|`. Sources are spin-plane offset and
  calibration-matrix errors. It is sometimes significant, including during CMEs
  (from gain and angle uncertainty), and can be mistaken for alpha-particle or
  other heavy-ion cyclotron waves.

.. _mag-cmad-vs-sw009:

Differences from SW-009
-----------------------

Only the L2 and in-flight calibration material overlaps. Where the two disagree:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Topic
     - SW-009
     - CMAD (ICL-013, ICL-017, 5.4.5)
   * - Quality bitmask
     - Eight named bits, ``SEC_SENS`` last, includes ``SCTONES`` and
       ``PIVOTPLATFORMINTERFERENCE``.
     - Four named bits, secondary sensor at bit 0. **Matches the code's YAML.**
   * - Offset method
     - "Leinweber method" for the offsets.
     - Leinweber for the spin axis only; Kepko plus spin-tone optimisation for
       the spin plane.
   * - In-flight gain/orthogonality
     - Identity until justified.
     - V9 matrices with fitted angles and gains that are not exactly 1.
   * - Spacecraft-field removal
     - Generic "several processes" for 0-64 Hz and DC steps.
     - Named processes for the pivot platform, Hi/Ultra and thrusters, with
       triggers and factors; TCMs and pivot motion flagged, not cleaned.
   * - L2 frames
     - DSRF, SRF, RTN, GSE.
     - Also GSM. The code already produces it; see
       :ref:`mag-implementation-status`.
   * - Provenance
     - Offsets file names the L1 file it applies to.
     - ICL-017 adds that the calibration input versions are "captured in the
       parent metadata field of the released L2 science data files". See the
       open question in :ref:`mag-implementation-status`.

Everything SW-009 specifies for **L0 to L1D, compression, I-ALiRT and the
ancillary file formats** is absent from the CMAD, so the rest of these pages keep
SW-009 as their primary source.
