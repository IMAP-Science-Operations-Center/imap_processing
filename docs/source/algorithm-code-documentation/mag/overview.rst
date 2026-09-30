.. _mag-overview:

Instrument and Mission Concepts
===============================

Everything on this page is background needed to read the algorithm pages. It is
mostly **[DOC]** (algorithm document sections 4 and 7.1).

What MAG measures
-----------------

MAG is a **conventional dual fluxgate magnetometer**. It measures the vector
interplanetary magnetic field at the spacecraft, continuously, from the L1
Lagrangian point. The magnetic field underpins nearly every other IMAP
measurement: it carries energy, supports waves and turbulence, and controls the
propagation of charged particles, so MAG data is used by SWAPI, SWE, CoDICE and
HIT analyses as well as on its own.

MAG also feeds the **I-ALiRT** near-real-time stream used for space weather
forecasting.

Hardware
--------

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Element
     - What it is
   * - **MAGo**
     - The **outboard** sensor - furthest from the spacecraft body on the boom,
       so it sees the least spacecraft-generated field. **The primary source of
       released science data.**
   * - **MAGi**
     - The **inboard** sensor - closer to the spacecraft. Its value is mostly in
       *characterising* the spacecraft field so it can be removed from MAGo
       (see :ref:`gradiometry <mag-gradiometry>`).
   * - **FOB**
     - Front End Electronics board connected to MAGo. Appears in telemetry field
       names (``FOB_TEMP``, ``FOB_RANGE``, ``FOB_SATURATED``) - read "FOB" as
       "MAGo".
   * - **FIB**
     - Front End Electronics board connected to MAGi. Read "FIB" as "MAGi".
   * - **ICU**
     - Instrument Control Unit. Runs MAG boot software (BSW) and application
       software (ASW); generates all telemetry.
   * - **PCU / ELB**
     - Power Control Unit; the Electronics Box housing ICU, FOB, FIB and PCU on
       the spacecraft platform (not on the boom).

Ranging
-------

MAG must cover sub-nT solar wind fields and the full Earth field (~40,000 nT)
for ground testing. Each sensor **autonomously changes range**. **[DOC]**

.. list-table::
   :header-rows: 1
   :widths: 10 24 24 42

   * - Range
     - Approx. coverage
     - Approx. resolution
     - Notes
   * - 0
     - +/-60,000 nT
     - 256 pT
     - Covers the Earth field; used for ground testing and at instrument start.
   * - 1
     - +/-2,048 nT
     - 64 pT
     -
   * - 2
     - +/-512 nT
     - 16 pT
     - Where MAGi may sit if spacecraft fields are large; MAGo may enter this
       range during a large CME.
   * - 3
     - +/-128 nT
     - 4 pT
     - Most sensitive. **Nominal range for MAGo at all times.**

.. important::

   **Every calibration parameter - offsets, gains, alignment - is different for
   each sensor AND each range.** This is why every calibration input in this
   pipeline is shaped ``(3, 3, 4)`` or ``(2, 4, 3)``: the trailing/leading axis
   is range 0-3. The range is carried as the fourth component of the L1A/L1B/L1C
   ``vectors`` array and as a separate ``range`` time series at L1D/L2.

   The two sensors range **independently**, and a range change can happen in the
   middle of a science packet - which is why the compression format has an
   optional per-vector range data section (see :ref:`mag-l1a`).

Cadence, science modes and operating modes
------------------------------------------

Sample rate is independently configurable per sensor. Permitted rates are
**1, 2, 4, 8, 16, 32, 64, 128 vectors/second** (``VecSec`` in
``mag/constants.py``). Packet duration is configurable from 1 to 256 seconds,
limited to 512 vectors per packet. **[DOC]**

.. list-table:: MAG science modes **[DOC]**
   :header-rows: 1
   :widths: 20 16 16 20 28

   * - Mode
     - Primary vec/s
     - Secondary vec/s
     - Default packet cadence (s)
     - Comment
   * - ``N_2_2``
     - 2
     - 2
     - 8
     - **Default normal mode**
   * - ``N_4_1``
     - 4
     - 1
     - 8
     -
   * - ``N_4_4``
     - 4
     - 4
     - 8
     -
   * - ``B_64_8``
     - 64
     - 8
     - 4
     - **Default burst mode**
   * - ``B_64_64``
     - 64
     - 64
     - 4
     -
   * - ``B_128_128``
     - 128
     - 128
     - 2
     -

.. warning::

   **Do not hardcode these six modes.** The document is explicit: the number of
   vectors and their cadence **must be inferred from the packet headers**. The
   table exists only so you recognise what you are looking at. The code follows
   this - ``PRI_VECSEC``/``SEC_VECSEC`` and ``PUS_SSUBTYPE`` are read per packet.
   **[CODE]**

Operating modes **[DOC]**:

* **Standby** - boot software only. Reduced housekeeping, no science, no I-ALiRT.
* **Config** - application software, no science telemetry. Used between science
  mode transitions.
* **Normal (NM)** - normal rate science telemetry, APID **1052**. ~23 h/day.
* **Burst (BM)** - burst rate science telemetry, APID **1068**. ~1 h/day. Entered
  by telecommand with a duration; exits back to NM by timeout, telecommand, or
  spacecraft DSN event flagging.

.. note::

   **NM and BM are mutually exclusive**, except at transitions where the last
   packet of one mode and the first packet of the other generally **overlap in
   sample time**. With certain rate/duration combinations, both packet types can
   be transmitted at the same time. This overlap is the reason L1C exists and is
   the reason L1C gap-filling has to tolerate duplicate timestamps.

The MAG science team wants a **continuous L2 normal-mode product**: real NM data
where it exists, and a synthesised low-cadence product built on the ground from
BM data where it does not. That synthesis is L1C. **[DOC]**

Science telemetry layout
------------------------

**[DOC]** Both NM and BM packets share the same header parameter structure, so
header decoding is a common operation. After the CCSDS header and the MAG data
field headers, the payload is:

.. code-block:: text

   [ PRIMARY   vectors: X(16b signed) Y(16b) Z(16b) range(2b) ] x N
   [ SECONDARY vectors: X(16b signed) Y(16b) Z(16b) range(2b) ] x M

Each sample is therefore **50 bits** and is *not* byte aligned.

**PRIMARY and SECONDARY are software labels, not sensors.** A boolean header
(``PRI_SENS``) is true when MAGo is PRIMARY. Nominally MAGo is PRIMARY, but this
**cannot be assumed** and can in principle change within a day. Every product
downstream of L0 is organised by **MAGo/MAGi**, never by primary/secondary.

Samples carry **no individual timestamps**. The data field header holds one
instrument time for the first PRIMARY vector and one for the first SECONDARY
vector; every other vector time is derived by adding
``1 / vectors_per_second``.

A note on timing
----------------

**[DOC]** IMAP is a spinning platform, so converting from the sensor frame to an
inertial frame needs a good measurement time. MAG timestamps vectors, but there
is a small systematic shift between the timestamp and the true measurement time.
The MAG team supplies a **time shift per sensor** (expected to be tens of
milliseconds at most, e.g. ``0.005`` s) that must be applied before any attitude
conversion. Positive shifts move times forward.

This is applied at **L1B** (:ref:`mag-l1b`), and again per-vector as a
``timedeltas`` correction at **L2** (:ref:`mag-l2`).

Reference frames
----------------

This chain is the spine of the whole pipeline. Each level's job is essentially
"advance one frame".

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - Frame
     - Where it appears
     - Meaning
   * - **MFO / MFI**
     - L0, L1A
     - **Measurement Frame** for MAGo / MAGi. Raw engineering units, three
       nearly-orthogonal sensor axes.
   * - **URFO / URFI**
     - L1B, L1C
     - **Unit Reference Frame** per sensor. Reached by applying the ground
       (engineering) calibration matrix. Data is now in **nT** and orthogonal.
   * - **ORFO / ORFI**
     - L1D, L2 (intermediate)
     - **Orthogonal Reference Frame**. Reached by applying the in-flight
       calibration matrix, which folds in gain/orthogonality corrections and the
       boom-derived mounting.
   * - **SRF**
     - L1D, L2
     - **Spacecraft Reference Frame** - spinning, spin axis aligned. Spin
       averaging is only possible here, in the two spin-plane axes.
   * - **DSRF**
     - L1D, L2
     - **De-spun Spacecraft Reference Frame**. Gradiometry is done here.
   * - **RTN, GSE, GSM**
     - L1D, L2, I-ALiRT
     - Inertial science frames for release.

**[CODE]** Frames are realised through SPICE. ``ValidFrames`` in
``mag/l2/mag_l2_data.py`` maps each name to an
``imap_processing.spice.geometry.SpiceFrame``:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - ``ValidFrames`` member
     - SPICE frame
     - Output variable name
   * - ``MAGO`` / ``MAGI``
     - ``IMAP_MAG_BASE``
     - ``vectors``
   * - ``MAGO_GROUND_CAL``
     - ``IMAP_MAG_O``
     - ``vectors``
   * - ``MAGI_GROUND_CAL``
     - ``IMAP_MAG_I``
     - ``vectors``
   * - ``SRF``
     - ``IMAP_SPACECRAFT``
     - ``b_srf``
   * - ``DSRF``
     - ``IMAP_DPS``
     - ``b_dsrf``
   * - ``GSE``
     - ``IMAP_GSE``
     - ``b_gse``
   * - ``GSM``
     - ``IMAP_GSM``
     - ``b_gsm``
   * - ``RTN``
     - ``IMAP_RTN``
     - ``b_rtn``

.. note::

   **MAGO and MAGI both map to ``IMAP_MAG_BASE``.** This is deliberate: the MAG
   team's in-flight calibration matrix transforms from each sensor's real
   mechanical mount into a single *idealised* frame, so by the time SPICE is
   involved both sensors share a frame. ``IMAP_MAG_O`` / ``IMAP_MAG_I``
   (``*_GROUND_CAL``) are retained for reference to the as-assessed ground
   mounting and are not used in the nominal path.

Calibration concept
-------------------

**[DOC]** Two distinct calibration campaigns feed two distinct file families.

**Ground calibration** (Magnetsrode facility, TU Braunschweig) characterises the
intrinsic sensor properties and produces what the pipeline calls the
**engineering (ENG) calibration**:

1. A nominal **scale factor** yielding nT per component, folded together with
   the **gain (sensitivity, sigma)** and **orthogonality (misalignment, omega)**
   matrix.
2. A **rotation from measurement frame (MF) to unit reference frame (URF)**,
   :math:`R_{MF \to URF}`.

These are combined into a single 3x3 matrix per (sensor, range) and applied at
L1B.

**In-flight calibration** (performed at Imperial College London) produces the
L1D and L2 inputs:

1. **URF to spacecraft frame rotation**, provided by the project after
   integration from the boom orientation. Possibly refined during commissioning,
   then fixed.
2. **Gain and orthogonality corrections** to the ground values. *Until there is
   evidence to justify a change these are the identity matrix.* Updated at
   roughly **monthly** cadence.
3. **Removal of spacecraft-generated fields** - several MAG-team processes that
   detect and remove low-frequency (0-64 Hz) and DC step changes caused by the
   spacecraft and other instruments.
4. **Offsets in spacecraft coordinates**, determined with the **Leinweber
   method** for magnetometer offsets in the solar wind. These combine the sensor
   offset and the spacecraft offset, are **time varying**, and are delivered
   **per vector, per day**. Data releases may include refined offsets later.

Only steps 1, 2 and 4 land in this repository, as delivered numbers in
calibration files. Step 3 happens entirely at Imperial College; its results
arrive as the per-vector offsets and the quality bitmask.

.. note::

   **[CMAD]** The public CMAD describes how this worked out in flight for Data
   Release 1, and it refines SW-009 on two points:

   * **Step 2:** the in-flight matrices (CalibrationMatricesV9) use fitted
     angles and gains that are *not* exactly identity.
   * **Step 4:** Leinweber is used only for the **spin-axis** offset. The
     spin-plane offsets use the Kepko method plus a spin-tone optimisation.

   Step 3 is now a set of named cleaning processes (IMAP-Lo pivot platform,
   Hi/Ultra heaters, thrusters). See :ref:`mag-cmad`.

Processing windows and the 30-minute buffer
--------------------------------------------

**[DOC/CODE]** MAG processes UTC-day windows. Because vectors are packetised
across midnight, L0, L1A, L1B and L1C files carry **an extra 30 minutes on each
side** (midnight - 30 min to midnight + 24 h + 30 min, i.e. a 25 hour file).

The buffer is **stripped to exactly 24 hours at L1D and L2**, after offsets and
time shifts have been applied - the order matters, because a time shift can move
a vector across the boundary.

**[CODE]** ``MagL2L1dBase.truncate_to_24h`` does the trimming; ``cli.py`` calls
``check_epochs_within_day_offsets`` on every MAG output, which raises if any
epoch is more than 24 hours outside the processing day.

Processing must also be **resilient to partial days**: dropped packets, MAG off
for part of the day, or partial downlink must not fail the run.
