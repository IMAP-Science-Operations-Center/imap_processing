.. _idex-overview:

Instrument Overview and Vocabulary
==================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is **[DOC]** unless marked otherwise - it is the
physical and telemetry context from chapters 2 and 3 of the algorithm document,
condensed. Read it once; the rest of the pages assume this vocabulary.

What IDEX measures
------------------

The Interstellar Dust Experiment is a **high-resolution impact-ionization
time-of-flight (TOF) mass spectrometer**. It provides the elemental composition
and mass distribution of interstellar dust (ISD) and interplanetary dust
particles (IDP). Scientifically it is the solid-phase counterpart to the gas
and pickup-ion composition measured by IMAP-Lo, CoDICE and SWAPI.

The measurement principle is **impact ionization**:

1. A dust grain enters the aperture through a set of grid electrodes and strikes
   the target. The effective target area is **600 cm²**.
2. The target is biased at **+3 kV**, accelerating the impact-generated positive
   ions away from it.
3. A shaped electrostatic field (biased rings plus a curved grid electrode)
   provides spatial and temporal focusing of those ions onto a centrally-located
   detector, with **reflectron-type** ion optics so that ions of the same mass
   but different initial energy arrive together.
4. The impact-generated **negative** charge is collected on the target itself.
5. Flight time from impact to detector scales as :math:`\sqrt{m}`, so the TOF
   waveform is a mass spectrum once it is calibrated. That calibration is what
   L2A's ``time_to_mass()`` attempts.

Target cleanliness is a first-order concern - condensed volatiles on the target
change the ion yield. IDEX has a one-time deployable door, and its flight
operations include a monthly-ish decontamination cycle that raises the target to
120 °C for 8 hours. **[CODE]** Nothing in this repository models, corrects for,
or flags decontamination cycles; they appear only as event-message log entries.

The six waveform channels
-------------------------

Every dust impact is recorded on **six** channels, from three physical signal
sources. The three TOF channels are three gain stages of the *same* detector
signal, present so that a single event can be measured across a wide dynamic
range without saturating.

.. list-table::
   :header-rows: 1
   :widths: 16 14 14 18 38

   * - Channel
     - Source
     - Rate
     - Digitization
     - What it is
   * - ``TOF_High``
     - TOF detector
     - high (260 MHz)
     - 10-bit
     - Highest-gain TOF stage. The primary mass-spectrum input at L2A;
       saturates first on large impacts.
   * - ``TOF_Mid``
     - TOF detector
     - high (260 MHz)
     - 10-bit
     - Mid-gain TOF stage.
   * - ``TOF_Low``
     - TOF detector
     - high (260 MHz)
     - 10-bit
     - Low-gain TOF stage. Survives the largest impacts.
   * - ``Target_High``
     - Target CSA
     - low (4.0625 MHz)
     - 12-bit
     - High-gain target charge-sensitive amplifier. The preferred impact-charge
       channel when unsaturated.
   * - ``Target_Low``
     - Target CSA
     - low (4.0625 MHz)
     - 12-bit
     - Low-gain target CSA. Used when Target High saturates. **[CODE]** It is
       also the channel L2B bins mass and charge on.
   * - ``Ion_Grid``
     - Ion grid CSA
     - low (4.0625 MHz)
     - 12-bit
     - Ion-grid charge-sensitive amplifier. Its signal may be positive or
       negative depending on polarity; L2A explicitly allows a negative fitted
       amplitude on this channel only.

The two sampling cadences:

.. math::

   \Delta t_{HS} = \frac{1}{260}\ \mu\mathrm{s} \approx 3.846\ \mathrm{ns},
   \qquad
   \Delta t_{LS} = \frac{1}{4.0625}\ \mu\mathrm{s} \approx 246.15\ \mathrm{ns}.

**[CODE]** These live as ``RawDustEvent.HIGH_SAMPLE_RATE`` and
``LOW_SAMPLE_RATE`` in ``idex_l1a.py``, expressed in **microseconds per
sample**. Note the naming trap: they are named "rate" but hold a *period*.

A separate constant, ``idex_constants.FM_SAMPLING_RATE =
0.0038466235767167234e-6`` seconds, is the flight-model quartz-oscillator
period used only by ``time_to_mass()``. It is the same 260 MHz cadence carried
to more digits and in seconds rather than microseconds.

How an event is captured
------------------------

IDEX has several instrument modes (boot, idle, science, transmit, plus
decontamination and door actuation). Only two matter to the pipeline:

**Science mode.** The instrument continuously samples the CSA channels and keeps
the six waveforms in a rolling buffer. Nothing is recorded as an event until an
impact-like **trigger** fires. On trigger, the flight software freezes the
pre-trigger and post-trigger portions of the buffer and stores the event to
onboard memory (an 8 GB NAND flash).

**Transmit mode.** Stored events are packetized into CCSDS telemetry: one
metadata/header packet per event, then waveform fragments.

The important consequence of the rolling buffer is that **the waveform starts
before the impact**. The number of pre-trigger "blocks" retained is in the
header, and L1A uses it to build a time axis whose zero is the trigger, with
negative times before the impact. The L2A baseline-noise windows (the first
5 µs of the low-rate record, and −7 µs to −5 µs on the high-rate record) exist
because of this pre-trigger data.

Blocks, samples and fragments
-----------------------------

Three different units of "chunk" appear in IDEX and they are easy to confuse.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Term
     - Meaning
   * - **Sample**
     - One digitized value from one ADC. The natural unit of the waveform
       arrays.
   * - **Block**
     - The FPGA's unit of buffer bookkeeping, ~1.969 µs of collection time.
       **[CODE]** A low-rate block is **8 samples**; a high-rate block is
       **512 samples** (8/4.0625 MHz = 512/260 MHz). The header reports
       pre-trigger counts in *blocks*, never in samples. ``MAX_HIGH_BLOCKS =
       16`` and ``MAX_LOW_BLOCKS = 64``, so a full event is nominally 8192
       high-rate and 512 low-rate samples per channel. ``DT_BLOCK = 8 /
       4.0625e6 ≈ 1.96923 µs`` in ``idex_constants.py`` is the block duration
       used for the dead-time calculation.
   * - **Fragment**
     - A telemetry-level chunk. One waveform channel does not fit in one CCSDS
       packet, so it arrives as several packets carrying ``IDX__SCI0RAW``
       payloads, ordered by ``IDX__SCI0FRAGOFF``. Fragments are a packetization
       artifact with no physical meaning, and L1A's job is to make them
       disappear.

Trigger and dead time
---------------------

The trigger does not return the instrument instantly to a ready state. After
the post-trigger samples are collected, the instrument observes a configured
**dead time** before it can accept another event. That interval is encoded in
the FPGA header and reconstructed at L1B as the ``dead_time`` variable.

Dead time matters for rate interpretation - it is time during which a real dust
impact could not have been recorded. **[CODE]** L1B computes and stores
``dead_time``, but **L2B's rate calculation does not use it**. L2B corrects only
for science-acquisition uptime derived from the event-message log. At ~16 events
per day the correction would be negligible, but the omission is undocumented.
See :ref:`idex-implementation-status`.

Trigger *origin* (which channel fired) and trigger *mode* (threshold, single
pulse, double pulse) are both decoded at L1B, and both feed the event
classification at L1A. See :ref:`idex-event-classification`.

Two telemetry streams
---------------------

IDEX produces two kinds of telemetry the pipeline cares about, and they are
processed on completely separate paths that only rejoin at L2B.

**Science telemetry (APID 1424).** The dust events. Header packet plus waveform
fragments, decommutated from ``idex_science_packet_definition.xml``. This is the
path that produces ``sci-10days`` at every level.

**Event-message telemetry (APID 1418).** Timestamped instrument log entries -
state changes, pulser activity, command responses. Not dust events. These are
decommutated from the housekeeping XTCE and rendered into human-readable strings
at L1A, then reduced at L1B to two state variables, ``science_on`` and
``pulser_on``. **[CODE]** L2B needs ``science_on`` to know what fraction of each
day the instrument was actually acquiring, which is the denominator of every
count rate it publishes. This is the only place the two streams meet.

A third stream, the **catalog list (APID 1419)**, is a packet-catalog summary
that the pipeline passes through with only a time conversion applied. It is not
described in the algorithm document.

Time systems
------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Quantity
     - Definition
   * - **Packet creation time**
     - ``SHCOARSE`` / ``SHFINE`` in the CCSDS secondary header. When the *packet*
       was generated, i.e. roughly when the event was downlinked from onboard
       storage. **This is not the event time, and this is the source of the L0
       file-grouping problem described in** :ref:`idex-data-products`.
   * - **Event time (MET)**
     - Reconstructed from the FPGA metadata header, not from the CCSDS header:

       .. math::

          t_{MET} = 2^{16}\,\mathtt{TXHDRTIMESEC1}
                  + \mathtt{TXHDRTIMESEC2}
                  + 20\times10^{-6}\,\mathtt{TXHDRTIMESUBS}

       The 32-bit seconds counter is split across two 16-bit telemetry words;
       the subsecond field is in units of 20 µs. **[CODE]**
       ``calculate_idex_event_time()`` in ``idex_l1a.py``. Mission elapsed time
       counts from 2010-01-01.
   * - **epoch**
     - **[CODE]** The CDF time coordinate on every IDEX product:
       **TT-J2000 nanoseconds**, produced by ``met_to_ttj2000ns()``. UTC is a
       derived, human-readable view of this and is never the stored basis.
   * - **Waveform time axes**
     - Per-event 1-D arrays ``time_low_sample_rate`` and
       ``time_high_sample_rate``, in **microseconds relative to the trigger**,
       so pre-trigger samples are negative. Stored as real variables on each
       event, not as global coordinates, because the pre-trigger offset varies
       per event. The integer index coordinates
       ``time_{low,high}_sample_rate_index`` are what the CDF dimensions
       actually depend on.
