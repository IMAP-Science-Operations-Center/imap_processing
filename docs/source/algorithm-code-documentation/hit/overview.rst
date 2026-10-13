.. _hit-overview:

Instrument and Measurement Concepts
===================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is background. Very little code depends on it
directly, but almost every design decision in :ref:`hit-l1a`,
:ref:`hit-l1b` and :ref:`hit-l2` only makes sense once you have it.

What HIT measures
-----------------

**[DOC]** HIT measures **~2-40 MeV/nucleon ion (H to Ni) composition, energy
spectra, angular distributions and temporal variations**. It is one of five
in-situ instruments on IMAP; together with SWAPI and CoDICE it gives
continuous ion coverage from 0.1 keV to 40 MeV/nucleon.

The design draws heavily on the **Low Energy Telescope (LET) on STEREO** and
on **EPI-Hi on Parker Solar Probe**. Most of the rate definitions, the onboard
matrices, the priority buffers and the rate compression scheme in this
document are literally the STEREO/LET ones.

The primary science products are **energetic ion intensities** in
:math:`\mathrm{cm^{-2}\,s^{-1}\,sr^{-1}\,(MeV/nuc)^{-1}}` for 16 species in
roughly 12 energy bins each.

The measurement technique
-------------------------

**[DOC]** HIT uses the standard **"dE/dx vs residual E"** technique:

* A particle passes *through* one detector, depositing :math:`\Delta E`.
* It then *stops* in a following detector, depositing :math:`E'`.
* The pair :math:`(\Delta E, E')` identifies the incident **species** and
  **kinetic energy**. Different elements trace out separate hyperbola-like
  tracks in that plane.
* Which detector **segments** fired identifies the **arrival direction**.

This technique has extensive flight heritage (ACE/SIS, Voyager/CRS,
Voyager/LECP, STEREO/LET, STEREO/HET, PSP/EPI-Hi).

.. important::

   **The species identification happens on board, not on the ground.** The
   flight software sorts each event through a lookup "matrix" and increments a
   counter. What reaches the SDC is already binned by species and energy.
   Ground processing at L1A/L1B/L2 is therefore bookkeeping, livetime
   correction and unit conversion - **not** particle identification. The only
   place ground code redoes the physics is L3, from the raw PHA events, and
   that is a different repository (see :ref:`hit-l3-scope`).

The sensor head
---------------

**[DOC]** 14 solid-state detectors (SSDs), each subdivided into segments, read
out by four custom **PHASIC** chips (16 channels each, dual gain).

.. list-table::
   :header-rows: 1
   :widths: 12 16 16 56

   * - Layer
     - Thickness
     - Active area
     - Notes
   * - **L1**
     - 24 um
     - 2 cm\ :sup:`2`, 3 segments each
     - Sits in the outer region of all 10 entrance apertures. Segments are
       named ``a``, ``b``, ``c`` (e.g. ``L1A2b``).
   * - **L2**
     - 50 um
     - 10.2 cm\ :sup:`2`, 10 segments
     - Two of them (``L2A0``-``L2A9``, ``L2B0``-``L2B9``), in the centre of
       the sensor head.
   * - **L3**
     - 1000 um
     - 15.6 cm\ :sup:`2`, 3 segments
     - Two of them. Segments are referred to as inner/outer (``L3Ai``,
       ``L3Ao``, ``L3Bi``, ``L3Bo``).
   * - **L4**
     - 1500 um
     - 2 active areas (inner ``i`` / outer ``o``)
     - **Only behind the 2 I-ALiRT apertures.** Optimised for energetic
       electrons.

**Apertures.** There are **10 entrance apertures** in two groups of five,
arranged along the arc of a circle:

* **8 "science" apertures** - ``A1``-``A4`` and ``B1``-``B4``. These feed the
  ion science rates and the sectored rates.
* **2 "I-ALiRT" apertures** - ``A0`` and ``B0``. These additionally have an
  L4 detector behind L1, used for real-time electron measurements.

The two-letter prefix throughout the telemetry is **side** (``A`` or ``B``)
followed by **aperture number** (0-4) and **segment** (``a``/``b``/``c``),
e.g. ``L1B3c`` = side B, layer 1, aperture 3, segment c.

Penetration ranges
------------------

**[DOC]** Events are classified by how deep they got. This is the single most
important organising concept in the HIT data, because the counters are laid
out by range first.

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 44

   * - Range
     - Detectors hit
     - Frame counters
     - Meaning
   * - **RNG2** / R2
     - L1 L2
     - ``l2fgrates`` (132), ``l2bgrates`` (12)
     - Stopped in an L2 detector. Lowest energies.
   * - **RNG3** / R3
     - L2 L3
     - ``l3fgrates`` (167), ``l3bgrates`` (12)
     - Stopped in one L3 detector.
   * - **RNG4** / PEN
     - L3A L3B
     - ``penfgrates`` (33), ``penbgrates`` (15)
     - Went through both L3s and possibly beyond ("penetrating").
   * - **RNG2I**
     - L1 L4 L2
     - ``l4fgrates`` (48), ``l4bgrates`` (24)
     - New for HIT: ions through an **I-ALiRT** aperture, so they lose extra
       energy in L4 and land at higher incident energy for the same range.
   * - **RNG3I**
     - L1 L4 L2 L3
     - (same arrays)
     - As above.
   * - **RNG4I**
     - L1 L4 L2 L3 L3
     - (same arrays)
     - As above.

R2, R3 and R4 match STEREO/LET exactly. The three ``I`` ranges are new.

.. note::

   **[CODE]** ``l4fgrates`` and ``l4bgrates`` are decommutated at L1A and
   carried through L1B, but **nothing at L2 uses them**. Only R2, R3 and R4
   feed the intensity products. See :ref:`hit-gap-l4rates`.

The matrices, Particle IDs, FGRATES and BGRATES
-----------------------------------------------

**[DOC]** Each penetration range has an onboard **matrix**: a lookup table
spanning **128 bins on the E' (x) axis and 400 bins on the dE (y) axis**,
covering Z = 1 to Z >= 40. The flight software computes the pair of indices
(**EPINDEX** 0-127 and **DEINDEX** 0-399) for each event, looks up which box
it lands in, and increments the corresponding counter.

* **Foreground rates (FGRATES)** - counters for boxes that lie along an
  element or isotope track. These are the real science: H, He-3, He-4, C, N,
  O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, each split into energy bins.
* **Background rates (BGRATES)** - counters for broad regions that are *not*
  on a track: the Li/Be/B region between He and C, the "backward moving
  particle" corner, STIM regions. **All background events get Particle ID
  255.**

**Particle ID** is simply the event's index into the FGRATES array for its
range. So Particle ID 0 in Range 2 is "H, 1.0-1.8 MeV/nuc". It is also
written into each PHA event record header, which is how you tie an event
back to a rate bin.

Two traps:

* **Particle IDs are not unique across ranges.** The same ID means different
  things in R2, R3 and R4. The document is explicit that deduplicating them
  on board was rejected as too expensive, and that the ground must handle it.
* **Some species exist in one range and not another.** Na, for example, is
  classified by the Range 3 matrix but not by Range 2 or Range 4; in those it
  falls into ID 255. This is why
  ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING`` has empty ``"R2"``/``"R4"``
  lists for many entries - that is correct, not an oversight.

Dynamic thresholds
------------------

**[DOC]** During a large SEP event the instrument would saturate: dead time
rises, chance coincidences rise, data quality falls. HIT mitigates this
automatically by **reducing its own geometry factor** - it disables high-gain
PHA channels on selected segments, which raises the effective energy
threshold. Light ions (H, He) stop being counted; heavy ions (Z >= 6), which
trigger low gain anyway, are largely unaffected.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - State
     - What is disabled
   * - **DT0**
     - Nothing. Nominal. All high gains functioning.
   * - **DT1**
     - High-gain PHA disabled on the **outer regions of all 16 L1 segments**
       in the science apertures. Geometry factor drops to that of the inner
       L1 detectors only.
   * - **DT2**
     - High-gain PHA disabled on **all science-aperture L1 detectors except
       the two centre ones** (``L1A2``, ``L1B2``).
   * - **DT3**
     - High-gain PHA additionally disabled on **all L2 detectors except
       ``L2A4``, ``L2A5``, ``L2B4``, ``L2B5``**, and on the **outer L3**
       detectors.

The transition up is driven by a commandable single-detector count rate
threshold; the transition back down happens at roughly half that rate
(hysteresis). **The I-ALiRT apertures are never affected** - they stay in
their nominal configuration in all four states.

.. important::

   **This is why L2 needs four ancillary tables per product.** The geometry
   factor and efficiency depend on the dynamic threshold state, so the state
   at the time of each science frame selects which table to use. The state is
   telemetered in **frame byte 1, bits 0-1**, decommutated as
   ``hdr_dynamic_threshold_state`` and carried into L1B and L2 as
   ``dynamic_threshold_state``.

Operating modes
---------------

**[DOC]** HIT's operating principle is "turn us on and leave us on" - it stays
on through spacecraft operations including thruster firings.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Mode
     - What it is
   * - **Science**
     - SSD high voltage bias on. The normal state, and the only one producing
       science frames.
   * - **Safe / standby**
     - SSD bias off. Entered by command or automatically on a fault such as
       high SSD current.
   * - **Boot**
     - FSW table/image upload to SRAM, then commit to MRAM, then a
       "Maintenance Status" packet, then a transition to Safe.

.. note::

   **[CODE]** Nothing in ``imap_processing/hit`` checks the operating mode.
   There is no mode field in the science frame header; the closest proxies are
   the housekeeping ``ENABLE_HVPS`` and ``FEE_RUNNING`` flags and the
   ``hdr_code_ok`` bit in the science frame.

The HIT Science Frame
---------------------

This is the structure that everything at L1A hangs off.

**[DOC]** One **Science Frame is one minute of data**. It is transmitted as
**20 CCSDS packets** on **APID 1252**, 262 bytes of payload each, at 20
packets per minute.

.. code-block:: text

   Packets 0-4    Science Frame Header, counters and rates
   Packet  5      remaining counters and rates, then the start of the Event Buffer
   Packets 6-19   Event Buffer (raw pulse-height events)

**[CODE]** ``decom_hit.assemble_science_frames`` splits at the packet
boundary: the **first 6 packets** (6 x 262 = 1572 bytes) are concatenated into
``count_rates_raw``, and the **last 14** into ``pha_raw``. That split is exact:
the fixed-format section is bytes 0-1571 and the Event Buffer header begins at
byte 1572.

The frame header is 5 bytes:

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Byte
     - Field
     - Meaning
   * - 0
     - ``hdr_unit_num`` (bits 6-7), ``hdr_frame_version`` (bits 0-5)
     - Unit: 00 = EM1, 01 = EM2, 10 = FM. Version is currently 11, modulo
       128 if it ever exceeds that.
   * - 1
     - ``hdr_code_ok`` (bit 7), ``hdr_heater_duty_cycle`` (bits 3-6),
       ``hdr_leak_conv`` (bit 2), ``hdr_dynamic_threshold_state`` (bits 0-1)
     - The dynamic threshold state lives here. **L2 depends on it.**
   * - 2
     - ``hdr_minute_cnt``
     - HIT internal minute counter. **Its value mod 10 selects which
       species/energy the sectored rates in this frame belong to.**
   * - 3-5
     - spare
     - Three spare bytes, dropped by the code.

.. note::

   The document describes a separate 5-byte *frame* header (SCID/version,
   SFLEN length, SFCHECK checksum) in section 5.3, and a *MISCBITS* block in
   Table 11. **Table 11 is what the packets actually contain and what the code
   parses.** Section 5.3's SFLEN/SFCHECK description also states the frame
   never exceeds 4160 bytes, while Table 27 places the last event-buffer byte
   at 5239. Treat the tables as authoritative; they agree with the code.

Sectored rates: how the sky is divided
--------------------------------------

**[DOC]** Anisotropy comes from combining the **8 science apertures** with the
**spacecraft spin**:

* **Declination** - 8 bins of 22.5 degrees each, covering the full 180
  degrees. Determined on board from *which* L1 segment and *which* L2 segment
  fired (algorithm document Table 2). 0 degrees is the spin axis in the
  sunward direction.
* **Inclination** - 15 bins of 24 degrees each, covering the 360 degrees of
  spin. Determined purely **by timing**, incrementing every second, assuming
  the nominal **4 rpm** spin. Zero inclination is the zero spin phase from the
  spacecraft Time and Status message; a new spin-phase-zero resets the index.

8 x 15 = **120 look directions** per species/energy combination.

Because 120 directions would blow the telemetry budget, the frame carries
**only one species/energy combination per minute**, cycling through 10 of
them:

.. list-table::
   :header-rows: 1
   :widths: 8 16 32 44

   * - mod 10
     - Species
     - Energy
     - Notes
   * - 0
     - H
     - 1.8 - 3.6 MeV
     -
   * - 1
     - H
     - 4.0 - 6.0 MeV
     -
   * - 2
     - H
     - 6.0 - 10.0 MeV
     -
   * - 3
     - 4He
     - 4.0 - 6.0 MeV/n
     -
   * - 4
     - 4He
     - 6.0 - 12.0 MeV/n
     -
   * - 5
     - CNO
     - 4.0 - 6.0 MeV/n
     - Element **group**, not a single species.
   * - 6
     - CNO
     - 6.0 - 12.0 MeV/n
     -
   * - 7
     - NeMgSi
     - 4.0 - 6.0 MeV/n
     - Element group.
   * - 8
     - NeMgSi
     - 6.0 - 12.0 MeV/n
     -
   * - 9
     - Fe
     - 4.0 - 12.0 MeV/n
     -

**[CODE]** This is ``MOD_10_MAPPING`` in ``hit/l0/constants.py``, keyed on
``hdr_minute_cnt % 10``.

Two consequences that trip people up, both handled in the code:

#. **A complete sectored set takes 10 minutes.** L1A only emits sectored data
   for runs of 10 consecutive frames whose ``hdr_minute_cnt % 10`` is exactly
   ``0,1,...,9``.
#. **Sectored counts are accumulated for 10 minutes and transmitted over the
   next 10 minutes.** So block *n*'s counts must be divided by block *n-1*'s
   livetime. See :ref:`hit-l1b-sectored`.

.. warning::

   **[DOC]** The inclination bins assume exactly 4 rpm. Real spin rates differ,
   which makes the **fifteenth inclination bin** narrower or wider than 24
   degrees. The document states plainly: *"This needs to be corrected on the
   ground. No onboard correction is planned."* **No such correction exists in
   the code.** See :ref:`hit-gap-spinrate`.

Naming: declination/inclination vs zenith/azimuth
-------------------------------------------------

**[CODE]** The code does **not** use the document's names:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Document
     - Code
     - Values
   * - Declination (8 bins of 22.5 deg)
     - ``zenith``
     - ``ZENITH_ANGLES`` = 11.25, 33.75, 56.25, 78.75, 101.25, 123.75,
       146.25, 168.75 (bin centres)
   * - Inclination (15 bins of 24 deg)
     - ``azimuth``
     - ``AZIMUTH_ANGLES`` = 12, 36, 60, ..., 348 (bin centres)

Also note the **transpose**: the frame stores ``sectorates`` as
``(8 declination, 15 inclination)``, and ``parse_count_rates`` transposes it to
``(epoch, azimuth, zenith)`` = ``(epoch, 15, 8)``. All downstream arrays are in
that order.

Boresight and pointing
----------------------

**[DOC]** HIT is rotated **30 degrees counterclockwise from the spacecraft
+Y axis**. The boresight is taken along the spacecraft **+Z** (sun-spacecraft
line) between the L0B and L1B apertures. The instrument vector in the
spacecraft frame is :math:`(-0.5,\ 0.866025,\ 0)`; use the transpose of the
rotation matrix to go from spacecraft to instrument.

**[CODE]** ``imap_processing/spice/geometry.py`` knows about HIT -
``SpiceFrame.IMAP_HIT = -43500``, a boresight lookup of ``[0, 1, 0]``, and a
spacecraft-to-instrument spin phase offset of ``119.6452/360`` (nominally
30 + 90 = 120 degrees). **No HIT processing code calls any of it.** Pointing
is needed at L3, not here.

Uncertainties
-------------

**[DOC]** Two families, handled at different levels:

* **Statistical.** Asymmetric Poisson, per Gehrels 1986 at 1-sigma
  (0.8413 confidence). Computed at L1A from raw counts, then carried through
  by the same arithmetic that transforms the counts:

  .. math::

     \delta_u = \sqrt{n+1} + 1, \qquad \delta_l = \sqrt{n}

  These populate ``DELTA_PLUS`` and ``DELTA_MINUS``. **Fill values must
  propagate to fill values.**

* **Systematic.** Two known sources, neither yet quantified:

  * **Chance coincidences** - two particles arriving close enough in time to
    look like one multi-hit event. Segmenting the detectors mitigates it;
    ground consistency checks are promised but unspecified.
  * **Livetime errors** at very high rates - the livetime counter does not
    account for the coincidence window opened by each trigger. The proposed
    correction is
    :math:`\mathrm{livetime_{corrected} = livetime} + (\Delta t \times N_{trig})`,
    with :math:`\Delta t` to be determined from the onboard "livestim" pulser
    data and accelerator calibration runs. **Neither the correction nor
    :math:`\Delta t` is defined yet.**

  **[DOC]** At launch all systematic uncertainties are **zero**, to be updated
  in flight. The total is the quadrature sum:

  .. math::

     \delta_{full} = \sqrt{\delta_{stat}^2 + \delta_{sys}^2}

**[CODE]** L1A computes the Gehrels values; L1B divides them by livetime; L2
runs them through the intensity equation; ``add_systematic_uncertainties``
writes zeros and ``add_total_uncertainties`` does the quadrature sum. The
chain is complete and honest - it just has zeros in the systematic slot, as
intended.

Vocabulary
----------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Term
     - Meaning
   * - **Science Frame**
     - One minute of HIT data, 20 CCSDS packets on APID 1252. The L1A record.
   * - **Aperture**
     - One of 10 entrance windows. ``A1``-``A4``/``B1``-``B4`` are science;
       ``A0``/``B0`` are I-ALiRT (they have an L4 detector).
   * - **Range (R2/R3/R4)**
     - Penetration depth class. Determines which matrix, which counter array,
       and which Particle ID namespace applies.
   * - **Matrix**
     - The onboard 128 x 400 lookup in (E', dE) space that assigns each event
       a species and energy bin. Drawn in Appendix C of the algorithm
       document.
   * - **FGRATES / BGRATES**
     - Foreground (on an element track - real science) / background (broad
       off-track regions, Particle ID 255) counter arrays.
   * - **Particle ID**
     - The index into a range's FGRATES array. 255 means unidentified or
       background. **Not unique across ranges.**
   * - **Standard rates**
     - Full-instrument (no look direction), 1-minute, native energy bins. The
       main product.
   * - **Summed rates**
     - Standard bins combined into wider bins, and species combined into
       groups, for better statistics during quiet times.
   * - **Sectored rates / macropixel**
     - The 120-look-direction anisotropy product. 10-minute cadence, 10
       species/energy combinations, reduced energy resolution. "Macropixel"
       is the name the L2 product uses.
   * - **Dynamic threshold (DT0-DT3)**
     - Automatic geometry-factor reduction during intense events. Selects
       which ancillary factor table L2 uses.
   * - **Livetime**
     - Fraction of the minute the FEE spent waiting for a trigger. Telemetered
       as a counter of 16 MHz clock cycles, compressed to 16 bits.
   * - **STIM**
     - Onboard pulser events. "Livetime STIM" pulses measure livetime
       independently; "ADC STIM" events calibrate the ADCs. Tagged in the
       event record header, counted in ``pbufrates`` 29-30.
   * - **HAZ (hazard)**
     - A trigger arriving within ~2.8 us of the previous one. Counted
       separately, currently rejected from analysis, and does **not** increment
       livetime when rejected.
   * - **Priority buffer**
     - One of 32 onboard queues that sample events for telemetry, weighted to
       favour rare/interesting particle classes. ``pbufrates`` counts them.
   * - **PHASIC**
     - Pulse Height Analysis System Integrated Circuit. Four of them, 16
       dual-gain channels each.
   * - **Event record / PHA word**
     - The variable-length raw pulse-height data in the Event Buffer. **Not
       decoded by this repository yet.**
