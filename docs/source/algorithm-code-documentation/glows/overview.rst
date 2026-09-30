.. _glows-overview:

Instrument and Measurement Overview
===================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is **[DOC]** unless marked otherwise. It exists so that
the algorithm pages can use GLOWS vocabulary without stopping to define it.

What GLOWS measures
-------------------

GLOWS (GLObal solar Wind Structure) observes the **helioglow**: the heliospheric
backscatter glow of interstellar neutral hydrogen (ISN H) in the solar Lyman-α
line at **121.567 nm**.

The physics chain is short:

1. ISN H atoms flow through the inner heliosphere, collisionless, within a few
   au of the Sun.
2. Intense solar Lyman-α resonantly excites them; they immediately re-emit in
   random directions. Those re-emitted photons are the helioglow.
3. The density and velocity distribution of ISN H is sculpted by solar
   gravity, Lyman-α radiation pressure, and **ionization losses** - charge
   exchange with solar wind protons and alphas, photoionization below ~91.2 nm,
   and (within 1-2 au) electron-impact ionization.
4. The solar wind has a **latitudinal structure that evolves over the solar
   cycle** (slow/dense ~400 km/s, ~5 cm⁻³ near the equator at low activity;
   fast/rarefied ~750 km/s, ~2.5 cm⁻³ at the poles). Different charge-exchange
   rates at different heliolatitudes carve a 3D structure into the ISN H
   density.
5. That structure shows up in the sky distribution of the helioglow. Observing
   the helioglow over the mission therefore **infers the latitude structure of
   the solar wind and how it evolves.**

Secondary objectives: the ISN H distribution itself, and the solar radiation
pressure acting on ISN H.

Helioglow intensity at ~1 au is of order **180-900 Rayleigh**, varying across
the sky and with observer position.

The instrument
--------------

Conceptually descended from the TWINS/LaD photometer (Nass et al. 2006;
McComas et al. 2009). Built by CBK PAN.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Element
     - Notes
   * - Collimator with baffle
     - Defines the field of view and suppresses stray light.
   * - Spectral filter
     - Narrow band around Lyman-α. Its **temperature** matters for sensitivity
       and is telemetered per block.
   * - Channeltron (CEM) detector
     - Single-pixel electron multiplier. Effective area ``A = 5.067 cm²``.
       Quantum efficiency ``QE ≈ 0.0041 cts/photon`` at 121.6 nm.
   * - Electronics block
     - Discriminates pulses, timestamps events, histograms them, packs
       telemetry.

Field of view: **~3.31° FWHM** (PSF report), nominal full diameter ~6.8-8°
depending on how you define the edge (transmission is 0.01 at 8°).

.. important::

   **GLOWS cannot distinguish photons from particles.** Any event above
   threshold is a count. Separating the helioglow from background is entirely a
   ground-processing problem, and it is why the flag machinery in L1B/L2 is as
   elaborate as it is.

Sources contributing to the observed counts (document §5), in rough order of
how much they annoy you:

1. Heliospheric backscatter glow in Lyman-α - **the science signal.**
2. Extraheliospheric sources: stars, the Milky Way, quasars, planets, comets.
   Bright stars are *also useful*: they are the in-flight photometric standards
   used to track detector aging.
3. Particle background - **bursty**, non-directional, detectable statistically
   as an anomalous total count rate in a block.
4. Solar Lyman-α scattered off interplanetary dust.
5. Reflections of solar flares/active regions off interplanetary hydrogen.
6. ENA glow (may itself become a science topic).
7. Stray light from strong EUV sources.
8. Detector dark counts.

Expected count rates: **~200-1000 cps** from the helioglow, up to ~1000 cps for
the brightest star, so the detector must not saturate below **~2000 cps**. The
document's nominal working number is ``s_mean = 600 cps``.

How an observation is built
---------------------------

.. code-block:: text

   photon -> CEM pulse -> "direct event" (timestamp + pulse length)
          -> binned by spin angle into a 3600-bin histogram
          -> accumulated over 8 spins  = one "block" = one CCSDS packet
          -> ~720 blocks = one "observational day" = one pointing

Scanning geometry
^^^^^^^^^^^^^^^^^

The IMAP spin axis points near the Sun, offset by **4°** towards lower ecliptic
longitudes. GLOWS' boresight is mounted at a fixed angle to the spin axis, so
one spin traces a **small circle of angular radius 75°** on the sky.

**[CODE]** ``GlowsConstants.SCAN_CIRCLE_ANGULAR_RADIUS = 75.0`` in
``glows/utils/constants.py``. Note that the value is currently *unused* by the
pipeline - bin sky positions are obtained from SPICE frame transforms instead
(see :ref:`glows-l1b`).

After one observational day the spin axis is re-pointed by ~1° to maintain the
4° Sun offset, and the observed strip of sky shifts accordingly. A given star
stays inside the FOV for **~7-8 consecutive days**, crossing at a different
distance from the boresight each day, which is what makes the extrapolation to
zero elongation (and hence absolute stellar brightness, and hence absolute
calibration) possible.

The key numbers
---------------

From the document's Table 0.1. Superscript **c** = configurable in flight,
**g** = configurable on the ground.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Quantity
     - Value
     - Notes
   * - IMAP day length ``T_IMAP_day``
     - 1.0 day
     - Bounded 0.5-3.0 days. Time between spin-axis changes.
   * - IMAP spin period ``P_IMAP``
     - 15 s
     - Bounded 14.63-15.38 s (4 ± 0.1 RPM).
   * - Spins per block ``n_block``
     - 8 :sup:`c`
     - 1-256. Set by particle-background detection capability (§10.2).
       ~120 s per block.
   * - Bins per histogram ``n_bin``
     - 3600 :sup:`c`
     - 225-3600. Set by star-calibration resolution needs (§10.3).
   * - Bin width ``b``
     - 0.1°
     - ``360°/n_bin``. Range 0.1°-1.6°.
   * - Bits per bin ``d``
     - 8 :sup:`g`
     - Fixed at 8 in practice; max ~66.67 counts/bin expected.
   * - Time per bin ``t_bin``
     - 4167 µs
     - ``P_IMAP/n_bin``; 4065-4272 µs.
   * - Typical counts per block ``C_block``
     - 72 000 ± 268
     - 0.37 % 1-σ Poisson scatter - the basis of background detection.
   * - Blocks per day ``N_block``
     - 720
     - 702-738 nominal; 351-2215 for extreme day lengths.
   * - GLOWS clock ``f_counter``
     - 2 MHz
     - 0.5 µs resolution. Subsecond limit = 2 000 000.
   * - DE blocks downlinked/day ``T_dirEv``
     - 13
     - 48 in an older revision of Table 0.1; §3.3.2 and §9.3 say 13.
   * - Direct events per day ``C_day``
     - 5.184 × 10⁷
     - Only a tiny fraction is downlinked.
   * - Low-res L3A bins ``n_bin_lores``
     - 90
     - 4° bins. **L3A is not produced in this repository.**
   * - GLOWS boresight azimuth ``ψ_GLOWS``
     - 217°
     - ``90 + 127`` in the IMAP frame, measured from the X axis. Exact value
       TBD from as-built measurement; **[CODE]** the pipeline reads it from
       SPICE instead of hard-coding it.

Vocabulary you must not mix up
------------------------------

Block vs. histogram vs. lightcurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Block** - ``n_block`` consecutive spins (~2 min). The atomic unit of
  culling. Everything at L1A/L1B is per-block.
* **Histogram** - the ``n_bin``-element array of **counts** accumulated over one
  block. L1A and L1B carry histograms.
* **Lightcurve** - the ``n_bin``-element array of **photon flux in Rayleighs**
  accumulated over one observational day. That is what L2 is. The document is
  deliberate about this distinction: histograms have counts, lightcurves have
  physical units.

Spin angle ψ vs. position angle ψ\ :sub:`PA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the single most confusing thing in GLOWS. There are two angular
coordinates around the scanning circle:

* **ψ (IMAP spin angle)** - defined in the GI ICD: zero is where the
  spacecraft Y-axis crosses the plane parallel to the ecliptic, moving
  north-to-south, with +Z rotation. This is what the observatory broadcasts and
  what the onboard histogramming uses. **Histograms at L0, L1A and L1B are
  organised by ψ.**
* **ψ**\ :sub:`PA` **(GLOWS position angle)** - same rotation sense, but
  measured **from the northernmost point of the GLOWS scanning circle**. This
  is what the science team wants. **The conversion happens at L2.**

.. math::

   \psi_{PA} = \left[\psi - \psi_{G,\mathrm{eff}}\right] \bmod 360^\circ
   \qquad\text{(Eq. 29)}

.. math::

   \psi_{G,\mathrm{eff}} = 360^\circ - \psi_{GLOWS} + \delta\psi_{G,\mathrm{eff}}
   \qquad\text{(Eq. 30)}

with ``ψ_GLOWS = 217°`` the boresight azimuth in the spacecraft frame. The
instrument team **decided to set** ``δψ_G,eff = 0``: precession and nutation
contribute <0.25° (3σ) and average out over a block, let alone a day. The
document therefore says explicitly that
``position_angle_offset_average = 360° - ψ_GLOWS`` and
``position_angle_offset_std_dev = 0``.

.. note::

   **[CODE]** ``position_angle_offset_std_dev`` is hard-coded to ``0.0`` at both
   L1B and L2, exactly as the document requires. But
   ``position_angle_offset_average`` is computed **two different ways** in two
   different places - see :ref:`glows-implementation-status`.

.. warning::

   Document §12.6.2 item 6 says the **L1B** histogram is re-arranged from ψ to
   ψ\ :sub:`PA`. Document §3.9.1 item 9 and §3.14 item 2 say the conversion
   happens at **L2**. §12 is the older text and the document states that §3
   supersedes it. **[CODE]** The code converts at L2, i.e. it follows §3.

Bad times vs. bad angles
^^^^^^^^^^^^^^^^^^^^^^^^

Burn this into memory; it structures the whole flag system.

* **Bad time** - the *entire block* is unusable. One flag set per block. 17 of
  them. A bad-time block is dropped completely before co-adding at L2.
* **Bad angle** - *individual bins* within an otherwise good block are
  unusable, because the instrument was looking at a star, the galactic plane, a
  comet, or something the instrument team flagged by hand. 4 flags, each a
  ``n_bin``-long array. Bad-angle bins are **masked, not removed**: the values
  stay in the L2 product and the mask travels with them. Removal happens at L3A.

Observational day, pointing, day mode, night mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Observational day** = **pointing** = the interval between two IMAP
  repointing maneuvers. Data products from L1A up are organised on this basis,
  and the file name carries a ``-repointNNNNN`` token. GLOWS follows the
  official POC/MOC repointing table.
* GLOWS keeps observing through daily repointings; the HV bias is normally
  **not** ramped down. It does get turned off for ΔV maneuvers (station
  keeping, TCMs), which produces genuine gaps of two hours or more.
* Around the repointing, GLOWS walks through a state sequence (document Fig.
  8.1):

  .. code-block:: text

     Day Mode                       Night Mode                         Day Mode
     ---------|--------------------------------------------------------|--------
      Evening | Sunset |        Night         | Sunrise (30 blocks)     |
              ^                                ^
              IsNight raised                   RepointingPending cleared
              (20 blocks ≈ 40 min after        (34 min after maneuver start)
               RepointingPending set,
               which is 1 h before maneuver)

  Data collection and histogramming continue throughout. The ``is_night`` flag
  simply marks the interval where spacecraft activity may perturb the spin.
  The extra 30-block Sunrise wait exists to let spin-axis instabilities from
  the **IMAP-Lo pivot platform** motion damp out.

* L2 excludes histograms near the repointing. The exclusion window is
  ``(t₀ⁿ + Δ₀, t₀ⁿ⁺¹ - Δ₁)``, driven by the ``is_night`` transitions and tuned
  by ``sunrise_offset`` / ``sunset_offset`` in the pipeline settings file. See
  :ref:`glows-l2`.

Operation modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Mode
     - Notes
   * - Normal science operations
     - Day and Night modes as above. The overwhelming majority of the data.
   * - Regular in-flight tests
     - **Monthly.** HV gain test, comparation voltage test, threshold voltage
       test. Data structures are *identical* to normal science, so they arrive
       in the same packets and must be flagged out:
       ``is_hv_test_in_progress`` / ``is_test_pulse_in_progress``.
   * - Initial calibration / HV ramp-up
     - Commissioning, and after any safing or reboot.

.. warning::

   **[DOC §3.9.1]** During HV tests the time-tagged command loads can cause
   **fake ``is_night`` transitions**. When detecting is_night transitions for
   the L2 day/night windowing, blocks with ``is_hv_test_in_progress`` raised
   must be excluded first. **[CODE]** This is not currently done - see
   :ref:`glows-implementation-status`.

Time systems
------------

Three clocks are in play and the pipeline touches all of them.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Clock
     - Notes
   * - **GLOWS internal clock**
     - Free-running, 2 MHz, **not synchronised** to the IMAP clock. All direct
       event timestamps are in this clock. Subseconds are counted in
       1/2 000 000 s ticks. The relationship to the IMAP clock is known only
       through the PPS timestamps carried in telemetry.
   * - **IMAP clock (MET/SCLK)**
     - Seconds since 2010-01-01T00:00:00 UTC. 3σ accuracy ±50 µs; distributed
       to instruments at 1 PPS with ±30 µs accuracy; aligned to UTC within
       ±500 ms (3σ) after ground post-processing.
   * - **CDF epoch**
     - **[CODE]** TT2000 nanoseconds, via
       ``imap_processing.spice.time.met_to_ttj2000ns``. Never hand-roll this.

Representation by level:

* **L0/L1A** - integer ``seconds`` + integer ``subseconds`` pairs. Modelled by
  ``TimeTuple`` in ``glows/utils/constants.py``, which normalises subseconds
  above the 2 000 000 limit into whole seconds.
* **L1B** - floats, subseconds as the decimal part.
* **L2 and above** - UTC / J2000.

Coordinate frames
-----------------

**[DOC §14]** GLOWS has two instrument frames and you must not confuse them:

* **Science (SPICE) frame** - ``+Z`` along the GLOWS boresight but *pointing
  opposite* to it, so the **boresight vector is (0, 0, -1)**. ``+Y`` lies in the
  plane defined by the boresight and the IMAP body ``Z`` axis, pointing
  anti-sunward. Right-handed, so ``X`` follows.
* **MICD ("mechanical") frame** - ``Z`` perpendicular to the mounting plane,
  ``X`` perpendicular to the EBOX PCBs. Used only for mechanical-interface
  discussions. Ignore it for data processing.

**[CODE]** The frames the pipeline actually names are
``SpiceFrame.IMAP_GLOWS``, ``SpiceFrame.IMAP_DPS`` (despun pointing frame),
``SpiceFrame.IMAP_SPACECRAFT`` and ``SpiceFrame.ECLIPJ2000``. Spacecraft state
vectors are taken relative to ``SpiceBody.SUN``.

What the data system must guarantee
-----------------------------------

The document (§6) states four objectives; they are a useful sanity check on any
change you make:

1. Identify and remove unwanted **particle background**.
2. Identify and remove the contribution from **extraheliospheric sources**.
3. Track the **day-to-day evolution of selected calibration stars** when they
   are in the FOV.
4. Bin the pure helioglow signal on the ground into **90 well-defined spin-angle
   bins without smearing**.

Objectives 1 and 2 are what L1B/L2 flagging is for. Objective 3 is not yet a
data product anywhere. Objective 4 is L3A, and therefore not this repository's
problem - but it is why L2 must stay at full 3600-bin resolution rather than
rebinning: masking must happen at high resolution first, so that as few good
counts as possible are thrown away.
