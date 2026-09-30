.. _lo-overview:

Instrument and Mission Concepts
===============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is background needed to read the algorithm pages. It is
mostly **[DOC]** (algorithm document sections 4 and 5).

What IMAP-Lo measures
---------------------

IMAP-Lo is a **single-pixel neutral atom imager**, built on IBEX-Lo heritage.
It has two distinct science jobs:

**ENA (Energetic Neutral Atom) imaging**
    Energy-resolved all-sky maps of heliospheric hydrogen ENAs from
    **40 eV to 1 keV**. These become flux/intensity maps.

**ISN (Interstellar Neutral) observation**
    Interstellar H, D, He, O and Ne flowing through the heliosphere, resolved
    by energy and angle and tracked over more than 180 degrees of ecliptic
    longitude. These become **rate** products, not fluxes, because a given
    signal often mixes species.

The single detector produces only two "observables" at the anode level:
**H-minus counts** and **O-minus counts**.

* Hydrogen ENAs, and ISN H and He, produce H-minus (by direct charge exchange
  or by sputtering off the conversion surface).
* ISN O and Ne (and some He) produce predominantly O-minus.

This is why products are split into "light atoms" (H, He, D) and "heavy atoms"
(O, Ne) rather than into individual species.

How a single measurement happens
--------------------------------

This chain explains nearly every variable name in the data products.

1. A neutral atom enters through the **collimator** (9 degree FWHM field of
   view).
2. It strikes the **conversion surface (CS)** and leaves as a **negative ion**.
   Some of what leaves is a *sputtered* product of the incoming atom rather
   than the atom itself; this is the physical origin of the sputter and
   bootstrap corrections at L2.
3. The ion is pre-accelerated into a toroidal **electrostatic analyzer (ESA)**,
   which passes only a narrow band of energy per charge. A ring of magnets
   removes electrons.
4. The ion is post-accelerated by the **PAC** voltage (nominal 12 kV, range
   ~7-16 kV) into the **time-of-flight (TOF) telescope**.
5. It passes through two thin carbon foils (**C-Foil a** and **C-Foil c**),
   each releasing secondary electrons. Those electrons are steered to the two
   annular **start** sections of a chevron microchannel plate (MCP).
6. The ion itself lands on the central **stop** section (anode B), which is
   split into four quadrants (positions 0-3).
7. The start/start/stop timing gives **three independent TOF measurements**,
   plus a redundant fourth.

Anodes, TOF channels, and coincidence
-------------------------------------

Four physical anodes: **A** and **C** (electron starts), **B0** and **B3**
(ion stops). Four TOF channels are formed from them:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Channel
     - Between
     - Meaning
   * - ``TOF0``
     - a and b0
     - Start (foil a) to stop
   * - ``TOF1``
     - b3 and c
     - Stop to start (foil c)
   * - ``TOF2``
     - a and c
     - Start-to-start; **this is the mass discriminator**
   * - ``TOF3``
     - b0 and b3
     - Stop-to-stop; used for position

**Golden triple events** satisfy the redundancy checksum

.. math::

   \mathrm{CKSM} = \mathrm{TOF0} + \mathrm{TOF3} - \mathrm{TOF1} - \mathrm{TOF2}

which is expected to be near zero (absolute value <= 1 ns). Requiring this
suppresses background to <= ~1 count/day. **[DOC]**

Which of the four TOF values are present defines the **coincidence type**, a
4-bit code carried through every level as the ``ABSENT`` / ``coincidence_type``
field. Cases and bit layouts are in :ref:`lo-l1a`.

Energy stepping, spins, and science cycles
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Concept
     - Definition
   * - **Spin**
     - One spacecraft rotation, ~4 RPM, so ~15 s nominal.
       ``LoConstants.NOMINAL_SPIN_PERIOD_SEC = 15.0``. **The true spin period
       is not exactly 15 s** and must be taken from the spin data.
   * - **ESA step / ESA level**
     - One of **7** programmable energy steps.
       ``LoConstants.N_ESA_LEVELS = 7``. Levels are 1-indexed in science
       products.
   * - **ESA mode**
     - ``HiRes`` (dE/E ~ 0.7, throughput ~0.14) or ``HiThru``
       (dE/E ~ 1.1, throughput ~0.29). Encoded ``0 = HiRes``, ``1 = HiThru``
       in the code. HiRes exists mainly to cross-calibrate against IBEX-Lo in
       year 1.
   * - **Sweep**
     - One pass through the 7 ESA steps, 2 spins per step = 14 spins.
   * - **Science cycle** (a.k.a. **ASC**, Aggregated Science Cycle)
     - **Two sweeps = 28 spins ~ 420 s.** Equivalently 4 spins per ESA level:
       ``LoConstants.N_SPINS_PER_ESA_LEVEL = 4``,
       ``HISTOGRAM_CYCLE_EPOCHS = 7 * 4 * 15 = 420`` s. This is the
       fundamental aggregation unit for all L1B rate products, and the
       granularity at which badtimes and goodtimes are declared.
   * - **Pointing / repointing period**
     - The interval between spacecraft repointing maneuvers (about one day).
       The spin axis is fixed within it, so the boresight traces one
       great circle. L1C makes **one pointing set per pointing**.
   * - **Pivot (platform) angle**
     - Angle between the boresight and the spin axis. Nominal 90 degrees;
       range 60-165 degrees. Enumerated in ``LoConstants.PIVOT_ANGLES`` at
       60, 75, 90, 105, 120, 135, 148, 160 degrees, each with a +/-5 degree
       acceptance window. Derived from the pivot platform potentiometers
       (``COARSE_POT_PRI``, ``FINE_POT_PRI``) in the PCC packet.
   * - **Spin angle / spin phase**
     - Azimuth around the spin axis. Binned three different ways depending on
       the product: 6 bins of 60 degrees, 60 bins of 6 degrees, or 3600 bins
       of 0.1 degrees.
   * - **Off angle**
     - Angle away from the nominal boresight in the direction perpendicular to
       the spin. Pointing sets use **40 bins of 0.1 degrees spanning -2 to +2
       degrees**.
   * - **Ram / anti-ram**
     - Ram is the hemisphere the spacecraft is moving into (~30 km/s). Ram
       observations dominate the released ENA products; anti-ram products are
       generally not public. In ``LoConstants``, ram is spin-angle histogram
       bins ``0:20`` plus ``50:60`` and anti-ram is ``20:50``.

.. warning::

   **Bin-count naming trap.** The document names exposure times by the *angular
   width* of a bin, not the number of bins:

   * "6 degree bins" means **60** bins per spin (used for histograms, triples,
     and direct-event rates).
   * "60 degree bins" means **6** bins per spin (used for singles and TOF
     monitor rates).

   Separately, ``LoConstants.N_SPIN_ANGLE_BINS = 60`` (L1B histogram bins)
   while ``lo_l1c.N_SPIN_ANGLE_BINS = 3600`` (pointing-set bins). Same name,
   different meaning, different module. **[CODE]**

Timing constants
----------------

Direct events carry a 12-bit tick counter relative to the last latched spin
start. **[DOC]**

.. code-block:: text

   SPIN_RATE        = 4                             # spins per minute, nominal
   SPIN_DURATION    = 60 / SPIN_RATE                # 15 s nominal; use real spin data
   CLOCK_RESOLUTION = 1 << 12                       # 4096 ticks per spin
   SECONDS_PER_TICK = SPIN_DURATION / CLOCK_RESOLUTION
   SIX_DEGREES      = (1 << 12) / 60                # ~68.27 ticks per 6-degree bin

Full event time is ``spin_start_time + de_time_ticks * SECONDS_PER_TICK``, and
the tick counter wrapping (a tick value smaller than the previous one) is what
advances the spin index within a science cycle.

What changed from IBEX-Lo, and why it matters
---------------------------------------------

**[DOC]** These are the reasons IMAP-Lo algorithms are not simply IBEX-Lo
algorithms.

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Change
     - What it is
     - Algorithmic consequence
   * - Pivot platform
     - Boresight articulates 60-165 degrees instead of being fixed at 90
     - ISN can be tracked all year. Products must be organized *by pivot
       angle*; geometric factors, background rates, sputter coefficients and
       the ISN mask are all tuned per pivot angle. Only 90 degrees is fully
       characterized pre-flight.
   * - Wider collimator
     - 9 degree FWHM instead of 7
     - Larger geometric factor; different angular response function.
   * - Programmable energy window
     - HiRes and HiThru modes selectable
     - Every calibration table is indexed by ESA mode; the sweep table
       ancillary says which mode was active when.
   * - No high-angular-resolution quadrant
     - Removed
     - ~1.2x geometric factor gain; no separate high-res pixel to handle.
   * - Better processor
     - More on-board tables and modes
     - Sweep tables can repeat energy steps, which forces the **"resweep"**
       logic that maps measured step indices onto true ESA levels.
   * - Star sensor
     - New PMT with an IR/red filter
     - Provides <= 0.1 degree absolute pointing knowledge, and is used as an
       independent pointing check for the badtimes list.

Reference frames
----------------

**[DOC]** Products name several frames. The ones that actually appear in
algorithms:

* **Spacecraft (S/C) frame** - de-spun, spin axis is roughly the S/C-to-Sun
  vector. Raw counts and rates live here.
* **Heliospheric / solar inertial frame** - the Sun rest frame. Fluxes are
  transformed into it by adding the ~30 km/s spacecraft velocity to the ENA
  velocity vector. This is the "solar frame transform".
* **HAE (Heliocentric Aries Ecliptic)** - the map frame for released ENA and
  ISN maps at 6 x 6 degree resolution.
* Also reported for convenience in some products: Earth equatorial, galactic,
  Equatorial J2000, Ecliptic J2000, HSE.

In the code, frames come from SPICE through
``imap_processing.spice.geometry.SpiceFrame`` and the map frame is parsed out
of the L2 map descriptor. **[CODE]**

Deliverables
========================================

Descriptor string grammar
-------------------------

Deliverable descriptors follow a positional pattern:

.. code-block:: text

   <mode+pivot>-<product>-<species>-<frame>-<sp>-<direction>-<coords>-<res>-<cadence>

   e.g.  l090-ena-h-sf-nsp-ram-hae-6deg-12mo
         t090-ena-h-hk-nsp-ram-hae-N/Adeg-5-7days
         lxxx-enanbs-h-sf-nsp-ram-hae-6deg-12mo
         l090-spx-h-sf-sp-ram-hae-6deg-12mo
         l090-isnnbkgnd-h-sf-nsp-ram-hae-6deg-12mo

.. note::

   The token meanings below are **inferred from the structure of the Appendix C
   table** (each row's descriptor read against its Frame / Quantity / SP-corr columns).
   The PDF does not define the grammar explicitly. Verify against whatever descriptor
   registry the SDC maintains before relying on it.

===================  ==========================================================
Token                Apparent meaning
===================  ==========================================================
``l090`` / ``t090``  ``l`` = HiRes, ``t`` = HiThru; ``090`` = pivot angle
``lxxx`` / ``txxx``  same, all/multiple pivot angles (separate file per angle)
``ena``              ENA intensity, sputter + bootstrap corrected
``enanbs``           ENA intensity, **n**\ o **b**\ ootstrap/**s**\ putter
                     correction (used where ESA ALL is requested)
``spx``              spectral index (L3 only)
``spxnbs``           spectral index, no bootstrap/sputter correction
``isn``              ISN count rates
``isnnbkgnd``        ISN count rates, background **not** subtracted
``h`` / ``o``        light ion (H) / heavy ion (O)
``sf``               spacecraft frame
``hk``               heliosphere, **k**\ inematic (non-CG-corrected)
``hf``               heliosphere frame, CG corrected
``nsp`` / ``sp``     no survival probability correction / SP corrected
``ram``              ram direction
``hae``              Heliographic Aries Ecliptic
``6deg``             6° pixels; ``N/Adeg`` for sky strips (no map resolution)
``5-7days``          sky strip accumulated per pointing
``12mo``             12-month accumulation
===================  ==========================================================

L2 deliverables
---------------

**All L2 maps are 6° resolution, ram direction, and carry no survival probability
correction.**

For maps, the quantities bundled with intensity are: statistical uncertainty,
non-statistical (systematic) uncertainty, exposure time, background intensity,
background uncertainty, observation date (and standard deviation of observation date),
and pivot angle range. **All L2 ENA maps are in the 90° pivot range.**

*ENA maps:*

=========  ==============================  =====================================  ==========
Cadence    Frame                           Quantity                               Descriptor
=========  ==============================  =====================================  ==========
5-7 days   spacecraft                      intensity, light ion, ESA 4-7          ``l090-ena-h-sf-nsp-ram-hae-N/Adeg-5-7days``
5-7 days   heliosphere (non CG)            intensity, light ion, ESA 4-7          ``l090-ena-h-hk-nsp-...-5-7days``
5-7 days   heliosphere (CG)                intensity, light ion, ESA 4-7          ``l090-ena-h-hf-nsp-...-5-7days``
5-7 days   spacecraft                      intensity, heavy ion, ESA 4-7          ``l090-ena-o-sf-nsp-...-5-7days``
12 mo      spacecraft                      intensity, light ion, ESA 4-7          ``l090-ena-h-sf-nsp-ram-hae-6deg-12mo``
12 mo      heliosphere (non CG)            intensity, light ion, ESA 4-7          ``l090-ena-h-hk-nsp-...-12mo``
12 mo      heliosphere (CG)                intensity, light ion, ESA 4-7          ``l090-ena-h-hf-nsp-...-12mo``
12 mo      spacecraft                      intensity, heavy ion, ESA 4-7          ``l090-ena-o-sf-nsp-...-12mo``
12 mo      spacecraft                      intensity, **ESA ALL**                 ``l090-enanbs-h-sf-nsp-...-12mo``
12 mo      heliosphere (non CG)            intensity **and energy**, ESA ALL      ``l090-enanbs-h-hk-nsp-...-12mo``
12 mo      spacecraft                      intensity, ESA ALL, **pivot angles**   ``lxxx-enanbs-h-sf-nsp-...-12mo``
12 mo      heliosphere (non CG)            intensity and energy, ESA ALL, pivot   ``lxxx-enanbs-h-hk-nsp-...-12mo``
=========  ==============================  =====================================  ==========

*ISN maps* (all spacecraft frame, count rates, all ESA steps, per mode):

=========  ================================  ==========
Cadence    Quantity                          Descriptor
=========  ================================  ==========
5-7 days   count rates, light ion, HiRes     ``l090-isn-h-sf-ram-hae-N/Adeg-5-7days``
5-7 days   count rates, heavy ion, HiRes     ``l090-isn-o-sf-nsp-...-5-7days``
5-7 days   count rates, light ion, HiThru    ``t090-isn-h-sf-ram-hae-N/Adeg-5-7days``
5-7 days   count rates, heavy ion, HiThru    ``t090-isn-o-sf-nsp-...-5-7days``
12 mo      count rates, light ion, HiRes     ``l090-isnnbkgnd-h-sf-nsp-...-6deg-12mo``
12 mo      count rates, heavy ion, HiRes     ``l090-isnnbkgnd-o-sf-nsp-...-6deg-12mo``
12 mo      count rates, light ion, HiThru    ``t090-isnnbkgnd-h-sf-...-6deg-12mo``
12 mo      count rates, heavy ion, HiThru    ``t090-isnnbkgnd-o-sf-...-6deg-12mo``
=========  ================================  ==========

The 12-month ISN rows are annotated **"Also add off-90, not for delivery."** ISN rate
maps will be provided for pivot angle pointing near 90°.

**Delivery classification** (Appendix C colour-codes every row):

===================  ==========================================================  =========
Class                Meaning                                                     Action
===================  ==========================================================  =========
Best for Science     Best choice for almost all scientific studies               Deliver
Science Quality      Good for scientific studies, complete sky sweep             Deliver
Quick look science   Good for part of the sky; will be replaced by full sweep     Deliver
Archival value       Marginal science value, most direct depiction of what the   **Don't
                     instrument saw                                              deliver**
Marginal             Not recommended for science, incomplete sky sweep           **Don't
                                                                                 deliver**
===================  ==========================================================  =========

L2 row counts by class: 8 / 0 / 10 / 0 / 2 = **20 products** (30 counting HiRes and
HiThru separately).

L3 deliverables
---------------

**All L3 ENA maps are 6° resolution, ram direction.** L3 adds two things over L2: the
``SP corr`` column (Y/N per row) and **spectral index** products (``spx``).

Spectral index maps bundle: map spectral index, statistical uncertainty, background
spectral index, background spectral index uncertainty, exposure time, observation date
(and its standard deviation), and pivot range.

L3 covers, at 5-7 day and 12-month cadences, across spacecraft / heliosphere-non-CG /
heliosphere-CG frames:

* **Intensity** maps, ESA 4-7, SP-corrected (``l090-ena-h-{sf,hk,hf}-sp-...``)
* **Spectral index** maps, ESA 4-7, both SP-corrected and not
  (``l090-spx-h-{sf,hk,hf}-{sp,nsp}-...``)
* **Spectral index** maps, ESA ALL, no bootstrap/sputter
  (``l090-spxnbs-h-{sf,hk}-nsp-...-12mo``)
* **Spectral index**, ESA ALL, all pivot angles (``lxxx-spxnbs-...``, separate file
  per pivot angle)
* **ISN count rates** at 12 mo, light and heavy, HiRes and HiThru, spacecraft frame,
  **final goodtimes, background subtracted** (``l090-isn-h-sf-ram-hae-6deg-12mo``) —
  contrast with the L2 ``isnnbkgnd`` variants

Pivot ranges for L3 differ by product: intensity and spectral index maps near 90° for
the ESA 4-7 rows, but "all pivot ranges (e.g., first year 75°, 90°, 105°)" for the
ESA ALL rows.

L3 row counts by class: 10 / 3 / 10 / 0 / 2 = **25 products** (40 counting HiRes and
HiThru separately).
