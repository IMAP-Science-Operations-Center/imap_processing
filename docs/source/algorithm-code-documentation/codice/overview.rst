.. _codice-overview:

Instrument Overview
===================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is background you need before any algorithm page makes
sense. If you only read one thing, read
:ref:`codice-esa-stepping` and :ref:`codice-modes` - those two are where almost
all CoDICE-specific processing complexity comes from.

What CoDICE measures
--------------------

**[DOC]** CoDICE combines an ElectroStatic Analyzer (ESA) with a common
Time-Of-Flight versus energy (TOF-E) subsystem to simultaneously measure:

1. the 3-D velocity distribution functions and ionic charge state and mass
   composition of ~0.5-80 keV/q ions (**CoDICE-Lo**), and
2. the mass composition and arrival direction of ~0.03-5 MeV/nuc ions
   (**CoDICE-Hi**).

The science goals are to determine Local InterStellar Medium composition and
flow properties, and to understand the origin of suprathermal tails and particle
acceleration in the heliosphere.

The final science products cover three populations: **solar wind heavy ions**,
**pickup ions** (interstellar and inner source), and **suprathermal particles**.

CoDICE-Lo
^^^^^^^^^

**[DOC]** Ions between ~0.5 and 80 keV/q enter a collimated aperture and are
selected by energy-per-charge in the ESA, which focuses them onto a carbon foil
at the entrance to the TOF-E subsystem.

* The ESA steps through **128 steps over 16 spacecraft spins (~4 minutes)**.
* The carbon-foil subassembly is biased at **-15 kV** (post-acceleration, PAC),
  accelerating ions before they strike the ~1 ug/cm2 foil.
* Secondary electrons from the foil go to the outer annulus of the **Start MCP**;
  the neutralized ion traverses the flight path and strikes one of **24 APDs**,
  whose secondary electrons go to the centre of the **Stop MCP**.
* TOF between Start and Stop gives velocity; the APD gives residual energy E.
* Combined (E/q, TOF, E) determines mass M, charge state q and M/q.
* Arrival direction in azimuth comes from a **delay-line anode** on the Start
  MCP ("position"), not from the APD ID - by design (section 4.1).

.. warning::

   **[DOC]** The post-launch data caveats (section 9.2) reverse that in
   practice: the delay-line position "is not an accurate identifier of particle
   direction" and "should only be used to support APD ID". The L3b 3-D VDFs now
   bin by APD ID. When the document says "position", check whether the team
   now means APD ID. See :ref:`codice-data-caveats`.

**[DOC]** One Lo azimuth sector always points sunward, so **one sector measures
the solar wind continuously** while the others sweep the sky as the spacecraft
spins. The 24 sectors cover 0 deg and 180 deg and +/-15, 30, 45, 60, 75, 90,
105, 120, 135, 150, 165 deg from the sunward direction.

CoDICE-Hi
^^^^^^^^^

**[DOC]** Ions between ~0.03 and 5 MeV/nuc enter through **12 separate 12 deg x
7 deg FOV collimators** with 150 nm Al-polyimide foils (UV attenuation to
< 0.1%, stops < 10 keV protons). Start/Stop MCP signals come from secondary
electrons produced at a ~1 ug/cm2 carbon foil, and the ion strikes one of **12
SSDs** (700 um thick, 15 x 15 mm active area) distributed over 360 deg azimuth.
(E, TOF) yields velocity and mass. **CoDICE-Hi covers 1.8 pi sr per spin.**

The Hi FOV is a cone centred **30 deg above the CoDICE-Lo FOV plane**, so
+/-~20 deg from the sun and anti-sun directions are not covered.

.. important::

   **[DOC]** There are 12 SSDs but **16 SSD ID values (0-15)**. The original
   design had four dual-pixel SSDs for electrons; that was not flown, but the
   flight software still emits 16 IDs. The **valid SSD IDs are 0, 1, 3, 4, 5, 7,
   8, 9, 11, 12, 13, 15**; the remaining four are not valid.

   **[CODE]** ``SSD_ID_TO_ELEVATION`` and ``SSD_ID_TO_SPIN_ANGLE`` in
   ``constants.py`` are 16-element arrays indexed by SSD ID with ``np.nan`` at
   indices 2, 6, 10 and 14. Any code that indexes by SSD ID must tolerate NaN.

Vocabulary
----------

You will see these subscripts and terms everywhere. The document uses single
letters; the code uses names.

.. list-table::
   :header-rows: 1
   :widths: 14 12 74

   * - Doc
     - Code
     - Meaning
   * - :math:`k`
     - ``inst_az`` / ``position`` / ``ssd_id``
     - Azimuthal look direction in the instrument frame. Lo: **position 1-24**
       (delay-line anode) in most of the document, but **APD ID 1-24** in the
       newer text (L1A species split, L3b VDFs) - see
       :ref:`codice-data-caveats`. Hi: **SSD ID**.
   * - :math:`n`
     - ``spin_sector`` / ``spin_angle``
     - Spin phase bin. Lo reports **12 half-spin sectors (0-11)** which de-spin
       to **24 spin angles (0-23, 15 deg each)**. Hi reports **24 sectors
       (15 deg)** for direct events and **12 sectors (30 deg)** for sectored
       counts.
   * - :math:`l`
     - ``esa_step`` / ``energy_step``
     - ESA energy-per-charge step, **0-127** (Lo only). Step 0 is the highest
       energy; the sweep descends.
   * - :math:`i`
     - ``energy_<species>``
     - Energy-per-nucleon bin index (Hi only). The number of bins **varies by
       species**.
   * - :math:`j`
     - species variable name
     - Ion species.
   * - :math:`m`
     - ``full`` / ``reduced``
     - Geometric-factor mode. See :ref:`codice-modes`.
   * - "half spin"
     - ``half_spin_per_esa_step``
     - Index 0-31 within the 16-spin cycle. The Lo ESA stepping table is
       expressed per half-spin.
   * - "collapsing"
     - collapse table
     - On-board summing of adjacent angle/spin bins to reduce telemetry. All
       collapsed regions must be rectangular and contiguous.
   * - "view"
     - ``view_id``
     - Selects, per APID, the collapse table and compression scheme in use.
   * - PHA
     - direct events
     - Pulse-Height-Analysis event, i.e. a single particle's full record.

.. _codice-frames:

Coordinate frames and angle conventions
---------------------------------------

**[DOC] CoDICE instrument frame**
  +Z points towards the Sun. +X points from the instrument towards the
  spacecraft. +Y completes the right-hand rule. The **azimuth angle**
  :math:`\varphi` is measured clockwise from +Z when viewing the Y-Z plane from
  the +X direction (i.e. looking towards the instrument from the spacecraft).
  This is in the plane of the APD FOVs.

**[DOC] IMAP spacecraft (SC) frame (de-spun)**
  +Z is the average spin vector, +X is the North Ecliptic Pole, +Y completes the
  right-hand rule.

**[DOC] Spin phase and the spin-angle reference**
  Spacecraft spin phase 0 deg is the moment the **+Y**\ :sub:`SC` axis crosses
  the ecliptic plane. The SC-frame spin angle is measured from +X\ :sub:`SC`
  (north ecliptic pole) to the detector's central look direction. Because the
  instrument frame rotates with the spacecraft, the **instrument-frame spin
  angle is measured from a fixed reference: the direction of the +X**\
  :sub:`Co` **axis at spin phase 0**. At spin phase 0, +X\ :sub:`Co` is offset
  **~46.0 deg** from +X\ :sub:`SC`.

**[DOC] Frame conversion**

.. math::

   \theta_{SC} = (\theta_{inst} + 46^\circ) \bmod 360^\circ

The elevation angle is identical in both frames. This applies to both Hi SSDs
and Lo APDs.

.. note::

   **[CODE]** The +46 deg rotation is **not applied anywhere in this
   repository.** All CoDICE L2 angle variables are in the **instrument frame**.
   The conversion appears in the document only in the L3 pitch-angle sections
   (13.1.2, 13.3.2), which are out of scope here. Section 4.2 now also says
   spin angles "SHOULD" be computed from the look-direction unit vectors with
   SPICE and the instrument kernel. The pipeline does not do that; it uses the
   tabulated constants below.

Lo azimuth (position) to angle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Position 1 is the sunward channel; positions increment clockwise in
15 deg steps:

.. math::

   \varphi_k = (k - 1) \times 15^\circ , \quad k = 1 \dots 24

Hi SSD azimuth
^^^^^^^^^^^^^^

**[DOC]**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - SSD ID
     - :math:`\varphi_k`
     - SSD ID
     - :math:`\varphi_k`
   * - 0
     - 180 deg
     - 8
     - 0 deg
   * - 1
     - 210 deg
     - 9
     - 30 deg
   * - 3
     - 240 deg
     - 11
     - 60 deg
   * - 4
     - 270 deg
     - 12
     - 90 deg
   * - 5
     - 300 deg
     - 13
     - 120 deg
   * - 7
     - 330 deg
     - 15
     - 150 deg

Lo position to elevation angle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Multiple positions sample the same physical elevation angle at
different spin phases. Products that retain angular information collapse the
24 positions to 13 elevation angles relative to the spin axis:

**[CODE]** ``LO_POSITION_TO_ELEVATION_ANGLE`` in ``constants.py``, split into
``"sw"`` and ``"nsw"`` sub-dictionaries.

.. list-table::
   :header-rows: 1
   :widths: 24 14 18 22 22

   * - Positions
     - Product
     - Elevation
     - SW array index
     - NSW array index
   * - 1
     - SW
     - 0 deg
     - 0
     - -
   * - 2, 24
     - SW
     - 15 deg
     - 1
     - -
   * - 3, 23
     - SW
     - 30 deg
     - 2
     - -
   * - 4, 22
     - NSW
     - 45 deg
     - -
     - 0
   * - 5, 21
     - NSW
     - 60 deg
     - -
     - 1
   * - 6, 20
     - NSW
     - 75 deg
     - -
     - 2
   * - 7, 19
     - NSW
     - 90 deg
     - -
     - 3
   * - 8, 18
     - NSW
     - 105 deg
     - -
     - 4
   * - 9, 17
     - NSW
     - 120 deg
     - -
     - 5
   * - 10, 16
     - NSW
     - 135 deg
     - -
     - 6
   * - 11, 15
     - NSW
     - 150 deg
     - -
     - 7
   * - 12, 14
     - NSW
     - 165 deg
     - -
     - 8
   * - 13
     - NSW
     - 180 deg
     - -
     - 9

**[DOC]** Positions 1 and 13 only observe half of the 24 spin angles for a given
ESA step, because they sit on the spin axis. Angular products must **replicate**
their counts into the unobserved half, choosing 0-11 or 12-23 depending on the
pixel orientation (A or B) of that half-spin.

Hi SSD to elevation angle
^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** = **[CODE]** ``SSD_ID_TO_ELEVATION`` (indexed by SSD ID) and
``HI_L2_ELEVATION_ANGLE`` (indexed by array position 0-11).

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 16 16 20

   * - Index
     - SSD ID
     - Elevation
     - Index
     - SSD ID
     - Elevation
   * - 0
     - 0
     - 150.0 deg
     - 6
     - 8
     - 30.0 deg
   * - 1
     - 1
     - 138.6 deg
     - 7
     - 9
     - 41.4 deg
   * - 2
     - 3
     - 115.7 deg
     - 8
     - 11
     - 64.3 deg
   * - 3
     - 4
     - 90.0 deg
     - 9
     - 12
     - 90.0 deg
   * - 4
     - 5
     - 64.3 deg
     - 10
     - 13
     - 115.7 deg
   * - 5
     - 7
     - 41.4 deg
     - 11
     - 15
     - 138.6 deg

Spin angles in the instrument frame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All spin angles below are measured from the +X\ :sub:`Co` reference defined in
:ref:`codice-frames` and are bin centres.

**[DOC] Lo** (section 4.2). The FOV of APDs 1 and 13 is perpendicular to the X-Y
plane, so their spin angle is meaningless. For de-spun data (spin-angle index
:math:`n` = 0-23 spanning 360 deg):

.. math::

   \theta_{Lo,n} = n \cdot 15^\circ + 277.5^\circ \quad (\bmod 360^\circ)

Sections 12.2.2 and 13.2.6 write the same constant as
:math:`7.5^\circ + 270^\circ`: the 270 deg is the spin angle of APDs 2-12 at
spin sector 0.

**[DOC] Hi** (section 4.2). The SSD FOVs are inclined 30 deg to the Lo FOV
plane, so each SSD sees a different spin angle for the same sector. Using the
look-direction unit-vector components :math:`n_{x,k}, n_{y,k}` for SSD
:math:`k`:

.. math::

   \theta_{Hi,0,k} = 180^\circ + \tan^{-1}\!\left(\frac{n_{y,k}}{n_{x,k}}\right)
   + 15^\circ

This is the centre of the first 30 deg (sectored) bin; add 30 deg per sector.
The direct-event table (section 12.1.1, 15 deg bins) is the same value minus
7.5 deg, incremented by 15 deg per sector. The draft used two expressions in
:math:`\varphi_k`, split by SSD group. Rev 3 Chg 1 replaces them with this
single unit-vector form.


.. _codice-esa-stepping:

The Lo ESA stepping scheme
--------------------------

This is the single most important CoDICE-Lo concept.

**[DOC]** The Lo FOV covers the full sky in **one half-spin**, so data is
accumulated on board at half-spin cadence for **32 half-spins = 16 spins ~ 4
minutes**. Different ESA steps are sampled during each half-spin, from 1 step
per half-spin at the top of the sweep to 6 at the bottom.

The **pixel orientation** (A for even half-spins, B for odd) describes the APD
FOV orientation relative to the start of a full spin: the spin angles measured
by a given APD are offset by 180 deg between A and B. Because accumulation is at
half-spin cadence, **the instrument always reports spin sector 0-11 regardless
of orientation** - this is what makes de-spinning necessary.

Terminology, from this point on:

* **spin sector** = the half-spin-relative bin the instrument reports, 0-11.
* **spin angle** = the physical spin angle, 0-23 indices covering 0-360 deg.

Two stepping schemes have been used:

**[DOC] Launch - 2025-12-18** - all 128 steps are sampled, 1/1/1/1 for
half-spins 0-3, then 2 per half-spin for 4-7, 3 for 8-11, 4 for 12-15, 5 for
16-23, and 6 for 24-31 (steps 80-127).

**[DOC] 2025-12-18 - current** - identical through half-spin 23 (steps 0-79),
then only **3 steps per half-spin** for half-spins 24-31, ending at **step 103**.
Steps 104-127 are never sampled.

.. note::

   **[CODE]** The pipeline does not hard-code either table. The per-ESA-step
   half-spin number is read from the SCI-LUT
   (``lo_stepping_tab["row_number"]["data"]``) and padded to 128 with
   ``HALF_SPIN_FILLVAL = 63`` when the table is shorter. Data at a fill-valued
   half spin is set to NaN. This is why the second scheme "just works" - the
   trailing 24 steps come back padded and get masked.

De-spinning
^^^^^^^^^^^

**[DOC]** After unpacking, Lo count arrays are (128 ESA steps x 5 or 19
positions x 12 spin sectors). They must be reformatted to (128 ESA steps x 24
spin angles x 5 or 19 positions) using:

.. list-table::
   :header-rows: 1
   :widths: 16 14 18 18 18 16

   * - Half-spin parity
     - Orientation
     - Positions
     - L0 spin sector
     - L1A spin angle
     - Position index
   * - Even
     - A
     - 1-12
     - 0-11
     - 0-11
     - SW 0-2 / NSW 0-9
   * - Even
     - A
     - 13-24
     - 0-11
     - 12-23
     - SW 3-4 / NSW 10-18
   * - Odd
     - B
     - 1-12
     - 0-11
     - 12-23
     - SW 0-2 / NSW 0-9
   * - Odd
     - B
     - 13-24
     - 0-11
     - 0-11
     - SW 3-4 / NSW 10-18

**[CODE]** ``LO_DESPIN_SPIN_SECTORS = 24``. This full de-spin mapping is only
required for the **angular** products, which are not implemented (see
:ref:`codice-implementation-status`). The species and priority products are
summed over position on board and keep the instrument's 0-11 (or fully
collapsed) spin-sector dimension.

Acquisition timing
------------------

.. note::

   In the CMAD (pages 623-709), Appendix C contains **only the first table**
   (values common to Hi and Lo). The Hi :math:`t_{acquire}` derivation and the
   Lo equations below are transcribed from the January 2026 draft, where they
   continued onto the next page. Sections 11.1 and 11.2 of Rev 3 Chg 1 still
   quote :math:`t_{acquire} = 0.59916` s for Hi and still point to "Appendix
   B" (a stale cross-reference; the timing appendix is C). The section 7
   pseudocode for Lo timing is unchanged and still omits the
   :math:`t_{minHvSettle}` floor.

**[DOC]** Appendix C. Values common to both sensors, all in microseconds unless
noted:

.. math::

   t_{sectorTime} = \mathrm{int}\!\left(\frac{P_{spinPeriod} \times 320}
   {\mathrm{NumSectors}}\right)

* ``NumSectors`` = 24 (configurable, but should not change).
* ``P_spinPeriod`` is in **spin ticks of 320 us**, initially 45687 ticks
  (14.62 s, deliberately shorter than the real spin), range 45687-48031.
  Sector times range 0.609-0.640 s; the baseline 15 s spin gives 0.625 s.
* ``t_sectorMargin`` nominal 5000 us - FSW dead time at the end of each sector.
* ``t_minHvSettle`` nominal 5000 us, ``t_maxHvSettle`` nominal 100000 us.

**CoDICE-Hi:**

.. math::

   t_{acquire} = (t_{sectorTime} - t_{sectorMargin} - t_{minHvSettle})
   \times 10^{-6} = 0.59916\ \mathrm{s}

**[CODE]** ``HI_ACQUISITION_TIME = 0.59916`` in ``constants.py``, hard-coded.
Hi does not actually need HV settling, but the same collection algorithm is used
for both sensors.

**CoDICE-Lo:** the number of ESA steps per sector varies, so:

.. math::

   t_{acquire} = \left(\frac{t_{sectorTime} - t_{sectorMargin}}
   {\mathrm{NumAcqSteps}} - t_{hvSettlePerStep}\right) \times 10^{-3}\ \mathrm{[ms]}

where

.. math::

   t_{nonAcquire} &= \mathrm{int}\!\left(\frac{t_{sectorTime}
     \times (100 - P_{dwellFraction})}{100}\right) \\
   t_{totalHvSettle} &= t_{nonAcquire} - t_{sectorMargin} \\
   t'_{hvSettlePerStep} &= \mathrm{int}\!\left(\frac{t_{totalHvSettle}}
     {\mathrm{NumAcqSteps}}\right) \\
   t_{hvSettlePerStep} &= \max\!\left(t_{minHvSettle},\
     \min(t'_{hvSettlePerStep},\ t_{maxHvSettle})\right)

``P_dwellFraction`` is nominally **95%** (to meet the L3 requirement of
collecting 95% of the time). ``NumAcqSteps`` varies 1-6 and is looked up from
the SCI-LUT Lo Stepping table.

**[CODE]** ``utils.calculate_acq_time_per_step`` implements exactly this, reading
``lo_stepping_tab["tunable_values"]`` (``spin_time_ms``, ``num_sectors_ms``,
``sector_margin_ms``, ``dwell_fraction_percentage``, ``min_hv_settle_ms``,
``max_hv_settle_ms``) and ``lo_stepping_tab["num_steps"]["data"]``. It returns a
128-element array **in seconds**, padded with NaN, and is written to L1A as
``acquisition_time_per_esa_step``.

Energy per charge
-----------------

**[DOC]** ESA sweep table entries are voltages. Energy-per-charge is:

.. math::

   E/q\ [\mathrm{keV/e}] = V \times k \times 10^{-3}, \qquad k = 5.76

**[CODE]** ``K_FACTOR = 5.76``. The voltage table for the active
``(plan_id, plan_step)`` is read from ``esa_sweep_tab`` in the SCI-LUT and
written to L1A as ``voltage_table``; ``energy_per_charge`` is derived at L1B.

.. _codice-modes:

RGFO and NSO operating modes
----------------------------

**[DOC]** Both are on-board count-rate protections on CoDICE-Lo. Neither applies
to CoDICE-Hi.

RGFO - Reduced Geometric Factor Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ratio of voltages on the upper and lower ESA plates is reduced from 1,
cutting the number of ions that pass the ESA. **The geometric factor changes**,
so the ground must know which :math:`G_m` (Full or Reduced) applied to every
bin. RGFO persists until NSO triggers or the 32 half-spin cycle ends. In the
launch configuration the FSW triggers RGFO on the **total counts on the
START-A and START-B MCP** over a half-spin (~7.5 s), with a limit tunable in
on-board LUTs. Section 4.1 gives the nominal reduction as the entrance ESA
running at 70% of the main ESA voltage (configurable).

NSO - No-Scan Operation
^^^^^^^^^^^^^^^^^^^^^^^

The ESA voltage is pinned to the highest value (step 0) and stays there until
the end of the cycle. **No further energy stepping happens**, so the summed
counts for the affected bins are not representative and **must be set to fill**.
NSO can only trigger while already in RGFO.

Telemetry
^^^^^^^^^

Every COUNTS packet carries ``RGFO_Half_Spin`` and ``NSO_Half_Spin`` (6 bits
each, 0-31). After the 2026-01-29 FSW load, both COUNTS **and PHA** packets also
carry ``RGFO_spin_sector``, ``RGFO_esa_step``, ``NSO_spin_sector`` and
``NSO_esa_step``, which pin the trigger to an exact bin rather than a half-spin.

**[CODE]** The FSW-version split is handled by having **two XTCE files**;
``codice_l1a.process_l1a`` picks one based on the date parsed out of the L0
filename:

.. code-block:: python

   if start_date >= datetime.datetime(2026, 1, 29):
       xtce_file = path / "imap_codice_packet-definition_20260129_v001.xml"
   else:
       xtce_file = path / "imap_codice_packet-definition_20250101_v001.xml"

Within a product, the branch is taken on ``packet_version`` (``<= 1`` = old
behaviour, ``> 1`` = exact-bin behaviour). When the new fields are absent they
are still created in L1A, filled with NaN, for SPDF consistency.

.. _codice-timeline:

Commissioning timeline (section 9.1)
------------------------------------

**[DOC]** Instrument behaviour has changed four times, giving five periods
(P0-P4). **Any algorithm that touches RGFO, NSO or the ESA sweep must branch on
the date.** This table is the authority for those branches. The document marks
the last period "Current" as of its publication and says to check with the
instrument team for later changes.

.. list-table::
   :header-rows: 1
   :widths: 8 22 70

   * - Period
     - Dates
     - Behaviour
   * - **P0**
     - Launch - 2025-11-24
     - RGFO triggers on total counts accumulated over a half spin; if the limit
       is passed the instrument enters RGFO on the **next half spin**. NSO
       triggers the same way while in RGFO.
   * - **P1**
     - 2025-11-24 - 2025-12-18
     - RGFO limit reduced so that **RGFO always triggers on half spin 0**, but
       the ESA voltage ratio remains 1, so the instrument is **not actually
       reduced-G**. ``RGFO_half_spin`` must be **ignored** and Full :math:`G_m`
       used. NSO still triggers on half-spin boundaries.
   * - **P2**
     - 2025-12-18 - 2026-01-29
     - ESA sweep table updated: only **3 ESA steps per half-spin from step 80**
       (576 V, 3.3 keV/e). RGFO/NSO as P1.
   * - **P3**
     - 2026-01-29 - 2026-04-03
     - FSW v1.5. RGFO/NSO trigger on **count rate within a single (ESA step,
       spin sector)** pair, switching on the following pair. New spin-sector and
       e-step fields added to all COUNTS and PHA packets. Hi/Lo priority counts
       are **lossless-only** (no lossy). **Fe highQ/lowQ labels swapped**
       (affects Lo I-ALiRT view 0, Lo SW species view 5, Lo SW angular view 7 -
       which also changed from species 13 to species 14). Two species added to
       Lo angular counts (SW He+ view 7 APID 0x486, NSW He+ view 8 APID 0x487).
       PHA allocation increased: Lo 5760 -> 11520 and Hi 5000 -> 10000 events
       per cycle. Hi priority scheme updated to P5 = Heavies, P4 = Helium,
       P3 = Protons.
   * - **P4**
     - 2026-04-03 - current
     - **Lo SCI_LUT update**: fixed an indexing issue that assigned species
       rates to the wrong rate box. **Lo on-board species classification LUT
       update**: refined mass boundaries for key species; binned **H+ is now
       TCRs only** (it was DCRs during P0-P3). **Hi on-board species
       classification LUT update**: fixed a bug that sent low-TOF events to low
       priority. Only LUT updates are listed; no FSW update.

.. note::

   The document lists **only LUT changes for P4, no FSW update**, so a new XTCE
   file or ``packet_version`` bump is not expected. (That is an inference; the
   document does not say so.) On the ground P4 arrives as a new
   ``l1a-sci-lut`` table. It does change what the Lo species (and H+) counts
   *mean* before and after 2026-04-03. The code has no date branch for P4, and
   none is needed for unpacking. Anyone comparing Lo species across that date
   should know about it. 2026-04-03 is also a mid-mission SCI-LUT change, the
   case the ``codice_l2.py`` TODO about SCI-LUT change days warns about (see
   :ref:`codice-implementation-status`).

.. warning::

   **[CODE]** The Fe highQ/lowQ label swap is handled indirectly: L1A reads the
   species ordering from the SCI-LUT (``desired_species_names`` vs
   ``actual_species_names``) and warns + fills with NaN when a wanted species is
   absent. The comment in ``codice_l1a_lo_species.py`` calls this out as
   handling "the bug in which the spacecraft was sending data down 'off by one'
   and getting mislabeled". If you see unexplained NaN species columns, check
   the SCI-LUT version first.

.. _codice-data-caveats:

Data caveats (section 9.2)
--------------------------

**[DOC]** New in Rev 3 Chg 1. "Due to issues identified post-launch", these
three Lo products are **not currently produced at any level (L1a-L2)**:
``lo-nsw-species``, ``lo-sw-angular``, ``lo-nsw-angular``. That matches the
repository, where none of the three is built. See
:ref:`codice-implementation-status`.

The caveats below apply to all CoDICE L2 data in the **first official IMAP data
release (Summer 2026)**. The document says most of them will be fixed or
mitigated in later releases. They describe the *data*, not the ground code, so
nothing here is a pipeline bug. Keep them in mind when validating output or
answering "why does this look wrong".

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - L2 product
     - Caveat
     - Effect on data
   * - ``lo-sw-species``
     - Counts are binned into SW/NSW products using **delay-line position
       instead of APD ID** (on board).
     - Lower counts than expected, so lower intensities.
   * - ``lo-sw-species``
     - Species classification boxes need fine tuning.
     - Charge-state species are not well separated / identified.
   * - ``lo-direct-events``
     - Delay-line position is **not an accurate identifier of particle
       direction**; position should only be used to support APD ID.
     - (as stated)
   * - ``lo-direct-events``
     - Automatic TOF correction based on position.
     - Larger TOF uncertainty.
   * - ``hi-omni``, ``hi-sectored``
     - Detection threshold depends on SSD ID.
     - Increased noise at lower energies.
   * - ``hi-omni``, ``hi-sectored``
     - ASIC baseline fluctuates with count rate, producing pseudo-TCRs for
       low-energy data.
     - Increased noise / false counts at lower energies.
   * - ``hi-omni``, ``hi-sectored``
     - On-board mass computation needs correction.
     - Mass tracks trend downward at low energy-per-nuc.
   * - ``hi-omni``, ``hi-sectored``, ``hi-direct-events``
     - MG and LG energy channels not well calibrated.
     - Upper part of MG usable; lower part of MG and **all of LG not currently
       usable**.

.. note::

   Two of these bear on ground-code choices already documented here:

   * The Lo **position vs APD ID** caveat is relevant to
     ``process_lo_direct_events``, which derives elevation from ``apd_id`` but
     de-spins spin sectors using ``position``. See
     :ref:`codice-implementation-status`.
   * The Hi **LG/MG** caveat means ``GAIN_ID_TO_STR`` values 1 (LG) and 2 (MG)
     label events whose energies are, for now, partly or wholly unreliable. The
     pipeline still converts them; it does not flag them.

Operational modes
-----------------

**[DOC]** CoDICE has four operational modes: **SAFE**, **LVENG** (low-voltage
engineering), **HVENG** (high-voltage engineering) and **Science**. Everything
in the algorithm document and in this pipeline applies to Science mode only.
