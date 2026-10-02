.. _swapi-overview:

Instrument and Mission Concepts
===============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Everything on this page is background needed to read the algorithm pages. It is
mostly **[DOC]** (algorithm document sections 4, 5, 6 and 7).

What SWAPI measures
-------------------

SWAPI (Solar Wind and Pickup Ion) is one of ten instruments on IMAP, built by
the Space Physics Group at Princeton University. It sits at the Sun-Earth L1
Lagrange point and measures four ion populations:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Population
     - Symbol
     - Notes
   * - Solar wind protons
     - H\ :sup:`+`
     - The bright core beam. Enters through the attenuating grid.
   * - Solar wind alphas
     - He\ :sup:`2+`
     - Sits at roughly twice the proton E/q. Also attenuated.
   * - Interstellar pickup helium
     - He\ :sup:`+`
     - The primary pickup-ion science target. Helium dominated.
   * - Interstellar pickup hydrogen
     - H\ :sup:`+`
     - Present but much weaker at 1 au than in the outer heliosphere.

The instrument is a modified version of New Horizons' **SWAP** (Solar Wind
Around Pluto). The heritage matters: the pickup-ion fitting code is inherited
from SWAP, and the I-ALiRT analytical model is the SWAP model of
Elliott et al. (2016). SWAP's own ground processing also stopped at L2.

.. note::

   **[DOC]** Heritage NH-SWAP code is *not* reused for L0-L2 processing. The
   processing *steps* were adapted, drawing on the SWAP team's decades of
   operations, but the code here is new. Only the SWAP **PUI** code is reused
   and modified, and that happens at L3 - outside this repository.

How a single measurement happens
--------------------------------

This chain explains nearly every variable name in the data products.

1. An ion enters through one of two paths:

   * the **sun-facing aperture grid** ("the sunglasses"), which is **0.1%
     transmissive**. This attenuates the solar wind so the detector is not
     saturated, without attenuating anything from the other directions.
   * the **open aperture**, unattenuated, which is where pickup ions come from.

2. It passes through the toroidal **electrostatic analyzer (ESA)**, which
   selects a narrow band of **energy per charge (E/q)**. The ESA also blocks UV
   light and neutrals.
3. It crosses a **field-free flight path** and is post-accelerated into the
   detector section.
4. It passes through an **ultrathin carbon foil**, liberating secondary
   electrons.
5. The ion itself lands on the **primary CEM (PCEM)**; the secondary electrons
   are steered onto the **secondary CEM (SCEM)**.
6. Pulses from both are amplified and accumulated in counters. If a PCEM and an
   SCEM event fall within a **100 ns window**, a **coincidence (COIN)** is also
   registered.

Three counters, one measurement
-------------------------------

Every SWAPI science measurement is the triple (PCEM, SCEM, COIN) for one ESA
step.

.. list-table::
   :header-rows: 1
   :widths: 14 18 68

   * - Counter
     - Also called
     - Meaning
   * - ``PCEM``
     - PRM, primary
     - Ions that passed through the carbon foil and hit the primary CEM.
   * - ``SCEM``
     - SEC, secondary
     - Secondary electrons liberated from the carbon foil.
   * - ``COIN``
     - coincidence
     - Both within 100 ns. **This is the science channel** - it is what the
       L3 fits and the I-ALiRT product use, because it has by far the lowest
       background.

**[DOC]** The three-counter arrangement lets absolute detection efficiency be
computed on orbit, independent of how the individual detectors age
(Funsten et al. 2005):

.. math::

   \varepsilon = \frac{\mathrm{COIN}^2}{\mathrm{PRM} \times \mathrm{SEC}}

This is why all three are carried all the way through L2 even though only COIN
is fit downstream.

High voltage
------------

**[DOC]** Three separate high-voltage power supplies (ESA, PCEM, SCEM), all
controlled by an 8051 microcontroller.

* The **ESA** steps its voltage through the sweep, up to **10.2 kV**.
* The **CEM** voltages are held at a steady value, up to **4 kV**. They are
  raised over the mission to compensate for CEM gain decay - that is what the
  periodic gain test is for (see :ref:`swapi-ancillary`).

Sweeps, steps, and the 12-second cadence
----------------------------------------

This is the single most important set of definitions in SWAPI processing.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Concept
     - Definition
   * - **Spin**
     - One spacecraft rotation about the Sun-pointing axis. **4 rpm, so ~15 s.**
       SWAPI does not use spin as an aggregation unit, but L3 needs spin phase
       per ESA step.
   * - **Sweep**
     - **12 seconds, 72 ESA steps.** The fundamental unit of L1 and L2: one
       record in the L1/L2 CDF is one sweep. ``NUM_ENERGY_STEPS = 72``.
       **[CODE]**
   * - **ESA step**
     - One energy setting. **6 steps per second, so 0.167 s per step.** Only
       **0.145 s** of that is actually counting - see live time below.
   * - **Packet**
     - One ``SWP_SCI`` packet holds **1 second = 6 ESA steps**, and is a fixed
       **54 bytes / 432 bits**. **12 packets make one sweep**;
       ``NUM_PACKETS_PER_SWEEP = 12``. **[CODE]**
   * - **Sequence number**
     - ``SWP_SCI.SEQ_NUMBER``, 0-11, identifies which 6-step group of the
       sweep a packet holds. ``0`` marks the start of a sweep, ``11`` the end.
       All 12 must be present to process the sweep.
   * - **Coarse sweep**
     - The **62 fixed E/q steps** covering **0.1-20 keV/q**, high energy to low
       energy. These have fixed, tabulated energies.
   * - **Fine sweep**
     - **9 variable steps**, placed according to the active *sweep plan*. Their
       energies are not fixed and must be solved for at L2.
   * - **Ramp-up step**
     - **1 step** at the start of the sweep, used to transition the ESA from
       the low fine-sweep voltage back to full voltage. It carries no usable
       science. **[CODE]** ``swapi_l1`` sets index 0 of all three count arrays
       to ``NaN``.
   * - **Live time**
     - ``SWAPI_LIVETIME = 0.145`` s. The actual accumulation time per energy
       bin, less than the 0.167 s step duration because the HVPS needs settling
       time. **[CODE]** ``swapi_l2.SWAPI_LIVETIME``.
   * - **Five-sweep chunk**
     - **60 s = 5 sweeps = 4 spacecraft spins.** 5 is the smallest number of
       12-second sweeps that closes on the ~15 s spin period, so a five-sweep
       chunk samples spin phase evenly. This is the L3a proton/alpha cadence
       and the I-ALiRT averaging window.
   * - **Fifty-sweep chunk**
     - **10 minutes.** The L3a pickup-helium and L3b cadence.

So: ``72 steps = 1 ramp-up + 62 coarse + 9 fine``, and
``12 s / 72 steps = 0.167 s/step``, of which ``0.145 s`` counts.

.. warning::

   **Step ordering trap.** The coarse sweep runs from the **highest** energy to
   the **lowest** (ESA step 0 is ~1.2 keV at the top of the ramp, step 62 is
   ~107 eV in the 2025-02-11 table). Energy is *decreasing* with step index
   through the coarse sweep, then the 9 fine steps jump back up around the
   solar wind peak. Never assume step index is monotonic in energy across the
   whole 72.

Sweep plans
-----------

**[DOC]** The 9 fine steps are programmable. There are **16 sweep plans**; the
active plan is telemetered per packet as ``SWP_SCI.PLAN_ID``, and the table
within the plan as ``SWP_SCI.SWEEP_TABLE``.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Plan
     - Fine-step allocation
   * - **Plan 4**
     - **The nominal operations plan since 2026-02-01.** 3 fine steps dedicated
       to low-energy background at ESA voltages **35 V, 20 V and 5 V**, plus
       **6 fine steps** distributed above and below the peak E/q bin.
   * - **Plan 5**
     - Used for electron-hoovering background tests. All 9 fine steps at ESA
       voltages **40, 35, 30, 25, 20, 15, 10, 4 and 0 V**.

.. important::

   The background fine steps exist because the L1 background at L1 is not
   constant - it varies with solar UV output (Bzowski et al. 2013; Sokol et al.
   2013) and with galactic/anomalous cosmic ray intensity (Leske et al. 2013).
   **[DOC]** L3 excludes these background bins from the solar wind fit and uses
   them to constrain the background level instead. At L1 and L2 they are
   ordinary energy steps and get no special treatment.

Both ``plan_id`` and ``sweep_table`` are carried into the L1 and L2 CDFs, and
``sweep_table`` is the key used to pick the right rows out of the ESA unit
conversion table at L2. **[CODE]**

Telemetry types
---------------

**[DOC]** There are seven SWAPI telemetry types. Note that SWAPI produces no
summary or histogram data (SWAP did), but does produce large science, I-ALiRT
and autonomy data.

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 58

   * - Telemetry
     - APID
     - Kind
     - Description
   * - ``SWP_HK``
     - 1184
     - Engineering
     - Housekeeping: instrument status, counters, ADC values. In HVSCI the HK
       packet nominally comes every **60 s**, but is sampled at 1 Hz. 102
       fields in the XTCE. **Processed.**
   * - ``SWP_SCI``
     - 1188
     - Science
     - PCEM, SCEM and COIN counts for each of 6 samples in a 1-second period.
       Reports the ESA level for a single sample (``ESA_LVL5``); the other
       levels come from the lookup table. 46 fields. **Processed.**
   * - ``SWP_IAL``
     - 1187
     - Science
     - I-ALiRT: a subset of the science packet carrying **coincidence counts
       only**. **Processed, under** ``imap_processing/ialirt/``.
   * - ``SWP_LGSCI``
     - --
     - Science
     - "Large" science - same as SCI but includes all 6 ESA levels rather than
       one. Used for testing. **Not processed; not in the XTCE.**
   * - ``SWP_AUT``
     - 1192
     - Engineering
     - Autonomy: minimum required per ICD (power off/cycle flags). **Enum
       exists in** ``SWAPIAPID`` **but not processed and not in the XTCE.**
   * - ``SWP_MG``
     - --
     - Engineering
     - Asynchronous event message packets. **Not processed.**
   * - ``SWP_MD``
     - --
     - Engineering
     - Memory dump, fixed 128-byte payload. **Not processed.**

**[DOC]** CCSDS packets have variable (even) length up to 4096 bytes including
headers, per the GI ICD. All ``SWP_SCI`` packets are fixed size. **All packet
types carry a CHKSUM parameter**, computed on board and intended to be
recomputed on the ground as a data check.

Instrument modes
----------------

``SWP_SCI.MODE`` / ``SWP_HK.MODE``. **[CODE]** ``swapi_utils.SWAPIMODE``:

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - Value
     - Name
     - Meaning
   * - 0
     - ``LVENG``
     - Low voltage, engineering
   * - 1
     - ``LVSCI``
     - Low voltage, science
   * - 2
     - ``HVENG``
     - High voltage, engineering
   * - 3
     - ``HVSCI``
     - **High voltage, science - the only mode that produces valid science.**
       Every packet of a sweep must be in HVSCI or the sweep is dropped.

Apertures, regions, and the response function
---------------------------------------------

**[DOC]** The instrument response is decomposed by *azimuthal region*, because
the sunglasses only cover part of the aperture. The azimuth angle
:math:`\phi` is measured in instrument coordinates:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Region
     - Azimuth range
     - Transmission
   * - Sunglasses (**SG**)
     - :math:`|\phi| \le 20^\circ`
     - :math:`T = 10^{-3}` across the flat central part
       (:math:`|\phi| \le 9^\circ`)
   * - Open aperture (**OA**)
     - :math:`20^\circ \le |\phi| \le 150^\circ`
     - :math:`T = 1` across the flat part
       (:math:`31^\circ \le |\phi| \le 115^\circ`)

The full effective-area function factorizes as

.. math::

   \mathcal{A}^s(v,\theta,\phi,V) =
   \mathcal{A}^s_0(V)\;
   P_{\mathrm{region}(\phi)}\!\left(\frac{v}{v_0^s},\theta,V\right)\;
   T(\phi)

where :math:`\mathcal{A}^s_0` is the central effective area,
:math:`P_r` is the region-specific energy-angle passband, :math:`T(\phi)` is
the azimuthal transmission tabulated from 0 to 180 degrees at 0.1 degree
spacing, and the central speed is
:math:`v_0^s = \sqrt{2 k^{*} q^s |V| / m^s}`.

**These three functions are ancillary CSVs used only by L3.** They are listed
here because they are the reason L2 must report ESA *energy* in a way that can
be converted back to ESA *voltage* - see the ``k`` factor below.

.. _swapi-k-factor:

The k factor
------------

**[DOC]** Two values are in play and they are not the same:

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Symbol
     - Value
     - Use
   * - :math:`k^{*}`
     - 1.89 eV/V/e
     - The peak :math:`(E/q)/|V|` at :math:`\theta = 0`, from high-resolution
       SIMION simulations. Used to normalize the L3 response functions.
   * - :math:`k_{L2}`
     - 1.93 eV/V/e
     - Estimated pre-launch from lab measurements (Rankin et al. 2025). **This
       is the factor used to convert the ESA energy in the L2 CDF files back to
       the actual ESA voltage of the instrument.**

The discrepancy is believed to come from inaccuracies in beam energy and
orientation in the lab measurements. It is still under investigation; as of the
initial IMAP data release, L3 uses the SIMION :math:`k^{*}`.

**[CODE]** The pipeline never applies a ``k`` factor. The ESA unit conversion
table carries a ``K factor`` column (1.88 in the 2025-02-11 rows, 1.93 in the
2025-05-19 rows) but ``swapi_l2`` reads only the ``Energy`` column and ignores
``K factor`` and ``Voltage`` entirely. This is correct as far as it goes -
energies in the table are already energies - but it means the L2 product does
not record which ``k`` was used to build them.

Energy resolution and effective area
------------------------------------

**[DOC]** Numbers quoted by the algorithm document for the solar wind:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Quantity
     - Value
     - Where used
   * - :math:`\Delta E / E` (FWHM)
     - 0.085
     - I-ALiRT passband width. **[CODE]**
       ``IalirtSwapiConstants.fwhm_width``
   * - Speed width :math:`\Delta v / v`
     - :math:`\tfrac{1}{2}\,\Delta E/E` = 0.0425
     - I-ALiRT. **[CODE]** ``IalirtSwapiConstants.speed_ew``
   * - Effective area :math:`A_{\mathrm{eff}}`
     - :math:`1.633 \times 10^{-4}`\  cm\ :sup:`2`
     - I-ALiRT analytical model. **[CODE]**
       ``IalirtSwapiConstants.eff_area`` (converted to m\ :sup:`2`)
   * - Azimuthal FOV :math:`\Delta\phi`
     - 30 degrees
     - I-ALiRT. **[CODE]** ``IalirtSwapiConstants.az_fov``
   * - Detector deadtime :math:`\tau`
     - 183.7 ns
     - **L3 only.** 5% correction at ~2.7e5 Hz, routine for high-flux solar
       wind. Not applied at L1 or L2.

Definitions of terms
--------------------

**[DOC]** Algorithm document section 5, reproduced because these units appear
verbatim in the CDF attributes.

.. list-table::
   :header-rows: 1
   :widths: 20 56 24

   * - Term
     - Definition
     - Unit
   * - Bulk velocity
     - Measure of the peak of the solar wind distribution
     - km/s
   * - Count
     - Number of particles recorded on the instrument
     - #
   * - Count rate
     - Counts per unit time
     - #/s
   * - Density
     - Number of particles per unit volume
     - #/cm\ :sup:`3`
   * - Efficiency
     - Correction factor applied in L1-L2 processing to correct for particle
       detection which may change over time
     - dimensionless
   * - Flux
     - Particle count rate per unit area per unit solid angle per unit
       energy/charge
     - #/[cm\ :sup:`2` s sr eV/q]
   * - Geometric factor
     - Energy width per energy multiplied by effective area,
       :math:`A_{\mathrm{eff}} \cdot \Delta E/E`
     - cm\ :sup:`2` sr eV/eV
   * - Temperature
     - Measure of the broadness of the distribution
     - K

.. note::

   The document's own definition of *efficiency* says it is "applied in L1-L2
   processing". **[CODE]** It is not. Efficiency is applied where the flux is
   formed, which is L3b (equation 13), and the L2 product is a bare count rate.
   Treat the definition table as a glossary, not as a specification of where
   the correction lives.

Reference frames
----------------

**[DOC]** Frames named by the algorithm document. None of them are used by any
code in this repository - L1 and L2 are frame-free (counts and rates per ESA
step). They are listed so that the L3 equations can be read.

* **Instrument frame** - speed :math:`v`, elevation :math:`\theta`, azimuth
  :math:`\phi`. The frame the response function is tabulated in.
* **Spacecraft RTN frame** - :math:`(v_R, v_T, v_N)`. The frame the L3a bulk
  velocity is fit in. Per-ESA-step SWAPI-to-RTN rotation matrices come from
  SPICE.
* **Solar inertial frame** - the Sun rest frame. L3a also reports velocity here,
  obtained by adding the spacecraft velocity from SPICE.
* **GSE** - used for the end-to-end model and for comparison with WIND/SWE
  (:math:`v_R \approx -V_{x,\mathrm{GSE}}`,
  :math:`v_T \approx -V_{y,\mathrm{GSE}}`,
  :math:`v_N \approx V_{z,\mathrm{GSE}}`).
* **Ecliptic J2000** - the frame the interstellar neutral inflow directions are
  quoted in (He from 255.7, 5.1 degrees at 25.4 km/s; H from 252.2, 9.0 degrees
  at 22 km/s).

Data volume
-----------

**[DOC]** The SWAPI team anticipates **4.786560 MB/day** of raw data and
**22 MB/day** of processed data across L1, L2 and L3.
