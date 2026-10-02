.. _mission-overview:

Mission Overview
================

This page is the mission-level context. It answers three questions: what IMAP is trying to find out, why
that requires these particular ten instruments, and why several of them appear
to measure the same thing.

It is deliberately short. Each instrument's own ``overview`` page covers how
that instrument works; this page only covers **why it is on the spacecraft.**

What IMAP is
------------

The Interstellar Mapping and Acceleration Probe is a NASA
Heliophysics Solar Terrestrial Probe built by a team of 25 partner institutions.
It carries **ten instruments** on a simple Sun-pointed spinner orbiting the
**Sun-Earth L1 Lagrange point**, and launched on **24 September 2025**. Its design is a follow up to the IBEX mission.

* Spins at **4 RPM**, like IBEX.
* Unlike IBEX, the spin axis is **repointed roughly 1° each day** to track the
  nominal solar wind aberration direction (~4° off the Sun in the ecliptic).
  Tracking the average solar wind direction, plus being far from terrestrial
  backgrounds, is why IMAP's ENA measurements are substantially cleaner than
  IBEX's.
* Three instruments have two sensor heads each, and one sits on its own
  dedicated pivot platform.
* IMAP was the top new mission priority in the 2013 Heliophysics Decadal
  Survey, formed by merging two separate white-paper proposals - one for
  expanded ENA imaging after IBEX, one for in-situ particle acceleration
  measurements.

That merger matters, and it is the reason this payload can look like two missions
bolted together. The Decadal Survey group judged the two proposals
*"not just complementary, but synergistic, as some of the particles accelerated
in the inner heliosphere are ultimately 'recycled' through charge exchange in
the outer heliosphere and return to L1 as ENAs."*


The science objectives
----------------------

IMAP is framed around **two coupled topics**:

1. the **acceleration of charged particles**, and
2. the **interaction of the solar wind with the local interstellar medium**
   (LISM, or VLISM for the *very* local interstellar medium).

These are coupled because particles accelerated in the inner heliosphere
propagate outward and then *mediate* that interaction.

Formally, the mission is organised around **four Science Objectives** from the
NASA Announcement of Opportunity, listed from the LISM inward. Instrument pages
refer to these as O1-O4:

.. list-table::
   :header-rows: 1
   :widths: 6 94

   * -
     - Objective
   * - **O1**
     - Improve understanding of the **composition and properties of the LISM**.
   * - **O2**
     - Advance understanding of the **temporal and spatial evolution of the
       boundary region** in which the solar wind and the interstellar medium
       interact.
   * - **O3**
     - Identify and advance understanding of processes related to the
       **interactions of the magnetic field of the Sun and the LISM**.
   * - **O4**
     - Identify and advance understanding of **particle injection and
       acceleration** near the Sun, in the heliosphere and heliosheath.

A third, operational goal explains a large amount of code in this repository.
**I-ALiRT** (IMAP Active Link for Real-Time) continuously
telemeters real-time space weather data from **SWAPI, CoDICE, HIT, SWE and
MAG**, which the SOC analyses and posts with a **latency of under 5 minutes**.

Imaging a boundary you cannot visit
-----------------------------------

The heliosphere is the bubble the solar wind inflates in the VLISM.
Its boundary region - the termination shock, the heliosheath beyond it, and the
heliopause separating heliospheric plasma from the VLISM - extends from hundreds
to roughly **1000 au** in the upwind direction. You cannot survey that by flying
to it.

Instead IMAP images it with **energetic neutral atoms**. An ion in the
heliosheath charge-exchanges with a neutral atom and becomes neutral; having no
charge, it is no longer steered by magnetic fields, so its arrival direction at
1 au still points back to where it was born. Three instruments image this
across overlapping energy ranges: **IMAP-Lo, IMAP-Hi and IMAP-Ultra**.

The difficulty is that an ENA map depends on both the ion population out there
*and* the neutral density it charge-exchanged with - and the latter depends on
the solar wind's history. **This is why the in-situ instruments exist.** They
are not a parallel experiment; they supply the boundary conditions that make the
maps interpretable.

That dependency is concrete, not rhetorical. GLOWS light curves
yield heliolatitude profiles of the ISN hydrogen ionization rate; those decompose
into photoionization and charge-exchange rates, and the charge-exchange rates
into **profiles of 3-D solar wind speed and density**. Those profiles are then
*"used to calculate survival probabilities of ENAs observed by IMAP."* In this
repository that appears as GLOWS L3 producing ENA survival probabilities for Lo,
Hi and Ultra. (:ref:`glows`)

Why pickup ions
---------------

The solar wind *"picks up locally ionized interstellar neutrals
drifting into the heliosphere, creating the PUI population."* Interstellar
neutrals cross into the heliosphere unimpeded because they carry no charge; once
ionized, the solar wind sweeps them up and carries them outward. They are
recognised in an E/q spectrum by a characteristic **PUI cutoff**, and SWAPI is
specified to observe the He+ distribution *"from low energies to beyond the PUI
cutoff."*

Pickup ions serve both goals at once.

* **They are a sample of the LISM delivered to 1 au.** Combining SWAPI, CoDICE,
  IMAP-Lo and GLOWS *"allows determination of the LISM flow properties with
  unprecedented accuracy"* (O1).
* **They are the seed population for acceleration.** *"Suprathermal ions,
  including PUIs, serve as seed populations for acceleration at interplanetary
  shocks and within the magnetosphere"* (O4).
* **They mediate the interaction itself.** The solar wind flows outward
  *"incorporating an increasing fraction of PUIs and PUI pressure all the way
  out to the termination shock"* (O2).

IMAP also distinguishes **interstellar** PUIs from **inner source** ones, and
asks explicitly *"what is the composition of dust at 1 au that provides a seed
population for the inner source PUIs?"* - which is part of why a dust instrument
shares the spacecraft.

.. _mission-overview-instruments:

The ten instruments
-------------------

Ranges, resolutions and cadences are from Table 4 of the mission
paper; the objective mapping is from each instrument's own section where the
paper states it explicitly.

.. list-table::
   :header-rows: 1
   :widths: 13 22 65

   * - Instrument
     - Range
     - Observable and contribution
   * - **SWAPI**
     - 0.1-20 keV/q
     - Solar wind H+ and He++ **and the lighter PUIs (H+ and He+)**, as 1-D
       VDFs. Solar wind bulk parameters at ~12 s; PUI He+ at ~10 min, enabling
       the PUI gravitational focusing cone. **O1, O4.**
   * - **CoDICE-Lo**
     - 0.5-80 keV/q
     - Solar wind and suprathermal ions **with composition and charge states**,
       3-D VDFs: solar wind He-Fe and **interstellar pickup He, O and Ne**.
       m/Δm ≥ 2.
   * - **CoDICE-Hi**
     - 0.03-5 MeV/nuc
     - Suprathermal and energetic ion mass composition and arrival direction.
       m/Δm ≥ 4. See :ref:`mission-overview-discrepancies`.
   * - **HIT**
     - Ions 2-50 MeV/nuc
       (species dependent);
       electrons 0.5-1 MeV
     - Energetic H-Ni composition, spectra, angular distributions and arrival
       times, linked to CoDICE suprathermals. Resolves 3He, a tracer of SEP
       acceleration. 8 of 10 apertures view the full sky; 2 are modified for
       electrons. **O4.**
   * - **SWE**
     - 1-5000 eV
     - Solar wind electron 3-D VDFs. Pitch-angle distributions diagnose
       magnetic topology. **[REPO]** (:ref:`swe-overview`)
   * - **MAG**
     - ±512 nT / ±60,000 nT
       (auto-ranging)
     - Vector interplanetary magnetic field, 2 Hz (64 Hz for ~8 hours/day).
       Needed by any pitch-angle calculation anywhere in the pipeline.
   * - **IMAP-Lo**
     - 5-1000 eV
     - ISN **and** ENA flux and composition. On a **pivot platform**, which
       breaks the measurement degeneracy in ISN flow parameters. Tracks ISN H,
       He, O, Ne and D over >180° of ecliptic longitude. **O1-O3.**
   * - **IMAP-Hi**
     - 0.41-15.6 keV FWHM
     - Hydrogen ENA flux maps in **nine contiguous energy passbands**. Two
       identical single-pixel cameras, 4.1° FWHM conical FOV: **Hi-90**
       perpendicular to the spin axis, **Hi-45** at 45° anti-sunward. Hi-90
       makes a full sky map every 6 months. **O2-O3.**
   * - **IMAP-Ultra**
     - ENA 3-300 keV;
       ions 3-5000 keV
     - The highest-energy ENAs, 2° angular resolution for H above 30 keV. Two
       slit-optics imagers at 45° and 90° covering **~3π sr per spin**; full sky
       map every 3 months. Near-copy of JUICE/JENI, with ~35x the collecting
       power of Cassini/INCA. **O2-O4.**
   * - **IDEX**
     - 2x10⁻¹³ - 5x10⁻¹¹ g;
       1-286 amu
     - Interstellar and interplanetary dust composition by impact-ionization
       TOF mass spectrometry, m/Δm > 120 at 56 amu. **Links the interstellar
       gas-phase composition from IMAP-Lo and the PUI measurements from CoDICE
       and SWAPI to the composition of dust grains.** **O1.**
   * - **GLOWS**
     - 120.5 ± 4.3 nm
     - Hydrogen Lyman-α helioglow light curves along Sun-centred rings, one per
       pointing. Yields heliolatitude profiles of 3-D solar wind speed and
       density, and thence ENA survival probabilities. **O1, O2.**

.. _mission-overview-ladder:

The energy ladder and deliberate overlaps
------------------------------------------

The three ENA cameras *"have overlapping energy ranges that roughly
match in-situ ion measurements above."* That matching is the design, not a
coincidence:

.. code-block:: text

   in-situ ions        SWAPI        0.1  - 20   keV/q
                       CoDICE-Lo    0.5  - 80   keV/q
                       CoDICE-Hi    0.03 - 5    MeV/nuc
                       HIT          2    - 50   MeV/nuc

   ENA imaging         IMAP-Lo      5    - 1000 eV
                       IMAP-Hi      0.41 - 15.6 keV
                       IMAP-Ultra   3    - 300  keV

The overlaps were engineered and then verified on the ground. Three
pairs - **SWAPI & CoDICE, IMAP-Lo & IMAP-Hi, and IMAP-Hi & IMAP-Ultra** - were
cross-calibrated *"as separate pairs in the same vacuum chamber at the same
time"*, rotated into steady ion and neutral beams across their overlapping
ranges. Instruments with overlapping ranges continue to cross-calibrate in
flight, and IMAP-Lo and IMAP-Hi are additionally cross-calibrated against
IBEX-Lo and IBEX-Hi for as long as IBEX survives.

This is what the ladder buys: following one population from thermal solar wind,
through the pickup shell, into the suprathermal tail and out to energetic
particle energies, as a single cross-calibrated spectrum. A gap loses the trail.

It also explains a structural oddity in this repository. **CoDICE-Hi's neighbour
in the spectrum is HIT, not CoDICE-Lo** - the paper says HIT links its
measurements to *"suprathermal measurements from CoDICE"*. Lo hands off to Hi,
Hi hands off to HIT. CoDICE is one box spanning a seam that falls in its middle;
see :ref:`codice-overview` for how deep that split runs in the code.

Why two instruments measure pickup ions
---------------------------------------

SWAPI and CoDICE both observe pickup ions and are the cross-calibrated pair in
that energy range - but they measure different properties, and the paper's own
wording separates them cleanly. SWAPI measures *"the lighter species
of PUIs (H+ and He+)"*; CoDICE-Lo measures *"interstellar pickup He, O, and Ne
ions"* with composition and charge state.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - SWAPI
     - CoDICE-Lo
   * - Measures
     - **E/q only** - no mass, no charge state
     - E/q **plus** TOF and residual energy, giving **M, q and M/q** per
       particle **[REPO]** (:ref:`codice-overview`)
   * - Species ID
     - Inferred from the *shape* of the E/q spectrum; L3 is model fitting
       **[REPO]** (:ref:`swapi-l3-scope`)
     - Measured directly, per event
   * - Range
     - 0.1-20 keV/q
     - 0.5-80 keV/q
   * - PUI species
     - H+ and He+ - the light ones
     - He, O and Ne - the heavy ones
   * - Cadence
     - ~12 s solar wind, ~10 min PUI He+
     - ≤ 1 hour
   * - Strength
     - Energy resolution and cadence on the dominant species
     - Breadth of species and charge states

So SWAPI supplies the precise *shape* of the distribution for the two species
that dominate - fast enough to resolve the PUI gravitational focusing cone - and
CoDICE supplies the *inventory* of which heavier elements and charge states are
present.

CoDICE has a second, unrelated solar wind job worth knowing about: charge-state
ratios such as O7+/O6+ and C6+/C5+ freeze in close to the Sun and are unchanged
by transport, so they reach L1 as a record of coronal conditions. The mission paper
notes these charge-state ratios as one of the novel I-ALiRT measurements
improving on ACE. In this repository they are the L3a ratio products and the
I-ALiRT pseudo-density ratios.(:ref:`codice-l3-scope`,
:ref:`codice-ialirt`)

.. _mission-overview-discrepancies:

Where the numbers disagree
--------------------------

.. warning::

   The mission paper and the instrument algorithm documents do not always agree
   on energy ranges, and **the mission paper is not internally consistent
   either.** Do not "fix" one to match the other.

   .. list-table::
      :header-rows: 1
      :widths: 16 28 28 28

      * - Instrument
        - Mission paper Table 4
        - Mission paper prose
        - Instrument pages here
      * - **CoDICE-Hi**
        - 0.05-2 MeV/nuc
        - ~0.03-5 MeV/nuc (§4.2 opening); *"~0.03 and >2 MeV/nuc"* a paragraph
          later
        - ~0.03-5 MeV/nuc (:ref:`codice-overview`)
      * - **HIT**
        - Ions 2-70 MeV/nuc
        - 2-50 MeV/nuc, *"species dependent"*
        - ~2-40 MeV/nuc (:ref:`hit-overview`)
      * - **IMAP-Lo**
        - 5-1000 eV
        - ENA maps *"down to 100 eV and below and up to 1 keV"*
        - ENAs 40 eV - 1 keV (:ref:`lo-overview`)

   These are mostly the same instrument described at different confidence
   levels and with different qualifiers - ``species dependent`` does a lot of
   work in the HIT row, and the IMAP-Lo rows differ because Table 4 covers ISN
   and ENA together while the other two quote the ENA range. **For anything
   that affects code, the instrument algorithm document wins**, per the
   convention on each instrument index page. This page quotes Table 4 because
   it is the only self-consistent mission-wide set.

.. _mission-overview-sources:

Sources
-------

**Primary source for this page:**

   D.J. McComas et al., *Interstellar Mapping And Acceleration Probe: The NASA
   IMAP Mission*, Space Science Reviews (2025) **221**:100, 82 pp.
   `doi:10.1007/s11214-025-01224-z
   <https://doi.org/10.1007/s11214-025-01224-z>`_

This paper is **open access**
(CC BY-NC-ND 4.0), so it can be linked and quoted freely. It is also the citable
reference for the mission-level **CMAD** (Calibration and Measurement Algorithms
Document), supplied as a supplemental file to the paper.

.. tip::

   If you hold a copy, put it in ``docs/reference/``. That directory is
   gitignored, so it will never be committed.

The paper is the lead article in a **17-paper IMAP collection** in Space Science
Reviews, which includes a dedicated paper for each of the ten instruments. When
an instrument page here lacks the background you need, the relevant paper is:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - SWAPI
     - Rankin et al. 2025
     - CoDICE - Livi et al. 2025
   * - HIT
     - Christian et al. 2025
     - SWE - Skoug et al. 2025
   * - MAG
     - Horbury et al. 2025
     - IMAP-Lo - Schwadron et al. 2025
   * - IMAP-Hi
     - Funsten et al. 2025
     - IMAP-Ultra - Gkioulidou et al. 2025
   * - IDEX
     - Horányi et al. 2025
     - GLOWS - Bzowski et al. 2025
   * - I-ALiRT
     - Lee et al. 2025
     - Observatory - Hegarty et al. 2025

The statements on this page come from the instrument overview pages
cited inline, each of which names its own algorithm document; see the
``Source documents`` section of any instrument index, for example
:ref:`codice-source-documents`.

Where to go next
----------------

* For how an instrument works: its ``overview`` page, linked from
  :ref:`algorithm-code-documentation`.
* For what it produces and what the files are called: its ``data-products``
  page.
* For what is actually built versus merely specified: its
  ``implementation-status`` page.
