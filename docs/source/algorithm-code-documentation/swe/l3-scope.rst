.. _swe-l3-scope:

L3 Scope - What Is Not in This Repository
=========================================

.. warning::

   **SWE L3 is not produced here.** ``imap-processing`` takes SWE to L2. A
   separate repository, run closer to the science team, takes L2 to L3.

   If a task sounds like fitting Maxwellians, finding a spacecraft potential,
   computing a pitch angle, or integrating a moment, **it does not belong in
   this repository.** Stop and check before writing code.

Section 3.4.6 of the algorithm document is four of its 38 pages and is by far
the most algorithmically dense part of the whole document. It is summarized
here for one reason only: **so you can tell what L2 owes L3**, and recognise an
L3 task when one arrives misfiled.

The contract - what L3 needs from L2
------------------------------------

**[DOC]** "Level 3 processing starts from the SWE Level 2 data files. In
particular, Level 3 processing will use Level 2 variables which contain the
electron phase space distributions as a function of energy, polar angle (CEM)
and spin angle (SWE azimuthal angle bin) **in despun spacecraft coordinates for
each individual nominal 80 millisecond measurement** (7 CEM detectors x 24 ESA
voltage x 30 spin angle bins per SWE full cycle). The Level 2 files also include
the energy corresponding to each ESA voltage step, and the polar angle
corresponding to each CEM detector."

The rationale is explicit: "For pitch angle calculations, starting from this
full data set will allow calculation of pitch angle using the magnetic field
vector measurement from MAG **during each nominal 80 millisecond SWE data
acquisition period**." Pitch angle is computed per measurement, not per bin -
so L2 must not throw away per-measurement resolution.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - What L3 needs
     - L2 variable
     - State
   * - Per-measurement phase space density
     - ``phase_space_density_spin_sector``
     - Present, ``(epoch, 24, 30, 7)``.
   * - Per-measurement intensity
     - ``flux_spin_sector``
     - Present.
   * - Energy per ESA step
     - ``energy`` coordinate; ``esa_energy`` at L1B
     - Present.
   * - Polar angle per CEM
     - ``inst_el`` coordinate
     - Present.
   * - Per-measurement spin angle
     - ``inst_az_spin_sector``
     - Present, **but in the instrument frame, not despun spacecraft**.
   * - Per-measurement time stamp
     - ``acquisition_time``
     - Present, center of the accumulation window, MET seconds.
   * - Accumulation duration
     - ``acq_duration``
     - Present, microseconds.
   * - Data quality
     - ``data_quality``
     - Present, one ``SweL1bFlags`` byte per full cycle.

.. important::

   The single open question on the L2-to-L3 contract is the **reference
   frame**. The document asks for despun spacecraft coordinates at L2 so that
   SWE and MAG can be combined without further rotation. The code produces
   instrument-frame angles. The difference is a fixed mounting offset that
   ``get_instrument_spin_phase`` already knows about, so this is a small change
   - but it is a change to a deliverable's definition and needs SWE team
   agreement, not a unilateral patch. See :ref:`swe-implementation-status`.

What L3 does
------------

Three stages. All of them need external inputs that this repository never
loads.

Stage 1 - spacecraft potential and the core/halo break
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** The spacecraft charges, which shifts the measured electron energies
and makes the low-energy end unreliable. The first L3 step is to find the
potential by **fitting two Maxwellians twice** to the 1D (angle-averaged) phase
space distribution versus energy:

* **below 30 eV** -> the spacecraft potential;
* **30 - 400 eV** -> the break between the thermal (**core**) and suprathermal
  (**halo**) populations.

Steps listed: compute fit weights; build the ``log(fv)`` vs ``E`` spectrum;
perform the fits on distributions **averaged over 7 full SWE cycles (~7 minute
running average)**; if no potential break is seen (potential below the minimum
SWE energy), **set the potential to 2.5 V**; refine using the bins adjacent to
the fitted break; if either fit fails, **fall back to a smoothed-spline maximum
curvature** method. Then correct the energies using the potential.

**[DOC]** notes the heritage ACE/SWEPAM code instead did a nonlinear
least-squares fit of **three** Maxwellians to find both break points, and that
the SWE break-point finder is tuned for Ultra deflectors at 3500 V, may fail
when they are off (a flag marks those times), and may fail when the potential
drops below SWE's ~3 eV floor.

Stage 2 - pitch angle and gyrophase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Requires the spacecraft potential, the **MAG** field vector, and the
**SWAPI** solar wind velocity, all in despun spacecraft coordinates. Steps:

* Correct energies for spacecraft potential.
* Compute velocity vector and energy in the **solar wind frame** per
  measurement.
* Compute pitch angle and gyrophase per measurement.
* Bin: **20 pitch angle bins of 9 degrees**, at the 24 nominal energies; and -
  not in the heritage code - **24 gyrophase bins of 15 degrees**.
* Fit the distribution versus energy in each pitch angle bin (and each
  pitch/gyrophase combination if gyrophase binning is used) and evaluate the
  fit at the nominal energies.
* Integrate the pitch angle distribution to get the 1D energy spectrum, and
  integrate the two halves (0-90 and 90-180 degrees) separately to define "in"
  and "out" from the Sun, where "out" is the direction of maximum phase space
  density.

**[DOC]** flags a genuine statistics problem with gyrophase binning: a full SWE
measurement gives only ``7 x 30 = 210`` measurements per energy step, against
``20 x 24 = 240`` pitch/gyrophase bins. Accumulating 5-10 full cycles may be
needed. Deferred to a future data release.

Stage 3 - moments
^^^^^^^^^^^^^^^^^

**[DOC]** Density, velocity, temperature and heat flux, computed **two ways**
(bi-Maxwellian fit and direct integration) and for **three populations** (core,
halo, total).

The integrals are considered more accurate because they assume no spectral
form, but SWE cannot measure down to zero energy, so **the integral densities
are corrected using the fitted moments to fill in the low-energy end**.

Every variant ends with: rotate V to RTN; compute the temperature tensor
eigenvalues and eigenvectors and pick the primary; rotate T to RTN; rotate T to
field-aligned coordinates using **B** from MAG to get ``Tpar`` and ``Tperp``.

**[DOC]** makes an important physical caveat: because spacecraft charging makes
low-energy electrons hard to measure, SWE will **ultimately assume the bulk
electron velocity equals the proton velocity and the electron density equals
the ion density, both from SWAPI**, and use the SWE moments analysis only for
**electron temperature and heat flux**.

External inputs L3 needs that this repository never loads
---------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Source
     - Used for
   * - **MAG L2**
     - Field vector per 80 ms measurement, in despun spacecraft coordinates.
       Pitch angle, gyrophase, and the field-aligned temperature rotation.
   * - **SWAPI L3**
     - Ion velocity (solar wind frame transformation), density and temperature
       (spacecraft potential context, and the substitution for electron bulk
       moments).
   * - **Ultra HK**
     - Deflector voltage state, to flag times when the break-point finder is
       unreliable.
   * - **SWE** ``config``
     - An SDC ancillary file of L3 tuning constants: geometric fractions, the
       pitch angle / gyrophase / energy bin definitions, the in-versus-out
       energy index, the core/halo breakpoint initial guess and similar.
       Delivered to the SDC under the ``config`` descriptor, but read only by
       the L3 repository.

If you are asked to add L3 here
-------------------------------

Push back, and route it to the L3 repository. If the request survives that,
the things to settle first are:

#. **Frame.** L3 needs despun spacecraft coordinates. Resolve the L2 frame
   question before anything else.
#. **Cross-instrument dependencies.** This repository's design principle is
   that each batch job is self-contained. L3 needs MAG and SWAPI products as
   inputs - that is a real change to the dependency model, not just new code.
#. **The ~7 minute running average.** L2 records are 1 minute. Any L3 step that
   averages 7 cycles needs data from neighbouring files, which crosses the
   day-boundary filtering done at L1A.
