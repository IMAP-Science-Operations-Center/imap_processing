.. _swapi-l3-scope:

L3 Scope - What Is Not in This Repository
=========================================

**Document:** sections 7.1.1, 9.5, 10.4, 10.5, 13.

.. important::

   **This repository takes SWAPI to L2 and stops.** L3a and L3b are produced by
   the SWAPI team's own code, delivered as a separate Docker container run on
   AWS. Section 13 of the algorithm document says so in one line: "SWAPI
   delivers the SWAPI L3 processing code using a Docker container via AWS."

   Roughly two thirds of the algorithm document describes L3. If you are
   reading the document and find yourself in a section full of Maxwellian
   integrals, Jacobians or Vasyliunas-Siscoe distributions, you are outside
   this repository's scope.

This page exists for two reasons: so that nobody starts implementing L3 here by
accident, and so that the **contract** between L2 and L3 is written down. That
contract is the only part of section 10.4 and 10.5 that constrains what we do.

The contract - what L3 needs from L2
------------------------------------

**[DOC]** Section 10.6, "Inputs Required" and "Ancillary files" for each L3
product. Consolidated:

.. list-table::
   :header-rows: 1
   :widths: 24 12 18 46

   * - L3 product
     - Cadence
     - L2 input
     - Other inputs
   * - Solar wind proton velocity, density, temperature
     - 1 min
     - Count rates and count rate uncertainties as a function of E/q for
       **62 coarse energy bins (5 sweeps)**
     - SPICE spin phase, SPICE frame kernel, efficiency LUT, instrument
       response LUT
   * - Solar wind alpha velocity, density, temperature
     - 1 min
     - as above, **coarse steps only**
     - as above, plus the **magnetic field vector** (MAG L2, L1D fallback)
   * - Pickup He\ :sup:`+` fit parameters, density, temperature
     - 10 min
     - Count rates and uncertainties as a function of E/q for **62 coarse
       energy bins (50 sweeps)**
     - as above, plus interstellar neutral He density LUT and interstellar
       neutral H and He flow vector LUT
   * - Combined differential flux :math:`J(E/q)`
     - 10 min
     - as above (50 sweeps)
     - :math:`\Delta E/E`, energy-dependent geometric factor
       :math:`G(E/q)`, efficiency :math:`\varepsilon`

Read off the consequences for L2:

1. **L3 consumes count rates keyed on E/q, not on step index.** That is why L2
   sets ``DEPEND_1 = "esa_energy"`` on every rate variable. If the energy solve
   is wrong, every L3 product is wrong and there is nothing downstream that
   can catch it.
2. **Rate uncertainties are a required input, not a nicety.** The proton and
   alpha fits use them for the covariance estimate and the PUI fit uses them
   in the Poisson likelihood.
3. **Chunking is L3's job, but the boundaries are ours.** L3 groups
   non-overlapping 5-sweep and 50-sweep chunks. It needs contiguous sweeps with
   trustworthy times, which is what ``sci_start_time`` and the epoch convention
   provide.
4. **L3 needs the sweep start time, per ESA step.** Hence
   ``sci_start_time`` - a UTC string of the first packet's epoch, added to L1
   "for L3 purposes per SWAPI requests" - alongside the centre-time ``epoch``.
5. **L3 needs to invert energy back to ESA voltage** using
   :math:`k_{L2} = 1.93` eV/V/e, because the response function is tabulated
   against voltage. See :ref:`swapi-k-factor`. L2 does not record the ``k`` it
   used.
6. **Coarse-only for some products.** The alpha fit uses coarse steps only; the
   proton fit uses coarse and fine. L2 must therefore keep the fine steps
   distinguishable, which is why ``plan_id`` and ``sweep_table`` are carried
   through and why the solve marks fine steps explicitly.
7. **L2 must not pre-average.** Every L3 product does its own chunking from
   12-second sweeps.
8. **L2 must not apply efficiency, deadtime or geometric factor.** Those live
   in the L3 forward model (:math:`\varepsilon`, :math:`\tau = 183.7` ns,
   :math:`G(E/q)`).

What L3a does, in one paragraph each
------------------------------------

Enough to recognize the algorithms, not enough to implement them. If you need
the equations, they are in document sections 10.4.1 (proton, pages 43-54),
10.4.2 (alpha, pages 55-57) and 10.4.3 (pickup helium, pages 58-67).

Solar wind protons (H\ :sup:`+`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** A **forward model fit**, not a moment integration. The proton VDF is
modelled as a drifting Maxwellian with parameters
:math:`\boldsymbol{x} = (\ln n, \ln T, v_R, v_T, v_N)` (logs so density and
temperature stay positive), and the model coincidence rate is the VDF
integrated against the instrument response over both azimuthal regions
(sunglasses and open aperture) by nested Gauss-Legendre quadrature with
:math:`(N_\theta, N_\phi, N_v) = (21, 21, 15)` points, with dynamic integration
limits from the passband and the VDF width. A deadtime correction
:math:`\mathcal{D} = 1/(1 + \tau C^{\mathrm{model}})` with
:math:`\tau = 183.7` ns maps the model rate to the observed rate.
Minimization is **unweighted** least squares - deliberately, because Poisson
inverse-variance weighting would over-weight the low-count wings where pickup
ions, alphas and the proton shoulder contribute, an effect that the sunglasses
exaggerate by attenuating the cold core. Uncertainties use the
heteroscedasticity-consistent **HC3** sandwich estimator rather than the
Jacobian covariance. Because the MSE has a spurious local minimum with the bulk
velocity flipped about the spin axis, the fit is re-run up to six times from
the flipped solution and the better minimum kept. Flags: ``FIT_ERROR`` if the
optimizer fails, ``BAD_FIT`` if :math:`R^2 < 0.9` or
:math:`T > 5 \times 10^5` K - and even then the speed is reported from the
peak of the sweep-averaged rates.

Solar wind alphas (He\ :sup:`2+`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Protons are fit first and **held fixed**, then three alpha parameters
are fit: :math:`(\log n^{He^{2+}}, \log T^{He^{2+}}, \Delta v^{He^{2+}})`,
where the alpha bulk velocity is constrained to
:math:`\boldsymbol{v}^{He^{2+}} = \boldsymbol{v}^{p} + \Delta v^{He^{2+}}\hat{B}`
- the drift lies **along the magnetic field**, which is where the MAG
dependency comes from. The alpha peak is found from the proton-subtracted
residual :math:`R_i = C_i - 2 C_i^p` (the factor of two so that few-percent
errors in the proton model are not mistaken for the alpha peak), searched
between 1.5 and 4 times the proton peak E/q; five points around it are fit with
``scipy.optimize.least_squares(method='lm')``. Coarse steps only.

Pickup helium (He\ :sup:`+`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** This is the part inherited from SWAP. The generalized
**Vasyliunas & Siscoe** model in the Chen et al. (2014) form is fit to 50
combined sweeps over the range
:math:`1.25 E_{b,H^+} < E_e < 1.2 E_{b,He^+}`:

.. math::

   f(r,w,\psi) = \frac{\alpha}{4\pi}\frac{\beta_E r_E^2}{r u_{\mathrm{sw}} v_b^3}
     w^{\alpha-3}\, n_{\mathrm{He}}(r w^\alpha, \psi)\, \Theta(1-w)

with free parameters cooling index :math:`\alpha`, ionization rate
:math:`\beta_E`, cutoff speed :math:`v_b` and a constant background rate
:math:`C_{\mathrm{bg}}`. The interstellar neutral helium density
:math:`n_{\mathrm{He}}` is precomputed from the **hot model** (Thomas 1978) and
supplied as a LUT. Optimization is Nelder-Mead on a bounds-transformed
parameter vector, maximizing the **Poisson likelihood**; density and
temperature then come from numerical integration of the best-fit distribution,
with uncertainties propagated by re-integrating at
:math:`j \pm \sigma_j`. Bounds and initial values are the document's Table 5
(:math:`\alpha \in [1,5]` starting 1.5;
:math:`\beta_E \in [0.6\times10^{-9}, 8\times10^{-7}]` s\ :sup:`-1` starting
:math:`10^{-7}`; :math:`v_b \in [0.8, 1.2] v_{\mathrm{sw}}` starting
:math:`v_{\mathrm{sw}}`; :math:`C_{\mathrm{bg}} \in [0, 10]` Hz starting
0.1 Hz). :math:`C_{\mathrm{bg}} > 1` Hz is reported as fill (suprathermal
contamination) while keeping the other parameters.

What L3b does
-------------

**[DOC]** Section 10.5. The one L3 equation worth having here, because it is
the only place efficiency and geometric factor enter and it is what a naive
reader might otherwise try to put at L2:

.. math::

   J\!\left(\frac{E}{q}\right) =
     \frac{C(E/q)}{\frac{E}{q} \cdot G(E/q) \cdot \varepsilon}

with :math:`J` the differential flux in #/[cm\ :sup:`2` s sr eV/q],
:math:`C` the count rate, :math:`G` the geometric factor and
:math:`\varepsilon` the time-varying efficiency (updated after each on-orbit
gain test and provided in a LUT). The calculation **assumes the same efficiency
for hydrogen and helium** - the H efficiency is used for both, efficiency
treated as mass-independent. Uncertainty:

.. math::

   \Delta J = J \sqrt{\left(\frac{\Delta C}{C}\right)^2
     + \left(\frac{\Delta (E/q)}{E/q}\right)^2}

Note the second term: **L3b needs an energy uncertainty**
:math:`\Delta(E/q)`. L2 does not produce one. The energy passband edges are in
the LUT notes table (``Lower Energy`` / ``Upper Energy`` columns, which no code
here reads), so the information exists but is not propagated.

If you are asked to add L3 here
-------------------------------

Push back, and point at section 13. If the answer is still yes, these are the
things that would have to change on our side of the boundary, in rough order:

1. L2 would need to publish an energy uncertainty (from the LUT notes
   passband edges).
2. L2 or L1 would need the thruster flag, since L3 rejects thruster-contaminated
   data.
3. The remaining L1 rejection criteria (checksum, saturation, ``RATE_ST``)
   would need implementing, because L3's flags assume L2 data is already clean.
4. The response-function ancillary files (central effective area, passbands,
   azimuthal transmission), the efficiency table and the interstellar-neutral
   LUTs would all need ingest paths here. The files themselves are delivered to
   the SDC and their descriptors are listed in :ref:`swapi-l3-descriptors`, but
   nothing in this repository reads any of them.
5. SPICE geometry would need wiring in: per-ESA-step SWAPI-to-RTN rotation
   matrices and spacecraft velocity in the solar inertial frame.
6. New logical sources and descriptors (``1m-sw-p``, ``1m-sw-a``,
   ``10m-pui-he``, ``10m-combined``) and a new ``PROCESSING_LEVELS`` entry.
