.. _codice-l3-scope:

Level 3 - Out of Scope Here
===========================

**This repository stops at L2.** CoDICE L3 products are produced by a separate
repository run closer to the science team. Nothing in section 13 of the
algorithm document should be implemented in ``imap_processing``.

This page exists so that you can **recognise an L3 request and redirect it**,
and so that you can tell which L2 variables exist purely to feed L3.

What L3 consists of
-------------------

**[DOC]** Section 13 and the L3 product table in section 6.

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Product
     - File prefix
     - Description
   * - Lo direct events
     - ``imap_codice_l3a_lo-direct-events_``
     - L2 direct events plus per-event **mass** and **mass-per-charge**, and a
       normalisation factor derived from the priority counts.
   * - Lo SW 3-D VDFs
     - ``imap_codice_l3a_lo-<species>-3d-distribution_``
       (was ``lo-sw-3d-vdf``)
     - Intensity vs (azimuthal sector, spin sector, energy) in the instrument
       frame, built from direct events. Binned by **APD ID**,
       output 128 E/q x 24 spin angles x 13 elevations.
   * - Lo SW partial densities
     - ``imap_codice_l3a_lo-partial-densities_``
     - Per-species partial densities from the L2 SW species intensities.
   * - Lo SW elemental abundance ratios
     - ``imap_codice_l3a_lo-sw-ratios_``
     - C/O, Mg/O, Fe/O at 12 minute cadence.
   * - Lo SW charge state ratios
     - (shares the partial-densities prefix)
     - O7+/O6+, C6+/C4+, C6+/C5+, Fe_low/Fe_high at 12 minute cadence.
   * - Lo SW charge state distributions
     - ``imap_codice_l3a_lo-sw-charge-state-distributions_``
     - Relative abundances of O charge states 5-8 and C charge states 4-6.
   * - Hi direct events
     - ``imap_codice_l3_hi-direct-events_``
     - L2 direct events plus energy-per-nucleon and estimated mass.
   * - Hi pitch angle distributions
     - ``imap_codice_l3_hi-pitch-angle_``
     - Intensity vs (energy, pitch angle) for H, 4He, O, Fe at 4 min cadence.
   * - Combined energy-time spectrograms
     - (L3c)
     - H, He, O, Fe intensities spanning the full Lo + Hi energy range.
   * - Combined pitch angle distributions
     - (L3c)
     - He and O pitch-angle distributions spanning Lo + Hi.

Why some of it looks like it belongs here
-----------------------------------------

Three things blur the line. Be alert to them:

**1. I-ALiRT computes ratios that look like L3a.**
   The real-time stream produces C/O, Mg/O, Fe/O, C6+/C5+, O7+/O6+ and
   Fe_low/Fe_high - the same quantities as the L3a ratio products. They are
   produced **here** because the entire I-ALiRT chain lives in this repository
   and must run in under five minutes. They use *pseudo*-densities (constant
   factors omitted, since they cancel in a ratio), whereas L3a uses real partial
   densities with the unit conversion factor
   :math:`C = 2.283 \times 10^{-8}`. **The two are not interchangeable.** See
   :ref:`codice-ialirt`.

**2. The L2 direct-event products carry fields L3 needs.**
   ``energy_per_charge``, ``spin_angle``, ``elevation_angle``,
   ``energy_per_nuc``, converted ``tof`` and converted energies exist at L2
   precisely so the L3 repository does not have to re-read calibration tables.
   Do not remove them because "nothing here uses them".

**3. Negative TOF filling.**
   ``process_lo_direct_events`` NaNs negative TOF values with a comment that it
   "mirrors Menlo's L3a handling". That is a deliberate coordination with the L3
   repository, not a stray heuristic.

Quantities L3 derives (for orientation only)
--------------------------------------------

Do not implement these. They are summarised so you can recognise them.

Lo mass and mass-per-charge (13.2.5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   MQ = \frac{2 \left(E/q + PAC - \alpha\right)}{K}
   \left(\frac{\tau}{d}\right)^{2}

with :math:`PAC` = 15 kV (post-acceleration), :math:`\alpha` an approximation
for carbon-foil energy loss (default 0), :math:`\tau` the TOF in ns, :math:`d` =
10.64 cm the path length, and :math:`K` the conversion constant assembled from

.. math::

   K = \frac{2 \times 1.602 \times 10^{-16}\,\mathrm{J/keV}}
            {1.673 \times 10^{-27}\,\mathrm{kg/AMU}}
       \left(\frac{1}{10.64\,\mathrm{cm}}\right)^{2}
       \left(\frac{100\,\mathrm{cm}}{\mathrm{m}}\right)^{2}
       \left(\frac{1\,\mathrm{s}}{10^{9}\,\mathrm{ns}}\right)^{2}

Mass comes from an empirical fit in log space:

.. math::

   X &= \ln E, \quad Y = \ln \tau \\
   Z &= A_0 + A_1 X + A_2 Y + A_3 XY + A_4 X^2 + A_5 Y^3 \\
   \mathrm{Mass} &= e^{Z} \quad [\mathrm{AMU}]

with preliminary coefficients :math:`A = [-0.85674, -0.363739, -1.36612,
0.30347, 0.0290419, 0.0557362]`.

Lo partial densities (13.2.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   dN_j(l) = C \cdot \Delta\vartheta \cdot \Delta\varphi \cdot \frac{\Delta E}{E}
   \cdot J_j(l) \cdot (E/q)_l \cdot \sqrt{(m/q)_j},
   \qquad N_j = \sum_{l=0}^{127} dN_j(l)

with :math:`C = 2.283 \times 10^{-8}` converting to cm-3. The m/q table for the
14 SW species is in section 13.2.1.

Pitch angles (13.1.2, 13.3.2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the one place the **instrument -> spacecraft frame rotation** is needed:

.. math::

   \theta_{SC} = (\theta_{inst} + 46^\circ) \bmod 360^\circ, \qquad
   \phi_{SC} = \phi_{inst}

.. math::

   \hat{n}_{SC} = (\cos\theta_{SC}\sin\phi_{SC},\;
                   \sin\theta_{SC}\sin\phi_{SC},\;
                   \cos\phi_{SC})

.. math::

   \alpha = \cos^{-1}\!\left(\frac{-\hat{n}_{SC} \cdot \vec{B}_{SC}}
   {|\vec{B}_{SC}|}\right) \cdot \frac{180}{\pi}

binned into 30 deg pitch-angle bins. Note the minus sign: the look direction is
where particles come *from*, so the direction of travel is its negative. The
magnetic field is averaged over the accumulation period, which makes L3 pitch
angles **dependent on MAG data** - a cross-instrument dependency the SDC would
have to broker if it were ever built here.

L3c combined products (13.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Hi + Lo combination converts Lo energy-per-charge to energy-per-nucleon
using an assumed charge state and mass, then re-bins Lo onto a **common grid of
20 sqrt(2)-spaced energy-per-nucleon bins** spanning
:math:`E_{min} = 10^{-4}` to :math:`E_{max} \approx 0.1` MeV/n (lower edges
:math:`E_{min} 2^{i/2}`, geometric centres). It then restricts the Hi range to avoid overlap
(H > 0.08, He > 0.04, O > 0.03, Fe > 0.02 MeV/n), time-averages Hi to the 4 min
Lo cadence, and concatenates.

If you are asked to build any of this
-------------------------------------

Say no, and point at this page. Then check:

1. Does the requested quantity actually need an **L2 change** to enable it? For
   example, if L3 needs a variable that L2 drops, that *is* work for this
   repository.
2. Is the request really about **I-ALiRT**? The ratio products exist in both
   places under similar names.
3. Is it a **validation** request - "compare our L2 to their L3"? That is fine
   and does not require implementing L3.
