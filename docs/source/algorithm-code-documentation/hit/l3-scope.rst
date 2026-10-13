.. _hit-l3-scope:

L3 Scope: What Is Not in This Repository
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Section 9 of the algorithm document, PDF pages 149-169.

.. warning::

   **HIT L3 is produced by a separate repository, run closer to the science
   team.** Nothing in ``imap_processing`` should compute a charge, a cosine
   correction, an incident ion energy, or a pitch angle. This page exists so
   that you can (a) recognise an L3 task when one is handed to you, and (b)
   know what L1A and L2 are obliged to hand L3.

There are **three** L3 products. Two of them consume L2 or I-ALiRT outputs
directly. The third, the PHA products, consumes **L1A direct events** - and
that is where this repository has an unfinished obligation.

.. _hit-l3-pha:

1. PHA products (section 9.1)
-----------------------------

**[DOC]** For each individual detected particle, L3 produces:

* time of detection
* calculated ion energy (MeV/nuc)
* calculated charge (Z)

**Input: HIT L1A raw event data.** That is the ``imap_hit_l1a_direct-events``
product from this repository.

The processing chain is roughly:

#. **Classify the event.** Table 40 is a 12-column truth table over "which
   layer had the largest signal" (``L1A14``, ``L1A0``, ``L2A``, ``L3A``,
   ``L3B``, ``L2B``, ``L1B0``, ``L1B14``, ``L4iA``, ``L4oA``, ``L4iB``,
   ``L4oB``) giving a class (``L12A``, ``L123A``, ``PENA``, ``2TEL``, ``ILA``,
   ``L142A``, ...) and a range (``2A``, ``3A``, ``4A``, ``2B``, ``3B``, ``4B``,
   or ``NOCALC``). "Largest signal on a layer" means comparing all high-gain
   values on that layer plus any non-zero low-gain values **multiplied by 20**.
#. **Convert ADC to MeV.** :math:`E[\mathrm{MeV}] = a \cdot \mathrm{PHA[ADC]} + b`,
   with six coefficient pairs (L1/L2/L3 x low/high gain) in Table 42. The
   coefficients derive from FSW ADC calibration data (J. Dumonthier, GSFC-672,
   version 2024-05-23). Note the erratum: on 2026-05-14 all L1A/B14 low-gain
   offsets were reduced by 8.
#. **Correct for the Kapton foils.** The apertures are shielded by dual foils
   with an effective thickness of ~14.8 um silicon-equivalent. A multiplicative
   correction is applied to the L1 signals from the ``WINCORR2`` (Range 2) or
   ``WINCORR3`` (Range 3/4) arrays, stored as fixed-point integers scaled by
   256.
#. **Cosine-correct.** Both :math:`\Delta E` and :math:`E'` are multiplied by
   :math:`K_\theta = L_\theta^{1/a}`, where :math:`L_\theta` is the ratio of
   nominal to actual path length through the detector. Tables 43-45 give
   :math:`K_\theta` for all 150 L1 x L2 segment combinations per range. The
   power-law index :math:`a` was tuned for best He-3/He-4 discrimination:
   **1.61 (R2), 1.68 (R3), 1.82 (R4)** for the science apertures and
   **2.36 / 1.73 / 1.76** for the A0/B0 I-ALiRT apertures.
#. **Calculate the charge.** Each of 15 species (H, 4He, C, N, O, Ne, Na, Mg,
   Al, Si, S, Ar, Ca, Fe, Ni - He-3 excluded) has a double-power-law fit to its
   track:

   .. math::

      f_t(E') = \left[\left(a_1 E'^{b_1}\right)^{\gamma}
                      + \left(a_2 E'^{b_2}\right)^{\gamma}\right]^{1/\gamma}

   Evaluate all 15 at the measured :math:`E'`, then interpolate the
   (:math:`Z_t`, :math:`f_t`) graph at the measured :math:`\Delta E` under a
   power-law assumption to get a **non-integer** Z:

   .. math::

      Z(\Delta E) = A\,(\Delta E)^B, \quad
      B = \frac{\log(Z_{t,2}/Z_{t,1})}{\log(f_t(E'_2)/f_t(E'_1))}, \quad
      A = \frac{Z_{t,1}}{[f_t(E'_1)]^B}

   Above the Ni track, extrapolate with the Fe/Ni pair; below the H track, with
   the H/4He pair. He-3 shows up as Z ~ 1.9 between the H and 4He tracks - that
   is the point of using a decimal charge.
#. **Compute the incident energy** as the sum of the energy deposits in all
   detectors the particle interacted with. Per-detector energies are also
   reported. Valid :math:`\Delta E` and :math:`E'` bounds per range are in
   Table 46.

**Ancillary files L3 needs:** four ADC-to-MeV conversion sets (one per detector
type) and three Z lookup tables (one per range), all supplied by the HIT team.

.. _hit-l3-sectored:

2. Sectored / pitch angle products (section 9.2)
------------------------------------------------

**[DOC]** Combines:

* ``imap_hit_l2_macropixel-intensity`` - the 120 look directions on a 10-minute
  cadence
* **MAG L1D** magnetic field vectors in the despun frame, **averaged over the
  same 10 minutes**

For each of the 120 (declination, inclination) look directions, compute the
angle between the particle acceptance direction and **B**. Output:

* an array of pitch angles matching the HIT sectored bins
* a second product carrying both pitch angle **and gyrophase**
* a 2-D "skymap" rebinned to **22.5 degrees in pitch angle x 24 degrees in
  gyrophase**

**No ancillary files are required for this product.**

.. _hit-l3-electrons:

3. Electron science products (section 9.3)
------------------------------------------

**[DOC]** Turns the 6 I-ALiRT electron rates (3 per side, see
:ref:`hit-ialirt`) into science-quality electron **intensities**.

* Low and medium energy products are single-parameter - inner-L4 energy deposit
  only.
* High energy products use L4 **and** L3.
* Requires **modelled response functions** to map measured deposit to incident
  energy.
* Requires **ion contamination subtraction** using high-energy proton
  measurements (range/bin TBD). Early in the mission the raw PHA data will also
  be used to characterise the contamination.

**Ancillary files L3 needs:** electron response matrices, one for the L4-only
products and one for the L3-vs-L4 products.

What this repository owes L3
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - L3 needs
     - From
     - Status here
   * - Decoded PHA events (per event: detector IDs, gain flags, ADC values,
       Particle ID, priority buffer, STIM/HAZ flags, DEINDEX/EPINDEX)
     - L1A
     - **Missing.** ``imap_hit_l1a_direct-events`` contains only the raw
       concatenated binary. See :ref:`hit-gap-events`.
   * - ``imap_hit_l2_macropixel-intensity`` with correct look directions
     - L2
     - **Produced**, but the spin-rate correction to the 15th inclination bin
       is missing. See :ref:`hit-gap-spinrate`.
   * - The 6 I-ALiRT electron rates
     - I-ALiRT
     - **Produced.**
   * - High-energy proton rates for contamination subtraction
     - I-ALiRT / L2
     - **Produced** (``hit_h_a_side_high_en``, ``hit_h_b_side_high_en``, and
       the L2 standard H bins).
   * - Correct energy bin identification (geometric mean + edges)
     - L1B / L2
     - **Deviates** - the code uses the arithmetic mean. See
       :ref:`hit-gap-geometric-mean`.

.. important::

   The **event record format is fully specified** in sections 4.2.5 - 4.2.8 of
   the algorithm document (PDF pages 25-33): the 32-bit Event Record Header
   (Table 4), the 20-bit ADC field layout (Table 5 and Figure 9), the byte
   padding rules (Table 6), the detector group flags (Table 7), the
   Extended Header Block, the STIM Information Block, and the detector address
   table (Table 8). **Decoding that is a Level 1A job and belongs in this
   repository** - it is the raw-to-reformatted step, not derived science. Only
   the *physics* applied to the decoded events (energy, charge, cosine
   correction) belongs to L3.
