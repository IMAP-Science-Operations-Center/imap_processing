.. _lo-l1c:

Level 1C - Pointing Sets
========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Module:** ``imap_processing/lo/l1c/lo_l1c.py``

A **pointing set (PSET)** is one repointing period's worth of counts and
exposure, accumulated onto a fixed instrument-relative sky grid. It is the
intended building block for L2 maps.

.. warning::

   **The L2 code does not currently read the pointing set.** ``lo_l2`` builds
   maps directly from the L1B ``histrates`` / ``bgrates`` / ``goodtimes``
   products. ``imap_lo_l1c_pset`` is produced but nothing consumes it. See
   :ref:`lo-implementation-status`. **[CODE]**

The grid
--------

**[CODE]** Defined at the top of ``lo_l1c.py``:

.. code-block:: python

   N_ESA_ENERGY_STEPS   = 7
   N_SPIN_ANGLE_BINS    = 3600          # 0.1 degree bins around the spin
   N_OFF_ANGLE_BINS     = 40            # 0.1 degree bins, -2 to +2 degrees
   PSET_SHAPE           = (1, 7, 3600, 40)
   PSET_DIMS            = ["epoch", "esa_energy_step", "spin_angle", "off_angle"]

   ESA_ENERGY_STEPS       = 1..7
   SPIN_ANGLE_BIN_EDGES   = linspace(0, 360, 3601)
   OFF_ANGLE_BIN_EDGES    = linspace(-2, 2, 41)

The leading dimension of length 1 is the pointing itself: one PSET covers one
repointing period, timestamped at the pointing start.

**[DOC]** The document describes the same grid and explicitly says nutation is
ignored, so only the 1-D longitude of the boresight is tracked. Its
``POINTINGDIR`` variable is ``[3600, 40, 2]`` of (RA, DEC) in HAE.

Inputs
------

**[CODE]** ``lo_l1c(sci_dependencies, anc_dependencies)`` expects a dict keyed
by logical source containing:

* ``imap_lo_l1b_de`` - the annotated direct events
* ``imap_lo_l1b_goodtimes`` - good-time intervals and the pivot angle
* ``imap_lo_l1b_bgrates`` - background rates
* ``imap_lo_l1b_histrates`` - the source of exposure time

The repointing ID is read from the ``Repointing`` global attribute of the L1B
DE dataset, and the pointing start/end MET from
``spice.repoint.get_pointing_times_from_id``. A missing ``Repointing``
attribute is a hard error.

Goodtime filtering
------------------

``filter_goodtimes(l1b_de, goodtimes_ds)`` keeps only events whose epoch falls
inside a ``[gt_start_met, gt_end_met]`` interval. An empty result is allowed:
the PSET is still written, with zero counts and zero exposure, and a warning is
logged.

Event selection filters
-----------------------

``FilterType`` selects which events go into which count array. Four count
arrays are produced from the same DE set.

Coincidence class
^^^^^^^^^^^^^^^^^

**[CODE]** Coincidence types are matched as 6-character bit strings.

.. code-block:: python

   triples = ["111111", "111100", "111000"]

   doubles = ["110100", "110000", "101101", "101100", "101000",
              "100100", "100101", "100000", "011100", "011000",
              "010100", "010101", "010000", "001100", "001101", "001000"]

   golden triple  =  coincidence_type == "111111"

.. note::

   These string codes are a different encoding of coincidence type from the
   4-bit ``ABSENT`` integer used in the algorithm document and in
   ``CASE_DECODER``. If you are cross-referencing the document's case table
   against this list, you must translate. This is a known readability wart.

Species
^^^^^^^

**[CODE]** Species selection at L1C is *not* the simple TOF2 test used at L1B.
It requires a **golden triple** and a 3-D box cut on PAC-corrected TOFs:

.. math::

   \mathrm{TOF0}_s = \mathrm{TOF0} + 0.5\,\mathrm{TOF3}, \qquad
   \mathrm{TOF1}_s = \mathrm{TOF1} - 0.5\,\mathrm{TOF3}

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * - Species
     - :math:`\mathrm{TOF0}_s`
     - :math:`\mathrm{TOF1}_s`
     - :math:`\mathrm{TOF2}`
   * - Hydrogen
     - 20 to 70
     - 10 to 50
     - 10 to 40
   * - Oxygen
     - 100 to 270
     - 60 to 150
     - 60 to 150

.. warning::

   L1B ``identify_species`` and L1C ``get_h_species`` / ``get_o_species`` use
   **different criteria and produce different answers**. L1B uses TOF2 alone on
   any event; L1C requires a golden triple and a 3-D cut. Neither the peak
   boxes nor the 0.5*TOF3 correction appear in the algorithm document. If you
   are asked "which events are hydrogen", the answer depends on which level is
   asking.

Count accumulation
------------------

``create_pset_counts(de, filter_type)`` bins the filtered events by
``(esa_energy_step, spin_bin, off_angle_bin)`` into the ``(1, 7, 3600, 40)``
grid. Four arrays result: ``triples_counts``, ``doubles_counts``, ``h_counts``,
``o_counts``.

Exposure time
-------------

**[CODE]** ``calculate_exposure_times(histrates_ds, goodtimes_ds)``:

1. Select the ``histrates`` epochs that fall inside a goodtime window.
2. Sum ``exposure_time_6deg`` over those epochs, giving shape ``(7, 60)``.
3. Expand 60 spin bins to 3600 by dividing each by 60 and repeating it 60
   times, so total exposure is conserved.
4. Divide by ``N_OFF_ANGLE_BINS`` and broadcast across the 40 off-angle bins.

.. note::

   Steps 3 and 4 spread exposure **uniformly**. This is an approximation: the
   real instrument response is not flat across a 6-degree bin, nor across the
   off-angle range. The document's version instead accumulates
   :math:`T_{0.1^\circ} = 4\langle \text{spin duration}\rangle / 3600` per
   cycle inside the goodtimes and sums. The two agree on the total but not on
   the distribution.

Background rates and pointing directions
----------------------------------------

* ``set_background_rates(filter_type, sci_dependencies, attr_mgr)`` carries the
  L1B background rates onto the PSET. **Background is never subtracted**; it is
  carried alongside so users can subtract case by case. **[DOC + CODE]**
* ``set_pointing_directions`` / ``compute_pointing_directions`` compute the sky
  direction of every grid cell at the pointing midpoint, using the pivot angle
  and ``spice.geometry.frame_transform_az_el``. ``lo_l2`` imports
  ``compute_pointing_directions`` directly.
* ``add_spacecraft_position_and_velocity_to_pset`` (from
  ``ena_maps.utils.corrections``) attaches the spacecraft state needed later
  for the solar-frame transform and Compton-Getting correction.

Rates and fluxes are deliberately absent
----------------------------------------

**[DOC]** As of a 21 July 2025 SDC directive, rates and fluxes were **removed
from L1C** and deferred to L2. L1C carries counts, exposure times, background
rates, and pointing directions only. The flux formula is documented at L1C for
reference but is applied at L2:

.. math::

   J_{i,s} = \frac{R_{i,s}}{G_{i,s}\, E_{i,s}\, \langle F_{\text{eff}} \rangle},
   \qquad \delta J = \frac{J}{\sqrt{N}}

See :ref:`lo-l2` for the version that is actually implemented.

Second output
-------------

``imap_lo_l1c_goodtimes`` is a reference copy of the goodtimes list, so a PSET
consumer does not need to go back to L1B for it.
