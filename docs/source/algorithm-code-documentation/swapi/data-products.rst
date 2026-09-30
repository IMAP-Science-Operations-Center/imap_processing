.. _swapi-data-products:

Data Products and What Feeds What
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This is the "what goes into what" map. It is the page to read before touching
``cli.py`` or adding a product.

Product inventory
-----------------

**[CODE]** Everything this repository produces for SWAPI. The
``logical_source`` strings come from
``imap_processing/cdf/config/imap_swapi_global_cdf_attrs.yaml`` and are what
``write_cdf()`` uses to name the output file.

.. list-table::
   :header-rows: 1
   :widths: 30 10 12 48

   * - ``logical_source``
     - Level
     - Cadence
     - Contents
   * - ``imap_swapi_l1_sci``
     - L1
     - 1 sweep (12 s)
     - PCEM/SCEM/COIN **counts** and uncertainties as 72-element arrays, plus
       quality flags and sweep metadata.
   * - ``imap_swapi_l1a_hk``
     - L1A
     - 1 s (raw)
     - Housekeeping, **raw** DN values. All 102 ``SWP_HK`` fields.
   * - ``imap_swapi_l1b_hk``
     - L1B
     - 1 s (derived)
     - Housekeeping, **derived/engineering-unit** values (same fields,
       ``use_derived_value=True``).
   * - ``imap_swapi_l2_sci``
     - L2
     - 1 sweep (12 s)
     - PCEM/SCEM/COIN **count rates** and uncertainties, plus the solved
       ``esa_energy`` (72 values per sweep).

Plus one non-CDF product:

.. list-table::
   :header-rows: 1
   :widths: 30 10 12 48

   * - Product
     - Level
     - Cadence
     - Contents
   * - SWAPI I-ALiRT record
     - --
     - 12 s
     - ``list[dict]`` of pseudo proton speed, density and temperature destined
       for the I-ALiRT database, **not** a CDF. See :ref:`swapi-ialirt`.

Not produced here
-----------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Product
     - Status
   * - L3a solar wind proton (``1m-sw-p``)
     - **SWAPI team's code.** Density, speed, temperature at 1 min.
   * - L3a solar wind alpha (``1m-sw-a``)
     - **SWAPI team's code.** Needs MAG L2.
   * - L3a pickup helium (``10m-pui-he``)
     - **SWAPI team's code.** Cooling index, cutoff speed, ionization rate,
       background rate, density, temperature at 10 min.
   * - L3b combined differential flux (``10m-combined``)
     - **SWAPI team's code.** :math:`J(E/q)` at 10 min.
   * - Quick-look plots
     - **[DOC]** Section 15 specifies colour-coded spectrograms of count rate
       vs energy/charge vs time (with MAG L1D overlaid on the daily plot).
       **No SWAPI quicklook code exists in this repository.**

See :ref:`swapi-l3-scope` for what those products need from L2.

The processing chain
--------------------

**[DOC]** Algorithm document figure 4, annotated with what is and is not here.

.. code-block:: text

                        L0  CCSDS packets  (SWP_SCI, SWP_HK)
                                 |
              +------------------+------------------+
              |                                     |
   Decommutation table                          Level 1A
      (the XTCE)                                Housekeeping   <-- HERE
              |                                     |
              v                                     v
           Level 1                              Level 1B
   PCEM/SCEM/COIN counts per            Housekeeping (derived)  <-- HERE
   ESA voltage step and time  <-- HERE
              |
   ESA Unit Conversion ADP -->|
              |
              v
           Level 2                                I-ALiRT
   PCEM/SCEM/COIN count rates  <-- HERE     (separate APID)     <-- HERE
   per ESA energy step and time
              |
              |   <-- B-field from MAG L2 (L1D fallback)
              |   <-- timing, attitude, ephemeris (SPICE)
              |   <-- instrument status summary
              |   <-- geometric factor, efficiency, instrument response
              |
      +-------+-------+
      |               |
      v               v
   Level 3a        Level 3b        <-- NOT HERE. SWAPI team's Docker container.
   SW + PUI        combined
   plasma fits     differential
                   intensity

Dependency wiring
-----------------

**[CODE]** ``imap_processing/cli.py``, ``class Swapi``. The CLI is the only
entry point; ``do_processing`` dispatches on ``self.data_level`` and
``self.descriptor``.

.. list-table::
   :header-rows: 1
   :widths: 14 12 34 40

   * - Level
     - Descriptor
     - Dependencies (count is checked)
     - Call
   * - ``l1`` / ``l1a``
     - ``sci``
     - **3**: SWAPI L0 ``raw``, SWAPI L1 ``hk`` CDF, time kernels
     - ``swapi_l1(dependencies, descriptor="sci")``
   * - ``l1`` / ``l1a``
     - ``hk``
     - **2**: SWAPI L0 ``raw``, time kernels
     - ``swapi_l1(dependencies, descriptor="hk")``
   * - ``l2``
     - (any)
     - **3**: SWAPI L1 ``sci`` CDF, ``esa-unit-conversion`` ancillary,
       ``lut-notes`` ancillary
     - ``swapi_l2(l1_dataset, esa_table_df, lut_notes_df)``

Two things worth knowing about this wiring:

* **The L1 science job depends on the L1 housekeeping job.** The HK CDF is
  loaded to populate the science quality flags, using nearest-epoch matching.
  You cannot produce ``l1_sci`` without first producing ``l1a_hk``/``l1b_hk``
  for the same day.
* **Both ``l1`` and ``l1a`` are accepted** as the data level for the same
  branch. ``l1a`` exists because the housekeeping products are named
  ``l1a_hk`` / ``l1b_hk`` while the science product is named ``l1_sci``.
* **The HK descriptor branch returns two datasets** (L1A raw and L1B derived)
  from a single invocation.
* **Anything other than l1/l1a/l2 raises** ``NotImplementedError``.

L1 output is passed through ``filter_day_boundary_data(ds, self.start_date)``
(``imap_processing/utils.py``) so that packets spilling over the UTC day
boundary are trimmed. L2 is **not** filtered - it inherits L1's boundaries.

Filenames and descriptors
-------------------------

**[DOC]** The algorithm document specifies:

.. code-block:: text

   imap_swapi_<data_level>_<descriptor>_<start_date>-<repointing>_<version>.<extension>

.. list-table::
   :header-rows: 1
   :widths: 20 16 64

   * - Field
     - Required?
     - SWAPI options per the document
   * - ``dataLevel``
     - required
     - ``l1``, ``l2``, ``l3a``, ``l3b``
   * - ``descriptor``
     - optional
     - ``12s`` (L1 or L2 data, default), ``1m-sw-p`` (L3a), ``1m-sw-a`` (L3a),
       ``10m-pui-he`` (L3a), ``10m-combined`` (L3b)
   * - ``extension``
     - required
     - ``cdf``

L0 files arrive from the SDC as
``imap_l0_sci_<instrument>_YYYYMMDD_vNN.pkts``.

.. warning::

   **[CODE]** The descriptors actually used are ``sci``, ``hk`` and the
   implicit ``l1a_hk``/``l1b_hk`` split - **not** ``12s``. The document's
   ``12s`` descriptor appears nowhere in the code or the CDF configs. This is
   a real deviation and is tracked in :ref:`swapi-implementation-status`. Do
   not "fix" it by renaming products; the SDC file catalog and every downstream
   consumer are keyed on the current strings.

Ancillary inputs
----------------

Summarized here; details in :ref:`swapi-ancillary`.

.. list-table::
   :header-rows: 1
   :widths: 24 12 20 44

   * - Ancillary
     - Used at
     - SDC descriptor
     - What for
   * - Decommutation table
     - L1
     - (the XTCE itself)
     - Packet field definitions.
   * - ESA unit conversion ADP
     - L2
     - ``esa-unit-conversion``
     - Fixed energies for ESA steps 0-62 and the fine-step offset indices.
   * - LUT notes table
     - L2
     - ``lut-notes``
     - The ESA DAC-to-energy ladder used to solve the fine-step energies.
   * - Efficiency LUT
     - L3
     - --
     - :math:`\varepsilon_H`, :math:`\varepsilon_{He}` vs time, from gain tests.
   * - Instrument response CSVs
     - L3
     - --
     - Central effective area, energy-angle passbands, azimuthal transmission.
   * - Gain test 1x2 LUT
     - operations
     - --
     - New PCEM/SCEM voltage settings. Not read by any pipeline code.
   * - MAG L2 (L1D fallback)
     - L3a alpha
     - --
     - Constrains the alpha-proton differential flow along **B**.
   * - SPICE kernels
     - L1, L3
     - (standard)
     - Time conversion at L1; spin phase and rotation matrices at L3.
   * - Thruster firing history
     - L1/L2
     - --
     - **[DOC]** Should set a thruster flag. **[CODE]** Not implemented.

CDF variable inventory
----------------------

**[CODE]** What is actually in the files.

L1 science (``imap_swapi_l1_sci``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinates: ``epoch`` (one per sweep), ``esa_step`` (0-71),
``esa_step_label``.

.. list-table::
   :header-rows: 1
   :widths: 44 14 42

   * - Variable
     - Shape
     - Notes
   * - ``swp_pcem_counts``
     - (sweep, 72)
     - Decompressed counts. Index 0 forced to ``NaN``.
   * - ``swp_scem_counts``
     - (sweep, 72)
     - as above
   * - ``swp_coin_counts``
     - (sweep, 72)
     - as above
   * - ``swp_pcem_counts_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - :math:`\sqrt{N}`; plus and minus are identical
   * - ``swp_scem_counts_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - as above
   * - ``swp_coin_counts_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - as above
   * - ``swp_l1a_flags``
     - (sweep, 72)
     - ``uint16`` bitfield, ``SWAPIFlags``
   * - ``sweep_table``
     - (sweep,)
     - ``SWP_SCI.SWEEP_TABLE``, taken from packet 0
   * - ``plan_id``
     - (sweep,)
     - ``SWP_SCI.PLAN_ID``, taken from packet 0
   * - ``sci_start_time``
     - (sweep,)
     - UTC string of the **first** packet's epoch, ms precision. Added at
       SWAPI's request for L3.
   * - ``esa_lvl5``
     - (sweep,)
     - ``SWP_SCI.ESA_LVL5`` from packet 11 (``SEQ_NUMBER == 11``). The key to
       the L2 fine-step solve.
   * - ``lut_choice``
     - (sweep,)
     - ``SWP_HK.LUT_CHOICE``, nearest HK packet
   * - ``fpga_type``
     - (sweep,)
     - ``SWP_HK.FPGA_TYPE``, nearest HK packet
   * - ``fpga_rev``
     - (sweep,)
     - ``SWP_HK.FPGA_REV``, nearest HK packet

.. note::

   The flag variable is named ``swp_l1a_flags`` even though the product is
   ``imap_swapi_l1_sci`` (not ``l1a``). It is carried through to L2 under the
   same name. Cosmetic, but it trips people up when grepping.

L2 science (``imap_swapi_l2_sci``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

L2 copies a subset of L1 - ``epoch``, ``esa_lvl5``, ``esa_step``,
``esa_step_label``, ``fpga_rev``, ``fpga_type``, ``lut_choice``, ``plan_id``,
``sci_start_time``, ``sweep_table``, ``swp_l1a_flags`` - and adds:

.. list-table::
   :header-rows: 1
   :widths: 44 14 42

   * - Variable
     - Shape
     - Notes
   * - ``esa_energy``
     - (sweep, 72)
     - eV/q. The solved energy of every step of every sweep.
       ``VALIDMIN``/``VALIDMAX`` = 0 / 21000.
   * - ``swp_pcem_rate``
     - (sweep, 72)
     - counts / 0.145 s. Negative values (L1 fill) become ``NaN``.
   * - ``swp_scem_rate``
     - (sweep, 72)
     - as above
   * - ``swp_coin_rate``
     - (sweep, 72)
     - as above
   * - ``swp_pcem_rate_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - L1 count uncertainty / 0.145 s
   * - ``swp_scem_rate_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - as above
   * - ``swp_coin_rate_stat_uncert_plus`` / ``_minus``
     - (sweep, 72)
     - as above

**The L1 counts themselves are not copied to L2.** Only rates.

Every rate variable, every rate uncertainty and ``swp_l1a_flags`` get
``DEPEND_1 = "esa_energy"`` at L2, replacing the L1 ``DEPEND_1 = "esa_step"``.
That is what makes the L2 file plottable against physical energy.

Epoch convention
----------------

**[CODE]** The L1/L2 ``epoch`` for a sweep is **the creation time of the 7th
packet** (``SEQ_NUMBER == 6``), i.e. the centre of the 12-second sweep, chosen
to line up with mission conventions. The *start* of the sweep is available
separately as the ``sci_start_time`` UTC string.

Do not assume ``epoch`` is the sweep start. **[DOC]** The L3 measurement-time
formula is written against the sweep start:

.. math::

   t_i = t_{\mathrm{start}} + i \cdot \frac{12}{72}\ \mathrm{s}
       = t_{\mathrm{start}} + (i+1) \cdot 0.16 - \frac{0.145}{2}\ \mathrm{s}

for 0-indexed ESA step :math:`i` (recalling that step 0 is skipped).
