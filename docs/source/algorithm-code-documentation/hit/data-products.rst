.. _hit-data-products:

Data Products and What Feeds What
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This is the "what goes into what" map. It is the page to read before touching
``cli.py`` or adding a product.

Product inventory
-----------------

**[CODE]** Everything this repository produces for HIT. The ``logical_source``
strings come from ``imap_processing/cdf/config/imap_hit_global_cdf_attrs.yaml``
and are what ``write_cdf()`` uses to name the output file.

.. list-table::
   :header-rows: 1
   :widths: 32 8 12 48

   * - ``logical_source``
     - Level
     - Cadence
     - Contents
   * - ``imap_hit_l1a_hk``
     - L1A
     - 1 min
     - Housekeeping, **raw** DN values. The 64 ``leak_i_NN`` fields are
       collapsed into a single 2-D ``leak_i`` on an ``adc_channels`` (64)
       dimension.
   * - ``imap_hit_l1a_counts-standard``
     - L1A
     - 1 min
     - Decompressed **counts** for every fixed-format counter in the science
       frame: singles, coincidence, event-processing, priority buffer, and
       all six FG/BG rate arrays, plus ``ialirtrates``, ``l4fgrates``,
       ``l4bgrates``, the livetime counter and the frame header fields. Each
       gets ``_stat_uncert_plus`` / ``_stat_uncert_minus`` companions.
   * - ``imap_hit_l1a_counts-sectored``
     - L1A
     - 1 min records, in complete 10-min sets
     - Sectored **counts** reorganised by species:
       ``h_sectored_counts``, ``he4_sectored_counts``,
       ``cno_sectored_counts``, ``nemgsi_sectored_counts``,
       ``fe_sectored_counts``, each ``(epoch, <species>_energy_mean,
       azimuth, zenith)``. Plus per-species energy mean/delta variables,
       ``livetime_counter`` on its own ``epoch_livetime`` coordinate, and
       ``hdr_dynamic_threshold_state``.
   * - ``imap_hit_l1a_direct-events``
     - L1A
     - 1 min
     - ``pha_raw`` only - the **undecoded** concatenated binary of packets
       6-19. See :ref:`hit-gap-events`.
   * - ``imap_hit_l1b_hk``
     - L1B
     - 1 min
     - Housekeeping, **derived / engineering-unit** values. Same packet,
       re-parsed with ``use_derived_value=True``.
   * - ``imap_hit_l1b_standard-rates``
     - L1B
     - 1 min
     - Every L1A standard counter divided by the livetime fraction, plus the
       scaled uncertainties and ``dynamic_threshold_state``.
   * - ``imap_hit_l1b_summed-rates``
     - L1B
     - 1 min
     - 17 species (``h``, ``he3``, ``he4``, ``he``, ``c``, ``n``, ``o``,
       ``ne``, ``na``, ``mg``, ``al``, ``si``, ``s``, ``ar``, ``ca``,
       ``fe``, ``ni``), **67 wide energy bins total**, summed across
       penetration ranges then divided by livetime.
   * - ``imap_hit_l1b_sectored-rates``
     - L1B
     - 1 min records, in complete 10-min sets
     - Sectored counts divided by ``15 x`` the **previous** 10 minutes'
       summed livetime. Variables lose the ``_sectored_counts`` suffix and
       become plain ``h``, ``he4``, ``cno``, ``nemgsi``, ``fe``.
   * - ``imap_hit_l2_standard-intensity``
     - L2
     - 1 min
     - 17 species, **204 native energy bins total**, in
       :math:`\mathrm{cm^{-2}s^{-1}sr^{-1}(MeV/nuc)^{-1}}`. Variables are
       renamed ``<species>_standard_intensity``, with ``_stat_uncert_*``,
       ``_sys_err_*`` and ``_total_uncert_*``.
   * - ``imap_hit_l2_summed-intensity``
     - L2
     - 1 min
     - Same, from the 67 summed bins. Variables
       ``<species>_summed_intensity``.
   * - ``imap_hit_l2_macropixel-intensity``
     - L2
     - **10 min**
     - 5 species (``h``, ``he4``, ``cno``, ``nemgsi``, ``fe``), 10
       species/energy combinations, 120 look directions. Variables
       ``<species>_macropixel_intensity``, dimensions ``(epoch,
       <species>_energy_mean, azimuth, zenith)``. Carries ``epoch_delta`` of
       5 minutes.

Plus one non-CDF product:

.. list-table::
   :header-rows: 1
   :widths: 30 10 12 48

   * - Product
     - Level
     - Cadence
     - Contents
   * - HIT I-ALiRT record
     - --
     - 1 min
     - ``list[dict]`` of 6 electron, 4 proton and 2 helium space-weather
       rates destined for the I-ALiRT database, **not** a CDF. See
       :ref:`hit-ialirt`.

Not produced here
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Product
     - Status
   * - L3 PHA products (time, ion energy in MeV/nuc, charge Z per event)
     - **Separate repository.** Algorithm document section 9.1. Needs the
       decoded event records from L1A, which do not exist yet either - see
       :ref:`hit-gap-events`.
   * - L3 sectored products (pitch angle, gyrophase, 22.5 x 24 degree
       skymaps)
     - **Separate repository.** Algorithm document section 9.2. Consumes
       ``imap_hit_l2_macropixel-intensity`` + MAG L1D.
   * - L3 electron science products
     - **Separate repository.** Algorithm document section 9.3. Consumes the
       6 I-ALiRT electron rates plus modelled response matrices.
   * - Quicklook plots
     - **No HIT quicklook code exists in this repository.** The algorithm
       document does not specify any either.

See :ref:`hit-l3-scope` for what those products need from L1A and L2.

The processing chain
--------------------

**[DOC]** Algorithm document Figure 8, annotated with what is and is not here.

.. code-block:: text

   CCSDS packets (APID 1251 hk, 1252 science, 1253 I-ALiRT)
        |
        |  decommutation table (the frame byte map)
        v
   L1A  raw particle counts
        |  imap_hit_l1a_hk
        |  imap_hit_l1a_counts-standard
        |  imap_hit_l1a_counts-sectored
        |  imap_hit_l1a_direct-events        <-- raw binary only, not decoded
        |
        |  livetimes
        v
   L1B  particle count rates
        |  imap_hit_l1b_hk                   <-- from L0 again, not from L1A
        |  imap_hit_l1b_standard-rates
        |  imap_hit_l1b_summed-rates
        |  imap_hit_l1b_sectored-rates
        |
        |  geometry factors, energy bin widths, efficiencies,
        |  selected by dynamic threshold state
        v
   L2   particle intensities
        |  imap_hit_l2_standard-intensity
        |  imap_hit_l2_summed-intensity
        |  imap_hit_l2_macropixel-intensity
        |
        |  MAG L1D B-field, ADC-MeV coefficients, charge lookup tables,
        |  electron response matrices
        v
   L3   pitch angles, ion charge, science-quality electrons
        *** NOT IN THIS REPOSITORY ***

Note the two things that are **not** a simple level chain:

#. **L1B housekeeping does not come from L1A housekeeping.** It re-reads the
   same L0 CCSDS file with ``use_derived_value=True``, letting the XTCE
   calibrators do the conversion. The L1A and L1B HK products are the same
   packet, raw and derived.
#. **L2 standard intensity does not come from an L1B "standard" *product* in
   the obvious way.** It takes the L1B standard rates (which are per-Particle-
   ID arrays) and *then* does the cross-range summation into species/energy
   bins. The equivalent summation for the summed product happens one level
   earlier, at L1B. See :ref:`hit-l2-where-summing-happens`.

How the CLI is wired
--------------------

**[CODE]** ``imap_processing/cli.py``, class ``Hit``.

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Level
     - Descriptor
     - Dependencies expected
     - Entry point
   * - ``l1a``
     - (any)
     - Exactly **2**: the L0 ``raw`` CCSDS file and the SPICE time kernels.
     - ``hit_l1a(science_file, start_date)``
   * - ``l1b``
     - ``hk``
     - One L0 ``raw`` file.
     - ``hit_l1b(path, "hk")``
   * - ``l1b``
     - ``standard-rates``, ``summed-rates``, ``sectored-rates``
     - Exactly one L1A CDF (loaded with ``load_cdf``).
     - ``hit_l1b(dataset, descriptor)``
   * - ``l2``
     - (derived from the input)
     - Exactly **5**: one L1B science CDF matching ``-rates``, plus **4**
       ancillary CSVs matching ``-dt`` (one per dynamic threshold state).
     - ``hit_l2(l1b_dataset, ancillary_files)``

Two things to know about this wiring:

* **L1A takes a ``start_date``.** ``hit_l1a`` raises ``ValueError`` without
  one. It is used to trim the day-boundary buffer - see
  :ref:`hit-l1a-day-boundary`.
* **L2 dispatches on the input's ``Logical_source``**, not on the descriptor.
  ``hit_l2`` inspects ``dependency_sci.attrs["Logical_source"]`` for
  ``imap_hit_l1b_summed-rates`` / ``standard-rates`` / ``sectored-rates`` and
  picks the processing function from that. If none match it silently returns
  ``None``.

Which ancillary files each L2 product needs
-------------------------------------------

**[CODE]** The CLI hands ``hit_l2`` all four files it was given; the code then
picks by filename substring ``dt<N>-factors``. The product determines which
*family* of four:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - L2 product
     - Ancillary family
   * - ``imap_hit_l2_standard-intensity``
     - ``imap_hit_standard-dt{0,1,2,3}-factors_<date>_v<NNN>.csv``
   * - ``imap_hit_l2_summed-intensity``
     - ``imap_hit_summed-dt{0,1,2,3}-factors_<date>_v<NNN>.csv``
   * - ``imap_hit_l2_macropixel-intensity``
     - ``imap_hit_sectored-dt{0,1,2,3}-factors_<date>_v<NNN>.csv``

Only the states actually present in the data are read - ``load_ancillary_data``
takes ``set(dataset["dynamic_threshold_state"].values)``. Passing the wrong
family will raise ``StopIteration`` from the ``next(...)`` generator lookup, not
a friendly error.

See :ref:`hit-ancillary` for the file format.

Dimensions and coordinates
--------------------------

**[CODE]** Worth having in front of you when reading the code.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Coordinate
     - Size
     - Where it comes from
   * - ``epoch``
     - n frames
     - Mean of the first and last packet epoch in the science frame
       (``calculate_epoch_mean``). For the L2 macropixel product it is
       recomputed to the centre of the **collection** window, 10 minutes
       before transmission.
   * - ``sc_tick``
     - n packets
     - Per-packet spacecraft time. Used as the dimension for the CCSDS header
       fields, which are per-packet not per-frame.
   * - ``epoch_livetime``
     - n frames
     - A shadow of ``epoch`` attached to ``livetime_counter`` so that
       filtering ``epoch`` for complete sectored sets does not also filter
       livetime. See :ref:`hit-l1b-sectored`.
   * - ``gain``
     - 2
     - 0 = high gain, 1 = low gain. Only used by ``sngrates``.
   * - ``zenith``
     - 8
     - Declination bin centres, 11.25 to 168.75 degrees.
   * - ``azimuth``
     - 15
     - Inclination bin centres, 12 to 348 degrees.
   * - ``<species>_energy_mean``
     - varies
     - Per-species energy bin identifier, with
       ``<species>_energy_delta_plus`` / ``_delta_minus`` giving the bin
       edges.
   * - ``<field>_index``
     - varies
     - Plain integer index into a raw counter array, e.g.
       ``l3fgrates_index`` runs 0-166. **This is the Particle ID** for the
       FG arrays.
