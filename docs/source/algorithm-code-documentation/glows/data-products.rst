.. _glows-data-products:

Data Products and Pipeline
==========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the map of **what exists, what feeds what, and what it is called**.
Use it to find the right module and the right ``Logical_source`` before diving
into an algorithm page.

Level definitions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 8 24 68

   * - Level
     - Units / organisation
     - Meaning for GLOWS
   * - L0
     - Raw bits, per pointing
     - Binary CCSDS packet stream,
       ``imap_glows_l0_raw_{date}-repoint{n}_v{vvv}.pkts``. **Not produced by
       this repository** - it is the input.
   * - L1A
     - Integer-encoded, per pointing
     - Unpacked telemetry blocks. Histogram bins and all ancillary values are
       still in onboard integer encoding. Direct events are decompressed from
       the timestamp/offset stream into explicit ``(sec, subsec, pulse_length,
       multi_event)`` tuples.
   * - L1B
     - Physical units, per pointing
     - Ancillary values decoded (°C, V, s, µs). Times become floats. Bad-time
       flags decoded and extended with ground-computed ones. Bad-angle flag
       arrays computed against the sky masks. Spacecraft position, velocity,
       spin axis and ground spin period added from SPICE. **Histogram counts
       are untouched** - no calibration happens at L1B.
   * - L2
     - Rayleighs, per pointing
     - One **daily lightcurve**: good-time L1B histograms co-added, divided by
       exposure, converted to photon flux in Rayleighs by the instrument team's
       cps-per-Rayleigh factor. Bins are re-indexed from IMAP spin angle to
       GLOWS position angle. Bad-angle flags are OR-ed across contributing
       blocks. Sky coordinates per bin added.
   * - L3A-L3E
     - --
     - **Not produced by this repository.** See below.

.. note::

   **Direct events stop at L1B.** The document is explicit (§3.8): "Automated
   processing of direct events in the SDC ends at Level-1B." There is no L2 DE
   product and none is planned in the current revision.

What happens after L2 (elsewhere)
---------------------------------

You do not implement any of this here, but you should know what your L2 output
feeds so you can reason about requirements.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Level
     - Product
   * - L3A
     - Daily **low-resolution** lightcurve of Lyman-α photon flux. Nominally 90
       bins of 4°, rebinned from L2's 3600 bins, with star- and
       region-contaminated bins **removed** (not just masked), plus estimates of
       the time-independent extra-heliospheric background and a time-dependent
       background. Uses SWE electron spectra for background correction.
   * - L3B
     - Carrington-period-averaged **ionization rate profiles** (charge exchange
       + photoionization) versus heliolatitude, via the WawHelioIon-MP model.
       Needs F10.7 and SWAPI/OMNI2.
   * - L3C
     - Latitudinal profiles of **solar wind speed and density**, Carrington
       averaged.
   * - L3D
     - Time **history of solar parameters** (WawHelioIon time series), using the
       composite Lyman-α index and OMNI2.
   * - L3E
     - Daily **ENA survival probabilities** for IMAP-Lo, Hi and Ultra. These are
       consumed by the other instruments' own L3 pipelines.

L3 also needs a **bad-day list** from the GLOWS team (days ruined by flares,
CMEs, etc.), and an **averaged spin-axis product** from the SDC shared with the
ENA instruments so that everyone uses identical pointing directions.

Source packets
--------------

**[CODE]** ``GlowsParams`` enum in ``glows/l0/decom_glows.py``. Fields are
defined in ``glows/packet_definitions/``.

.. list-table::
   :header-rows: 1
   :widths: 12 16 20 52

   * - APID
     - Hex
     - Enum
     - Contents
   * - 1480
     - ``0x5c8``
     - ``GlowsParams.HIST_APID``
     - One block histogram plus its ancillary data. 24 fields after the CCSDS
       header. Defined in ``P_GLX_TMSCHIST.xml``.
   * - 1481
     - ``0x5c9``
     - ``GlowsParams.DE_APID``
     - One second of direct events (possibly split across several packets).
       4 fields after the header. Defined in ``P_GLX_TMSCDE.xml``.

Both are loaded from the single master document
``packet_definitions/GLX_COMBINED.xml``, which is what ``decom_packets`` passes
to ``packet_generator``. Any other APID in the file (housekeeping,
telecommands, memory dumps, boot reports) is **silently ignored** - GLOWS
housekeeping is out of scope for this repository.

Histogram packet fields (APID 1480)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §3.4.1]** The uppercase name is the telemetry-definition mnemonic; the
lowercase name is what the GLOWS Python bundle and these docs call it. Big
endian throughout.

.. list-table::
   :header-rows: 1
   :widths: 16 30 8 46

   * - Mnemonic
     - Name
     - Bits
     - Meaning
   * - ``STARTID``
     - ``first_spin_id_in_block``
     - 32
     - Ordinal ID of the first IMAP spin in the block.
   * - ``ENDID``
     - ``diff_spin_id_in_block``
     - 16
     - **Difference** from the first ID, not the last ID. The mnemonic is
       misleading and the document says so.
   * - ``FLAGS``
     - ``histogram_validity_flags``
     - 16
     - The 10 onboard bad-time flags; upper 6 bits reserved, zero.
   * - ``SWVER``
     - ``software_version``
     - 24
     - Flight software version used to generate the histogram.
   * - ``SEC`` / ``SUBSEC``
     - ``imap_start_time_second`` / ``_subsecond``
     - 32 / 24
     - IMAP-clock block start. Subseconds are interpolated onboard from the
       GLOWS clock; limit 2 000 000.
   * - ``OFFSETSEC`` / ``OFFSETSUBSEC``
     - ``imap_diff_second`` / ``_subsecond``
     - 16 / 24
     - IMAP-clock **end-time offset** (duration), not an absolute end time.
   * - ``GLXSEC`` / ``GLXSUBSEC``
     - ``glows_start_time_second`` / ``_subsecond``
     - 32 / 24
     - Same, from the GLOWS internal SCIENCE timer.
   * - ``GLXOFFSEC`` / ``GLXOFFSUBSEC``
     - ``glows_diff_second`` / ``_subsecond``
     - 16 / 24
     - Same, GLOWS clock.
   * - ``SPINS``
     - ``number_of_spins_per_block``
     - 8
     - ``n_block``. Taken from FSW configuration; included so the ground can
       cross-check it against ``ENDID``.
   * - ``NBINS``
     - ``number_of_bins_per_histogram``
     - 16
     - ``n_bin``.
   * - ``TEMPAVG`` / ``TEMPVAR``
     - ``filter_temperature_average`` / ``_variance``
     - 8 / 16
     - Encoded per Eq. 37 / Eq. 43.
   * - ``HVAVG`` / ``HVVAR``
     - ``hv_voltage_average`` / ``_variance``
     - 16 / 32
     - CEM high voltage.
   * - ``SPAVG`` / ``SPVAR``
     - ``spin_period_average`` / ``_variance``
     - 16 / 32
     - The spin period **used onboard for histogramming**.
   * - ``ELAVG`` / ``ELVAR``
     - ``pulse_length_average`` / ``_variance``
     - 8 / 16
     - Event impulse length.
   * - ``EVENTS``
     - ``number_of_events``
     - 32
     - Total events in the histogram.
   * - ``HISTTAB``
     - ``histogram``
     - ``NBINS`` × 8
     - The counts, in **IMAP spin angle** order.

Direct-event packet fields (APID 1481)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 34 12 38

   * - Mnemonic
     - Name
     - Bits
     - Meaning
   * - ``SEC``
     - ``imap_start_time_second``
     - 32
     - IMAP-clock whole second this packet's data belongs to.
   * - ``LEN``
     - ``number_of_de_packets``
     - 16
     - How many CCSDS packets carry this one second of data.
   * - ``SEQ``
     - ``seq_num_of_de_packet``
     - 16
     - Index of this packet within that sequence.
   * - ``DATA``
     - ``de_data``
     - variable
     - Payload. Structure depends on ``LEN``/``SEQ`` - see :ref:`glows-l1a`.

``LEN == 0 and SEQ == 0`` means the packet carries **no useful data** (e.g. a
requested second was not found in instrument memory).

Products produced by this repository
------------------------------------

**[CODE]** Exact ``Logical_source`` strings from
``imap_processing/cdf/config/imap_glows_global_cdf_attrs.yaml``. Anything not in
this table does not exist.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - ``Logical_source``
     - Descriptor
     - Contents
   * - ``imap_glows_l1a_hist``
     - ``hist``
     - Per-block histograms, integer-encoded ancillary, raw onboard flag word.
   * - ``imap_glows_l1a_de``
     - ``de``
     - Per-second direct-event arrays plus the ``data_every_second``
       housekeeping structure.
   * - ``imap_glows_l1b_hist``
     - ``hist``
     - Per-block histograms in physical units, 17 bad-time flags, 4 × ``n_bin``
       bad-angle flags, SPICE-derived spacecraft state.
   * - ``imap_glows_l1b_de``
     - ``de``
     - Per-second direct-event times and pulse lengths, decoded housekeeping,
       11 DE flags.
   * - ``imap_glows_l2_hist``
     - ``hist``
     - The daily lightcurve. One epoch per file.

.. note::

   ``docs/source/filename-convention/naming-conventions.rst`` lists ``hist``,
   ``de``, ``lightcurve``, ``ionization-rate`` and ``survival-probabilities`` as
   GLOWS descriptors. Only ``hist`` and ``de`` are produced here; the L2 product
   uses ``hist``, **not** ``lightcurve``, despite being a lightcurve.

Data flow
---------

.. code-block:: text

   ┌─────────────────────────── instrument team supplies ────────────────────────┐
   │ l1b-conversion-table-for-anc-data   l1b-map-of-uv-sources                   │
   │ l1b-map-of-excluded-regions         l1b-exclusions-by-instr-team            │
   │ l1b-suspected-transients            pipeline-settings        l2-calibration │
   └────────────────────────────────────────────────────────────────────────────┘
             │                                        │                │
   L0 .pkts  │                                        │                │
      │      │                                        │                │
      ▼      │                                        │                │
   ┌──────────────────┐                               │                │
   │  glows_l1a()     │  decom_packets -> APID split  │                │
   └────────┬─────────┘                               │                │
            ├──────────────► imap_glows_l1a_hist ─────┼──┐             │
            └──────────────► imap_glows_l1a_de   ──┐  │  │             │
                                                   │  │  │             │
                       conversion table only ──────┘  │  │             │
                              │                       │  │             │
                              ▼                       ▼  ▼             │
                    ┌──────────────────┐   ┌────────────────────────┐  │
                    │  glows_l1b_de()  │   │      glows_l1b()       │  │
                    └────────┬─────────┘   └───────────┬────────────┘  │
                             ▼                         ▼               │
                    imap_glows_l1b_de        imap_glows_l1b_hist       │
                       (pipeline ends)                 │               │
                                                       ▼               ▼
                                             ┌────────────────────────────┐
                                             │        glows_l2()          │
                                             │ + pipeline-settings        │
                                             │ + l2-calibration           │
                                             └────────────┬───────────────┘
                                                          ▼
                                                imap_glows_l2_hist
                                                          │
                                                          ▼
                                            L3A..L3E  (separate repository)

   SPICE (spin table, CK, SPK, frame kernels) feeds glows_l1b() and glows_l2().

CLI wiring
----------

**[CODE]** ``class Glows(ProcessInstrument)`` in ``imap_processing/cli.py``.
Supported ``data_level`` values are ``l1a``, ``l1b`` and ``l2``; anything else
raises ``NotImplementedError``.

L1A
^^^

.. code-block:: python

   science_files = dependencies.get_file_paths(source="glows", data_type="l0")
   # exactly one file required, else ValueError
   datasets = glows_l1a(science_files[0])

Returns a list of **one or two** datasets - histogram and/or direct event,
whichever the L0 file contained. A single call produces both products.

L1B
^^^

Requires exactly one L1A CDF. The conversion table is loaded for **both**
branches:

.. code-block:: python

   conversion_table_file = dependencies.get_processing_inputs(
       descriptor="l1b-conversion-table-for-anc-data")[0]
   with open(conversion_table_file.imap_file_paths[0].construct_path()) as f:
       conversion_table_dict = json.load(f)

   current_day = np.datetime64(...)          # from self.start_date
   day_buffer  = current_day + np.timedelta64(3, "D")

Then the branch is chosen on ``"hist" in self.descriptor``:

* **Histogram branch** additionally pulls five ancillary inputs, each wrapped in
  a ``GlowsAncillaryCombiner`` with the 3-day buffer:
  ``l1b-map-of-excluded-regions``, ``l1b-map-of-uv-sources``,
  ``l1b-suspected-transients``, ``l1b-exclusions-by-instr-team``,
  ``pipeline-settings``. All five ``.combined_dataset`` values plus the
  conversion table go into ``glows_l1b(...)``.
* **Direct-event branch** is just
  ``glows_l1b_de(input_dataset, conversion_table_dict)``.

.. warning::

   The branch condition is a substring test on the **descriptor**, so a
   descriptor containing "hist" anywhere selects the histogram path. There is no
   validation that the input CDF's ``Logical_source`` matches the descriptor.

L2
^^

Requires exactly one L1B CDF plus ``pipeline-settings`` and ``l2-calibration``,
again through ``GlowsAncillaryCombiner`` with the 3-day buffer:

.. code-block:: python

   datasets = glows_l2(
       input_dataset,
       pipeline_settings_combiner.combined_dataset,
       calibration_combiner.combined_dataset,
   )

``glows_l2`` returns an **empty list** - and therefore no file is written - when
there are no good-time L1B blocks, or when flux, uncertainties and exposure are
all zero. That is intentional and not an error.

.. note::

   The 3-day ``day_buffer`` exists because instrument-team ancillary files are
   time-ranged and open-ended: the combiner needs an end date to close the last
   validity interval. It is not a data selection window.

CDF attribute configuration
---------------------------

**[CODE]** ``imap_processing/cdf/config/``:

* ``imap_glows_global_cdf_attrs.yaml`` - **the definitive product list**. All
  five ``Logical_source`` values above.
* ``imap_glows_l1a_variable_attrs.yaml``
* ``imap_glows_l1b_variable_attrs.yaml``
* ``imap_glows_l2_variable_attrs.yaml``

Each level's module calls ``ImapCdfAttributes.add_instrument_global_attrs("glows")``
and ``.add_instrument_variable_attrs("glows", "<level>")``. If a variable name is
missing from the YAML, ``get_variable_attributes`` raises - which is the usual
first failure when adding a new output variable.
