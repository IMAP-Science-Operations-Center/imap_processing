.. _codice-data-products:

Data Products and Pipeline
==========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the map of **what exists, what feeds what, and what it is called**.
Use it to find the right module and the right ``Logical_source`` before diving
into an algorithm page.

Level definitions
-----------------

**[DOC]** Section 2.1 and section 5.2:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Level
     - Meaning for CoDICE
   * - L0
     - Raw CCSDS packets. A ``.pkts`` file, not produced by this repository.
   * - L1A
     - Raw data decommutated into formatted variables. Decompressed and
       un-collapsed per the SCI-LUT, de-spun where applicable, NSO-masked, but
       **counts, not rates**. Electrons are written as Flashed minus Unflashed.
   * - L1B
     - Counts converted to rates using accumulation times and spin rates. For
       housekeeping, raw DN converted to engineering units.
   * - L2
     - Geometric factors, efficiencies and other conversion factors applied to
       give physical units. Species separated into distinct CDF variables.
       **The lowest level usable for science analysis.**
   * - L3
     - Higher-level science: partial densities, ratios, VDFs, pitch angles,
       Hi+Lo combinations. **Not produced by this repository** - see
       :ref:`codice-l3-scope`.

Source packets (APIDs)
----------------------

**[CODE]** ``CODICEAPID`` in ``imap_processing/codice/constants.py``. Fields for
the science and housekeeping APIDs are defined in the two XTCE files in
``imap_processing/codice/packet_definitions/``.

.. list-table::
   :header-rows: 1
   :widths: 8 9 33 12 38

   * - Dec
     - Hex
     - Name
     - L1A?
     - Contents / cadence
   * - 1136
     - 0x470
     - ``COD_NHK``
     - yes
     - Nominal housekeeping: voltages, currents, digital settings.
   * - 1152
     - 0x480
     - ``COD_LO_IAL``
     - I-ALiRT only
     - Lo I-ALiRT trickle: a subset of SW species counts, 15 data bytes per
       packet, 232 packets per set.
   * - 1153
     - 0x481
     - ``COD_LO_PHA``
     - yes
     - Lo direct events, 8 priorities, up to 11520 events/cycle (P3).
       Segmented across CCSDS packets.
   * - 1155
     - 0x483
     - ``COD_LO_SW_PRIORITY_COUNTS``
     - yes
     - Sunward priority counts, priorities 0-4.
   * - 1156
     - 0x484
     - ``COD_LO_SW_SPECIES_COUNTS``
     - yes
     - Sunward species counts, 16 species, 128 ESA steps.
   * - 1157
     - 0x485
     - ``COD_LO_NSW_SPECIES_COUNTS``
     - **no**
     - Non-sunward species counts, 8 species.
   * - 1158
     - 0x486
     - ``COD_LO_SW_ANGULAR_COUNTS``
     - **no**
     - Sunward angular counts, 128 x 12 spin sectors x 5 positions.
   * - 1159
     - 0x487
     - ``COD_LO_NSW_ANGULAR_COUNTS``
     - **no**
     - Non-sunward angular counts, 128 x 12 spin sectors x 19 positions.
   * - 1160
     - 0x488
     - ``COD_LO_NSW_PRIORITY_COUNTS``
     - yes
     - Non-sunward priority counts, priorities 5-6.
   * - 1161
     - 0x489
     - ``COD_LO_INST_COUNTS_AGGREGATED``
     - yes
     - Lo engineering rates (TCR, DCR, STA, STB, SP, total position).
   * - 1162
     - 0x48A
     - ``COD_LO_INST_COUNTS_SINGLES``
     - yes
     - Lo per-APD singles, 24 rates.
   * - 1168
     - 0x490
     - ``COD_HI_IAL``
     - I-ALiRT only
     - Hi I-ALiRT trickle: H only, 5 data bytes per packet, 199-239 packets per
       set.
   * - 1169
     - 0x491
     - ``COD_HI_PHA``
     - yes
     - Hi direct events, 6 priorities, up to 10000 events/cycle (P3).
   * - 1170
     - 0x492
     - ``COD_HI_INST_COUNTS_AGGREGATED``
     - yes
     - Hi engineering counters summed over all spin sectors and SSDs.
   * - 1171
     - 0x493
     - ``COD_HI_INST_COUNTS_SINGLES``
     - yes
     - Hi per-SSD counters (TCR, SSDO, STSSD).
   * - 1172
     - 0x494
     - ``COD_HI_OMNI_SPECIES_COUNTS``
     - yes
     - Hi omni-directional species counts, 9 species, sqrt(2)-spaced E/n bins,
       summed over 4 spins, 1 min cadence.
   * - 1173
     - 0x495
     - ``COD_HI_SECT_SPECIES_COUNTS``
     - yes
     - Hi sectored species counts, 4 species, x2-spaced E/n bins, 12 spin
       sectors x 12 SSDs, 16 spins, 4 min cadence.
   * - 1174
     - 0x496
     - ``COD_HI_INST_COUNTS_PRIORITIES``
     - yes
     - Hi priority counts, 6 priorities.

Other APIDs are defined in ``CODICEAPID`` (``COD_AUT``, ``COD_BOOT_HK``,
``COD_MEMDMP``, ``COD_SHK``, the ``COD_DIAG_*`` family, ``COD_CSTOL_CONFIG``,
etc.) but are **not processed** by ``process_l1a``.

.. warning::

   **[CODE]** APIDs 1157, 1158 and 1159 have ``Logical_source`` entries, CDF
   variable attributes and species name lists in the repository, but **there is
   no processing module for them and ``process_l1a`` has no branch for them**.
   The Lo species, angular and NSW products described in sections 10.3.3 and
   10.3.4 of the document are therefore **only half built**. See
   :ref:`codice-implementation-status`.

   **[DOC]** Rev 3 Chg 1 section 9.2 now says the instrument team is **not
   currently producing** ``lo-nsw-species``, ``lo-sw-angular`` or
   ``lo-nsw-angular`` at any level, "due to issues identified post-launch".
   The missing code therefore matches current operations. The algorithms are
   still fully specified in sections 10-12, and "not currently" suggests the
   products may return. Ask the team before building them. See
   :ref:`codice-data-caveats`.

Products this repository can emit
---------------------------------

**[CODE]** These are the exact ``Logical_source`` strings, all defined in
``imap_processing/cdf/config/imap_codice_global_cdf_attrs.yaml``. A string that
is not in that file cannot be written - ``get_global_attributes`` raises.

Level 1A
^^^^^^^^

Produced by ``codice_l1a.process_l1a(dependencies)``. One call processes every
APID in the L0 file and returns a list of datasets.

.. list-table::
   :header-rows: 1
   :widths: 36 10 54

   * - ``Logical_source``
     - APID
     - Status / notes
   * - ``imap_codice_l1a_hskp``
     - 1136
     - Raw DN. Written straight from ``packet_file_to_datasets``.
   * - ``imap_codice_l1a_lo-counters-aggregated``
     - 1161
     - Implemented.
   * - ``imap_codice_l1a_lo-counters-singles``
     - 1162
     - Implemented.
   * - ``imap_codice_l1a_lo-sw-priority``
     - 1155
     - Implemented.
   * - ``imap_codice_l1a_lo-nsw-priority``
     - 1160
     - Implemented.
   * - ``imap_codice_l1a_lo-sw-species``
     - 1156
     - Implemented.
   * - ``imap_codice_l1a_lo-nsw-species``
     - 1157
     - **Declared but not produced.**
   * - ``imap_codice_l1a_lo-sw-angular``
     - 1158
     - **Declared but not produced.**
   * - ``imap_codice_l1a_lo-nsw-angular``
     - 1159
     - **Declared but not produced.**
   * - ``imap_codice_l1a_lo-direct-events``
     - 1153
     - Implemented.
   * - ``imap_codice_l1a_lo-ialirt``
     - 1152
     - Declared, but the I-ALiRT path deliberately writes **no L1A CDF**; the
       intermediate dataset borrows the ``lo-sw-species`` global attributes.
   * - ``imap_codice_l1a_hi-counters-aggregated``
     - 1170
     - Implemented.
   * - ``imap_codice_l1a_hi-counters-singles``
     - 1171
     - Implemented.
   * - ``imap_codice_l1a_hi-priority``
     - 1174
     - Implemented.
   * - ``imap_codice_l1a_hi-omni``
     - 1172
     - Implemented.
   * - ``imap_codice_l1a_hi-sectored``
     - 1173
     - Implemented.
   * - ``imap_codice_l1a_hi-direct-events``
     - 1169
     - Implemented.
   * - ``imap_codice_l1a_hi-ialirt``
     - 1168
     - Declared; produced only inside the I-ALiRT pipeline, no CDF written.

Level 1B
^^^^^^^^

Produced by ``codice_l1b.process_codice_l1b(l1a_file)``, **one file in, one file
out**. The descriptor is derived from the input file's ``Logical_source``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``Logical_source``
     - Status / notes
   * - ``imap_codice_l1b_hskp``
     - **Produced during the L1A run**, not by ``process_codice_l1b``. See below.
   * - ``imap_codice_l1b_lo-counters-aggregated``
     - Implemented.
   * - ``imap_codice_l1b_lo-counters-singles``
     - Implemented.
   * - ``imap_codice_l1b_lo-sw-priority``
     - Implemented.
   * - ``imap_codice_l1b_lo-nsw-priority``
     - Implemented.
   * - ``imap_codice_l1b_lo-sw-species``
     - Implemented.
   * - ``imap_codice_l1b_lo-nsw-species``
     - **Declared, no L1A input, no rate branch.**
   * - ``imap_codice_l1b_lo-sw-angular``
     - **Declared, no L1A input, no rate branch.**
   * - ``imap_codice_l1b_lo-nsw-angular``
     - **Declared, no L1A input, no rate branch.**
   * - ``imap_codice_l1b_lo-ialirt``
     - Computed in memory by the I-ALiRT pipeline; no CDF written.
   * - ``imap_codice_l1b_hi-counters-aggregated``
     - Implemented.
   * - ``imap_codice_l1b_hi-counters-singles``
     - Implemented.
   * - ``imap_codice_l1b_hi-priority``
     - Implemented.
   * - ``imap_codice_l1b_hi-omni``
     - Implemented.
   * - ``imap_codice_l1b_hi-sectored``
     - Implemented.
   * - ``imap_codice_l1b_hi-ialirt``
     - Computed in memory by the I-ALiRT pipeline; no CDF written.

.. note::

   **There is no L1B direct-event product**, by design. Direct events go
   straight from L1A to L2, where the bit values are converted to physical
   units. The document agrees (section 12.1.1 / 12.2.1 convert "L1A event data"
   to L2).

.. important::

   **[CODE]** ``imap_codice_l1b_hskp`` is written by ``process_l1a``, not by the
   L1B CLI branch. When the housekeeping APID is present, ``process_l1a`` calls
   ``packet_file_to_datasets`` a **second** time with ``use_derived_value=True``
   so that the XTCE polynomial calibrators produce engineering units, and
   appends that as a separate dataset. This is why a single ``l1a`` invocation
   returns both ``imap_codice_l1a_hskp`` and ``imap_codice_l1b_hskp``.

Level 2
^^^^^^^

Produced by ``codice_l2.process_codice_l2(descriptor, dependencies)``.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - ``Logical_source``
     - Status / notes
   * - ``imap_codice_l2_lo-sw-species``
     - **Implemented.** Solar-wind and pickup-ion intensities with geometric
       factors and efficiencies.
   * - ``imap_codice_l2_lo-nsw-species``
     - **Declared only.** No branch in ``process_codice_l2``.
   * - ``imap_codice_l2_lo-sw-angular``
     - **Declared only.** No branch.
   * - ``imap_codice_l2_lo-nsw-angular``
     - **Declared only.** No branch.
   * - ``imap_codice_l2_lo-direct-events``
     - **Implemented.** Physical-unit conversion.
   * - ``imap_codice_l2_hi-omni``
     - **Implemented.** Omni-directional intensities.
   * - ``imap_codice_l2_hi-sectored``
     - **Implemented.** Sectored intensities plus spin/elevation angles.
   * - ``imap_codice_l2_hi-direct-events``
     - **Implemented.** Physical-unit conversion.

``process_codice_l2`` also has a **pass-through branch** listing
``hi-counters-singles``, ``hi-counters-aggregated``, ``lo-counters-singles``,
``lo-counters-aggregated``, ``lo-sw-priority`` and ``lo-nsw-priority`` with the
comment "No changes needed. Just save to an L2 CDF file. TODO: May not even need
L2 files for these products". **None of those six descriptors has a
``Logical_source`` entry**, and the branch as written raises - see
:ref:`codice-implementation-status`.

Product flow
------------

.. code-block:: text

   L0 .pkts
     |
     +-- 1136 --> l1a_hskp ------------------> l1b_hskp (derived values, same run)
     |
     +-- 1153 --> l1a_lo-direct-events -------------------------> l2_lo-direct-events
     +-- 1169 --> l1a_hi-direct-events -------------------------> l2_hi-direct-events
     |
     +-- 1156 --> l1a_lo-sw-species --> l1b_lo-sw-species ------> l2_lo-sw-species
     +-- 1155 --> l1a_lo-sw-priority --> l1b_lo-sw-priority ----> (none)
     +-- 1160 --> l1a_lo-nsw-priority --> l1b_lo-nsw-priority --> (none)
     +-- 1161 --> l1a_lo-counters-aggregated --> l1b_... -------> (none)
     +-- 1162 --> l1a_lo-counters-singles --> l1b_... ----------> (none)
     |
     +-- 1172 --> l1a_hi-omni --> l1b_hi-omni ------------------> l2_hi-omni
     +-- 1173 --> l1a_hi-sectored --> l1b_hi-sectored ----------> l2_hi-sectored
     +-- 1174 --> l1a_hi-priority --> l1b_hi-priority ----------> (none)
     +-- 1170 --> l1a_hi-counters-aggregated --> l1b_... -------> (none)
     +-- 1171 --> l1a_hi-counters-singles --> l1b_... ----------> (none)
     |
     +-- 1157/1158/1159 --> NOT PROCESSED

   I-ALiRT stream (separate entry point, imap_processing/ialirt/l0/process_codice.py)
     +-- 1152 --> l1a_lo_species --> convert_to_rates --> ratios (DynamoDB items)
     +-- 1168 --> l1a_ialirt_hi  --> convert_to_rates --> intensities

CLI wiring
----------

**[CODE]** ``imap_processing/cli.py``, class ``Codice``:

.. code-block:: python

   if self.data_level == "l1a":
       datasets = codice_l1a.process_l1a(dependencies)
       for i, ds in enumerate(datasets):
           datasets[i] = filter_day_boundary_data(ds, self.start_date)

   elif self.data_level == "l1b":
       science_files = dependencies.get_file_paths(source="codice")
       if len(science_files) != 1:
           raise ValueError(...)
       datasets = [codice_l1b.process_codice_l1b(science_files[0])]

   elif self.data_level == "l2":
       datasets = [codice_l2.process_codice_l2(self.descriptor, dependencies)]

Things worth knowing about this:

* **L1A is one-to-many.** A single ``--level l1a`` run emits every product
  present in the L0 file. ``--descriptor`` is not used to select a product.
* **``filter_day_boundary_data`` is applied to every L1A dataset**, trimming
  records that fall outside the requested UTC day. It is applied only at L1A.
* **L1B is one-to-one** and requires exactly one CoDICE science file.
* **L2 dispatches on ``self.descriptor``**, which is used both to fetch the
  input file (``dependencies.get_file_paths(descriptor=descriptor)``) and to
  build the output ``Logical_source``. It then re-parses the descriptor out of
  the returned filename with ``ScienceFilePath``.

Input dependencies per level
----------------------------

**[CODE]** What the SDC must supply in the ``ProcessingInputCollection``:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Level
     - Descriptor / type
     - Purpose
   * - L1A
     - ``data_type="l0"``
     - The ``.pkts`` file.
   * - L1A
     - ``descriptor="l1a-sci-lut"``
     - The SCI-LUT JSON. **Required for every APID except the two PHA
       (direct-event) APIDs**, which are unpacked without it.
   * - L1B
     - ``source="codice"``
     - Exactly one L1A CDF.
   * - L2
     - ``descriptor=<product descriptor>``
     - The L1B (or, for direct events, L1A) CDF.
   * - L2 Lo
     - ``l2-lo-gfactor``, ``l2-lo-efficiency``
     - Geometric factors and efficiencies for species/angular intensities.
   * - L2 Lo DE
     - ``l2-lo-onboard-energy-table``, ``l2-lo-onboard-energy-bins``,
       ``l2-lo-onboard-mpq-cal``
     - APD energy, ESA step and TOF conversions.
   * - L2 Hi
     - ``l2-hi-omni-efficiency``, ``l2-hi-sectored-efficiency``
     - Efficiencies and the geometric factor (row ``GF``).
   * - L2 Hi DE
     - ``l2-hi-energy-table``, ``l2-hi-tof-table``
     - SSD energy, TOF and energy-per-nucleon conversions.

See :ref:`codice-ancillary` for the contents of each of these.

Species inventories
-------------------

**[CODE]** ``constants.py``. These lists drive which CDF variables exist; the
**actual** species ordering in the telemetry comes from the SCI-LUT and is
cross-checked at L1A.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Constant
     - Members
   * - ``LO_SW_SPECIES_VARIABLE_NAMES`` (16)
     - ``hplus``, ``heplusplus``, ``cplus4``, ``cplus5``, ``cplus6``,
       ``oplus5``, ``oplus6``, ``oplus7``, ``oplus8``, ``ne``, ``mg``, ``si``,
       ``fe_loq``, ``fe_hiq``, ``heplus``, ``cnoplus``
   * - ``LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES`` (14)
     - the above minus ``heplus`` and ``cnoplus``
   * - ``LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES`` (2)
     - ``heplus``, ``cnoplus``
   * - ``LO_SW_ANGULAR_VARIABLE_NAMES`` (5)
     - ``hplus``, ``heplusplus``, ``oplus6``, ``fe_loq``, ``heplus``
   * - ``LO_NSW_ANGULAR_VARIABLE_NAMES`` (2)
     - ``heplusplus``, ``heplus``
   * - ``LO_SW_PRIORITY_VARIABLE_NAMES`` (5)
     - ``p0_tcrs``, ``p1_hplus``, ``p2_heplusplus``, ``p3_heavies``, ``p4_dcrs``
   * - ``LO_NSW_PRIORITY_VARIABLE_NAMES`` (2)
     - ``p5_heavies``, ``p6_hplus_heplusplus``
   * - ``LO_COUNTERS_AGGREGATED_VARIABLE_NAMES`` (6)
     - ``tcr``, ``dcr``, ``sta``, ``stb``, ``sp``, ``total_position_count``
   * - ``LO_COUNTERS_SINGLES_VARIABLE_NAMES``
     - ``apd_singles`` (one variable, 24 APDs on an axis)
   * - ``HI_OMNI_VARIABLE_NAMES`` (9)
     - ``h``, ``he3``, ``he4``, ``c``, ``o``, ``ne_mg_si``, ``fe``, ``uh``,
       ``junk``
   * - ``HI_SECTORED_VARIABLE_NAMES`` (4)
     - ``h``, ``he3he4``, ``cno``, ``fe``
   * - ``HI_PRIORITY_VARIABLE_NAMES`` (6)
     - ``priority0`` ... ``priority5``
   * - ``HI_COUNTERS_AGGREGATED_VARIABLE_NAMES`` (7)
     - ``dcr``, ``mst``, ``starts_only``, ``stops_only``, ``singles_starts``,
       ``singles_stops``, ``low_tof_cutoff``
   * - ``HI_COUNTERS_SINGLES_VARIABLE_NAMES`` (3)
     - ``tcr``, ``ssdo``, ``stssd``
   * - ``LO_IALIRT_VARIABLE_NAMES`` (9)
     - ``heplusplus``, ``cplus5``, ``cplus6``, ``oplus6``, ``oplus7``,
       ``oplus8``, ``mg``, ``fe_hiq``, ``fe_loq``
   * - ``HI_IALIRT_VARIABLE_NAMES``
     - ``h``

**[DOC]** The document's non-sunward Lo species list is H+, He++, O5-8, C4-6,
Ne+Mg+Si, Fe, He+ and CNO+ (8 species). There is no corresponding constant in
the code, consistent with the NSW products not being implemented (and not
currently produced by the team, per section 9.2).

**[DOC]** Table 2 of the document lists the Lo angular species as He++, O+6,
C+5, Fe+10 and PUI He+. The code's ``LO_SW_ANGULAR_VARIABLE_NAMES`` is
``hplus``, ``heplusplus``, ``oplus6``, ``fe_loq``, ``heplus``. Resolve this
against the SCI-LUT, not either list, if the angular products are ever built.
