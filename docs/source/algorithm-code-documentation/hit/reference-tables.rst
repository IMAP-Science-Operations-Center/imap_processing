.. _hit-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

Rule of thumb
-------------

Big tables are **not** reproduced in these pages. They go stale, and most of
them already exist in a machine-readable form that the code actually reads.
This page tells you which file, and which page of the algorithm document if you
have a copy.

HIT is an extreme case: **roughly 90 of the algorithm document's 169 pages are
lookup tables.** Tables 17-27 alone (the frame byte map) run to 25 pages of
"frame byte N to frame byte N+1, H (4.5-5.0 MeV/nuc), Particle ID 7". Tables
32-37 (geometry factors) run to another 38. Loading any of that into an agent's
context is almost always waste - you need three numbers, not fifteen hundred
rows.

Authority order, highest first:

#. **The code and its ancillary files.** What runs.
#. **The XTCE** (``hit_packet_definitions.xml``). What is parsed, and where the
   housekeeping conversions live.
#. **The algorithm document.** What the instrument team intends - and note it
   is still marked **Draft**, with a revision as recent as June 2026 that
   changed a packet table "to reflect what is actually in the packets".

.. note::

   If a table you need is genuinely required for development, add it as a new
   RST file under this directory rather than inlining it into one of the
   narrative pages. That keeps it out of the default context.

Machine-readable tables in the repository
-----------------------------------------

The frame byte map
^^^^^^^^^^^^^^^^^^

**This is the important one.** Algorithm document Tables 11-27 - 25 pages of
byte assignments - exist in full as
``COUNTS_DATA_STRUCTURE`` in ``imap_processing/hit/l0/constants.py``. It has
been verified byte-for-byte against the document. The per-field byte ranges are
summarised in :ref:`hit-l1a`; the ordered dict itself is the authority.

What it does **not** contain is the *meaning* of each index within an array -
which Particle ID is which species and energy. For that you still need the
document, or the two mapping dicts below.

Particle ID to species/energy mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - File
     - Contains
   * - ``imap_processing/hit/l2/constants.py``
       ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
     - 17 species, **204 energy bins**, each with the R2/R3/R4 index lists.
       This is the machine-readable form of document Table 31 **and** of the
       relevant subset of Tables 17, 19 and 21.
   * - ``imap_processing/hit/l1b/constants.py``
       ``SUMMED_PARTICLE_ENERGY_RANGE_MAPPING``
     - 17 species, **67 energy bins**, with R2/R3/R4 index lists. The
       machine-readable form of document Table 28.
   * - ``imap_processing/hit/l0/constants.py`` ``MOD_10_MAPPING``
     - The 10 sectored species/energy combinations. Document Table 3.

Ancillary CSVs
^^^^^^^^^^^^^^

``imap_processing/tests/hit/test_data/ancillary/`` holds the twelve
``imap_hit_{standard,summed,sectored}-dt{0,1,2,3}-factors_*.csv`` files -
the machine-readable form of document Tables 32-37. Format and quirks are in
:ref:`hit-ancillary`.

Housekeeping conversions
^^^^^^^^^^^^^^^^^^^^^^^^

``imap_processing/hit/packet_definitions/hit_packet_definitions.xml``. Document
Table 29 (voltage conversions) is 156 ``PolynomialCalibrator`` elements;
document Table 30 (191 thermistor rows, -40 to +150 degrees C) is eight
``ContextCalibrator`` chains of 20-22 ranges each. **Edit the XTCE, not
Python.**

Packet definitions
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - File
     - Contains
   * - ``imap_processing/hit/packet_definitions/hit_packet_definitions.xml``
     - ``HIT_HSKP`` (APID 1251) and ``HIT_SCIENCE`` (APID 1252). 133
       parameters. Note that the science packet's payload is a single opaque
       ``science_data`` field - the internal structure is in
       ``COUNTS_DATA_STRUCTURE``, not the XTCE.
   * - ``imap_processing/ialirt/packet_definitions/ialirt_hit.xml``
     - The HIT I-ALiRT fields: ``HIT_MET``, ``HIT_SC_TICK``, ``HIT_STATUS``,
       ``HIT_SUBCOM``, ``HIT_FAST_RATE_1``, ``HIT_FAST_RATE_2``,
       ``HIT_SLOW_RATE``, ``HIT_EVENT_DATA_00`` .. ``_10``, ``HIT_SPARE``.

APIDs
-----

**[DOC]** Table 9. Only two of the six are parsed by the XTCE in this
repository.

.. list-table::
   :header-rows: 1
   :widths: 10 10 26 14 14 26

   * - Dec
     - Hex
     - Description
     - Bytes
     - Cadence
     - Status here
   * - 1250
     - 0x4e2
     - Autonomy / Aliveness
     - 2
     - 1/sec
     - Not parsed.
   * - 1251
     - 0x4e3
     - Housekeeping
     - 140
     - 1/min
     - ``HitAPID.HIT_HSKP``. Feeds ``imap_hit_l1a_hk`` and
       ``imap_hit_l1b_hk``.
   * - 1252
     - 0x4e4
     - Science
     - 262
     - 20/min
     - ``HitAPID.HIT_SCIENCE``. Feeds everything else.
   * - 1253
     - 0x4e5
     - I-ALiRT
     - 54
     - 1/sec
     - ``HitAPID.HIT_IALRT``. Defined in the enum but handled entirely by the
       I-ALiRT machinery, not by ``hit_utils``.
   * - 1254
     - 0x4e6
     - Message Log
     - 0-360
     - as needed
     - Not parsed.
   * - 1255
     - 0x4e7
     - Memory Dump
     - 360
     - on command
     - Not parsed.

**[DOC]** The science APID also carries a **packet number 0-19 in bits 0-4 of
the CCSDS subseconds field** (Table 10), overwriting those bits. The code does
not use it - it identifies frames from the grouping flags and sequence counters
instead.

Index to the algorithm document
-------------------------------

Page numbers are **PDF page numbers** in
``HIT_Algorithm_Document_v1p11p00_06_02_2026.pdf`` (169 pages). The printed page
number in the footer is one lower.

Narrative sections - worth reading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 46 12 30

   * - Section
     - Title
     - Pages
     - Summarised in
   * - 2
     - HIT Instrument Description
     - 10-16
     - :ref:`hit-overview`
   * - 3
     - Operating Modes (incl. dynamic thresholds)
     - 16-18
     - :ref:`hit-overview`
   * - 4.1
     - Data Product Level Definitions, data flow
     - 19-21
     - :ref:`hit-data-products`
   * - 4.2
     - Science Data (rates, sectors, event buffer, PHA word)
     - 21-36
     - :ref:`hit-overview`, :ref:`hit-l1a`, :ref:`hit-l3-scope`
   * - 4.3
     - Uncertainties
     - 36-37
     - :ref:`hit-overview`
   * - 4.4
     - I-ALiRT Data
     - 37-38
     - :ref:`hit-ialirt`
   * - 5.1
     - Rates Compression/Decompression Algorithm
     - 38-40
     - :ref:`hit-l1a`
   * - 5.3
     - HIT Science Frame Header
     - 40-42
     - :ref:`hit-overview`
   * - 5.5
     - Lev1A Uncertainty
     - 78
     - :ref:`hit-l1a`
   * - 6.1-6.2
     - Energy Bins, Livetime Corrected Rates
     - 79-81
     - :ref:`hit-l1b`
   * - 6.3
     - Summed Rates (prose)
     - 81
     - :ref:`hit-l1b`
   * - 7.1
     - Conversion to Intensity (prose)
     - 98-99
     - :ref:`hit-l2`
   * - 7.2-7.3
     - Sectored and Summed intensity (prose)
     - 138, 142
     - :ref:`hit-l2`
   * - 8
     - I-ALiRT Algorithms
     - 145-150
     - :ref:`hit-ialirt`
   * - 9
     - Lev3 Algorithms
     - 150-169
     - :ref:`hit-l3-scope`

Tables - look up, do not read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 50 14 26

   * - Table
     - Contents
     - Pages
     - Machine-readable at
   * - 2
     - Sector directions (L1 x L2 segment to declination bin, 0-7)
     - 24
     - -- (onboard only)
   * - 3
     - Species/energy bins for sector rates
     - 25
     - ``MOD_10_MAPPING``
   * - 4
     - Event Record Header bit layout
     - 27
     - -- **needed for** :ref:`hit-gap-events`
   * - 5, 6
     - ADC field bit allocations; event record bit padding
     - 29
     - -- **needed for** :ref:`hit-gap-events`
   * - 7
     - Detector group flags (Extended Header Block)
     - 31
     - -- **needed for** :ref:`hit-gap-events`
   * - 8
     - Detector name to address, 0-63
     - 32-34
     - -- **needed for** :ref:`hit-gap-events`
   * - 9, 10
     - APIDs; science packet numbers
     - 41
     - ``HitAPID``
   * - 11
     - MISCBITS (frame header)
     - 43
     - ``COUNTS_DATA_STRUCTURE``
   * - 12
     - ERATES (livetime and trigger counters)
     - 43
     - ``COUNTS_DATA_STRUCTURE``
   * - 13
     - SNGRATES (116 singles rates by detector address)
     - 44-48
     - ``COUNTS_DATA_STRUCTURE``
   * - 14
     - EVPRATES (event processing counters)
     - 49
     - ``COUNTS_DATA_STRUCTURE``
   * - 15
     - COINRATES (26 coincidence rates)
     - 50
     - ``COUNTS_DATA_STRUCTURE``
   * - 16
     - PBUFRATES (32 priority buffers, with descriptions)
     - 51-52
     - ``COUNTS_DATA_STRUCTURE``
   * - **17**
     - **L2FGRATES - 132 Range 2 foreground rates with Particle IDs**
     - **53-57**
     - ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
   * - 18
     - L2BGRATES - 12 Range 2 background rates
     - 57-58
     - --
   * - **19**
     - **L3FGRATES - 167 Range 3 foreground rates with Particle IDs**
     - **58-65**
     - ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
   * - 20
     - L3BGRATES - 12 Range 3 background rates
     - 65
     - --
   * - **21**
     - **PENFGRATES - 33 Range 4 foreground rates with Particle IDs**
     - **66-67**
     - ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
   * - 22
     - PENBGRATES - 15 Range 4 background rates
     - 67-68
     - --
   * - 23
     - IALIRTRATES - the 20 I-ALiRT rates in the science frame
     - 69
     - :ref:`hit-ialirt`
   * - 24
     - SECTORRATES - 120 look direction byte assignments
     - 70-74
     - ``COUNTS_DATA_STRUCTURE``
   * - 25, 26
     - L4FGRATES (48) and L4BGRATES (24) - the I-ALiRT-aperture ion ranges
     - 74-77
     - -- see :ref:`hit-gap-l4rates`
   * - 27
     - Event Buffer byte range
     - 77
     - ``COUNTS_DATA_STRUCTURE``
   * - **28**
     - **Full table of summed Lev1B rates - which L1A bins feed which summed
       bin**
     - **81-89**
     - ``SUMMED_PARTICLE_ENERGY_RANGE_MAPPING``
   * - 29
     - Housekeeping raw-to-EU conversion equations
     - 89-91
     - the XTCE
   * - 30
     - Thermistor conversion table (191 rows, -40 to +150 degrees C)
     - 91-98
     - the XTCE ``ContextCalibrator`` chains
   * - **31**
     - **HIT Level 2 Standard Rate products - 204 entries with contributing
       ranges**
     - **99-107**
     - ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
   * - 32-35
     - Geometry factors, bin widths, efficiencies for Standard Rates, DT0-DT3
     - 107-138
     - ``imap_hit_standard-dt<N>-factors_*.csv``
   * - 36
     - Same for Sectored Rates, all 8 declination sectors
     - 138-142
     - ``imap_hit_sectored-dt<N>-factors_*.csv``
   * - 37
     - Same for Summed Rates
     - 142-145
     - ``imap_hit_summed-dt<N>-factors_*.csv``
   * - 38
     - I-ALiRT subcommutation map (60 slots x 3 rate types)
     - 145-147
     - ``HIT_PREFIX_TO_RATE_TYPE`` - **partially stale**, see
       :ref:`hit-gap-ialirt-slots`
   * - 39
     - The 20 I-ALiRT rates, described
     - 148
     - :ref:`hit-ialirt`
   * - 40-52
     - **L3 only.** Event classification, WINCORR arrays, ADC-MeV
       coefficients, cosine corrections, energy bounds, double-power-law fit
       parameters
     - 153-167
     - -- separate repository, see :ref:`hit-l3-scope`

Appendices and figures
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Item
     - Note
   * - **Appendix C**
     - The HIT matrix maps - graphical renderings of the 128 x 400 dE vs E'
       lookup for each range, with the foreground element tracks and
       background regions drawn on. Referenced repeatedly by sections 4.2.1
       and 4.2.5. **Not present in the v1.11.00 PDF** despite being cited;
       request it separately if you need it.
   * - Figure 8
     - The data flow diagram, reproduced in :ref:`hit-data-products`.
   * - Figure 13
     - Visualisation of the energy bins for all ion species. The quickest way
       to see the shape of the product inventory.
   * - Figure 14
     - The sectored livetime timeline - the clearest statement of the
       10-minute offset.
   * - Figure 15
     - Simulated L4 energy-loss distributions with the I-ALiRT rate boxes
       drawn on. See :ref:`hit-ialirt`.
   * - Figures 17-21
     - L3 ion tracks, charge resolution, charge histograms.
   * - Figures 22-23
     - Sectored rate geometry, and the definition of pitch angle and
       gyrophase (after van den Berg et al. 2020).
