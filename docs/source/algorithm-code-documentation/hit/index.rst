:orphan:
.. _hit-index:

HIT
===

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

.. currentmodule:: imap_processing.hit

This is the HIT (High-energy Ion Telescope) instrument module, which contains
the code for processing data from the HIT instrument.

Purpose of these pages
----------------------

These pages are a **condensed, self-contained working reference** for the HIT
processing algorithms, written so that a developer (human or AI agent) can get
productive without reading the full algorithm document.

They are a summary of the source document below plus what the code in
``imap_processing/hit`` actually does. Where the two disagree, that is called
out explicitly in :ref:`hit-implementation-status`.

.. _hit-source-documents:

Source documents
----------------

**None of these are redistributed in this repository.** This is an open-source
repository and the mission documents are not ours to publish. Request them from
the HIT instrument team at NASA Goddard Space Flight Center or the SDC document
store.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Short name
     - Reference
   * - **Algorithm document**
     - *IMAP/HIT Science Algorithms*, Version 1.11.00, dated 2 June 2026.
       Prepared by J. Grant Mitchell, Eric Christian and Alessandro Bruno,
       NASA Goddard Space Flight Center. 169 pages, marked **Draft**. The
       primary source for these pages. Referred to below as "the algorithm
       document".
   * - **HIT_L1A_L1B_Table_JGM_2_6_24**
     - Cited by section 6.2 of the algorithm document as the authoritative
       "full list of all of the data products expected to be livetime
       corrected". Not held by the SDC; the code derives the same list from
       the frame format tables instead.
   * - **HIT-FSW-DESC-004**
     - Flight software description cited by section 4.2.2 for the
       ``SECTORRATES`` subcommutation scheme (120 look directions, one
       species/energy per minute).
   * - **HIT-ELEC-HDBK-0008**
     - Electronics handbook; the source of the housekeeping voltage and
       thermistor conversions reproduced in algorithm document Tables 29-30.
       In this repository those conversions live in the XTCE, not in Python.
   * - **STEREO-CIT-CIT-002.F**
     - *SEP HIT and Central MISC Processors Flight Software Requirements*.
       Heritage document cited for Particle ID semantics.
   * - **HIT-SYS-TRD-0004**
     - FPGA trade study. Background only; nothing in the code depends on it.
   * - **Gehrels 1986**
     - DOI `10.1086/164079 <https://doi.org/10.1086/164079>`_. The source of
       the asymmetric Poisson uncertainties used at L1A.
   * - **Heritage instruments**
     - STEREO/LET is the direct ancestor (matrices, rate definitions, the
       rate compression scheme, the priority buffers). Parker Solar
       Probe/EPI-Hi contributed the PHASICs. ACE/SIS, Voyager/CRS,
       Voyager/LECP and STEREO/HET are cited for the dE/dx vs residual-E
       technique.

.. tip::

   If you hold a copy of the algorithm document, put it in ``docs/reference/``.
   That directory is gitignored, so it will never be committed, and the section
   index in :ref:`hit-reference-tables` is written against that location.

.. important::

   Two conventions used throughout these pages:

   * **[DOC]** marks a statement taken from the algorithm document. It
     describes the intended behavior, which may not be what the code does yet.
   * **[CODE]** marks a statement verified against ``imap_processing/hit``
     (or ``imap_processing/ialirt/l0/process_hit.py`` for the real-time
     product).

   When those conflict, the code is what runs and the document is what the
   instrument team expects. Both matter; do not silently "fix" one to match
   the other without asking. The document is also still marked **Draft**, and
   its most recent revision note says Table 38 was changed "to reflect what is
   actually in the packets" - so for HIT the document is a moving target.

.. warning::

   **This repository stops at L2.** HIT's L3 (pitch angle and gyrophase
   distributions from L2 sectored intensities plus MAG, ion charge and energy
   from the raw PHA events, and science-quality electron intensities) is
   produced by a separate repository run closer to the science team. Section 9
   of the algorithm document - 16 of its 169 pages, and by far the most
   algorithmically dense part - describes work that does **not** belong here.
   See :ref:`hit-l3-scope` before starting anything that looks like a charge
   calculation, a cosine correction, or a pitch angle.

   The one thing that *is* ours and looks like L3 work is **decoding the PHA
   event records at L1A**. That is currently unimplemented. See
   :ref:`hit-gap-events`.

Which page to read
------------------

Read only what you need. Each page is designed to be loaded on its own.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Read it when you need to know...
   * - :ref:`hit-overview`
     - What HIT physically is, how a measurement happens, and the vocabulary
       (aperture, range, matrix, Particle ID, FGRATES/BGRATES, dynamic
       threshold, sector, science frame). **Start here if you are new.**
   * - :ref:`hit-data-products`
     - The full product inventory, exact ``logical_source`` strings, what
       feeds what, and how the CLI is wired. **The "what goes into what"
       map.**
   * - :ref:`hit-l1a`
     - Science frame assembly from 20 packets, the 16-to-32 bit rate
       decompression, the byte-for-byte frame layout, the sectored-rate
       subcommutation, Poisson uncertainties, and the day-boundary buffer
       rules.
   * - :ref:`hit-l1b`
     - The livetime fraction piecewise fit, counts to rates, the summed-rate
       definitions, the 10-minute sectored livetime shift, and housekeeping
       engineering-unit conversion.
   * - :ref:`hit-l2`
     - The intensity equation, geometry factors and efficiencies, dynamic
       threshold state selection, range summation, and the macropixel
       10-minute regrouping.
   * - :ref:`hit-ancillary`
     - The twelve ``*-dt<N>-factors`` CSVs, their exact column format, the
       XTCE calibrators, and what SPICE is (and is not) used for.
   * - :ref:`hit-ialirt`
     - The 60-second subcommutated real-time product and the 12 space-weather
       rates derived from it.
   * - :ref:`hit-l3-scope`
     - What L3 is, why it is not in this repository, and exactly what L1A and
       L2 have to hand it. **Read before writing any charge or pitch angle
       code.**
   * - :ref:`hit-implementation-status`
     - What is implemented, what is stubbed, where the code deviates from the
       document, and what is not written at all. **Read before proposing
       work.**
   * - :ref:`hit-reference-tables`
     - Where the big tables live (PDF page ranges, ancillary CSVs, XTCE).
       Deliberately *not* reproduced inline.

.. toctree::
   :maxdepth: 1

   overview
   data-products
   l1a
   l1b
   l2
   ancillary
   ialirt
   l3-scope
   implementation-status
   reference-tables

Ten-second orientation
----------------------

* HIT is a **stack of segmented solid-state detectors (SSDs)** measuring
  ~2-40 MeV/nucleon ions from H to Ni. It is a direct descendant of
  STEREO/LET, with PHASIC front-end chips from Parker Solar Probe/EPI-Hi.
  Together with SWAPI and CoDICE it gives IMAP continuous ion coverage from
  0.1 keV to 40 MeV/nucleon.
* The measurement technique is **dE/dx vs residual E**: the energy a particle
  deposits in the detector it passes *through*, plotted against the energy it
  deposits in the detector it *stops* in, identifies the species and energy.
  Which detectors were hit also gives the arrival direction.
* **Almost all the species identification happens on board.** The flight
  software sorts each event through a *matrix* (a 128 x 400 lookup in
  dE vs E' space) and increments a counter. What comes down is therefore
  already "counts of Fe between 12 and 15 MeV/nuc that stopped in L3" - not
  raw physics. Ground processing is overwhelmingly **bookkeeping, livetime
  correction, and unit conversion**, not particle identification.
* Three **penetration ranges** define three sets of counters:
  ``R2`` (L1L2, stopped in L2), ``R3`` (L2L3, stopped in L3), ``R4``
  (L3AL3B, penetrating). Their counters are ``l2fgrates``/``l2bgrates``,
  ``l3fgrates``/``l3bgrates``, ``penfgrates``/``penbgrates``. "FG" =
  foreground (identified species and energy), "BG" = background (broad
  regions of the matrix that are not on an element track).
* The unit of telemetry is the **HIT Science Frame**: one minute of data,
  5240 bytes, spread over **20 CCSDS packets** (APID 1252, 262 bytes each).
  The first 6 packets are the fixed-format counters and rates (bytes
  0-1571); the last 14 are the variable-length **Event Buffer** of raw pulse
  heights.
* Every rate in the frame is **compressed from 24 (or 32) bits to 16** by a
  biased-exponent / hidden-one scheme inherited from STEREO/LET, and must be
  expanded on the ground.
* **Sectored rates** give anisotropy: the 8 science apertures plus the
  spacecraft spin divide the sky into **8 declination x 15 inclination = 120
  look directions**. Only one of **10 species/energy combinations** is sent
  per minute, so a complete sectored set takes **10 minutes** - and it is
  telemetered **10 minutes after it was collected**, so it must be paired
  with the *previous* block's livetime.
* Processing chain in this repository:
  ``CCSDS packets -> L1A (decompressed counts) -> L1B (livetime-corrected
  rates) -> L2 (intensities in cm^-2 s^-1 sr^-1 (MeV/nuc)^-1)``. L3 is not
  ours.
* There is also a **1-minute I-ALiRT** product built from a 60-slot
  subcommutated 1 Hz packet (APID 1253), living under
  ``imap_processing/ialirt/``.

Where the code lives
--------------------

.. code-block:: text

   imap_processing/hit/
     hit_utils.py                        HitAPID, CDF attr manager, housekeeping
                                         reshaping, energy-bin variables, and the
                                         cross-range summing helpers shared by L1B
                                         summed rates and L2 standard intensity
     l0/constants.py                     COUNTS_DATA_STRUCTURE (the frame byte map),
                                         FLAG_PATTERN, FRAME_SIZE=20, MOD_10_MAPPING,
                                         ZENITH_ANGLES, AZIMUTH_ANGLES,
                                         MANTISSA_BITS=12, EXPONENT_BITS=4
     l0/decom_hit.py                     science frame assembly, bit parsing,
                                         16->32 bit rate decompression
     l1a/hit_l1a.py                      L1A products: housekeeping, standard counts,
                                         sectored counts, direct events; Gehrels
                                         uncertainties; processing-day filtering
     l1b/constants.py                    SECTORS=15, fill values,
                                         SUMMED_PARTICLE_ENERGY_RANGE_MAPPING
     l1b/hit_l1b.py                      livetime fraction, counts->rates, summed
                                         rates, sectored rates, housekeeping (derived)
     l2/constants.py                     VALID_SPECIES, VALID_SECTORED_SPECIES,
                                         STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
                                         SECONDS_PER_MIN/SECONDS_PER_10_MIN, N_AZIMUTH
     l2/hit_l2.py                        intensity calculation, ancillary loading by
                                         dynamic threshold state, systematic and total
                                         uncertainties, macropixel 10-minute regrouping
     packet_definitions/
       hit_packet_definitions.xml        HIT_HSKP (APID 1251) + HIT_SCIENCE (APID 1252),
                                         including all housekeeping EU calibrators

   imap_processing/ialirt/
     l0/process_hit.py                   the whole HIT I-ALiRT algorithm
     packet_definitions/ialirt_hit.xml   HIT I-ALiRT packet fields

   imap_processing/cdf/config/imap_hit_global_cdf_attrs.yaml
   imap_processing/cdf/config/imap_hit_l1a_variable_attrs.yaml
   imap_processing/cdf/config/imap_hit_l1b_variable_attrs.yaml
   imap_processing/cdf/config/imap_hit_l2_variable_attrs.yaml
   imap_processing/cli.py (class Hit)    dependency wiring per level
   imap_processing/tests/hit/            tests, L0 test data, ancillary CSVs,
                                         validation CSVs
