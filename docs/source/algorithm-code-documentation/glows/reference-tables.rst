.. _glows-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

The algorithm document's large tables are **deliberately not reproduced** here:
they go stale, and in almost every case a machine-readable version already
exists in the repository that the code actually reads.

The document itself is **not in this repository** - see
:ref:`glows-source-documents`.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - If you need...
     - Go to
   * - A packet field's name, bit offset, width or type
     - ``imap_processing/glows/packet_definitions/P_GLX_TMSCHIST.xml`` (APID
       1480) or ``P_GLX_TMSCDE.xml`` (APID 1481). ``GLX_COMBINED.xml`` is the
       master document ``decom_packets`` loads.
   * - The list of bad-time flags, in order
     - ``BAD_TIME_FLAG_NAMES`` in ``imap_processing/glows/__init__.py``. This is
       authoritative; the L2 ``flags_label`` variable is written from it.
   * - The bad-angle flag bit values
     - ``GLOWSL1bFlags`` in ``imap_processing/quality_flags.py``.
   * - Fill values, the scan-circle radius, the standard bin count, the
       subsecond limit
     - ``GlowsConstants`` in ``imap_processing/glows/utils/constants.py``.
   * - The integer-to-physical conversion ranges
     - The ``l1b-conversion-table-for-anc-data`` file. Bundled example:
       ``imap_processing/glows/ancillary/l1b_conversion_table_v001.json``.
   * - Which flags are active, and every threshold
     - The ``pipeline-settings`` file. Bundled example:
       ``imap_processing/glows/ancillary/imap_glows_pipeline-settings_20250923_v002.json``;
       test copy in ``imap_processing/tests/glows/validation_data/``.
   * - The star catalogue or the excluded-region point set
     - ``imap_processing/glows/ancillary/imap_glows_map-of-uv-sources_*.dat``
       and ``imap_glows_map-of-excluded-regions_*.dat``.
   * - A CDF variable's units, fill value, valid range or description
     - ``imap_processing/cdf/config/imap_glows_l{1a,1b,2}_variable_attrs.yaml``.
   * - The complete list of GLOWS products
     - ``imap_processing/cdf/config/imap_glows_global_cdf_attrs.yaml``.
   * - The direct-event compression scheme
     - :ref:`glows-l1a` - fully transcribed.
   * - The encoding/decoding equations
     - :ref:`glows-l1b` - fully transcribed.
   * - The L2 co-adding, exposure, flux and uncertainty equations
     - :ref:`glows-l2` - fully transcribed.
   * - The spin angle ↔ position angle conversion
     - :ref:`glows-overview` - fully transcribed.
   * - Expected numeric values for a level
     - The GLOWS team's JSON in
       ``imap_processing/tests/glows/validation_data/``.
   * - Anything else
     - the algorithm document, using the section index below.

Machine-readable tables in the repository
------------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   grep -o 'name="[^"]*"' imap_processing/glows/packet_definitions/P_GLX_TMSCHIST.xml | sort -u
   grep -o 'name="[^"]*"' imap_processing/glows/packet_definitions/P_GLX_TMSCDE.xml | sort -u

Both APIDs are reachable from ``GLX_COMBINED.xml``, which is the only file
``decom_packets`` references. The variable-length payloads
(``HISTOGRAM_DATA``, ``DE_DATA``) are declared as dynamically sized byte
sequences (``BYTEHIST`` / ``BYTEDE``).

CDF metadata
^^^^^^^^^^^^

``imap_processing/cdf/config/``:

* ``imap_glows_global_cdf_attrs.yaml`` - **the definitive product list.** The
  five ``Logical_source`` values are ``imap_glows_l1a_hist``,
  ``imap_glows_l1a_de``, ``imap_glows_l1b_hist``, ``imap_glows_l1b_de`` and
  ``imap_glows_l2_hist``. If a string is not in this file,
  ``get_global_attributes`` will raise.
* ``imap_glows_l1a_variable_attrs.yaml``
* ``imap_glows_l1b_variable_attrs.yaml``
* ``imap_glows_l2_variable_attrs.yaml``

Validation data
^^^^^^^^^^^^^^^

``imap_processing/tests/glows/validation_data/``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Contents
   * - ``glows_test_packet_20110921_v01.pkts``
     - 505 histogram + 1088 direct-event CCSDS packets.
   * - ``glows_l1a_hist_validation.json``
     - Expected L1A histogram values from the GLOWS team's bundle.
   * - ``imap_glows_l1b_hist_full_output.json``
     - Expected L1B histogram values.
   * - ``imap_glows_l1b_de_output.json``
     - Expected L1B direct-event values.
   * - ``imap_glows_pipeline-settings_20251112_v001.json``
     - Pipeline settings used by the tests. Differs materially from the bundled
       production example - see :ref:`glows-ancillary`.

**[DOC §3.11]** The team's Python bundle emits JSON for every level from L1A to
L3A, and the SDC's CDF output is compared to it "number vs number". The document
is explicit that *"looking into the source code in the scripts (and/or asking
the GLOWS team) is presumably the best way of understanding the intentions of
the instrument team, when something in this document needs clarification."*
Treat the bundle as the tie-breaker, not the prose.

Algorithm document section index
---------------------------------

Revision 4.4.7, 25 July 2025, 123 pages. Page numbers are the document's own.

.. list-table::
   :header-rows: 1
   :widths: 10 12 78

   * - Section
     - Pages
     - Contents
   * - Table 0.1
     - 2
     - **The parameter table.** Every nominal value and its bounds. Start here
       for any "what is the nominal X" question.
   * - 1
     - 5-6
     - Introduction: what the helioglow is and why it maps the solar wind.
   * - 2
     - 6-9
     - Onboard processing, science-team view. Table 2.1 operation modes,
       Table 2.2 data types.
   * - 3
     - 9-32
     - **The primary specification for L0-L2.** Supersedes §12 where they
       conflict.
   * - 3.2
     - 10-11
     - Inputs: telemetry, SDC-provided ancillary, GLOWS-team-provided files.
   * - 3.3.3
     - 13-14
     - "Per pointing" organisation; repointing and ΔV handling.
   * - 3.4
     - 14-19
     - **L0 packet layouts.** Table 3.1 CCSDS header, §3.4.1 histogram fields,
       §3.4.2 DE fields with Tables 3.2 (data_every_second), 3.3 (full
       timestamp) and 3.4 (time offset).
   * - 3.5
     - 19-20
     - L0→L1A processing, including the DE parsing loop.
   * - 3.6
     - 20-21
     - **L1A data products.** Tables 3.5 (histogram), 3.6 (DE), 3.7 (single DE).
   * - 3.7
     - 21-23
     - L1A→L1B processing; bad-angle masking algorithm.
   * - 3.8
     - 23
     - L1B data products. Tables 3.8 (histogram), 3.9 (header),
       **3.10 (17 bad-time flags)**, **3.11 (4 bad-angle flags)**, 3.12 (DE).
   * - 3.9
     - 23-27
     - **L1B→L2 processing.** The nine steps, day/night offsets, bad-angle
       masking for L2.
   * - 3.10
     - 28-29
     - **L2 data products.** Tables 3.13 (product), 3.14 (header),
       3.15 (daily_lightcurve).
   * - 3.11
     - 30
     - The Python-script bundle.
   * - 3.12
     - 30-31
     - **The nine ancillary files.** Names and contents.
   * - 3.13
     - 31
     - Validation data sets; §3.13.1 describes Validation Set 1 in detail.
   * - 3.14
     - 32
     - Comments: quick-look nature of L3, ψ vs ψ\ :sub:`PA`, HK needs, the
       SWE/HIT background question.
   * - 4
     - 32-49
     - **L3 pipeline.** Not this repository. §4.1 lists internal (SWE, SWAPI,
       averaged spin axis) and external (F10.7, composite Lyman-α, OMNI2)
       dependencies; §4.15 lists the L3 ancillary files.
   * - 5
     - 49
     - **Sources of the observed signal** - the eight contributions.
   * - 6
     - 50
     - **What the data system must guarantee** - the four objectives.
   * - 7
     - 50-51
     - Characteristics of the observed signal; count-rate expectations; star
       visibility windows; Eqs. 1-2 for day length and spin period; onboard time
       keeping and attitude accuracy.
   * - 8
     - 51-54
     - **Operation modes.** §8.1 and Figure 8.1 are the definitive description
       of the Evening/Sunset/Night/Sunrise sequence. §8.2 in-flight tests.
       §8.3 ground histogram generation from DEs.
   * - 9
     - 54-61
     - Direct events in depth: collection, onboard file format, downlink
       selection (min/max blocks, ±k\ :sub:`DE` neighbours, bisection ordering),
       data volume.
   * - 10
     - 61-74
     - **Histograms in depth.** §10.2 block length from background-detection
       capability (Eqs. 13-20). §10.3 bin width from star calibration
       (Eqs. 21-23). §10.4 the two onboard histogramming algorithms and the χ²
       testing. §10.5 bit rate. **§10.6 the spin-angle offset, Eqs. 29-30 -
       essential reading.**
   * - 11
     - 74-77
     - Ancillary onboard data. §11.2 average/variance definitions (Eqs. 31-33).
       §11.3 event signals. **§11.4 encoding/decoding, Eqs. 35-44.**
       **Table 11.1 conversion coefficients** (superseded in practice by the
       delivered JSON).
   * - 12
     - 77-91
     - The older, more discursive L0-L2 treatment. §12.3 the flag taxonomy.
       §12.7.1 initial culling and **Eqs. 45-46, the count-based background
       rejection**. §12.7.3 co-adding, Eqs. 47-49. §12.7.4 count rate, Eq. 50.
       **§12.7.5 the calibration factor, Eqs. 51-53.** §12.7.6 bad-angle masking
       rationale. §12.7.7 uncertainties, Eqs. 54-55.
   * - 13
     - 91-102
     - L3 in depth, including the survival-probability programs. Not this
       repository.
   * - 14
     - 102-104
     - **Coordinate frames.** §14.1.1 the science (SPICE) instrument frame,
       §14.1.2 the MICD frame.
   * - 15
     - 104-105
     - Concluding remarks: five future data products the team intends to add.
   * - Change log
     - 107-123
     - Revision history. Useful for working out when a definition changed.

Equations worth knowing by number
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Eq.
     - What it is
   * - 29, 30
     - ψ\ :sub:`PA` = mod[ψ - ψ\ :sub:`G,eff`, 360°] and ψ\ :sub:`G,eff` = 360° -
       ψ\ :sub:`GLOWS` + δψ\ :sub:`G,eff`. Implemented at L2.
   * - 35-39
     - Onboard integer encoding and ground decoding of a scalar.
   * - 40
     - Decoding an **averaged** quantity - same form as Eq. 39.
   * - 43, 44
     - Decoding an encoded **variance**, then taking the square root.
   * - 45, 46
     - Upper and lower count-based rejection thresholds for particle background.
       **Not implemented.**
   * - 47
     - Daily histogram = sum of good-time block histograms.
   * - 48, 49
     - Per-block and daily exposure time.
   * - 50
     - Daily count rate S\ :sub:`m` = H\ :sub:`m` / Δ\ :sub:`m`.
   * - 51, 52
     - The calibration factor α\ :sub:`0` = 3.37 cps/R and its parameterised
       form α(HV, THRS, COMP, t).
   * - 53
     - Intensity in Rayleighs. **Printed as a multiplication; the code divides,
       which is dimensionally correct.**
   * - 54, 55
     - Poisson uncertainty on the count rate and on the intensity.

Glossary
--------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Term
     - Meaning
   * - **Block**
     - ``n_block`` = 8 consecutive spins, ~120 s. The unit of bad-time culling.
       One histogram CCSDS packet.
   * - **Bin**
     - One of ``n_bin`` = 3600 spin-angle slots, 0.1° wide. The unit of
       bad-angle culling.
   * - **Observational day / pointing**
     - Interval between IMAP repointing maneuvers. The unit of L2 accumulation.
   * - **DE**
     - Direct event: a single photon detection - GLOWS-clock timestamp plus
       impulse length.
   * - **HB**
     - Histogram block - the document's abbreviation for the histogram data
       structure.
   * - **CEM**
     - Channeltron electron multiplier - the detector.
   * - **Helioglow**
     - The heliospheric backscatter glow of ISN H in Lyman-α. The science
       signal.
   * - **ISN H**
     - Interstellar neutral hydrogen.
   * - **Rayleigh (R)**
     - Surface-brightness unit. 1 R corresponds to a radiance of
       ``10⁶/(4π)`` photons s⁻¹ cm⁻² sr⁻¹.
   * - **cps/R**
     - Counts per second per Rayleigh - the calibration factor's unit.
       ``α₀ = 3.37``.
   * - **ψ (spin angle)**
     - IMAP spin phase as defined in the GI ICD. Used at L0/L1A/L1B.
   * - **ψ**\ :sub:`PA` **(position angle)**
     - Angle from the northernmost point of the GLOWS scanning circle. Used at
       L2 and above.
   * - **Bad time**
     - Whole block unusable. 17 flags. Blocks are dropped.
   * - **Bad angle**
     - Individual bins unusable. 4 flags. Bins are masked, not dropped.
   * - **PPS**
     - Pulse per second - the spacecraft timing signal that lets GLOWS relate
       its free-running clock to the IMAP clock.
   * - **BT / PT**
     - The document's "Bad Times" and "Prohibited Times" data structures
       (§12.7.1). Conceptual only; the implementation uses the flag array.
   * - **WawHelioGlow / WawHelioIon**
     - The GLOWS team's forward models, used to generate synthetic validation
       data and, at L3, to invert lightcurves into ionization rates.
