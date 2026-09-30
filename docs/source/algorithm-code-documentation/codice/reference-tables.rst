.. _codice-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

The algorithm document's large tables are **deliberately not reproduced** here:
they go stale, and in almost every case a machine-readable version already
exists that the code actually reads.

The document itself is **not in this repository** - see
:ref:`codice-source-documents`. CoDICE is unusual in that the single most
important table set - the SCI-LUT - is neither in the document nor in this
repository: it is an operational ancillary file that changes in flight.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - If you need...
     - Go to
   * - A packet field's name, bit offset, width or type
     - ``imap_processing/codice/packet_definitions/imap_codice_packet-definition_20250101_v001.xml``
       (pre-2026-01-29) or ``..._20260129_v001.xml`` (post)
   * - A housekeeping field or its engineering-unit calibrator
     - ``imap_processing/codice/packet_definitions/P_COD_NHK.xml``
   * - The plan / view / collapse / ESA sweep / Lo stepping tables
     - The ``l1a-sci-lut`` JSON ancillary file. A test copy is at
       ``imap_processing/tests/codice/data/l1a_lut/imap_codice_l1a-sci-lut_20251007_v005.json``
       (and ``..._20260129_v002.json``). Structure documented in
       :ref:`codice-ancillary`.
   * - The lossy A/B decompression tables
     - ``LOSSY_A_TABLE`` / ``LOSSY_B_TABLE`` in
       ``imap_processing/codice/constants.py`` (256 entries each)
   * - APIDs
     - ``CODICEAPID`` in ``imap_processing/codice/constants.py``
   * - Species lists, position groupings, angle tables, acquisition time
     - ``imap_processing/codice/constants.py`` - see the inventory on
       :ref:`codice-data-products`
   * - Direct-event bit layouts
     - ``DE_DATA_PRODUCT_CONFIGURATIONS`` and ``DE_METADATA_FIELDS`` in
       ``constants.py``
   * - The I-ALiRT packet layout
     - ``IAL_BIT_STRUCTURE`` in ``constants.py``, plus
       ``imap_processing/ialirt/utils/constants.py``
   * - A CDF variable's units, fill value, valid range or description
     - ``imap_processing/cdf/config/imap_codice_l1a_variable_attrs.yaml``,
       ``imap_codice_l1b_variable_attrs.yaml`` and the six
       ``imap_codice_l2-*_variable_attrs.yaml`` files
   * - The complete list of CoDICE products
     - ``imap_processing/cdf/config/imap_codice_global_cdf_attrs.yaml``
   * - Geometric factors or efficiencies
     - The ``l2-lo-gfactor`` / ``l2-*-efficiency`` ancillary CSVs; test copies
       in ``imap_processing/tests/codice/data/l2_lut/``
   * - The spin-angle reference values (now equal to Rev 3 Chg 1; 90 deg below
       older drafts)
     - :ref:`codice-spin-angle-offset` and
       ``imap_processing/tests/codice/test_codice_spin_angles.py``
   * - The commissioning timeline / RGFO / NSO rules
     - :ref:`codice-timeline` and :ref:`codice-modes` - transcribed
   * - The instrument team's known data caveats
     - :ref:`codice-data-caveats` - transcribed
   * - The acquisition-time equations (Appendix C)
     - :ref:`codice-esa-stepping` - transcribed
   * - Anything else
     - the algorithm document, using the section index below

Machine-readable tables in the repository
------------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

Two XTCE files, selected by date in ``codice_l1a.process_l1a``:

.. code-block:: text

   imap_codice_packet-definition_20250101_v001.xml   ~335 KB, pre-2026-01-29 FSW
   imap_codice_packet-definition_20260129_v001.xml   ~372 KB, post-2026-01-29 FSW
   P_COD_NHK.xml                                     ~37 KB, housekeeping only

The 2026-01-29 file adds ``RGFO_SPIN_SECTOR``, ``RGFO_ENERGY_STEP``,
``NSO_SPIN_SECTOR`` and ``NSO_ENERGY_STEP`` to the COUNTS and PHA packets.

.. code-block:: bash

   # every parameter defined for one APID
   grep -o 'name="COD_LO_SW_SPECIES_COUNTS\.[^"]*"' \
     imap_processing/codice/packet_definitions/imap_codice_packet-definition_20260129_v001.xml

   # diff the two FSW versions
   diff <(grep -o 'name="[^"]*"' .../20250101_v001.xml | sort -u) \
        <(grep -o 'name="[^"]*"' .../20260129_v001.xml | sort -u)

CDF metadata
^^^^^^^^^^^^

``imap_processing/cdf/config/``:

* ``imap_codice_global_cdf_attrs.yaml`` - **the definitive product list.** Every
  ``Logical_source`` CoDICE can emit has an entry here; if a string is not in
  this file, ``get_global_attributes`` raises. 41 entries: 18 L1A, 15 L1B,
  8 L2.
* ``imap_codice_l1a_variable_attrs.yaml`` - L1A counts and metadata variables,
  including the ``lo-species-attrs`` / ``lo-species-unc-attrs`` templates with
  ``{species}`` and ``{direction}`` placeholders.
* ``imap_codice_l1b_variable_attrs.yaml`` - rates, ``energy_per_charge``.
* ``imap_codice_l2-lo-species_variable_attrs.yaml`` - the ``lo-sw-species-attrs``,
  ``lo-pui-species-attrs`` and matching ``-unc-attrs`` templates.
* ``imap_codice_l2-lo-angular_variable_attrs.yaml`` - **written, but the product
  is not.**
* ``imap_codice_l2-lo-direct-events_variable_attrs.yaml``
* ``imap_codice_l2-hi-omni_variable_attrs.yaml``
* ``imap_codice_l2-hi-sectored_variable_attrs.yaml``
* ``imap_codice_l2-hi-direct-events_variable_attrs.yaml``

.. code-block:: bash

   grep 'Logical_source:' imap_processing/cdf/config/imap_codice_global_cdf_attrs.yaml

Placeholder substitution is done by ``utils.apply_replacements_to_attrs``, which
uses plain ``str.replace`` rather than ``str.format`` so that braces elsewhere in
a template do not raise.

Validation data
^^^^^^^^^^^^^^^

``imap_processing/tests/codice/data/``:

.. code-block:: text

   l0_data/           imap_codice_l0_raw_20241110_v001.pkts
   l1a_input/         per-product .pkts slices, dated 20250814
                      + imap_codice_l0_raw_20260130_v001.pkts (FSW change)
   l1a_lut/           SCI-LUT JSON, v005 (20251007) and v002 (20260129)
   l1b_validation/    per-product L1B CDFs, 20250814 v015
   l2_lut/            the seven L2 ancillary CSVs

Pinned in ``conftest.py`` as ``VALIDATION_FILE_DATE = "20250814"`` and
``VALIDATION_FILE_VERSION = "v015"``. The ``codice_lut_path`` fixture maps
``(descriptor, data_type)`` to a path and **raises on anything unknown** - the
quickest way to enumerate what a code path needs.

Algorithm document section index
--------------------------------

Section numbers are those of **Rev 3 Chg 1**, as embedded in the CMAD
(``IMAP_CMAD_20260722.pdf`` section 4.3.2, PDF pages 623-709). To find a
printed body page in the CMAD, add 628 to it (printed page 1 = CMAD page 629).
The front matter is on CMAD pages 625-628, and Appendices A, B and C start on
CMAD pages 704, 705 and 709. Printed page numbers moved by one or two pages
relative to the January 2026 draft, and sections 4-5 were re-numbered as shown.

.. list-table::
   :header-rows: 1
   :widths: 12 52 36

   * - Section
     - Title
     - Covered on
   * - 1-3
     - Introduction, product overview, heritage instruments
     - :ref:`codice-overview`
   * - 4.1
     - Physical description
     - :ref:`codice-overview`
   * - 4.2
     - Angular mappings in sensor coordinates (look-direction unit vectors,
       azimuth tables, spin-phase / +X\ :sub:`Co` reference, spin-angle tables,
       46 deg SC frame offset)
     - :ref:`codice-frames`
   * - 5.1
     - Algorithm description: the four packet IDs and the SCI-LUT (was 4.3
       "Algorithm Input" in the draft)
     - :ref:`codice-l1a`, :ref:`codice-ancillary`
   * - 5.2
     - Processing pipeline (was 5.1)
     - :ref:`codice-data-products`
   * - 5.3
     - Data validation: pre-launch synthetic-data validation, independent
       re-implementation by the SDC, post-launch re-validation on flight data.
     - -
   * - 6
     - Data products overview (Tables 1-4)
     - :ref:`codice-data-products`
   * - 7
     - Common processing algorithms: CCSDS header, packet structures,
       compression, the unpacking algorithm, acquisition timing
     - :ref:`codice-l1a`
   * - 8
     - Worked unpacking example
     - :ref:`codice-l1a`
   * - 9.1
     - Instrument operation timeline and changes (P0-P4)
     - :ref:`codice-timeline`
   * - 9.2
     - Data caveats for the Summer 2026 release; products not being produced
     - :ref:`codice-data-caveats`
   * - 10.1
     - L1A housekeeping
     - :ref:`codice-l1b`
   * - 10.2
     - L1A CoDICE-Hi (counters, direct events, omni, sectored, priority)
     - :ref:`codice-l1a`
   * - 10.3
     - L1A CoDICE-Lo (counters, direct events, species, angular, priority),
       ESA stepping tables, de-spinning
     - :ref:`codice-l1a`, :ref:`codice-esa-stepping`
   * - 10.4
     - L1A I-ALiRT packet formats
     - :ref:`codice-ialirt`
   * - 11
     - L1B rates for both sensors
     - :ref:`codice-l1b`
   * - 12.1
     - L2 Hi: direct events, sectored intensities, omni intensities
     - :ref:`codice-l2`
   * - 12.2
     - L2 Lo: direct events, angular intensities, species intensities, RGFO
       rules
     - :ref:`codice-l2`
   * - 13
     - L3 (all)
     - :ref:`codice-l3-scope` - **out of scope**
   * - 14
     - I-ALiRT rates, intensities, pseudo-densities, ratios
     - :ref:`codice-ialirt`
   * - 15
     - Operation modes
     - :ref:`codice-overview`
   * - App. A
     - Acronyms
     - -
   * - App. B
     - Hi true omni-directional intensity (solid-angle correction)
     - :ref:`codice-implementation-status` - **not implemented**
   * - App. C
     - Spin-sector acquisition times. **Truncated in the CMAD** after the
       common-values table; the Hi/Lo equations are only in an older draft of the algorithm document.
     - :ref:`codice-esa-stepping` - transcribed

Things you genuinely need the document for
-------------------------------------------

Everything else on this page has a machine-readable equivalent. These do not:

* **The look-direction unit vectors** (section 4.2) for all 12 SSDs and 24 APDs
  in the instrument frame. Not in the code - the document notes they can be
  obtained from SPICE with the instrument kernel instead.
* **The full Hi spin-angle table** for all 12 SSDs x 12 spin sectors (section
  4.2). The code stores only the sector-0 reference column and increments by
  30 deg.
* **The 30 Lo aggregated rate types and 15 Hi rate types** (sections 10.3.1 and
  10.2.1). Only the nominal subsets are named in ``constants.py``.
* **The PHA event-type field tables** (section 10.2.2) showing which fields are
  populated for TCR vs DCR vs SSD events at what resolution.
* **Appendix B** in full - the solid-angle derivation and the
  :math:`\Omega_k` / :math:`\Omega_{12,k}` table.
* **Section 13** in full, if you need to understand what the L3 repository
  expects from our L2 products.
