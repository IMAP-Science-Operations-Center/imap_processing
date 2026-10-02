.. _swapi-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

The algorithm document's large tables are **deliberately not reproduced** in
these pages: they are long, they go stale, and in almost every case a
machine-readable version already exists in the repository that the code
actually reads.

The document itself is **not in this repository** - see
:ref:`swapi-source-documents`. This page tells you where to look instead, and
gives a page index for the parts that only exist in the PDF.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - If you need...
     - Go to
   * - A packet field's name, bit offset, width, type or enumeration
     - ``imap_processing/swapi/packet_definitions/swapi_packet_definition.xml``
   * - An I-ALiRT packet field
     - ``imap_processing/ialirt/packet_definitions/ialirt_swapi.xml``
   * - A fixed ESA step energy, or a fine-step offset
     - the ``esa-unit-conversion`` ancillary CSV
   * - The ESA DAC-to-energy ladder
     - the ``lut-notes`` ancillary CSV
   * - A quality flag's bit position
     - ``imap_processing/quality_flags.py``, ``class SWAPIFlags``
   * - A CDF variable's units, fill value, valid range or description
     - ``imap_processing/cdf/config/imap_swapi_variable_attrs.yaml``
   * - A ``logical_source`` string or global attribute
     - ``imap_processing/cdf/config/imap_swapi_global_cdf_attrs.yaml``
   * - A pipeline constant (12 packets, 72 steps, 0.145 s)
     - ``imap_processing/swapi/constants.py``, ``swapi_l2.SWAPI_LIVETIME``
   * - An I-ALiRT physical constant
     - ``imap_processing/ialirt/constants.py``,
       ``class IalirtSwapiConstants``
   * - An equation from L1, L2 or I-ALiRT
     - :ref:`swapi-l1` / :ref:`swapi-l2` / :ref:`swapi-ialirt` - the
       load-bearing ones are transcribed
   * - An L3 equation
     - :ref:`swapi-l3-scope` for the shape of it, then the document
   * - Anything else
     - the algorithm document, using the page index below

Machine-readable tables in the repository
-----------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

``imap_processing/swapi/packet_definitions/swapi_packet_definition.xml`` is the
authoritative field definition for the two processed APIDs. It supersedes the
document's telemetry description and, unlike it, cannot be out of date with
respect to processing - it is what the code parses.

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - Container
     - Fields
     - Notes
   * - ``SWP_HK``
     - 102
     - APID 1184. Includes ``PKT_APID``. Carries the enumerated status bits,
       the ADC monitors, and ``CHKSUM``.
   * - ``SWP_SCI``
     - 46
     - APID 1188. ``SHCOARSE``, ``SEQ_NUMBER``, ``SWEEP_TABLE``, ``PLAN_ID``,
       ``MODE``, ``ESA_LVL5``, 18 ``*_RNG_ST0..5`` bits, 18 ``*_CNT0..5``
       counts, ``CHKSUM``.

Query it directly rather than trusting a transcription:

.. code-block:: bash

   # every SWP_HK field name
   grep -o 'parameterRef="SWP_HK[^"]*"' \
     imap_processing/swapi/packet_definitions/swapi_packet_definition.xml

   # the enumeration for a status field
   grep -A 12 'name="SWP_HK.MODE"' \
     imap_processing/swapi/packet_definitions/swapi_packet_definition.xml

The I-ALiRT packet is separate:
``imap_processing/ialirt/packet_definitions/ialirt_swapi.xml``, APID **1187**,
13 fields, coincidence counts only.

APIDs
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - Telemetry
     - APID
     - In ``SWAPIAPID``?
     - In an XTCE?
   * - ``SWP_HK``
     - 1184
     - yes
     - yes (``swapi_packet_definition.xml``)
   * - ``SWP_IAL``
     - 1187
     - **no**
     - yes (``ialirt_swapi.xml``)
   * - ``SWP_SCI``
     - 1188
     - yes
     - yes (``swapi_packet_definition.xml``)
   * - ``SWP_AUT``
     - 1192
     - yes
     - **no**
   * - ``SWP_LGSCI``
     - unknown
     - no
     - no
   * - ``SWP_MG``
     - unknown
     - no
     - no
   * - ``SWP_MD``
     - unknown
     - no
     - no

Ancillary CSVs
^^^^^^^^^^^^^^

Vendored **test copies** live in ``imap_processing/tests/swapi/lut/``. These are
test fixtures, not the operational ancillary files - the real ones come from the
SDC's ancillary store by descriptor.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - File
     - Rows
     - Contents
   * - ``imap_swapi_esa-unit-conversion_20250626_v001.csv``
     - 288
     - Two stacked versions of sweep 0 plus one each of sweeps 1 and 2
       (72 steps each). Columns as in
       :ref:`swapi-esa-unit-conversion-adp`.
   * - ``imap_swapi_lut-notes_20250626_v006.csv``
     - 1024
     - The DAC-to-energy ladder. ``ESA Index Number``, ``ESA Voltage``,
       ``Energy``, ``Lower Energy``, ``Upper Energy``, ``ESA Range``,
       ``ESA DAC (Dec)``, ``ESA DAC (Hex)``.

.. code-block:: bash

   # what sweep tables and table versions does the ESA table cover?
   cut -d, -f1,6,8 imap_processing/tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv \
     | sort -u

Validation and test data
^^^^^^^^^^^^^^^^^^^^^^^^

``imap_processing/tests/swapi/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Contents
   * - ``l0_data/imap_swapi_l0_raw_20240924_v001.pkts``
     - Raw CCSDS packets, **pre-launch idle data, 2024-09-24**.
   * - ``l0_validation_data/idle_export_raw.SWP_SCI_20240924_080204.csv``
     - SWAPI-supplied decommutation truth for the first science packet.
   * - ``l0_validation_data/idle_export_raw.SWP_HK_20240924_080204.csv``
     - Same for the first housekeeping packet.

Constants
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Constant
     - Value
     - Where
   * - ``NUM_PACKETS_PER_SWEEP``
     - 12
     - ``swapi/constants.py``
   * - ``NUM_ENERGY_STEPS``
     - 72
     - ``swapi/constants.py``
   * - ``SWAPI_LIVETIME``
     - 0.145 s
     - ``swapi/l2/swapi_l2.py``
   * - ``NUM_IALIRT_ENERGY_STEPS``
     - 63
     - ``ialirt/l0/process_swapi.py``
   * - ``eff_area``
     - 1.633e-4 cm\ :sup:`2`
     - ``ialirt/constants.py``
   * - ``az_fov``
     - 30 deg
     - ``ialirt/constants.py``
   * - ``fwhm_width``
     - 0.085
     - ``ialirt/constants.py``
   * - ``temporary_density_factor``
     - :math:`e^1`
     - ``ialirt/constants.py``

.. note::

   ``swapi/constants.py`` is four lines long. Numbers that arguably belong
   there - the live time, the I-ALiRT step count, the overflow sentinel, the
   ``4.0`` MHz saturation threshold if it is ever implemented - are scattered
   across the modules that use them. If you add a tunable, consider whether it
   belongs in ``constants.py``.

Document page index
-------------------

Page numbers are the **PDF page**, which is the printed document page + 1
(the PDF has one unnumbered cover). Use these when you need something these
pages deliberately do not reproduce.

.. list-table::
   :header-rows: 1
   :widths: 12 34 12 42

   * - Section
     - Title
     - PDF pages
     - In scope here?
   * - 1
     - Scope
     - 6-7
     - background
   * - 2
     - Applicable documents
     - 8
     - --
   * - 3
     - Abbreviations
     - 9
     - --
   * - 4
     - Instrument description
     - 10-12
     - :ref:`swapi-overview`
   * - 5
     - Definitions of terms
     - 14
     - :ref:`swapi-overview`
   * - 6.1
     - Data volume, filenames
     - 15
     - :ref:`swapi-data-products`
   * - 6.2
     - Data product definitions (L0-L3)
     - 15-16
     - :ref:`swapi-data-products`
   * - 6.3
     - L0 data content and format (the 7 telemetry types)
     - 16-17
     - :ref:`swapi-overview`
   * - 6.4
     - L1/L2/L3 grouping, sweep structure, sweep plans
     - 17
     - :ref:`swapi-overview`
   * - 6.5
     - Processing flow (figure 4)
     - 18
     - :ref:`swapi-data-products`
   * - 7
     - Heritage instrument (NH SWAP)
     - 19-23
     - background; 7.1.1 is L3
   * - 8
     - External dependencies
     - 24
     - :ref:`swapi-ancillary`
   * - 9.1-9.4
     - Calibration: ETE model, SIMION, lab, CoDICE, on-orbit
     - 25-30
     - :ref:`swapi-ancillary`
   * - 9.5
     - Instrument response function, efficiency, passbands
     - 30-33
     - **L3.** :ref:`swapi-ancillary` summarizes.
   * - 10.1
     - **L0 to L1 processing**
     - 35-40
     - :ref:`swapi-l1`
   * - 10.1.3
     - L1 CDF contents (table 2) and flag array (table 3)
     - 37-39
     - :ref:`swapi-l1`
   * - 10.2
     - **L1 to L2 processing**
     - 40-41
     - :ref:`swapi-l2`
   * - 10.3
     - **ESA Unit Conversion ADP** (table 3, and the solve procedure)
     - 41-43
     - :ref:`swapi-l2`, :ref:`swapi-ancillary`
   * - 10.4.1
     - L3a solar wind protons
     - 44-56
     - **L3.** :ref:`swapi-l3-scope`
   * - 10.4.2
     - L3a solar wind alphas
     - 56-59
     - **L3.** :ref:`swapi-l3-scope`
   * - 10.4.3
     - L3a pickup helium
     - 59-69
     - **L3.** :ref:`swapi-l3-scope`
   * - 10.5
     - L3b combined differential flux
     - 69
     - **L3.** :ref:`swapi-l3-scope`
   * - 10.6
     - **Summary of SWAPI data products** (inputs and ancillaries per product)
     - 69-72
     - :ref:`swapi-data-products`, :ref:`swapi-l3-scope`
   * - 11
     - Maintenance: the gain test 1x2 LUT (table 6)
     - 73
     - :ref:`swapi-ancillary`
   * - 12
     - Recommended testing approaches
     - 74
     - :ref:`swapi-ancillary`
   * - 13
     - **Code delivery** (L3 is a SWAPI Docker container)
     - 75
     - :ref:`swapi-l3-scope`
   * - 14
     - **I-ALiRT products**
     - 76-78
     - :ref:`swapi-ialirt`
   * - 15
     - Quick look products
     - 79-80
     - not implemented
   * - 16
     - References
     - 81
     - --

Figures and tables worth knowing about
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 14 70

   * - Item
     - PDF page
     - What it shows
   * - Figure 2
     - 10
     - Instrument block diagram: three HVPS, three counters, 8051.
   * - Figure 3
     - 12
     - Electro-optics cross-section - the clearest single picture of the
       sunglasses / open-aperture / ESA / carbon foil / PCEM / SCEM path.
   * - Figure 4
     - 18
     - **The data pipeline flow diagram.** Worth having open when reading
       :ref:`swapi-data-products`.
   * - Table 1
     - 34
     - **NH/SWAP vs IMAP/SWAPI operations comparison.** The concise statement
       of the 12 s / 72 step / 0.167 s cadence and the sweep-plan regime.
   * - Table 2
     - 37-38
     - L1 data product contents.
   * - Table 3 (flags)
     - 39
     - L1 flag array contents.
   * - Table 3 (ADP)
     - 41-42
     - Start and end of the ESA Unit Conversion ADP. Note the document reuses
       the number "Table 3" for both the flag array and the ADP.
   * - Figure 12
     - 31
     - Central effective area and azimuthal transmission curves.
   * - Figure 13
     - 33
     - Interpolated energy-angle passbands with integration limits.
   * - Table 4
     - 52
     - SWAPI proton fitted parameters vs WIND/SWE - the flight-data sanity
       check for L3a.
   * - Table 5
     - 66
     - He\ :sup:`+` PUI fitting parameter bounds and initial values.
   * - Table 6
     - 73
     - The 1x2 gain test LUT format.
   * - Figure 24
     - 78
     - **The I-ALiRT validation case** (550 km/s, 5.27 cm\ :sup:`-3`,
       1e5 K in; 545.3 km/s, 4.67 cm\ :sup:`-3`, 1.18e5 K out).
   * - Figure 25
     - 80
     - The daily quick-look plot layout.
