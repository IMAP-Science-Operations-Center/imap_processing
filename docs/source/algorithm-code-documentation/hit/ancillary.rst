.. _hit-ancillary:

Ancillary Data and External Dependencies
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

HIT is unusually light on ancillary data. There is exactly **one** family of
ancillary files, and it is only used at L2.

The factor CSVs
---------------

**[CODE]** Twelve files: three product families x four dynamic threshold
states.

.. code-block:: text

   imap_hit_standard-dt0-factors_<YYYYMMDD>_v<NNN>.csv
   imap_hit_standard-dt1-factors_<YYYYMMDD>_v<NNN>.csv
   imap_hit_standard-dt2-factors_<YYYYMMDD>_v<NNN>.csv
   imap_hit_standard-dt3-factors_<YYYYMMDD>_v<NNN>.csv
   imap_hit_summed-dt0-factors_...  (and dt1, dt2, dt3)
   imap_hit_sectored-dt0-factors_... (and dt1, dt2, dt3)

Copies used by the test suite live in
``imap_processing/tests/hit/test_data/ancillary/`` at version
``20250219_v002``.

The CLI selects them by the descriptor substring ``-dt``; ``load_ancillary_data``
then picks the right one per state by the substring ``dt<N>-factors``.

Format
^^^^^^

**Standard** and **summed** share a 7-column layout:

.. code-block:: text

   Species,Lower Energy (MeV),Upper Energy (MeV),Delta E (MeV),Geometry Factor (cm2 sr),Efficiency,b
   H,1.8,2.2,0.4,3.4146,1,0
   H,2.2,2.7,0.5,3.44142,1,0
   ...

**Sectored** inserts a ``Sector`` column after the species:

.. code-block:: text

   Species, Sector, Lower Energy (MeV), Upper Energy (MeV), Delta E (MeV), Geometry Factor (cm2 sr), Efficiency, b
   H      ,0,1.8,4,2.2,0.41686,1,0
   H      ,1,1.8,4,2.2,0.49253,1,0
   ...

Row counts, which are a useful sanity check:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Family
     - Data rows
     - Structure
   * - ``standard``
     - **204**
     - one per (species, energy bin); matches
       ``STANDARD_PARTICLE_ENERGY_RANGE_MAPPING``
   * - ``summed``
     - **67**
     - one per (species, energy bin); matches
       ``SUMMED_PARTICLE_ENERGY_RANGE_MAPPING``
   * - ``sectored``
     - **80**
     - 10 species/energy combinations x 8 declination sectors

Quirks to be aware of
^^^^^^^^^^^^^^^^^^^^^

* **The files are UTF-8 with a BOM.** ``pandas.read_csv`` handles it, but a
  naive reader will see ``﻿Species`` as the first column name.
* **Whitespace is inconsistent.** The sectored file has spaces after commas in
  the header and trailing spaces in the species values. The code compensates:
  ``load_ancillary_data`` does ``df.columns.str.lower().str.strip()`` and
  lowercases the species, and ``get_species_ancillary_data`` additionally
  strips every string cell.
* **The ``Sector`` column is never read by name.** ``get_species_ancillary_data``
  groups by ``lower energy (mev)`` and relies on the **row order within the
  group** being sector 0-7. If a future delivery reorders the rows, the
  declination assignment will silently scramble. See
  :ref:`hit-gap-sector-order`.
* **``Efficiency`` is 1 everywhere and ``b`` is 0 everywhere**, exactly as the
  document says they should be until measured in flight.

Where the numbers come from
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Algorithm document Tables 32-37:

.. list-table::
   :header-rows: 1
   :widths: 14 20 22 44

   * - Table
     - Family
     - State
     - PDF pages
   * - 32
     - standard
     - DT0
     - 107-115
   * - 33
     - standard
     - DT1
     - 115-122
   * - 34
     - standard
     - DT2
     - 122-130
   * - 35
     - standard
     - DT3
     - 130-138
   * - 36
     - sectored
     - all (8 declination sectors)
     - 138-142
   * - 37
     - summed
     - all
     - 142-145

Page numbers are **PDF page numbers** in
``HIT_Algorithm_Document_v1p11p00_06_02_2026.pdf``; the printed page number in
the footer is one lower. These tables are deliberately **not** reproduced
here - the CSVs are the machine-readable form the code actually reads. See
:ref:`hit-reference-tables`.

Housekeeping conversions
------------------------

Not a separate file: the raw-to-engineering-unit conversions for housekeeping
(algorithm document Tables 29 and 30, originally from HIT-ELEC-HDBK-0008) live
inside the XTCE at
``imap_processing/hit/packet_definitions/hit_packet_definitions.xml``:

* **156 ``PolynomialCalibrator`` elements** for the linear voltage
  conversions, including the 64 ``LEAK_I_NN`` channels which all share
  ``0.00488758553274682``.
* **Eight ``ContextCalibrator`` chains** for the thermistors (Table 30 is
  191 rows, -40 to +150 degrees C) -
  ``TEMP0``-``TEMP3`` with 22 context ranges each (FEE board),
  ``ANALOG_TEMP``, ``HVPS_TEMP``, ``IDPU_TEMP``, ``LVPS_TEMP`` with 20 each
  (analog board). Together these express the piecewise Table 30 lookup.

``hit_l1b`` gets engineering units purely by re-parsing the L0 packet with
``use_derived_value=True``. **To change a housekeeping conversion, edit the
XTCE, not Python.**

SPICE
-----

**[CODE]** HIT uses SPICE for **time conversion only**:

* ``hit_l1a`` imports ``et_to_datetime64``, ``met_to_datetime64`` and
  ``ttj2000ns_to_et`` for the processing-day filter.
* ``ialirt/l0/process_hit.py`` imports ``met_to_ttj2000ns``.
* The CLI declares SPICE time kernels as one of the two L1A dependencies.

**No geometry is used.** ``imap_processing/spice/geometry.py`` does define
``SpiceFrame.IMAP_HIT = -43500``, a boresight of ``[0, 1, 0]``, a mounting
normal of ``[0, 1, 0]`` and a spacecraft-to-instrument spin phase offset of
``119.6452 / 360`` (nominally 30 + 90 = 120 degrees, from the frame kernel
``imap_130.tf``) - but nothing in HIT processing calls any of it. Pointing is
needed at L3, which is a different repository.

.. note::

   The one place SPICE geometry *would* be needed inside this repository is the
   spin-rate correction to the 15th inclination bin (see
   :ref:`hit-gap-spinrate`). That correction is required by the document and is
   not implemented.

Ancillary data needed by L3 but not by this repository
------------------------------------------------------

Listed here so that nobody adds them to ``imap_processing`` by mistake. All of
these belong to the L3 repository; see :ref:`hit-l3-scope`.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Algorithm document reference
   * - ADC-to-MeV coefficients (6 sets: L1/L2/L3 x low/high gain)
     - Table 42, section 9.1.2
   * - ``WINCORR2`` / ``WINCORR3`` Kapton foil correction arrays
     - Table 41, section 9.1.1
   * - Cosine correction :math:`K_\theta` tables (150 L1 x L2 segment
       combinations, per range)
     - Tables 43-45, section 9.1.3
   * - Double-power-law ion-track fit parameters (15 species x 5 parameters x
       3 ranges, x2 for the A0/B0 apertures)
     - Tables 47-52, section 9.1.4
   * - Charge (Z) lookup tables per range
     - section 9.1
   * - Event classification / range table
     - Table 40, section 9.1
   * - MAG L1D despun magnetic field vectors
     - section 9.2
   * - Electron response matrices (L4-only and L3 vs L4)
     - section 9.3
