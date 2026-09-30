.. _swapi-ancillary:

Ancillary Files, Calibration and External Dependencies
======================================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Document:** sections 8, 9, 10.3 and 11.

SWAPI needs remarkably little ancillary data to reach L2 - two tables, both
derived from the same delivered workbook. Everything else on this page is
either L3's problem or operations' problem, and is documented so you can tell
which is which.

What L2 actually reads
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Ancillary
     - SDC descriptor
     - Read by
   * - ESA unit conversion table
     - ``esa-unit-conversion``
     - ``swapi_l2.solve_full_sweep_energy`` (and the I-ALiRT code)
   * - LUT notes table
     - ``lut-notes``
     - ``swapi_l2.solve_full_sweep_energy``

Both L2 tables are loaded through ``swapi_utils.read_swapi_lut_table(path)``,
which is a thin ``pd.read_csv`` plus one important cleanup:

.. code-block:: python

   df["Energy"] = (df["Energy"].astype(str)
                               .str.replace(",", "", regex=False)
                               .replace("Solve", -1)
                               .astype(np.float64))

Two things that will bite you: the ``Energy`` column arrives as
**comma-thousands-separated strings** (``"1,163"``), and the sentinel
``"Solve"`` becomes **-1**. Everything downstream detects solve steps as
``Energy < 0``. Only the ``Energy`` column is cleaned - ``Voltage`` keeps its
literal ``"Solve"`` strings.

.. warning::

   The CSVs are read with a UTF-8 BOM, so the first column name is
   ``"﻿timestamp"``, not ``"timestamp"``. Pandas handles it; a hand-rolled
   ``csv.DictReader`` will not.

.. _swapi-l3-descriptors:

What the SDC holds but this repository never reads
---------------------------------------------------

SWAPI delivers a further **thirteen** ancillary files to the SDC. All of them
are inputs to the SWAPI team's L3 container (:ref:`swapi-l3-scope`), and none
of them appear anywhere in this codebase. Their descriptors are recorded here
because the algorithm document refers to most of them only by the symbol they
carry, and that mapping is otherwise written down nowhere.

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - SDC descriptor
     - Symbol
     - Purpose
   * - ``central-effective-area``
     - :math:`\mathcal{A}_0^s(V)`
     - Central effective area vs ESA voltage.
       :ref:`swapi-instrument-response`.
   * - ``passband-fit-coefficients``
     - :math:`P_r(v/v_0, \theta, V)`
     - Energy-angle passband, one set for the sunglasses region and one for the
       open aperture. :ref:`swapi-instrument-response`.
   * - ``azimuthal-transmission``
     - :math:`T(\phi)`
     - Azimuthal transmission. :ref:`swapi-instrument-response`.
   * - ``instrument-response-lut``
     - n/a
     - The three functions above combined into the tabulated response that the
       L3 forward model evaluates directly.
   * - ``efficiency-lut``
     - :math:`\varepsilon_H`, :math:`\varepsilon_{He}`
     - Time-varying detection efficiency. :ref:`swapi-efficiency-lut`.
   * - ``energy-gf-sw-lut``
     - :math:`G(E/q)`
     - Energy-dependent geometric factor, solar wind.
   * - ``energy-gf-pui-lut``
     - :math:`G(E/q)`
     - Energy-dependent geometric factor, pickup ions.
   * - ``density-of-neutral-helium-lut``
     - :math:`n_{\mathrm{He}}`
     - Hot-model interstellar neutral helium density, the source population for
       the L3 pickup helium fit.
   * - ``helium-inflow-vector``
     - n/a
     - Interstellar neutral helium inflow vector.
   * - ``hydrogen-inflow-vector``
     - n/a
     - Interstellar neutral hydrogen inflow vector.
   * - ``proton-density-temperature-lut``
     - n/a
     - **Unidentified.**
   * - ``alpha-density-temperature-lut``
     - n/a
     - **Unidentified.**
   * - ``clock-angle-and-flow-deflection-lut``
     - n/a
     - **Unidentified.**

.. warning::

   The last three have no counterpart in any section of the algorithm document
   summarised on these pages, and nothing in the L2-to-L3 contract explains
   them. Ask the SWAPI team what they contain rather than inferring it from the
   descriptor names.

.. _swapi-esa-unit-conversion-adp:

ESA Unit Conversion ADP
-----------------------

**[DOC]** Section 10.3. Delivered by the SWAPI Instrument Team "as needed".

**Naming.** The ADP file naming convention is the IMAP standard:

.. code-block:: text

   imap_<instrument>_<description>_<start_date>_<end_date>_<version>.<extension>

so the ESA Unit Conversion ADP is

.. code-block:: text

   imap_swapi_esa-unit-conversion_<start_date>_<end_date>_<version>.xlsx

The initial ground-test delivery was
``imap_swapi_esa-unit-conversion_20250211_v00.xlsx``.

**[CODE]** The SDC ingests it as CSV. The vendored test copies are
``imap_processing/tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv``
and ``.../imap_swapi_lut-notes_20250626_v006.csv``.

**Columns** (main sheet):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``timestamp``
     - Validity start of this table version, ``M/D/YYYY H:MM``. Parsed with
       ``format="%m/%d/%Y %H:%M"``.
   * - ``ESA Step #``
     - 0-71.
   * - ``K factor``
     - eV/V/e. 1.88 in the 2025-02-11 rows, 1.93 in the 2025-05-19 rows.
       **Read but never used.** See :ref:`swapi-k-factor`.
   * - ``Voltage``
     - ESA voltage in V, or the literal ``Solve``. **Not used.**
   * - ``Energy``
     - eV/q, or ``Solve``. **This is what L2 uses.**
   * - ``Sweep #``
     - Which sweep table this row applies to. Matched against the L1
       ``sweep_table`` variable. The test file contains sweeps 0, 1 and 2.
   * - ``ESA Index Number``
     - For fixed steps, the row on the DAC ladder. For solve steps, the
       **offset** from the final solve step (``-16 ... +16`` in the nominal
       table). This is the column that drives the fine-step walk.
   * - ``LUT version number``
     - Which ``LUT_Notes_vx`` sheet the offsets refer to.

.. note::

   The table is keyed on **(timestamp, Sweep #)**, and L2 selects the latest
   version at or before the sweep's start time. A single CSV therefore contains
   several stacked table versions - the 288-row test file holds two versions of
   sweep 0 plus one each of sweeps 1 and 2. When adding a new version, append
   rows; do not replace the file's history, or reprocessing older data will
   silently pick the wrong energies (or fall back to the earliest version).

LUT notes table
---------------

**[DOC]** The ``LUT_Notes_vx`` tab of the ADP workbook, one tab per LUT
version. **[CODE]** ingested as ``imap_swapi_lut-notes_<date>_v<NNN>.csv`` -
note the file version tracks the **LUT** version (``v006`` in the test file),
not the delivery.

**Columns:**

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Meaning
   * - ``ESA Index Number``
     - Row index on the ladder, 0-based, ascending.
   * - ``ESA Voltage``
     - ESA voltage in V, descending down the table (10240 V at row 0).
   * - ``Energy``
     - eV/q. **The column L2 reads.**
   * - ``Lower Energy``, ``Upper Energy``
     - Passband edges. Not used by any code here.
   * - ``ESA Range``
     - HV range selector.
   * - ``ESA DAC (Dec)``, ``ESA DAC (Hex)``
     - The commanded DAC value. **``ESA DAC (Hex)`` is what
       ``esa_lvl5`` is matched against**, formatted as 4 uppercase hex digits.

The test file has 1024 data rows. Adjacent rows can share a DAC value (rows 0
and 1 are both ``1FFE``), which is why "first match wins" in
``solve_full_sweep_energy`` is a real behavioural choice.

Calibration data
----------------

**[DOC]** Section 9. Three ground sources plus two on-orbit ones. None of this
is read by L1 or L2; it feeds the L3 response function and the operations
voltage updates.

Ground calibration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Source
     - What it produces
   * - **End-to-end (ETE) model**
     - A full simulation of instrument plus environment, driven by real 1 au
       solar wind (ACE and Wind, via the OMNI database), used to generate the
       LUTs and to validate the processing. Five modules: OMNI observations,
       SW+PUI distribution function model, coordinate transformation matrices,
       a SIMION-based electro-optical model, and a telemetry formatter. **This
       is also how SWAPI processing code was verified** - see below.
   * - **SIMION module**
     - Ions flown through the instrument geometry at a range of energies, entry
       locations and angles, per ESA voltage. Integrated over a uniform beam to
       give the response function, then tabulated: ion energies, entry angles,
       and relative intensities per ESA voltage. 72 response files per ESA step.
   * - **Princeton lab**
     - Instrument response over 0.4-18 keV/q required (0.1-20 keV/q tested), a
       broad range of angles, each optical element (through the attenuation
       grid as well as the open aperture), and H and He. Energy-angle
       passbands from azimuthal positioning; entrance-aperture response from
       elevation-angle rolls. Also CEM gain curves, absolute detection
       efficiency, carbon-foil active area, transmission and scattering, UV
       suppression, rate dependence and backgrounds. Details in Rankin et al.
       (2025).
   * - **CoDICE cross-calibration**
     - Both instruments in the same chamber at the same time, rotating in and
       out of the beam, with an absolute beam monitor. Primarily 2 keV/q and
       16 keV/q protons and helium, at several elevation angles and at fixed
       azimuths of 0 degrees (through the sunglasses) and 90 degrees (through
       the open aperture). Rankin et al. (2025), Livi et al. (2025).

On-orbit calibration
^^^^^^^^^^^^^^^^^^^^

**[DOC]** Two activities:

1. **Gain curve tests**, every ~3 to 6 months (multiple times during
   commissioning), to find the best PCEM and SCEM voltages as the CEMs age.
   The optimal setting is where the count rate "flattens out" - above some
   bias, rates become independent of further voltage increase. CEM gain decays
   with total output charge over its lifetime, so the voltage must be raised
   to compensate.
2. **In-flight comparison with CoDICE**, once both instruments' data are
   validated and processed, using overlapping species and energy ranges.

The absolute efficiency is computed from the three counters:

.. math::

   \varepsilon = \frac{\mathrm{COIN}^2}{\mathrm{PRM} \times \mathrm{SEC}}

which is independent of individual detector performance and aging
(Funsten et al. 2005).

.. _swapi-efficiency-lut:

The efficiency LUT
^^^^^^^^^^^^^^^^^^

**[DOC]** Section 9.5.1. The efficiency calibration table has **two columns**,
:math:`\varepsilon_H` for hydrogen and :math:`\varepsilon_{He}` for helium,
versus time. It scales the central effective area:

.. math::

   \mathcal{A}_0^{H^+}(V) = \mathcal{A}_{0,\mathrm{lab}}^{H^+}(V)
     \frac{\varepsilon_H(t)}{\varepsilon_H(t_{\mathrm{lab}})}

.. math::

   \mathcal{A}_0^{He^{2+}}(V) = \mathcal{A}_0^{He^{+}}(V)
     = \mathcal{A}_{0,\mathrm{lab}}^{H^+}(V)
       \frac{\varepsilon_{He}(t)}{\varepsilon_H(t_{\mathrm{lab}})}

where :math:`\varepsilon_H(t_{\mathrm{lab}})` is the **first proton entry in
the table on or after 2025-11-01** and :math:`\varepsilon_H(t)` is the most
recent entry preceding :math:`t`. Only relative values matter, so the table is
agnostic to whether it is ever rescaled to an absolute efficiency.

Initial values: hydrogen column **1**, helium column **1.05**. The 1.05 comes
from the high-energy limit observed in the lab for He\ :sup:`+` versus
H\ :sup:`+`; above a few keV per charge the ratio was consistently 1.05. Since
both solar wind alphas and pickup ions tend to be above that threshold, the
increase of the ratio at low energies has not been accounted for.
He\ :sup:`+` and He\ :sup:`2+` are assumed to have the same efficiency.

**[CODE]** No efficiency table is read anywhere in this repository. It is an
L3 input, delivered to the SDC under the ``efficiency-lut`` descriptor.

.. _swapi-instrument-response:

Instrument response CSVs
^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Three functions, stored as CSV files and loaded by the production
(L3) code from ancillary files:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Notes
   * - :math:`\mathcal{A}_0^s(V)`
     - Central effective area vs ESA voltage. ~0.75 cm\ :sup:`2` at 100 V
       falling to ~0.29 cm\ :sup:`2` near 3 kV, rising slightly above that.
       Carries an uncertainty of **up to 20%**, which is a systematic floor on
       every density measurement.
   * - :math:`P_r(v/v_0, \theta, V)`
     - Region-specific energy-angle passband, one set for SG and one for OA.
       Interpolated on speed ratio and elevation.
   * - :math:`T(\phi)`
     - Azimuthal transmission, tabulated from 0 to 180 degrees at 0.1 degree
       spacing. Missing entries treated as zero; the interpolator uses
       :math:`|\phi|` after wrapping azimuth into
       :math:`[-180^\circ, 180^\circ)`. The flat portions are hard-coded for
       performance: :math:`T = 10^{-3}` for :math:`|\phi| \le 9^\circ` and
       :math:`T = 1` for :math:`31^\circ \le |\phi| \le 115^\circ`.

Normalizations of :math:`\mathcal{A}_0^s` and :math:`P_r` are aligned at
:math:`\theta = 0` and :math:`k^{*} = 1.89` eV/V/e.

**[CODE]** None of these files are read here. They are delivered to the SDC as
``central-effective-area``, ``passband-fit-coefficients`` and
``azimuthal-transmission`` - see :ref:`swapi-l3-descriptors`.

The gain test LUT
-----------------

**[DOC]** Section 11, "Maintenance of SWAPI ground data processing". This is
the *only* regular maintenance item the document names:

   The only regular maintenance of the SWAPI ground data processing is updating
   the PCEM and SCEM voltages periodically.

The gain test is SWAPI's only calibration activity during routine operations -
multiple times during commissioning, then every ~3 months. Its output is a
**1x2 LUT**:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - PCEM setting (V)
     - SCEM level (V)
   * - (value)
     - (value)

LUT improvements are made asynchronously with SWAPI data production, and
**"the ground processing will also need to incorporate the resulting efficiency
changes acquired after gain curve tests."**

**[CODE]** Nothing reads this LUT. Its effect reaches the pipeline only
indirectly - via the efficiency table at L3, and via ``SWP_HK.PCEM_LVL`` /
``SWP_HK.SCEM_LVL`` in housekeeping. Since L2 does not apply efficiency, no L2
reprocessing is triggered by a gain test.

External dependencies
---------------------

**[DOC]** Section 8. Three, and their status here is worth being precise about.

Spacecraft thruster data
^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** SWAPI data will be flagged in the L1 data product **and beyond** if
obtained during spacecraft thruster activity, for data quality reasons.

* SWAPI is **off** for at least 25 minutes before, during, and 1 hour after
  delta-V and stationkeeping maneuvers - so no data exists to flag for those.
* SWAPI is **fully operational** for the daily repointing maneuvers - so those
  are exactly the ones that need flagging.
* Thruster firing history comes from the MOC.

**[CODE]** Not implemented. ``swapi_l2.py`` carries
``# TODO: add thruster firing flag`` and there is no bit for it in
``SWAPIFlags``. Section 10.2 of the document places the flag at L1-L2 (as
``SWP_SC_THRUSTER``, 0 or 1), so L2 is the right place for it.

MAG Level 2 data
^^^^^^^^^^^^^^^^

**[DOC]** The magnetic field direction is needed to constrain the solar wind
**alpha** flow vector, because the alpha-proton differential drift is modelled
as lying along :math:`\hat{B}`. Where MAG L2 is unavailable, MAG **L1D** is
used and the product is reprocessed once L2 exists; **only the L2-processed
data is released publicly**, and the interim product carries a
``PRELIMINARY_MAG`` flag.

**[CODE]** L3a's concern. No SWAPI code in this repository reads MAG.

Timing, attitude, ephemeris (SPICE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** The L3 solar wind flow deflection angle at ~1 min resolution needs
**IMAP/SWAPI spin phase for each ESA energy bin of the individual sweep** and
the spacecraft pointing in a standard coordinate system. IMAP SPICE kernels are
used. At L3 this becomes 72 SWAPI-to-RTN rotation matrices per sweep; if the
matrices are unavailable the chunk is skipped and assigned fill values, and if
the spacecraft velocity is unavailable the fit still runs but the Sun-frame
outputs cannot be computed.

**[CODE]** In this repository SPICE is used only for **time conversion**: the
L1 and L2 CLI branches require time kernels as a dependency, and
``swapi_l1`` uses ``met_to_utc`` / ``ttj2000ns_to_met`` to write
``sci_start_time``. No geometry, no spin phase, no rotation matrices.

Instrument status summary
^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Figure 4 shows an "instrument status summary" feeding L3 alongside
MAG and SPICE. The document does not define its format. The nearest thing that
exists is the L1A/L1B housekeeping products and the ``swp_l1a_flags`` bitfield.

Testing and code delivery
-------------------------

**[DOC]** Sections 12 and 13, quoted because they define the working
relationship:

* **Testing.** "SWAPI processing code was tested by the SDC working with the
  SWAPI data team. The SWAPI team ran the ETE model and provided inputs and
  outputs to the SDC to allow the SDC to verify the implementation of the SWAPI
  ground processing code." So the intended validation path for L1/L2 is
  ETE-model input/output pairs from the instrument team, not synthetic data
  invented here.
* **Code delivery.** "SWAPI delivers the SWAPI L3 processing code using a
  Docker container via AWS." This is the document's own statement that L3 is
  not the SDC's to write.

**[CODE]** The tests that exist validate decommutation against SWAPI CSV
exports of a single pre-launch idle packet, plus unit tests of the grouping,
decompression and energy-solve logic. There are **no ETE-model
input/output validation pairs** in the repository. See
:ref:`swapi-implementation-status`.
