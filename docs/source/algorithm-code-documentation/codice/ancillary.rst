.. _codice-ancillary:

Ancillary and Calibration Files
===============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

CoDICE depends more heavily on ancillary files than any other IMAP instrument in
this repository: **it cannot even unpack its own telemetry without one.** This
page lists every file the pipeline asks for, what it contains, and which level
consumes it.

All of these arrive through ``ProcessingInputCollection.get_file_paths(
descriptor=...)``. **None of them are vendored in this repository** - the two
CSVs under ``imap_processing/codice/data/`` are historical snapshots that no
pipeline code reads.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 10 22 38

   * - Descriptor
     - Level
     - Format
     - Purpose
   * - ``l1a-sci-lut``
     - L1A
     - JSON
     - **The SCI-LUT.** Plan, ESA sweep, Lo stepping, views and collapse tables.
       Required for every product except direct events.
   * - ``l2-lo-gfactor``
     - L2 Lo
     - CSV
     - Geometric factors per (mode, ESA step, position).
   * - ``l2-lo-efficiency``
     - L2 Lo
     - CSV
     - Efficiencies per (species, product, ESA step, position).
   * - ``l2-lo-onboard-energy-table``
     - L2 Lo DE
     - CSV
     - APD energy channel -> energy bin index, per (APD ID, gain).
   * - ``l2-lo-onboard-energy-bins``
     - L2 Lo DE
     - CSV
     - Energy bin index -> keV.
   * - ``l2-lo-onboard-mpq-cal``
     - L2 Lo DE
     - CSV
     - k-factor, ESA step voltages, and the TOF channel -> ns quadratic.
   * - ``l2-hi-omni-efficiency``
     - L2 Hi
     - CSV
     - Average efficiency per (species, energy bin) plus a ``GF`` row.
   * - ``l2-hi-sectored-efficiency``
     - L2 Hi
     - CSV
     - Efficiency per (species, energy bin, inst_az) plus a ``GF`` row.
   * - ``l2-hi-energy-table``
     - L2 Hi DE
     - CSV
     - SSD energy channel -> MeV, per (SSD ID, gain).
   * - ``l2-hi-tof-table``
     - L2 Hi DE
     - CSV
     - TOF index -> (ns, MeV/nuc).

The SCI-LUT
-----------

**[DOC]** Sections 5.1, 7 and 8. Originates as ``26850.03-SCI-LUT-01.xls`` with
tabs ``Plan``, ``ESA Sweep``, ``Lo Stepping``, ``Views``, ``Collapse_Lo``,
``Collapse_Hi``, ``Data Products - Lo`` and ``Data Products - Hi``. The
``Table_ID`` in each science packet says which spreadsheet version to use.

**[CODE]** The SDC receives a JSON rendering. Structure, as read by ``utils.py``:

.. code-block:: text

   {
     "<table_id>": {
       "view_tab": {
         "(<view_id>, 0x<APID hex>)": {
            "sensor": 0|1,             # 0 = Lo, 1 = Hi
            "collapse_table": <int>,   # index into collapse_lo / collapse_hi
            "3d_collapse": <int>,      # spins per matrix (Hi)
            "compression": 0..6        # CoDICECompression
         }, ...
       },
       "collapse_lo": { "<id>": { "matrix": [[...]], "variables": {name: [...]} } },
       "collapse_hi": { "<id>": { "matrix": [[...]], "variables": {name: [...]} } },
       "plan_tab":  { "(<plan_id>, <plan_step>)": { "lo_stepping": <int>, ... } },
       "esa_sweep_tab": { "<table_number>": [128 voltages] },
       "lo_stepping_tab": {
          "row_number": {"data": [half-spin index per ESA step]},
          "num_steps":  {"data": [ESA steps sampled in that half-spin]},
          "tunable_values": {
             "spin_time_ms", "num_sectors_ms", "sector_margin_ms",
             "dwell_fraction_percentage", "min_hv_settle_ms", "max_hv_settle_ms"
          }
       },
       "data_product_lo_tab": {
          "0": {
            "species": {"sw": {"species_names": [...],
                               "desired_species_names": [...]}},
            "ialirt":  {"sw": {"species_names": [...],
                               "desired_species_names": [...]}}
          }
       },
       "data_product_hi_tab": { ... }
     }
   }

Things to know:

* **The view key is a literal string** ``"(3, 0x484)"``. If the APID hex casing
  or the spacing changes, the lookup returns ``None`` and processing fails with
  a ``TypeError`` rather than a helpful message.
* ``collapse_*[id]["matrix"]`` is the raw collapse pattern.
  ``get_collapse_pattern_shape`` derives the *reduced* array shape from it;
  ``index_to_position`` recovers which physical positions survived.
* ``collapse_*[id]["variables"]`` is used only by the aggregated-counters
  products, where each row is one counter and a row of zeros means "counter
  turned off in flight".
* ``species_names`` is what the instrument actually sent;
  ``desired_species_names`` is what the team wants written. L1A takes the union
  of ``desired_species_names`` and the hard-coded ``*_VARIABLE_NAMES``, and
  NaN-fills anything that is not in ``species_names``. This is how the P3 Fe
  highQ/lowQ label swap is absorbed.
* ``lo_stepping_tab["row_number"]["data"]`` is the half-spin index per ESA step
  and can be **shorter than 128** (it is under the post-2025-12-18 scheme). L1A
  pads with ``HALF_SPIN_FILLVAL = 63`` and masks those steps to NaN.

Test copies live at
``imap_processing/tests/codice/data/l1a_lut/imap_codice_l1a-sci-lut_20251007_v005.json``
(pre-FSW-change) and ``..._20260129_v002.json`` (post).

Lo geometric factors (``l2-lo-gfactor``)
----------------------------------------

**[DOC]** :math:`G_m` in cm2 sr eV/eV for one angular bin, provided for **each
APD ID and ESA step**, i.e. a (128 x 24) array **per mode**. Two modes: ``full``
and ``reduced``.

**[CODE]** ``get_geometric_factor_lut`` reads a CSV with columns ``mode``,
``esa_step``, ``position_1`` ... ``position_24``, filters ``mode == "full"`` and
``mode == "reduced"``, sorts by ``esa_step``, sorts the position columns
numerically, and returns ``{"full": (128, 24), "reduced": (128, 24)}``.

Test file: ``imap_codice_l2-lo-gfactor_20251212_v003.csv``.

Lo efficiencies (``l2-lo-efficiency``)
--------------------------------------

**[DOC]** :math:`\varepsilon_{jlk}` - efficiency by species, ESA step and
position.

**[CODE]** ``get_efficiency_lut`` returns the whole DataFrame; columns are
``species``, ``product``, ``esa_step``, ``position_1`` ... ``position_24``. The
``product`` column selects ``"sw"`` or (for the unimplemented NSW path)
``"nsw"``. ``get_species_efficiency(species, df)`` filters by species, sorts by
ESA step, and returns an ``xr.DataArray`` with dims ``("esa_step", "inst_az")``.

A species with no rows produces a warning and is **skipped** - its intensity is
left as a rate. Watch for that.

Test file: ``imap_codice_l2-lo-efficiency_20251212_v003.csv``.

Lo direct-event calibration
---------------------------

Three files, all consumed by ``process_lo_direct_events``.

``l2-lo-onboard-energy-table``
  **[CODE]** ``pd.read_csv(header=None, skiprows=1)``. Rows are APD energy
  channels, columns are ordered ``APD-1-LG, APD-1-HG, APD-2-LG, ...`` so the
  column index is ``apd_id * 2 + gain``. The value is an **energy bin index**,
  not an energy.

``l2-lo-onboard-energy-bins``
  **[CODE]** Column 1 is the energy in keV for each bin index. Chained after the
  table above.

``l2-lo-onboard-mpq-cal``
  **[CODE]** Read positionally, which makes it fragile:

  * ``df.loc[0, 10]`` - the k-factor.
  * ``df.loc[4, 4:]`` - the 128 ESA step voltages;
    ``esa_kev = esa_v * k_factor / 1000``.
  * ``df.loc[2, 1]``, ``df.loc[3, 1]``, ``df.loc[4, 1]`` - the quadratic
    coefficients :math:`a`, :math:`b`, :math:`c` for
    :math:`\tau_{ns} = a\,\tau_{ch}^2 + b\,\tau_{ch} + c`.
  * ``df.loc[6:, 0]`` - the TOF channel numbers.

  **Any change to the row/column layout of this file silently produces wrong
  numbers.** There is no header validation.

Hi efficiencies
---------------

``l2-hi-omni-efficiency``
  **[CODE]** Columns include ``species`` and ``average_efficiency``, one row per
  (species, energy bin). Species names use hyphens where the code uses
  underscores (``ne-mg-si`` vs ``ne_mg_si``); the code translates. A special row
  with ``species == "GF"`` carries the geometric factor, read as
  ``.values[0][-1]`` (**last column of the first GF row**). See the warning on
  :ref:`codice-l2` about whether this is :math:`G_k` or :math:`\sum_k G_k`.

``l2-hi-sectored-efficiency``
  **[CODE]** Columns ``species``, ``energy_bin``, then 12 ``inst_az`` columns.
  The ``GF`` row is likewise 12 values, one per SSD, which is used as a
  per-``inst_az`` ``DataArray``.

For I-ALiRT, ``convert_to_intensities`` reads a *different* layout from the same
descriptor slot: columns ``group_0`` ... ``group_3`` plus a ``GF`` row, sorted by
``energy_bin``.

Test files: ``imap_codice_l2-hi-omni-efficiency_20251212_v003.csv``,
``imap_codice_l2-hi-sectored-efficiency_20251212_v003.csv``.

Hi direct-event calibration
---------------------------

``l2-hi-energy-table``
  **[CODE]** First column is an index and is dropped. Rows are SSD energy
  channels; columns are ordered ``ssd0-LG, ssd0-MG, ssd0-HG, ssd1-LG, ...`` so
  the column index is ``ssd_id * 3 + (gain - 1)``. Values are **MeV** directly
  (no second-stage bin lookup, unlike Lo).

``l2-hi-tof-table``
  **[CODE]** First column dropped. Column 0 of the remainder is **TOF in ns**;
  column 1 is **energy-per-nucleon in MeV/nuc**. Both are indexed by the raw
  10-bit TOF value.

Files in the repository that are *not* used
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Path
     - Status
   * - ``imap_processing/codice/data/esa_sweep_values.csv``
     - Historical ESA sweep values. **Not read by any pipeline code** - the
       sweep now comes from ``esa_sweep_tab`` in the SCI-LUT.
   * - ``imap_processing/codice/data/lo_stepping_values.csv``
     - Historical Lo stepping table. **Not read by any pipeline code** - the
       stepping now comes from ``lo_stepping_tab`` in the SCI-LUT.

Both predate the SCI-LUT-driven design. Treat them as documentation of what a
nominal table looks like, not as inputs.

CoDICE SPICE Usage
------------------

**[CODE]** CoDICE processing uses ``imap_processing.spice.time.met_to_ttj2000ns``
to convert acquisition times to CDF epochs. It does **not** use SPICE kernels,
pointing frames or spin data - all CoDICE L2 angles are in the instrument frame
and the +46 deg rotation to the spacecraft frame (+316 deg in the January 2026
draft) is not applied here (see :ref:`codice-frames`). Rev 3 Chg 1 section 4.2
says spin angles "SHOULD" be computed with SPICE from the look-direction unit
vectors and the instrument kernel. The pipeline uses tabulated constants
instead, which the document also provides. CoDICE L2 jobs therefore do not need a metakernel beyond
what the leapsecond/SCLK conversion requires.
