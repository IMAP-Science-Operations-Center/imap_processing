.. _lo-ancillary:

Ancillary and Calibration Data
==============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

IMAP-Lo processing depends on a lot of external tables. Some ship with this
package, some arrive as CLI dependencies from the SDC, and some are described
in the algorithm document but do not exist yet.

The reader
----------

**[CODE]** ``imap_processing/lo/lo_ancillary.py`` provides a single function,
``read_ancillary_file(path) -> pd.DataFrame``. It is a thin ``pd.read_csv``
wrapper with three behaviors worth knowing:

* **Date conversion.** Columns named ``YYYYDDD``, ``#YYYYDDD``,
  ``YYYYDDD_strt`` and ``YYYYDDD_end`` are parsed from year + day-of-year into
  datetimes, and renamed to ``Date``, ``StartDate``, ``EndDate``.
* **esa-mode-lut** files skip the first row, which is a comment.
* **geometric-factor** files are sniffed for a **legacy format**. If the text
  contains ``Hi_Thr,,,`` the file uses the old layout: header rows at lines 1
  and 38 are skipped, and an ``esa_mode`` column is synthesized (rows 0-35 are
  mode 0 / HiRes, rows 36+ are mode 1 / HiThru). Newer files carry an
  ``esa_mode`` column and use ``#`` comment lines instead.

Ancillaries shipped with the package
------------------------------------

**[CODE]** ``imap_processing/lo/ancillary_data/``. These are read directly by
path glob from ``ANCILLARY_DATA_DIR``, not passed in as CLI dependencies.

Geometric factors
^^^^^^^^^^^^^^^^^

``imap_lo_hydrogen-geometric-factor_v004.csv``,
``imap_lo_oxygen-geometric-factor_v004.csv`` (plus ``_v001`` in the legacy
format). Read by ``load_geometric_factor_data`` /
``reduce_geometric_factor_data`` / ``_esa_calibration``.

Current (v004) columns:

.. code-block:: text

   esa_mode, incident_E-Step, Observed_E-Step, Cntr_E, Cntr_E_unc,
   GF_Trpl_H, GF_Trpl_H_unc_minus, GF_Trpl_H_unc_plus,
   Cntr_E_delta_minus, Cntr_E_delta_plus

Units: energies in keV, geometric factors in cm^2 sr keV/keV. 14 rows: 7 ESA
levels for each of the two ESA modes.

The asymmetric ``unc_minus`` / ``unc_plus`` columns are what produce the
asymmetric systematic error on L2 intensity.

Legacy (v001) columns are richer but not indexed by ESA mode in the file
itself:

.. code-block:: text

   incident_E-Step, Observed_E-Step, Cntr_E, Cntr_E_unc,
   GF_Dbl_all, GF_Dbl_all_unc, GF_Trpl_all, GF_Trpl_all_unc,
   GF_Dbl_H, GF_Dbl_H_unc, GF_Trpl_H, GF_Trpl_H_unc

**[DOC]** The document specifies (incident step ``i``, observed step ``k``)
pairs with :math:`k \le i`, because an ESA level can also detect particles that
entered at a higher level. The v001 file has this structure (36 rows per mode =
1+2+...+7 pairs + extras); **v004 has only the diagonal** ``i == k``. If you
need the off-diagonal response, it is in v001 and not in v004.

Sputter correction factors
^^^^^^^^^^^^^^^^^^^^^^^^^^

``imap_lo_sputter-correction-factors_v002.csv``, read by
``load_sputter_correction_data`` / ``_sputter_correction``.

.. code-block:: text

   source_species,target_species,target_esa,source_esa,sputter_factor
   o,h,1,4,0.236
   o,h,2,4,0.372
   o,h,3,4,0.898
   o,h,4,4,0.891
   o,h,5,4,0.037
   o,h,5,6,0.32
   o,h,6,6,0.32
   o,h,7,6,0.22

Bootstrap correction factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``imap_lo_bootstrap-correction-factors_v001.csv``, read by
``load_bootstrap_correction_data`` / ``_bootstrap_correction``. Columns
``esa_step_i, esa_step_k, bootstrap_factor``. This is Appendix A Table 3 in
long form and matches it exactly, including the virtual level 8 terms
(``6,8,0.061`` and ``7,8,0.75``).

Remember that ``LoConstants.BOOTSTRAP_SCALE = 0.5`` halves these before use.

ISN mask parameters
^^^^^^^^^^^^^^^^^^^

``imap_lo_isn-mask-parameters_v001.csv``, read by
``load_isn_mask_parameters`` / ``_isn_mask_parameters``.

.. code-block:: text

   pivot_angle,esa_step,intensity_threshold_fraction,angular_width_deg,outlier_percentile

Only pivot angles **75, 90 and 105** are tuned. Requesting an ISN-masked map at
any other pivot angle raises ``ValueError``. Not in the algorithm document.

Ancillaries supplied as CLI dependencies
----------------------------------------

**[CODE]** These arrive in ``anc_dependencies`` and are matched by substring in
the filename.

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Matched on
     - Level
     - Used for
   * - ``sweep-table``
     - L1B
     - ``set_esa_mode`` looks up the row covering the data's date and returns
       ``0`` (``HiRes``) or ``1`` (``HiThr``).
   * - ``esa-mode-lut``
     - L1B
     - ``_get_esa_level_indices`` builds the measured-step to true-ESA-level
       mapping used by the resweep.
   * - ``bg-rates-anti-ram-overrides``
     - L1B
     - Optional. Per-``(year, doy)`` overrides of the anti-ram background
       threshold, for anomalous days.
   * - ``efficiency-factor``
     - L2
     - ``load_efficiency_data``. Optional.
   * - ``esa-eta-fit-factors``
     - L2
     - ``_flux_corrector``. The 5th-order polynomial coefficients for the ESA
       transmission bias :math:`\eta_{\mathrm{esa},k}(\gamma)`. **Required** for
       any Compton-Getting corrected map; its absence raises ``ValueError``
       before accumulation starts.

Ancillaries the document defines
--------------------------------

**[DOC]** These are specified in document section 11 as IMAP-Lo team
deliverables. Some have no counterpart in the code.

Goodtimes list
^^^^^^^^^^^^^^

Time-interval list of usable data, separate lists for ENA and ISN, declared on
science-cycle boundaries.

Columns: ``YYYYDDD``, ``START``, ``END``, ``BIN_START``, ``BIN_END``, ``LO``,
``BADTIME_FLAG[N,7]`` (1 = good, per ESA step), ``COMMENTS``.

The document says the derivation algorithm is **TBD**, done manually for
IBEX-Lo. **[CODE]** The repository generates goodtimes automatically at L1B;
see :ref:`lo-l1b`.

Efficiency factor
^^^^^^^^^^^^^^^^^

The MCP has two voltage tracks. Calibration and geometric factors assume the
**upper track**. When the instrument cannot reach it, efficiency drops:

.. math::

   F_{\text{eff}} = \frac{\text{TOF triple efficiency (current)}}
                         {\text{TOF triple efficiency (upper track)}}

typically ~1.0, lower in non-optimal periods. It is tracked via ``MCP_TOF_V``
and eventually to be automated as a fit curve in ``MCP_TOF_V``.

Columns: ``YYYYDDD``, ``START``, ``END``, ``FEFF[N,7,2]`` (per ESA step, plus
systematic uncertainty), ``COMMENTS``.

Background files
^^^^^^^^^^^^^^^^

``Background_ENA.txt`` and ``Background_ISN.txt``, separate per species and per
ESA mode.

Columns: ``YYYYDDD``, ``START``, ``END``, ``Bin_strt``, ``Bin_end``, ``Lo``,
``Background_Rate_H[N,14,3]``, ``Background_Rate_O[N,14,3]``. The 14 is
HiRes levels 1-7 followed by HiThru levels 8-14; the 3 is
[value, systematic, statistical].

.. note::

   The "14 energy steps" convention - HiRes in slots 1-7 and HiThru in slots
   8-14 - recurs in the efficiency, background and geometric factor file
   specifications. The code instead uses a separate ``esa_mode`` column with 7
   levels. Watch for this when ingesting a team-delivered file. **[CODE
   deviates]**

Map-pointing files
^^^^^^^^^^^^^^^^^^

Two files, maintained by the IMAP-Lo data team from the operations plan.

``imap_lo_pointing-Index_20250101_20271231_v001.txt`` - maps to pivot-angle
ranges and date ranges: ``MAP_INDEX``, ``PIVOT_ANGLE[M,3]`` (nominal, min,
max), start ``YYYYDDD``, end ``YYYYDDD``.

``imap_lo_pointing-file_20250101_20271231_v001.txt`` - one row per operational
day: ``YYYYDDD``, ``MAP_INDEX``, ``PIVOT_ANGLE``.

**[CODE]** The repository does not read these. Pivot angle is derived from
housekeeping (``pcc_coarse_pot_pri`` median) and map membership is decided by
``Lo.pre_processing`` comparing the goodtimes ``pivot`` variable against the
map descriptor's pivot angle within ``PSET_PIVOT_ANGLE_TOLERANCE``.

Compression tables
^^^^^^^^^^^^^^^^^^

``IMAP-Lo_Compression_Tables.xlsx`` in the document. **[CODE]** These are
already vendored as CSVs in
``imap_processing/lo/l0/decompression_tables/``:
``log10_8_to_12_bit_uncompress.csv``, ``log10_8_to_16_bit_uncompress.csv``,
``log10_12_to_16_bit_uncompress.csv``.

Constants that behave like calibration
--------------------------------------

**[CODE]** A number of values that would normally be calibration live in
``imap_processing/lo/constants.py`` as ``LoConstants``. When someone says
"the calibration changed", check here as well as the CSVs.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Constant
     - Meaning
   * - ``PIVOT_ANGLES``
     - The 8 nominal pivot angles (60, 75, 90, 105, 120, 135, 148, 160), their
       +/-5 degree acceptance windows, and per-angle ram / anti-ram background
       thresholds. Only 75, 90, 105 have their own thresholds.
   * - ``PSET_PIVOT_ANGLE_TOLERANCE = 5.0``
     - Degrees of slack when matching a pointing to a map's pivot angle.
   * - ``PIVOT_RAM_OFFSET = 4.0``
     - Empirical degrees added to the measured pivot angle when projecting onto
       ram.
   * - ``BG_RATES``
     - Nominal background counts/s: H 0.0014925, O 0.000136635.
   * - ``BG_RATE_FALLBACK_SCALE`` / ``BG_RATE_FLOOR_DIVISOR``
     - Fallbacks when goodtime exposure or the computed rate is zero.
   * - ``RAM_ESA_LEVELS = (6, 7)``
     - 1-indexed ESA levels used for the ram background estimate.
   * - ``RAM_HISTOGRAM_BINS`` / ``ANTI_RAM_HISTOGRAM_BINS``
     - Spin-angle bin slices (0:20 + 50:60) and (20:50) out of 60.
   * - ``N_CYCLE_SUM = 1``, ``N_CYCLE_AVE = 7``
     - Goodtime boundary granularity and the background averaging window.
   * - ``DELAY_MAX = 100``, ``GOODTIME_PADDING = 2.0``, ``EXPOSURE_FACTOR = 0.5``
     - Goodtime interval construction.
   * - ``BOOTSTRAP_SCALE`` and friends
     - Bootstrap scaling and its bracketing systematic error.
   * - ``ESA_8_ENERGY_RATIO = 2.1``
     - Virtual ESA level 8 energy, as a multiple of level 7.
   * - ``CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV = 4.661``
     - 1/2 m_H U^2 at ~30 km/s, in eV.
   * - ``CG_MAX_ITERATIONS = 20``, ``CG_CONVERGENCE_TOLERANCE = 0.005``
     - Predictor-corrector limits.
   * - ``STAR_MIN_COUNT_THRESHOLD = 700``, ``STAR_END_BINS_TO_EXCLUDE = 2``,
       ``STAR_BIN_OFFSET_BY_SYNC``
     - Star sensor processing.
   * - ``PIVOT_HK_HOUR_RANGE = (0.5, 22.5)``
     - Hours of the day the pivot-angle median is taken over.

Also in ``lo/l1b/tof_conversions.py``: the four TOF DN-to-ns linear
coefficients, which came from a Word document and an email rather than the
algorithm document.
