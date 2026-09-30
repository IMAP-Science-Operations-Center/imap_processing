.. _swe-ialirt:

I-ALiRT - Real-Time Bidirectional Electrons
===========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page covers algorithm document sections 3.3.5, 3.4.2.2 and 3.4.5.

.. important::

   **None of this lives in** ``imap_processing/swe``. The entire SWE I-ALiRT
   algorithm is ``imap_processing/ialirt/l0/process_swe.py``, which imports
   three things from the SWE package:
   ``decompressed_counts``, ``deadtime_correction`` and
   ``read_in_flight_cal_data``. If you change any of those three, you have
   changed the real-time product.

What it is
----------

**[DOC]** SWE is part of the IMAP Active Link for Real-Time (i-ALiRT) system.
The downlinked subset is **electron counts from all 7 CEMs at all spin angles
for 8 of the 24 energy channels**, chosen to cover the suprathermal range
**100 - 1000 eV** where the distribution is governed by field topology.

Two products come out:

#. **BDE** - the bidirectional electron flag. One value per time step: **1 =
   counterstreaming**, **0 = nominal unidirectional flow**. Counterstreaming is
   typically the signature of a coronal mass ejection, where the field is
   connected to the Sun at both ends. **[DOC]** cautions that it can also come
   from connection to Earth's bow shock, so it is "not a definitive CME
   signature but must be considered in the context of other measurements". The
   algorithm derives from the onboard search flown on Genesis/GEM
   [Neugebauer et al., 2003].
#. **Normalized counts** - counts at each of the 8 ESA steps summed over all
   azimuths and all CEMs, giving a time series that shows suprathermal
   variability and the energy distribution.

Which 8 energies
----------------

**[DOC]** "for the nominal stepping table ... the SWE i-ALiRT packet includes
ESA steps **12-19**" (1-based).

**[CODE]** Zero-based rows 11-18 of ``ESA_VOLTAGE_ROW_INDEX_DICT``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - row (0-based)
     - ESA V
     - E (eV)
     - ``ialirt.utils.constants.swe_energy``
   * - 11
     - 21.13
     - 100.4
     - 100.4
   * - 12
     - 29.39
     - 139.6
     - 140.0
   * - 13
     - 40.88
     - 194.2
     - 194.0
   * - 14
     - 56.87
     - 270.1
     - 270.0
   * - 15
     - 79.10
     - 375.7
     - 376.0
   * - 16
     - 110.03
     - 522.6
     - 523.0
   * - 17
     - 153.05
     - 726.9
     - 727.0
   * - 18
     - 212.89
     - 1011.2
     - 1011.0

The same selection is encoded a **third** time as the ``ialirt`` column of the
ESA LUT CSV, where exactly 16 of table 0's 48 rows are flagged (8 energies
appearing twice each). Three independent encodings of one fact; see
:ref:`swe-implementation-status`.

The packet
----------

**[DOC]** Section 3.4.2.2. Unlike ``SWE_SCIENCE``, the I-ALiRT packet gives
**each 8-bit value its own named field**. Starting at byte 18, bit 0 there are
**28 fields**: 4 ESA steps × 7 CEMs, laid out CEM-major (CEM1's four steps,
then CEM2's four, and so on).

Per nominal 1-second packet: 7 CEMs × 4 energies, with **2 energies in one
spin-angle bin and 2 in the next**. Field names follow
``SWE_IALIRT.ELECTRON_COUNTS_SPIN_I_POL_<cem>_E_<n>J``.

The counts use the **same 8-bit compression** as the science packet - see
:ref:`swe-decompression`.

**[CODE]** ``imap_processing/ialirt/packet_definitions/ialirt_swe.xml`` names
them ``SWE_CEM<n>_E<m>`` (n = 1-7, m = 1-4), plus ``SWE_SHCOARSE``,
``SWE_ACQ_SEC``, ``SWE_ACQ_SUB``, ``SWE_SEQ``, ``SWE_NOM_FLAG``,
``SWE_OPS_FLAG``.

Cadence
-------

**[DOC]** "2 quarter-cycles are required for i-ALiRT measurements at all 8
energy steps. Since one criterion for defining the SWE i-ALiRT bidirectional
electron parameter will be the identification of counterstreaming electrons
over a range of energies, this means that the appropriate time cadence for SWE
i-ALiRT data will be **30 seconds**."

**[CODE]** ``process_swe()`` accumulates a full minute (``swe_seq`` 0-59, one
packet per second), then splits it into two halves - ``swe_seq`` 0-29 and
30-59 - and emits **two records per minute**. A group with any missing or
duplicate sequence number is skipped and logged.

The algorithm as implemented
----------------------------

.. code-block:: text

   60 accumulated 1-second packets
     |
     | drop groups where swe_nom_flag == 0
     | require swe_seq to be exactly 0..59
     |
     +-- first half  (swe_seq  0-29)  --+
     +-- second half (swe_seq 30-59)  --+
                                        |
     1. prepare_raw_counts     -> (8 energies, 7 CEMs, 30 phi bins)
     2. decompress_counts      -> 16-bit
     3. deadtime_correction    -> acq_duration hard-coded to 80 ms
     4. normalize_counts       -> * cal_factor / geometric_factor
     5a. sum over CEMs         -> azimuthal_check_counterstreaming
     5b. sum over azimuth      -> polar_check_counterstreaming
     6. BDE = max(azimuthal, polar)
     7. sum over CEMs and azimuth -> swe_normalized_counts (8 values)

Step 1 - building the (8, 7, 30) array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``prepare_raw_counts()``. Two lookups do all the work:

.. code-block:: python

   ENERGY_BINS = np.array([
       [1, 5, 7, 3],   # swe_seq  0-14  (Q1)
       [2, 6, 4, 0],   # swe_seq 15-29  (Q2)
       [3, 7, 5, 1],   # swe_seq 30-44  (Q3)
       [0, 4, 6, 2],   # swe_seq 45-59  (Q4)
   ])

   phi = [(12 + 24*seq) % 360,     # energy fields e1, e2
          (24 + 24*seq) % 360]     # energy fields e3, e4
   bin = ((phi - 12) // 12) % 30

The parity is the thing to notice: ``(12 + 24*seq)`` always maps to an **even**
bin index and ``(24 + 24*seq)`` to an **odd** one. Combined with
``ENERGY_BINS``, each of the 8 energies is written into **only even or only odd
bins**, 15 of the 30, within a half cycle. The other 15 stay zero.

This is not a defect - it reflects the instrument, where a given energy is
sampled in only the odd or only the even spin-angle bins of a quarter cycle -
and the peak-offset arithmetic below preserves parity, so the zeros are never
read.

Steps 2-4 - counts to normalized counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** "For i-ALiRT calculations, we do not need to convert counts to phase
space distribution or intensity, but do need to **normalize the counts based on
the geometric factors** for each CEM detector", and to apply the in-flight
calibration, using **the most recent factors** rather than an interpolation.

.. code-block:: text

   norm_counts[i][j][k] = ccounts[i][j][k] * cal_factor[j] / g[j]

**[CODE]**

* ``deadtime_correction(counts, 80 * 10**3)`` - ``ACQ_DURATION`` is not in the
  I-ALiRT packet, so the nominal 80 ms is hard-coded in microseconds.
* Calibration factor: ``searchsorted(cal_met, group_mid, side="right") - 1``,
  i.e. the last row at or before the midpoint of the half cycle. No
  interpolation, as specified.
* ``normalize_counts(counts, interp_cal)`` -
  ``counts * (interp_cal / GEOMETRIC_FACTORS)[:, np.newaxis]``, then clamps
  negatives to 0 (the equivalent of the heritage ``ccounts < 0`` guard, which
  is **absent** from the science L2 path).

.. note::

   **[DOC]** The I-ALiRT C fragment assigns a **different** geometric factor
   array (``435.0e-6, 599.0e-6, 808.0e-6, 781.0e-6, 876.0e-6, 548.0e-6,
   432.0e-6``) from the one in its own comment header and from the science
   ``fspace()`` fragment. **[CODE]** uses the single shared
   ``swe_constants.GEOMETRIC_FACTORS``, i.e. the SWE nominal set. The document
   is internally inconsistent here; the code's choice (one set of geometric
   factors for the instrument) is the defensible one, but it is worth
   confirming with SWE.

Step 5a - the azimuthal search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** Sum normalized counts over all CEMs, leaving counts as a function of
energy and azimuth. At each energy:

* Find the azimuth ``Apeak`` with maximum counts ``Cpeak``. **This direction is
  assumed to be the magnetic field direction.**
* ``C180`` = counts 180 degrees away, i.e. 15 bins - but since only every other
  bin is measured, use the **average of the two neighbours**: bins ``N+14`` and
  ``N+16``.
* ``C90`` = counts 90 degrees away, the average of ``N+6``, ``N+8``, ``N+22``
  and ``N+24``. All arithmetic mod 30.
* ``Cmin = min(C180, C90)``.
* **Bidirectional at this step if both ``Cpeak/Cmin`` and ``C180/Cmin`` exceed
  the threshold**, initially **1.75** from Genesis experience.

The physical reasoning: for unidirectional flow, both 90 and 180 degrees away
are low. For bidirectional flow, 180 degrees away is *also* high, and only the
90-degree directions are low.

**[CODE]** ``find_min_counts()`` computes **three** offset averages rather than
two:

.. code-block:: python

   counts_90     = average_counts(peak_bin, summed, ( 6,  8))
   counts_180    = average_counts(peak_bin, summed, (14, 16))
   counts_neg_90 = average_counts(peak_bin, summed, (-6, -8))
   cmin = np.min(np.hstack([counts_90, counts_180, counts_neg_90]), axis=1)

Note ``-6 mod 30 == 24`` and ``-8 mod 30 == 22``, so the code's
``counts_neg_90`` is the document's other half of ``A90``. The deviation is
that the code takes the **minimum of the two 90-degree sides separately**
rather than their average. That makes ``Cmin`` smaller or equal, so the ratios
are larger or equal, so the code is **slightly more willing to declare
bidirectional flow** than the document. All offsets are even, which preserves
the bin parity noted above.

``determine_streaming(cpeak, counts_180, cmin, threshold=1.75)`` then applies
the two-ratio test elementwise over the 8 energies.

Step 5b - the polar search
^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** "Depending on the direction of the interplanetary magnetic field, it
is possible that counterstreaming electrons could be observed in the polar
angle direction rather than the azimuthal angle direction. Thus if
counterstreaming is not observed in the above analysis, a second search will be
done in the other dimension."

Sum over azimuth instead of over CEMs, leaving counts as a function of energy
and CEM. Then ``Cmin`` = the mean of CEMs 3, 4 and 5 (the middle three), and
bidirectional is declared if both ``C_CEM1/Cmin`` and ``C_CEM7/Cmin`` exceed
1.75 - i.e. both **end** detectors are enhanced relative to the middle.

**[CODE]** ``polar_check_counterstreaming()``:
``summed[:, 2:5].mean(axis=1)`` for ``Cmin`` (zero-based indices 2, 3, 4 =
CEMs 3, 4, 5), then ``determine_streaming(summed[:, 0], summed[:, 6], cmin)``.

**[CODE]** The polar search runs **unconditionally**, not only when the
azimuthal search fails, and the results are combined with ``max()``. That is
logically equivalent to the document's "if not observed, do the second search",
and simpler to vectorize.

Step 6 - the BDE flag
^^^^^^^^^^^^^^^^^^^^^

**[DOC]** "The bidirectional electron parameter BDE will then be set to 1 if
bidirectional electrons are identified in **at least a minimum number of ESA
steps** ... with an initial value of **3 of the 8** ESA steps."

**[CODE]** ``compute_bidirectional(streaming_first, streaming_second,
min_esa_steps=3)`` sums each half's per-energy flags and compares to 3.
``bde_first_half = max(bde_first_search[0], bde_second_search[0])`` combines the
azimuthal and polar searches.

Step 7 - normalized counts product
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** "counts at each ESA step will be summed over both azimuthal and polar
angles ... this data product will have a 30 second cadence."

**[CODE]** ``normalized_first_half.sum(axis=(1, 2))`` -> 8 values, cast to
``int`` for the output record.

Output record
-------------

**[CODE]** Each half cycle appends a dict:

.. code-block:: python

   {
       "instrument": "swe",
       "swe_epoch": int(met_to_ttj2000ns(group_time_mid)),
       "swe_normalized_counts": [int(v) for v in summed],     # 8 values
       "swe_counterstreaming_electrons": bde,                  # 0 or 1
   }

merged with ``_populate_instrument_header_items(met)``.
``ialirt/utils/create_xarray.py`` writes these into the shared I-ALiRT dataset
against dimensions ``("swe_epoch", "swe_electron_energy")`` and
``("swe_epoch",)``, with dtypes ``int64`` and ``uint8``
(``ialirt/utils/constants.py``).

.. note::

   **[DOC]** section 3.4.5 warns: "if a SWE quarter-cycle is sufficiently
   longer than a spacecraft spin, it may be necessary to leave out the last
   azimuthal step of each quarter cycle in the analysis". **[CODE]** does not
   do this, and there is no flag or diagnostic that would tell you whether it
   has become necessary. That decision was explicitly deferred to flight
   experience.

Tests
-----

``imap_processing/tests/ialirt/unit/test_process_swe.py`` covers
``prepare_raw_counts``, ``decompress_counts``, ``normalize_counts``,
``find_bin_offsets``, ``average_counts``, ``find_min_counts``,
``determine_streaming``, ``compute_bidirectional`` and the end-to-end
``process_swe``.
