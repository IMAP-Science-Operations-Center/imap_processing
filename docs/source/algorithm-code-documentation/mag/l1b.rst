.. _mag-l1b:

L1B - Engineering Calibration
=============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/mag/l1b/mag_l1b.py``.

**Document:** section 7.3.3.

L1B is the smallest step in the pipeline. **One L1A product maps to exactly one
L1B product**, and the four sensor/mode combinations are processed
independently. Only the vectors and the timestamps change; every other variable
and attribute is carried through unmodified.

Inputs
------

**[DOC/CODE]** The **engineering (ENG) calibration** file supplies:

* **8 x 3x3 matrices** - one per (sensor, range) - transforming from the
  measurement frames (MFO, MFI) to the unit reference frames (URFO, URFI).
* **2 time shifts** - one per sensor, in seconds.

The time validity of the calibration file is specified as **pre-timeshift**
values.

**[CODE]** Variable names in the calibration CDF:

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Variable
     - Shape
     - Meaning
   * - ``MFOTOURFO``
     - ``(epoch, 3, 3, 4)``
     - MAGo MF -> URF, indexed ``[:, :, range]`` after day selection.
   * - ``MFITOURFI``
     - ``(epoch, 3, 3, 4)``
     - MAGi MF -> URF.
   * - ``OTS``
     - ``(epoch,)``
     - MAGo time shift, seconds.
   * - ``ITS``
     - ``(epoch,)``
     - MAGi time shift, seconds.

The file is combined across multiple ancillary inputs by
``MagAncillaryCombiner`` so that every day in the range has an entry, then
``retrieve_matrix_from_l1b_calibration`` does
``calibration_dataset.sel(epoch=day)``.

**[CODE]** If ``calibration_dataset`` is ``None``, ``mag_l1b`` falls back to the
bundled ``imap_processing/mag/l1b/imap_mag_l1b-calibration_20240229_v002.cdf``
and logs "Using default test calibration file." This is a **test convenience**;
the CLI always supplies a real file.

Processing steps
----------------

1. Select the calibration
^^^^^^^^^^^^^^^^^^^^^^^^^

Sensor is determined from the ``Logical_source`` string (``"mago"`` /
``"magi"``); a raw L1A file raises ``ValueError``. The output logical source is
the input with ``l1a`` replaced by ``l1b``.

.. warning::

   **[DOC]** says the valid ENG calibration may span **multiple files applying
   to non-overlapping time ranges within the processing window**, and that when
   two files are valid for the same vector the most recently generated one wins.

   **[CODE]** applies exactly one calibration matrix and one time shift to the
   whole day, chosen by ``sel(epoch=day)``. There is a
   ``# TODO: Check validity of time range for calibration`` at
   ``mag_l1b.py:73``. See :ref:`mag-implementation-status`.

2. Rescale compressed vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rescale_vector``. Vectors that came from compressed packets with a
``COMPRESSION_WIDTH`` other than 16 bits must be rescaled to 16-bit-equivalent
engineering values:

.. math::

   M = 2^{16 - \text{width}}, \qquad \mathbf{v}' = M \cdot \mathbf{v}

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Width
     - M
     - Example
   * - 14
     - 4
     -
   * - 15
     - 2
     - ``[32766, -2, 1]`` -> ``[65532, -4, 2]``
   * - 16
     - 1
     - unchanged
   * - 18
     - 1/4
     - ``[10, -2000, 0]`` -> ``[2.5, -500, 0]``

The result **must be floating point** - widths above 16 produce fractional
values, widths below 16 can overflow 16-bit integers. The code casts to
``np.float64`` and uses ``np.float_power``.

Uncompressed vectors (``compression_flags[0] == 0``) are returned unchanged.

3. Transform to URF
^^^^^^^^^^^^^^^^^^^

``calibrate_vector``. For each vector, take the **range** from the fourth
component, select ``calibration_matrix[:, :, range]``, and

.. math::

   \mathbf{v}_{URF} = T_{\text{sensor},\,\text{range}} \; \mathbf{v}_{MF}

Data is now in **nT**. The range component is preserved untouched in position 3.
A non-integer range raises ``ValueError``.

Steps 2 and 3 are combined in ``update_vector`` and applied with
``xr.apply_ufunc(..., vectorize=True)`` over the ``direction`` and
``compression`` core dimensions.

4. Apply the time shift
^^^^^^^^^^^^^^^^^^^^^^^

``shift_time``. One value per sensor, in seconds, applied to every vector for
the whole validity period:

.. code-block:: python

   time_shift_ns = np.int64(round(time_shift.item() * 1e9))
   shifted = epoch_times + time_shift_ns

Positive shifts move times **forward**, negative shifts backward, zero is a
no-op. A time shift with more than one element raises ``ValueError``.

This can move vectors across the day boundary, which is precisely why the L1
files carry the 30-minute buffer on each side.

``timeshift_vectors_per_second`` applies the same shift to the timestamps
embedded in the ``vectors_per_second`` global attribute string, so L1C's cadence
lookup stays aligned with the shifted epochs.

5. Export
^^^^^^^^^

Output variables are ``vectors`` (unchanged shape, now nT + range) and
``compression_flags`` (copied verbatim). Global attributes ``is_mago``,
``is_active``, ``all_vectors_primary``, ``vectors_per_second`` and
``missing_sequences`` are propagated; a missing one is logged at INFO level and
skipped rather than raising.

.. note::

   **[DOC]** requires the ENG calibration file to appear in the output CDF's
   ``Parents`` header. **[CODE]** this is handled generically by
   ``ProcessInstrument.post_processing`` in ``cli.py`` from the dependency list,
   not inside ``mag_l1b``.

Validation
----------

``imap_processing/tests/mag/test_mag_validation.py::test_mag_l1b_validation``
covers cases T009-T012 in
``imap_processing/tests/mag/validation/L1b/``. Each case has an input CSV, a
per-sensor expected output CSV, and (for T012) a bespoke calibration CDF.
