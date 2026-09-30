.. _mag-ancillary:

Ancillary and Calibration Files
===============================

**Document:** section 7.2.

Everything the MAG team delivers to the SDC arrives as a **CDF ancillary file**.
The document is explicit that **the filename must never be needed to determine
any property of the data inside** - all validity information lives in the file's
data or metadata. It is also explicit that **if two calibration files are valid
for the same data point, the most recently generated file wins**.

The IMAP magnetometer requires dynamic calibration to remove the magnetic field 
of the spacecraft. The calibration matrices should be applied based on sensor 
and range and the offsets added based on sensor and range. The gradiometer factor 
should be used to calculate time varying offsets via gradiometry during the 
calibration process, and the spin averaging factors should be used to calculate 
time varying offsets.

How ancillary files are loaded
------------------------------

**[CODE]** ``MagAncillaryCombiner`` in
``imap_processing/ancillary/ancillary_dataset_combiner.py`` - a thin subclass of
``AncillaryCombiner`` with no MAG-specific behaviour.

The base class assumes ancillary CDFs have **no time-varying variables inside
them**: a file is valid from the first second of its start date to the last
second of its end date. It takes a collection of files (different dates,
different versions), works out the total span, and produces a **single combined
dataset with one ``epoch`` entry per day**, where each day points at the values
from the file that is valid for that day. That is why every MAG level does
``calibration_dataset.sel(epoch=day)``.

Two details worth remembering:

* Ancillary files can have **no end date**, so the constructor requires an
  ``expected_end_date``. ``cli.py`` passes ``start_date + 3 days`` for MAG.
* Validity dates come from the ``AncillaryFilePath`` filename, and version
  ordering resolves overlaps. This is the one place the filename *is* used, and
  it is a repository-wide convention rather than a MAG choice.

.. _mag-eng-calibration:

Engineering (ENG) calibration - ``l1b-calibration``
---------------------------------------------------

Consumed by :ref:`mag-l1b` and by the I-ALiRT path.

Derived from **ground calibration** at the Magnetsrode facility of TU
Braunschweig, folding together the nominal scale factor, the gain (sigma) and
orthogonality (omega) matrix, and the Measurement Frame -> Unit Reference Frame 
rotation.

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - Variable
     - Shape
     - Meaning
   * - ``MFOTOURFO``
     - ``(epoch, 3, 3, 4)``
     - MAGo measurement frame -> unit reference frame, per range.
   * - ``MFITOURFI``
     - ``(epoch, 3, 3, 4)``
     - MAGi measurement frame -> unit reference frame, per range.
   * - ``OTS``
     - ``(epoch,)``
     - MAGo time shift, **seconds**.
   * - ``ITS``
     - ``(epoch,)``
     - MAGi time shift, seconds.

**[DOC]** the validity of an ENG calibration file is specified in
**pre-timeshift** values, and multiple ENG calibrations may apply to
non-overlapping ranges within one processing window.

A fallback copy, ``imap_mag_l1b-calibration_20240229_v002.cdf``, is bundled in
``imap_processing/mag/l1b/`` and used only when ``mag_l1b`` is called with
``calibration_dataset=None``. Do not rely on it outside tests.

Calibration matrices (in-flight) - ``l2-calibration``
------------------------------------------------------

Consumed by :ref:`mag-l2`. **[DOC]** generated approximately **monthly**;
nothing inside varies with time.

Document contents:

* Start and end time of validity
* Generation time
* **Time shift, in seconds, for each sensor**
* For each sensor, for each range, a dimensionless **3x3 matrix T** transforming
  from the L1A data frame to the (spinning) spacecraft frame
* Additional metadata (TBD)

**[CODE]** variables actually read:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Variable
     - Shape
     - Meaning
   * - ``URFTOORFO``
     - ``(epoch, 3, 3, 4)``
     - MAGo URF -> ORF, per range.
   * - ``URFTOORFI``
     - ``(epoch, 3, 3, 4)``
     - MAGi URF -> ORF, per range.

The per-sensor time shift described in the document is **not** read from this
file at L2; the per-vector ``timedeltas`` in the offsets file serves that role
instead.

Offsets - ``l2-norm-offsets`` and ``l2-burst-offsets``
-------------------------------------------------------

Consumed by :ref:`mag-l2`. **[DOC]** generated **daily**. These are the output of
the MAG team's spacecraft-field removal and offset determination. How Imperial
produces them is described in :ref:`mag-cmad`.

Document contents, **for each sensor and for every timestamped vector in the L0
data**:

* A **3x1 offset matrix H** in nT to be removed. May be NaN when no good offset
  can be determined; in the CDF, invalid values use ``FILLVAL`` placed outside
  ``VALIDMIN``/``VALIDMAX`` so the intent is unambiguous.
* A **quality flag** (see :ref:`mag-quality-flags`).
* A **quality bitmask**, with some bits reserved for in-flight calibration.
  Bit definitions are in :ref:`mag-quality-flags`.
* A **Delta-T** in +/- milliseconds adjusting the vector timestamp.
* Additional metadata (TBD) that may include time-varying information about how
  the calibration was determined - for example when interference was occurring
  and was removed. **This metadata should be copied into the final science
  file**, so that the MAG team can annotate products with calibration steps that
  are not yet defined.

**[CODE]** variables read: ``epoch``, ``offsets`` ``(n, 3)``, ``timedeltas``
``(n,)`` in seconds, ``quality_flag`` ``(n,)``, ``quality_bitmask`` ``(n,)``.

.. important::

   The offsets file also carries a ``Parents`` global attribute naming the
   **exact L1B/L1C files** the offsets were generated against.
   ``retrieve_mag_l1_inputs_from_l2_offsets`` downloads those files and L2 uses
   them, ignoring anything passed in as a dependency. Epochs must match exactly
   or ``mag_l2`` raises.

L1D calibration - ``l1d-calibration``
--------------------------------------

Consumed by :ref:`mag-l1d` and by the I-ALiRT path.

**[DOC]** a **single file** whose offsets must be estimated *before the fact*,
making them considerably less precise than the L2 offsets. Contents:

* Start and end time of validity; generation time
* For each sensor, for each range, a dimensionless **3x3 matrix T** (L1A frame ->
  spinning spacecraft frame)
* For each sensor, for each range, a **3x1 offset matrix H** in nT
* A **gradiometer factor K**
* Additional metadata (TBD)

**[CODE]** ``MagL1dConfiguration`` reads:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Variable
     - Shape
     - Meaning
   * - ``URFTOORFO``
     - ``(3, 3, 4)``
     - MAGo URF -> ORF per range. Read with the shared
       ``retrieve_matrix_from_l2_calibration``.
   * - ``URFTOORFI``
     - ``(3, 3, 4)``
     - MAGi URF -> ORF per range.
   * - ``offsets``
     - ``(2, 4, 3)``
     - ``[sensor, range, axis]``; sensor ``0 = MAGo``, ``1 = MAGi``. nT, ORF.
   * - ``number_of_spins``
     - scalar
     - Spin Count Calibration Value. Nominally 240 spins (~1 hour).
   * - ``spin_average_application_factor``
     - scalar
     - How much of the computed spin offset to apply, in ``[-1, 1]``.
   * - ``quality_flag_threshold``
     - scalar
     - Gradiometer offset magnitude above which data is flagged.
   * - ``gradiometer_factor``
     - ``(3, 3)``
     - Kappa.

.. note::

   **Kappa is a 3x3 matrix, not a scalar.** The document's section 7.2.1 still
   writes the gradiometer equation with a scalar :math:`\mathrm{K}`, but Issue 5
   Revision 1 changed L1D specifically to a matrix so each axis can be handled
   independently and a small rotation of the measured gradiated offset can be
   absorbed. Section 7.3.5 and the code both use the matrix form.

   It is possible that :math:`\mathrm{K} = 0` will be used in flight, which
   disables gradiometry without a code change.

SDC configuration (not an ancillary file)
------------------------------------------

**[CODE]** ``imap_processing/mag/imap_mag_sdc_configuration_v001.py``:

.. code-block:: python

   L1C_INTERPOLATION_METHOD = "linear_filtered"
   ALWAYS_OUTPUT_MAGO = True

The algorithm document is explicit that the interpolation method is a **software
configuration variable and not a MAG-provided input file**, which is why this is
a Python module in this repository rather than an ancillary CDF. Changing
``ALWAYS_OUTPUT_MAGO`` also requires dependency-system changes so MAGi files
become upstream dependencies of L2.

Ancillary files produced *by* this repository
----------------------------------------------

**[CODE]** L1D emits three ancillary CDFs alongside its science products. They
are written by ``Mag.post_processing`` in ``cli.py`` with ``istp=False`` and
``terminate_on_warning=False``, bypassing ``write_cdf``; any failure is logged
and swallowed.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``Logical_source``
     - Contents
   * - ``imap_mag_l1d_spin-offsets``
     - ``epoch``, ``x_offset``, ``y_offset``, ``validity_start_time``,
       ``validity_end_time``, ``start_spin_counter``, ``end_spin_counter``. One
       row per spin-averaging chunk (nominally ~1 hour).
   * - ``imap_mag_l1d_gradiometry-offsets-norm``
     - ``epoch``, ``gradiometer_offsets`` ``(n, 3)``,
       ``gradiometer_offset_magnitude``, ``quality_flags``. One row per MAGo
       vector.
   * - ``imap_mag_l1d_gradiometry-offsets-burst``
     - Same, for burst mode.

**[DOC]** section 7.3.5 expects **two** spin-offset files, one per sensor.
Only one is produced. See :ref:`mag-implementation-status`.
