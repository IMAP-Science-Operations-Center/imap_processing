.. _mag-l1d:

L1D - Rapid Near-L2 Product
===========================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/mag/l1d/mag_l1d.py``,
``imap_processing/mag/l1d/mag_l1d_data.py``, with the shared machinery in
``imap_processing/mag/l2/mag_l2_data.py``.

**Document:** section 7.3.5. *Note: L1D was formerly called L2Pre, and the
workflow diagram in the document still labels it L2P.*

Why L1D exists
--------------

The MAG team's L2 offsets are determined **after the fact** from a full day of
data using the Leinweber method, and are delivered as a hand-checked ancillary
file. That takes time. Other instrument teams need a magnetic field product
*quickly*.

L1D produces a near-L2-quality product using only information available at
processing time: predicted per-range offsets from a slowly-varying calibration
file, plus two offset-estimation techniques computed on the fly - **spin
averaging** and **gradiometry**. It uses more processing steps than L2, not
fewer.

.. important::

   L1D is a **Level 1** product, produced in this repository, on purpose. It is
   not a stand-in for L2 and it is not a partial L2. L2 does **not** consume it.

Inputs
------

**[DOC/CODE]**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Input
     - Notes
   * - L1C normal-mode MAGo and MAGi
     - **Required.** ``mag_l1d`` raises ``ValueError`` without both.
   * - L1B burst-mode MAGo and MAGi
     - Optional. Both or neither; burst output is skipped if absent.
   * - ``l1d-calibration`` ancillary
     - See :ref:`mag-ancillary`.
   * - SPICE kernels
     - For the SRF/DSRF/GSE/RTN transforms and the spin data.
   * - SDC config ``ALWAYS_OUTPUT_MAGO``
     - Default ``True``.

Datasets are routed by the last token of ``Logical_source``
(``norm-magi``, ``norm-mago``, ``burst-magi``, ``burst-mago``); anything else
raises.

Configuration read from the calibration file
--------------------------------------------

**[CODE]** ``MagL1dConfiguration`` (``mag_l1d_data.py``) selects the day with
``sel(epoch=day)`` and exposes:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Attribute
     - Shape
     - Meaning
   * - ``mago_calibration``
     - ``(3, 3, 4)``
     - ``URFTOORFO`` - URF -> ORF per range, MAGo.
   * - ``magi_calibration``
     - ``(3, 3, 4)``
     - ``URFTOORFI`` - URF -> ORF per range, MAGi.
   * - ``calibration_offsets``
     - ``(2, 4, 3)``
     - ``offsets``, indexed ``[sensor, range, axis]`` where sensor
       ``0 = MAGo``, ``1 = MAGi``. In nT, in ORF.
   * - ``spin_count_calibration``
     - scalar
     - ``number_of_spins`` - spins per averaging chunk. Nominally **240**
       (~1 hour).
   * - ``spin_average_application_factor``
     - scalar
     - How much of the computed spin offset to apply, in ``[-1, 1]``.
   * - ``quality_flag_threshold``
     - scalar
     - Gradiometer offset magnitude above which data is flagged.
   * - ``gradiometer_factor``
     - ``(3, 3)``
     - Kappa. **A matrix, not a scalar** - Issue 5 Revision 1 of the document
       changed this specifically so each axis can be treated independently and
       a small rotation of the gradiated offset can be absorbed.
   * - ``apply_gradiometry``
     - bool
     - Set ``False`` by ``mag_l1d`` when the MAGo L1C file's
       ``all_vectors_primary`` attribute is falsy.

Processing steps
----------------

**[CODE]** All of this happens in ``MagL1d.__post_init__``, so constructing the
dataclass *is* running the algorithm.

.. code-block:: text

   frame = MAGO
   truncate_to_24h(day)                     # strip the 30-minute buffers
   calibrate + apply per-range offsets      # URF -> ORF
   rotate_frame(SRF)                        # SPICE
   calculate_spin_offsets()                 # NORM only; reused for BURST
   apply_spin_offsets()                     # to MAGo and MAGi
   rotate_frame(DSRF)                       # SPICE despin
   calculate_gradiometry_offsets()          # if apply_gradiometry
   apply_gradiometry_offsets()
   magnitude = |B|

1. Truncate to 24 hours
^^^^^^^^^^^^^^^^^^^^^^^

``MagL2L1dBase.truncate_to_24h``. Removes the 30-minute buffers. Raises if
nothing remains.

2. Calibration and per-range offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``_calibrate_and_offset_vectors``. Range is re-attached as a fourth component,
``MagL2L1dBase.apply_calibration`` applies the per-range 3x3 matrix (URF ->
ORF), then ``apply_calibration_offset_single_vector`` **adds**
``offsets[sensor, range, :]``. Applied to both sensors.

Quality flags for the L1D science product are hardcoded to ``0``, which the
document asks for explicitly so that L1D and L2 have the same shape.

3. Rotate to SRF
^^^^^^^^^^^^^^^^

``MagL1d.rotate_frame`` overrides the base implementation to rotate **both**
sensors. It is careful about one thing: ``self.frame`` always describes the MAGo
data, so when the starting frame is ``MAGO``, the MAGi vectors are rotated from
``MAGI`` instead. Setting ``self.frame = MAGI`` raises.

FILLVAL and NaN entries survive rotation - they are masked back to FILLVAL after
``frame_transform``. ``allow_spice_noframeconnect=True`` lets the transform
degrade gracefully when a frame chain is unavailable.

.. _mag-spin-averaging:

4. Spin-average offsets
^^^^^^^^^^^^^^^^^^^^^^^

``calculate_spin_offsets``. **Only meaningful in SRF and only on normal-mode
data.**

The physical idea: in a spin-aligned frame the spacecraft-generated field is
static in the two spin-plane axes while the ambient field rotates, so averaging
over a whole number of spins leaves the spacecraft contribution behind. The
third (spin-aligned) axis gets no such benefit and is left alone.

Implementation:

* ``spice.spin.get_spacecraft_spin_phase(met)`` gives per-vector spin phase.
  Vectors where the phase is NaN are marked NaN.
* Spin starts are where the phase decreases (wrap through zero) **plus** every
  NaN-to-number transition, so gaps do not merge two spins.
* For a NaN gap longer than one median spin period (from
  ``spice.spin.get_spin_data()``), synthetic spin starts are inserted so the
  spin count stays honest across the gap.
* Spins are grouped into chunks of ``spin_count_calibration``. The mean of the
  x and y components is taken per chunk with ``np.nanmean``.
* A chunk is **rejected** and the previous chunk's averages reused if more than
  half of x or y is NaN, or if it contains fewer than half the expected samples.

The result is an ``xr.Dataset`` with ``epoch``, ``x_offset``, ``y_offset``,
``validity_start_time``, ``validity_end_time``, ``start_spin_counter``,
``end_spin_counter`` - written out as the ``imap_mag_l1d_spin-offsets``
ancillary product.

5. Apply spin offsets
^^^^^^^^^^^^^^^^^^^^^

``apply_spin_offsets``. For each chunk interval, subtract
``offset * spin_average_application_factor`` from the x and y components of
every vector in that interval; z is copied through unchanged. The first chunk
catches everything before its own start time and the last chunk catches
everything after, so no vector is left unassigned.

The same offsets object computed from normal mode is passed into the burst-mode
``MagL1d``, which is why ``mag_l1d`` constructs norm first.

6. Despin
^^^^^^^^^

``rotate_frame(ValidFrames.DSRF)`` - SPICE ``IMAP_DPS``.

.. _mag-gradiometry:

7. Gradiometry
^^^^^^^^^^^^^^

**[DOC]** section 7.2.1. Far from any magnetic source the field looks like a
dipole, so a sensor further from a time-varying source sees a smaller variation.
MAGi is closer to every spacecraft source than MAGo, so the difference between
them estimates the spacecraft contribution:

.. math::

   \mathbf{B}_O(t) \;\leftarrow\; \mathbf{B}_O(t) - \mathrm{K}\,
   \bigl(\mathbf{B}_I(t) - \mathbf{B}_O(t)\bigr)

MAGi and MAGo are not sampled simultaneously and MAGo usually has the higher
cadence, so **MAGi must be interpolated onto the MAGo timeline first**.

It is possible that a value of :math:`\mathrm{K} = 0` will be used in flight,
which disables the correction without a code change.

**[CODE]** ``calculate_gradiometry_offsets``:

* Uses ``interpolation_methods.linear`` with ``extrapolate=True`` to put MAGi on
  the MAGo epochs. There is a ``# TODO: should this extrapolate or should
  non-overlapping data be removed?`` at ``mag_l1d_data.py:725``.
* ``diff = aligned_magi - mago`` per axis.
* Records the magnitude of ``diff`` and a quality flag
  ``magnitude > quality_flag_threshold``.
* Emits the ``imap_mag_l1d_gradiometry-offsets-{norm,burst}`` ancillary
  datasets with ``gradiometer_offsets``, ``gradiometer_offset_magnitude`` and
  ``quality_flags``.

``apply_gradiometry_offsets`` computes ``np.dot(offset, K)`` per vector and
subtracts it.

.. warning::

   ``np.apply_along_axis(np.dot, 1, offset_value, gradiometer_factor)`` computes
   the **row-vector product** :math:`\mathbf{o}^{T} \mathrm{K}`, i.e.
   :math:`\mathrm{K}^{T}\mathbf{o}`, not :math:`\mathrm{K}\mathbf{o}`. If
   ``gradiometer_factor`` is symmetric or diagonal this is immaterial, but the
   convention needs confirming with the MAG team before any off-diagonal kappa
   is delivered. See :ref:`mag-implementation-status`.

Gradiometry is skipped entirely when MAGo was not the primary sensor for all
vectors - the document requires this, and ``mag_l1d`` enforces it via
``all_vectors_primary``.

8. Magnitude and output
^^^^^^^^^^^^^^^^^^^^^^^

``magnitude = np.linalg.norm(vectors, axis=1)`` over the three components.

``mag_l1d`` then walks the frames, calling ``rotate_frame`` then
``generate_dataset`` for SRF, DSRF, GSE and RTN in that order. **This mutates
the dataclass in place**, so each rotation starts from the previous frame and
the order is not arbitrary.

``MagL1d.generate_dataset`` overrides the base method to swap in the MAGi
vectors, epochs and ranges when ``ALWAYS_OUTPUT_MAGO`` is ``False``, then
restores them.

Output variables per science file: ``b_srf`` / ``b_dsrf`` / ``b_gse`` /
``b_rtn`` (from ``ValidFrames.var_name``), ``quality_flags``,
``quality_bitmask``, ``range``, ``magnitude``. Range is a **separate time
series**, not a fourth vector component, as the document requires.

Outputs
-------

**[DOC]** expects:

* 2 spin-average offset CDFs (MAGo and MAGi), normal mode
* 2 gradiometer offset CDFs (normal and burst)
* L1D burst in DSRF, SRF, RTN, GSE
* L1D normal in DSRF, SRF, RTN, GSE

**[CODE]** produces all of these except that **only one spin-offsets file is
produced**, computed from MAGo and applied to both sensors. See
:ref:`mag-implementation-status`.

Ancillary datasets are written by ``Mag.post_processing`` in ``cli.py``, which
intercepts the three ancillary ``Logical_source`` values, generates an
``AncillaryFilePath`` filename, and calls ``xarray_to_cdf`` with
``istp=False`` and ``terminate_on_warning=False``. Failures are logged and
swallowed - an ancillary file will never fail the run.
