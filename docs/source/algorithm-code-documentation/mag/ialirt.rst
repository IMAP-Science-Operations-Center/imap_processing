.. _mag-ialirt:

I-ALiRT - Real-Time Space Weather Stream
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/ialirt/l0/parse_mag.py`` (the whole algorithm),
``imap_processing/ialirt/l0/mag_l0_ialirt_data.py`` (status bit decoders),
``imap_processing/ialirt/packet_definitions/ialirt_mag.xml``.

**Document:** section 7.4.

.. note::

   The MAG I-ALiRT code does **not** live under ``imap_processing/mag/``. It
   lives with the other instruments' I-ALiRT parsers, and imports the pieces it
   needs from the MAG modules (``TimeTuple``, ``calibrate_vector``,
   ``shift_time``, ``MagL1d.calculate_gradiometry_offsets``,
   ``MagL1d.apply_gradiometry_offsets``, ``ValidFrames``). Its output is a
   **list of dicts destined for the I-ALiRT database, not a CDF.**

What it is
----------

**[DOC]** IMAP provides near-real-time interplanetary magnetic field
measurements from upstream of the Earth for space weather forecasting. MAG
contributes a fixed **1 vector per sensor every 4 seconds**. The rate is not
configurable.

The algorithm is: decommutate the I-ALiRT-specific packet format, then reuse the
main science steps to get to an L1D-equivalent product.

.. code-block:: text

   1. Decommutate I-ALiRT data (one vector per sensor spread over 4 packets)
   2. Output a raw I-ALiRT data product
   3. Apply L1A steps  (gaps, convert sample times to MET/J2000)
   4. Apply L1B steps  (ENG calibration, time shift)
   5. Apply L1C steps  (gaps)             <- the document notes there are none
   6. Apply L1D steps  (calibration, truncation, offsets, gradiometry, magnitude)

Packet format
-------------

**[DOC]** ``MAG_SCI_IALIRT``:

.. list-table::
   :header-rows: 1
   :widths: 26 16 16 42

   * - Mnemonic
     - Bits
     - Start bit
     - Notes
   * - CCSDS primary header
     - 48
     - 0
     - ``PHVERNO``, ``PHTYPE``, ``PHSHF``, ``PHAPID``, ``PHGROUPF``,
       ``PHSEQCNT``, ``PHDLEN``
   * - ``SHCOARSE``
     - 32
     - 48
     -
   * - ``ACQ_TM_COARSE``
     - 32
     - 80
     - Science acquisition time, whole seconds
   * - ``ACQ_TM_FINE``
     - 16
     - 112
     - Science acquisition sub-second counter
   * - ``STATUS``
     - 24
     - 128
     - Mixed status and science; layout depends on the packet ID
   * - ``DATA``
     - 24
     - 152
     - Science payload
   * - ``CHECKSUM``
     - 16
     - 176
     -

**One science sample is spread over four sequential packets.** The leading two
bits of ``STATUS`` are a 0-3 packet ID identifying which of the four structures
this packet carries.

.. code-block:: text

   Primary   (MAGo) components: xx, yy, zz
   Secondary (MAGi) components: aa, bb, cc

   P1 = xx y
   P2 =  y zz
   P3 = aa b
   P4 =  b cc

Timestamps come from packets 0 and 2:

* Packet ID **0** carries ``ACQ_TM_COARSE``/``ACQ_TM_FINE`` for the
  **primary** sensor.
* Packet ID **2** carries them for the **secondary** sensor.

STATUS field layout
^^^^^^^^^^^^^^^^^^^

**[DOC]** Table 7-2. Each packet ID unpacks to a different set of fields.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Packet
     - Fields (after the 2-bit packet ID and 1 validity bit)
   * - 0
     - ``p1V5V``, ``p1V5C``, ``p1V8V``, ``p1V8C`` (2 bits each), ICU temp
       (7 bits), MAGo saturation flag, MAGi saturation flag, mode (4 bits)
   * - 1
     - ``p2V5V``, ``p2V5C`` (2 bits each), ``p3V3V`` (8 bits), ``p3V3C``
       (9 bits)
   * - 2
     - ``p8V5V``, ``p8V5C`` (2 bits each), ``n8V5V`` (8 bits), ``n8V5C``
       (9 bits)
   * - 3
     - MAGo temp (8 bits), MAGi temp (8 bits), MAGo range (2 bits), MAGi range
       (2 bits), multiple-bit error status

**[CODE]** These are decoded by ``Packet0`` .. ``Packet3`` in
``mag_l0_ialirt_data.py``. Several 2-bit fields in the document are decoded as
individual warning/danger flags in the code (``hk1v5_warn``,
``hk1v5_danger``, ...), and the 7/8/9-bit engineering values are left-shifted
back into a 16-bit scale (``icu_temp = ((status >> 6) & 0x7F) << 5``, and so on).

Validity
^^^^^^^^

**[DOC]** ``PRI_ISVALID = 1`` only if the validity bit is set in **both**
packets 0 and 1; ``SEC_ISVALID = 1`` only if set in **both** packets 2 and 3.

**[CODE]** ``pri_isvalid`` is read only from ``Packet1`` bit 21 and
``sec_isvalid`` only from ``Packet3`` bit 21. The document's AND across two
packets is not implemented. See :ref:`mag-implementation-status`.

Processing
----------

**[CODE]** ``process_packet(accumulated_data, engineering_calibration_dataset,
l1d_calibration_dataset)`` runs on roughly one minute of accumulated packets.

1. Group packets
^^^^^^^^^^^^^^^^

``get_pkt_counter`` extracts the 2-bit packet ID from ``mag_status >> 22``.
``find_groups`` collects runs of ``pkt_counter`` 0..3. A group whose counters are
not exactly ``[0, 1, 2, 3]`` is skipped and logged as an "incomplete group".
Groups with both ``pri_isvalid`` and ``sec_isvalid`` zero are dropped.

.. note::

   **[DOC]** says a missing packet within a group should NaN only the fields it
   would have carried. **[CODE]** drops the whole group.

2. Extract the vectors
^^^^^^^^^^^^^^^^^^^^^^

``extract_magnetic_vectors`` reassembles six 16-bit signed integers from the
four 24-bit ``DATA`` fields using the ``xxy | yzz | aab | bcc`` layout above.

3. Timestamps
^^^^^^^^^^^^^

``get_time`` builds ``TimeTuple(coarse, fine)`` for each sensor and converts with
``to_j2000ns()``, then applies the per-sensor ENG time shift via
``mag.l1b.mag_l1b.shift_time``. This is the L1A + L1B timing step reused
verbatim.

.. warning::

   The document says a second is split into **1/65535** fine-time units for
   I-ALiRT; ``TimeTuple`` uses ``MAX_FINE_TIME = 65536``. See
   :ref:`mag-implementation-status`.

4. L1B equivalent
^^^^^^^^^^^^^^^^^

``calculate_l1b`` uses ``retrieve_matrix_from_single_l1b_calibration`` to pull
``MFOTOURFO``/``OTS`` and ``MFITOURFI``/``ITS`` from a **single** engineering
calibration dataset (no ``MagAncillaryCombiner`` day selection), then calls
``mag.l1b.mag_l1b.calibrate_vector`` with the range from ``fob_range`` /
``fib_range`` in the STATUS field.

Invalid sensors are filled with ``-32768``.

5. L1D equivalent
^^^^^^^^^^^^^^^^^

``calibrate_and_offset_vectors`` applies ``URFTOORFO``/``URFTOORFI`` and the
per-(sensor, range) ``offsets`` from the L1D calibration file.

Despinning is then done **without SPICE attitude**, using the alternative the
document explicitly permits:

* ``sc_spin_phase`` (uint16 -> radians via ``2*pi/65535``),
  ``sc_inertial_right`` (``0.0055 deg`` per count) and ``sc_inertial_decline``
  (``0.0027 deg`` per count) come from the **spacecraft** I-ALiRT packet.
* ``interpolate_spherical`` converts RA/Dec into a unit Cartesian vector,
  ``CubicSpline`` interpolates each component to the vector's acquisition time
  (chosen over linear interpolation because the vector moves along a curved arc
  on the unit sphere), then converts back. Spin phase is unwrapped before
  ``np.interp`` and rewrapped modulo 360.
* ``transform_instrument_vectors_to_inertial`` produces an **ECLIPJ2000** vector.

``apply_gradiometry_correction`` then reuses
``MagL1d.calculate_gradiometry_offsets`` and
``MagL1d.apply_gradiometry_offsets`` with the ``gradiometer_factor`` from the
L1D calibration file - applied in ECLIPJ2000 rather than DSRF, since that is the
despun frame this path produces.

``transform_to_frames`` uses SPICE ``frame_transform`` from ``ECLIPJ2000`` to
``IMAP_GSE``, ``IMAP_GSM`` and ``IMAP_RTN``.

6. Clock angles
^^^^^^^^^^^^^^^

**[DOC]** section 7.4.5, added in Issue 5 Revision 2:

.. math::

   \theta_B = \arctan\!\left(\frac{B_z}{\sqrt{B_x^2 + B_y^2}}\right)

:math:`\theta_B = 0` in the x-y plane, :math:`+90` degrees along :math:`+z`, GSE
coordinates.

.. math::

   \varphi_B = \operatorname{atan2}(B_y,\, B_x)

Angles in the x-y plane, counterclockwise from the Earth-Sun line
(:math:`+x` axis), GSE coordinates.

**[CODE]** ``cartesian_to_spherical`` returns ``(r, azimuth, elevation)``. The
elevation is ``arcsin(z / |v|)``, which is mathematically identical to the
document's :math:`\theta_B`. The azimuth is ``arctan2(y, x)`` wrapped to
**[0, 360)** degrees, whereas the document's MATLAB ``atan2`` convention gives
**[-180, 180]**. Downstream consumers need to know which they are getting.

Angles are computed for **both** GSE and GSM.

Output
------

**[CODE]** A list of dicts, one per complete group (the first group is always
skipped because its attitude interpolation is extrapolated):

.. code-block:: text

   instrument         "mag"
   mag_epoch          TTJ2000 ns of the MAGo vector
   mag_B_GSE          [Bx, By, Bz] to 3 decimal places
   mag_B_GSM          [Bx, By, Bz]
   mag_B_RTN          [Br, Bt, Bn]
   mag_B_magnitude
   mag_phi_B_GSM      azimuth,   degrees
   mag_theta_B_GSM    elevation, degrees
   mag_phi_B_GSE      azimuth,   degrees
   mag_theta_B_GSE    elevation, degrees
   mag_hk_status      dict of ~30 decoded STATUS fields

Values are ``Decimal`` for database storage.

.. note::

   **[DOC]** step 2 of section 7.4 asks for a **raw I-ALiRT data product** to be
   written, named "MAG I-ALiRT RAW", with headers for the first/last sequence
   counter and any sequence gaps. **[CODE]** no raw product is produced; the
   pipeline goes straight to the L1D-equivalent dict. Sequence gaps
   (``PHSEQCNT``) are not tracked either. See
   :ref:`mag-implementation-status`.
