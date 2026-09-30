.. _lo-l1a:

Level 1A - Packet Decommutation
===============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Module:** ``imap_processing/lo/l1a/lo_l1a.py``, with the real unpacking in
``imap_processing/lo/l0/``.

L1A is deliberately dumb: it turns CCSDS packets into CDF variables with the
same names and the same values. **No calibration, no engineering-unit
conversion, no science.** The only transformations allowed are the ones needed
to recover what the instrument actually measured before it compressed it for
downlink. This code is mature and, with the spacecraft launched, unlikely to
change much.

Read this page when you are debugging a value that looks wrong at L1A, or when
you need to know the shape or meaning of a raw field.

The three pieces of real logic
------------------------------

Everything else at L1A is field extraction driven by the XTCE definition.

1. **Log decompression via lookup tables** - science counts and star sensor.
2. **Direct event bit unpacking** - variable-length, case-dependent.
3. **Segmented packet reassembly** - a science direct event record can span
   multiple CCSDS packets.

1. Log decompression
--------------------

**[CODE]** ``imap_processing/lo/l0/utils/bit_decompression.py``

Counters are compressed on board with a pseudo-logarithmic scheme and expanded
at L1A by table lookup. Three tables ship with the package:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - ``Decompress`` enum
     - Table file
     - Used for
   * - ``DECOMPRESS8TO12``
     - ``log10_8_to_12_bit_uncompress.csv``
     - 8-bit counters expanded to 12 bits
   * - ``DECOMPRESS8TO16``
     - ``log10_8_to_16_bit_uncompress.csv``
     - 8-bit TOF / triple / histogram counters
   * - ``DECOMPRESS12TO16``
     - ``log10_12_to_16_bit_uncompress.csv``
     - 12-bit singles and position counters

The tables are loaded once into ``DECOMPRESSION_TABLES``. Decompression is a
single fancy-index:

.. code-block:: python

   data = decompression_lookup[decompression]
   decompressed = data[compressed_values, 2]   # column 2 is the bin MEAN

.. note::

   Column 2 is the **mean** of the compression bin. The tables also carry the
   bin minimum and maximum, which are not used. If someone ever needs a
   compression-induced uncertainty, it comes from those columns.

2. Science counts (APID 705) layout
------------------------------------

**[CODE]** ``lo_science.HIST_DATA_META`` defines the whole packet as a
concatenation of fixed-length bit sections. Shapes are ``(esa_step, azimuth)``
per packet, and the packet dimension is prepended.

.. list-table::
   :header-rows: 1
   :widths: 30 12 14 14 30

   * - Fields
     - bits
     - section bits
     - shape
     - Meaning
   * - ``start_a``, ``start_c``, ``stop_b0``, ``stop_b3``
     - 12
     - 504
     - (7, 6)
     - Singles rates on each anode
   * - ``tof0_count`` .. ``tof3_count``
     - 8
     - 336
     - (7, 6)
     - Double coincidences within threshold
   * - ``disc_tof0`` .. ``disc_tof3``
     - 8
     - 336
     - (7, 6)
     - Events discarded for being outside threshold
   * - ``pos0`` .. ``pos3``
     - 12
     - 504
     - (7, 6)
     - Position counters, derived on board from a TOF3 lookup
   * - ``tof0_tof1``, ``tof0_tof2``, ``tof1_tof2``, ``silver``
     - 8
     - 3360
     - (7, 60)
     - Triple coincidences
   * - ``hydrogen``, ``oxygen``
     - 8
     - 3360
     - (7, 60)
     - On-board species histograms

So: **7 ESA steps** always; **6 spin bins of 60 degrees** for the monitor
rates, **60 spin bins of 6 degrees** for triples and the species histograms.
One packet covers one science cycle (~420 s).

3. Direct events (APID 706)
---------------------------

This is the only genuinely intricate parsing in Lo L0/L1A.

**[CODE]** ``lo_science.parse_events`` / ``parse_packet_events``, with the
layout tables in ``lo/l0/decompression_tables/decompression_tables.py``.

Every event begins with fixed fields, then carries only the variable fields
that its case and mode call for:

.. code-block:: python

   FIXED_FIELD_BITS    = FixedFields(coincidence_type=4, de_time=12,
                                     esa_step=3, mode=1)
   VARIABLE_FIELD_BITS = VariableFields(tof0=10, tof1=9, tof2=9,
                                        tof3=6, cksm=4, pos=2)
   PACKET_FIELD_BITS   = PacketFields(passes=32)

``CASE_DECODER`` is keyed by ``(case, mode)`` and says which variable fields
are present:

.. code-block:: python

   # (case, mode): VariableFields(tof0, tof1, tof2, tof3, cksm, pos)
   (0,  1): (True,  False, True,  True,  True,  False)   # golden triple
   (0,  0): (True,  True,  True,  True,  False, False)
   (1,  0): (True,  True,  True,  False, False, False)
   (2,  0): (True,  True,  False, False, False, True)
   (3,  0): (True,  False, False, False, False, False)
   (4,  1): (True,  False, True,  False, False, True)
   (4,  0): (True,  False, False, True,  False, False)
   (5,  0): (True,  False, True,  False, False, False)
   (6,  1): (True,  False, False, False, False, True)
   (6,  0): (True,  False, False, True,  False, False)
   (7,  0): (True,  False, False, False, False, False)
   (8,  0): (False, True,  True,  False, False, True)
   (9,  0): (False, True,  True,  False, False, False)
   (10, 1): (False, True,  False, False, False, True)
   (10, 0): (False, True,  False, True,  False, False)
   (11, 0): (False, True,  False, False, False, False)
   (12, 1): (False, False, True,  False, False, True)
   (12, 0): (False, False, True,  True,  False, False)
   (13, 0): (False, False, True,  False, False, False)

Cases 14 and 15 never appear: 15 is impossible, 14 is the common single, which
is not telemetered.

**Bit shift.** Every TOF and checksum value is transmitted with the low bit
dropped, so unpacked values must be shifted left by one. ``DE_BIT_SHIFT``
encodes this: ``tof0..tof3`` and ``cksm`` shift by 1, ``pos`` by 0. Forgetting
this halves every TOF.

**Coincidence type semantics.** The 4-bit ``coincidence_type`` (called
``ABSENT`` in the document) is a bitmask of which TOF channels are *missing*.
Case 0 with the checksum present is the **golden triple**, the cleanest event
class. Cases that keep three TOFs are triples; cases that keep one or two are
doubles. **[DOC]**

**Checksum and TOF3 recovery.** **[DOC]** For the golden triple, TOF3 is not
transmitted; it is reconstructed from the checksum:

.. math::

   \mathrm{TOF3} = \mathrm{CKSM} - \mathrm{TOF0} + \mathrm{TOF1} + \mathrm{TOF2}

and the recovered TOF3 is then compared against per-anode threshold bands to
assign ``pos`` in 0..3. The threshold bands are supposed to come from a lookup
table. **[CODE]** In ``lo_l1b.py`` the left checksum boundary is currently
hardcoded to ``-21`` with a TODO to read it from the LUT when available.

4. Segmented packet reassembly
------------------------------

**[CODE]** ``lo_science.combine_segmented_packets`` and ``find_valid_groups``.
A science cycle's direct events can exceed the maximum CCSDS packet size, so
they arrive as a sequence of segments. The code:

1. Uses the CCSDS grouping flags to find segment starts and ends.
2. Validates that sequence counters within a group are strictly sequential
   (``is_sequential``, shared with HIT).
3. Concatenates the raw bytes of a valid group and takes the MET from the
   first packet.
4. Drops groups that fail validation.

5. Star sensor (APID 707)
-------------------------

**[CODE]** ``lo_star_sensor.process_star_sensor``. 720 samples per spin,
transmitted 8-bit, expanded to 12-bit through ``DECOMPRESS8TO12``. That is all
that happens at L1A; the corrections live at L1B ``prostar``.

6. Spin data (APID 708)
-----------------------

**[CODE]** ``lo_science.organize_spin_data``. One packet carries **28 spins**
with 7 fields each: start time seconds and subseconds, ESA positive and
negative DAC settings, validity of period and phase, and source. The packet
itself carries the number of spins completed and the acquisition end time.

This is the source of truth for real spin durations, which the exposure-time
calculations at L1B depend on. Do not substitute the nominal 15 s.

Housekeeping (APIDs 676, 677, 725)
----------------------------------

Straight field extraction from the XTCE. NHK has roughly 157 fields, SHK
roughly 27. They are written raw at L1A as ``imap_lo_l1a_nhk`` /
``imap_lo_l1a_shk``, and again in engineering units as ``imap_lo_l1b_nhk`` /
``imap_lo_l1b_shk``.

**[DOC]** The engineering-unit conversion is the standard linear range map:

.. math::

   \mathrm{value_{EU}} = \mathrm{range_{min}}
       + \mathrm{raw} \cdot
         \frac{\mathrm{range_{max}} - \mathrm{range_{min}}}{2^{b}}

where :math:`b` is the field's bit width. Example from the document:
``cdh_12vp`` is 12 bits over [-20, +20] V, so
``value = -20 + raw * 0.009765625``.

Validation expectations
-----------------------

**[DOC]** Stated L1A checks:

* Packets grouped by APID and sorted by MET; reject mismatched APIDs.
* Only packets inside the pointing window are accepted.
* Sequence counters monitored for gaps and duplicates.
* Direct events: ``coincidence_type`` must be 0..13; ``esa_step`` must be 0..6;
  the bit reader must not run past the packet boundary.
* Golden triples: verify the checksum relation.
* Spin numbers must be sequential; ESA steps must sweep 1..7 in order.

Where the big tables are
------------------------

The per-field housekeeping tables are **not** reproduced here on purpose. See
:ref:`lo-reference-tables`; the machine-readable version is
``imap_processing/lo/packet_definitions/lo_xtce.xml``, which is the thing the
code actually reads.
