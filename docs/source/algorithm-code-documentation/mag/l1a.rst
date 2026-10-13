.. _mag-l1a:

L1A - Decommutation and Decompression
=====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Code:** ``imap_processing/mag/l0/decom_mag.py``,
``imap_processing/mag/l1a/mag_l1a.py``,
``imap_processing/mag/l1a/mag_l1a_data.py`` (the large one).

**Document:** section 7.3.2 and Appendix 1.

L1A takes a file of CCSDS packets and produces two things: a near-verbatim
"raw" record of each packet, and a per-vector time series split by physical
sensor. **No calibration and no unit conversion happen here.**

Inputs and outputs
------------------

**[DOC]** Inputs are 25 hours of CCSDS packets (23:30 on the previous day to
00:30 on the next), the MAG telemetry spreadsheet, and the SPICE clock kernel.

**[CODE]** The CLI passes a single L0 file path. The XTCE at
``mag/packet_definitions/MAG_SCI_COMBINED.xml`` replaces the spreadsheet, and
``imap_processing.spice.time.met_to_ttj2000ns`` replaces manual clock kernel
handling.

The processing sequence
-----------------------

1. Split and de-duplicate
^^^^^^^^^^^^^^^^^^^^^^^^^

``decom_mag.decom_packets`` iterates the packet file, keeps APIDs 1052 and 1068,
and builds a ``MagL0`` per packet. Packets are collected in a ``dict`` keyed on
the ``MagL0`` object itself, whose hash is ``(SHCOARSE, APID, SRC_SEQ_CTR)`` -
this **silently drops duplicate downlinks**. Returns
``{"norm": [...], "burst": [...]}``.

2. Export raw
^^^^^^^^^^^^^

``decom_mag.generate_dataset`` writes one row per packet:

* ``epoch`` = ``met_to_ttj2000ns(SHCOARSE)`` - the **packet** time, not a vector
  time.
* ``raw_vectors`` = the undecoded byte block, zero-padded so all packets share
  the widest length in the file.
* Every other header field becomes its own ``epoch``-dimensioned variable,
  except ``SHCOARSE``, ``VECTORS`` and the PUS/spare fields.

3. Decommutate vectors
^^^^^^^^^^^^^^^^^^^^^^

``mag_l1a.process_packets`` loops over packets. Per packet:

.. code-block:: text

   primary_start_time   = TimeTuple(PRI_COARSETM, PRI_FNTM)
   secondary_start_time = TimeTuple(SEC_COARSETM, SEC_FNTM)
   mago_is_primary      = (PRI_SENS == PrimarySensor.MAGO.value)

   seconds_per_packet   = PUS_SSUBTYPE + 1
   total_vectors        = seconds_per_packet * vectors_per_second

``MagL1aPacketProperties.__post_init__`` computes ``seconds_per_packet``,
``total_vectors`` and, for compressed packets, ``compression_width`` from the
first six bits of the first vector byte.

Vectors are unpacked (see below), timestamped by
``MagL1a.calculate_vector_time`` -- first vector at the header time, each
subsequent vector ``+ 1/vectors_per_second`` -- and appended to a ``MagL1a``
object for MAGo and one for MAGi, chosen by ``mago_is_primary``.

``MagL1a.append_vectors`` also tracks CCSDS sequence-counter gaps into
``missing_sequences``.

.. warning::

   ``TimeTuple`` uses ``MAX_FINE_TIME = 65536`` when converting the fine-time
   counter to seconds. The algorithm document says a second is split into
   **1/65535** fractions for I-ALiRT (section 7.4.2). The two conventions differ
   by one part in 65536 (~15 microseconds). See
   :ref:`mag-implementation-status`.

4. Export per sensor
^^^^^^^^^^^^^^^^^^^^

``mag_l1a.generate_dataset`` writes ``vectors`` (``epoch`` x ``direction`` of
size 4: x, y, z, range) and ``compression_flags`` (``epoch`` x ``compression`` of
size 2: is_compressed, compression_width), plus the global attributes described
in :ref:`mag-data-products`.

Uncompressed vector unpacking
-----------------------------

``MagL1a.process_uncompressed_vectors``. Each sample is 50 bits - three 16-bit
signed components and a 2-bit unsigned range - and is **not byte aligned**, so
the pattern repeats every 4 vectors (200 bits = 25 bytes).

The implementation is an explicit four-case ``i % 4`` bit-shuffle over
``uint8``/``int32`` data, written by the MAG instrument team, followed by
``to_signed16``. Vectors ``0 .. primary_count-1`` go to the primary list, the
rest to the secondary list.

.. tip::

   Do not "clean up" this function. It is a direct transcription of the
   instrument team's reference implementation and is covered by the eight L1A
   validation cases (T001-T008) in ``imap_processing/tests/mag/validation/L1a/``.

.. _mag-compression:

Compression and decompression
-----------------------------

**[DOC]** Appendix 1 and section 7.3.2.1. The compression is **lossless** and
applies only to the vector data block. It differs from the uncompressed format
in four ways:

1. **Deltas**, not absolute vectors. MAG fields are large but vary slowly.
2. **Zig-zag encoding** maps signed deltas onto non-negative integers.
3. **Fibonacci encoding** gives those non-negative integers variable bit widths.
4. **Range data is only included if a range change occurs** within the packet.

Compressed block layout
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   [ 6 bits ] COMPRESSION_WIDTH      0-20, width in bits. Usually 16.
   [ 1 bit  ] HAS_RANGE_DATA_SECTION 1 = per-vector range section present
   [ 1 bit  ] spare

   Primary vectors section
     P1_X, P1_Y, P1_Z   COMPRESSION_WIDTH bits each, signed  (full vector)
     P1_RNG             2 bits unsigned
     P2..PN             3 variable-width Fibonacci/zig-zag deltas each

   Secondary vectors section
     S1_X, S1_Y, S1_Z, S1_RNG  same structure as P1
     S2..SM                    deltas

   [ 0-7 bits padding to the next byte boundary ]

   Primary range data section   (only if HAS_RANGE_DATA_SECTION == 1)
     P2_RNG .. PN_RNG   2 bits each,  (N - 1) values
   Secondary range data section (only if HAS_RANGE_DATA_SECTION == 1)
     S2_RNG .. SM_RNG   2 bits each,  (M - 1) values

   [ 0-7 bits padding ]

Expected counts:

.. code-block:: text

   Expected Primary Vectors   = PRIMARY_RATE   * (PUS_SSUBTYPE + 1)
   Expected Secondary Vectors = SECONDARY_RATE * (PUS_SSUBTYPE + 1)

where the encoded ``PRI_VECSEC``/``SEC_VECSEC`` values 0-7 map to rates
1, 2, 4, 8, 16, 32, 64, 128. **[CODE]** ``MagL0.__post_init__`` already applies
``2 ** value``.

If ``HAS_RANGE_DATA_SECTION == 0``, every vector in that section shares the
first vector's range.

Zig-zag encoding
^^^^^^^^^^^^^^^^

Interleaves signed integers onto non-negative ones so that small magnitudes
(regardless of sign) need few bits.

.. code-block:: text

   encode:  C = (n << 1) ^ (n >> 31)      # arithmetic shift extracts the sign bit
   decode:  n = (C >> 1) ^ -(C & 1)

Fibonacci encoding
^^^^^^^^^^^^^^^^^^

By Zeckendorf's theorem, every positive integer is a unique sum of
non-consecutive Fibonacci numbers. Set a bit for each Fibonacci number used
(least significant bit first), then **append a terminal 1**. Because the
representation never has two consecutive set bits, the terminal 1 creates the
only ``0b11`` in the code and makes it self-delimiting.

Example, encoding 19: :math:`19 = 13 + 5 + 1` -> ``101001`` -> append the
terminator -> ``1101001``.

Decoding, with the 40-element ``FIBONACCI_SEQUENCE`` in ``mag/constants.py``
(1, 2, 3, 5, 8, ..., 165580141):

.. code-block:: python

   def decode_fib_zig_zag(code):          # code is an array of bits ending in 1, 1
       code = code[:-1]                   # drop the terminator
       value = sum(FIBONACCI_SEQUENCE[: len(code)] * code) - 1   # Fibonacci decode
       return int((value >> 1) ^ (-(value & 1)))                 # zig-zag decode

Note the ``- 1``: the encoder biases by one so that zero is representable.

Worked example from the document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw vectors, ``COMPRESSION_WIDTH = 16``, all in range 3:

.. code-block:: text

   P1(0,0,0,3)  P2(-5,-5,-5,3)  P3(-15,-15,-15,3)  P4(5,-15,-20,3)

As deltas:

.. code-block:: text

   P1(0,0,0,3)  P2(-5,-5,-5)  P3(-10,-10,-10)  P4(+20,0,-5)

Encoded, with ``-5 = 0b010011`` (6 bits), ``-10 = 0b0101011`` (7 bits),
``+20 = 0b010100011`` (9 bits), ``0 = 0b11`` (2 bits):

.. code-block:: text

   P1 = 0b00000000000000000000000000000000000000000000000011   (48 bits + 2 range)
   P2 = 0b010011010011010011
   P3 = 0b010101101010110101011
   P4 = 0b01010001111010011

The high dynamic field escape hatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC]** If a compressed vector's three axes together exceed **60 bits**,
compression has failed because the field is too variable. From that point on,
**every remaining vector in that section is written uncompressed** at
``COMPRESSION_WIDTH`` bits per axis, with no range data. Range data is never in
the vector section after the first vector, compressed or not.

**[CODE]** ``constants.MAX_COMPRESSED_VECTOR_BITS = 60``.

How the code implements it
^^^^^^^^^^^^^^^^^^^^^^^^^^

``MagL1a.process_compressed_vectors``. The strategy is vectorised rather than a
bit-by-bit loop:

1. ``np.unpackbits`` the whole block into a bit array.
2. Read the 8-bit header; unpack the first primary vector with
   ``unpack_one_vector``.
3. Find **every** ``0b11`` in one shot by summing the bit array with a
   ``np.roll`` of itself and looking for the value 2. Those indices are the
   candidate Fibonacci terminators.
4. Walk the terminators to build ``primary_boundaries`` and
   ``secondary_boundaries`` (three boundaries per vector), watching for the
   60-bit condition to switch into uncompressed handling and for the boundary
   between the primary and secondary sections.
5. ``np.split`` on those boundaries, ``decode_fib_zig_zag`` each piece, and
   accumulate deltas with ``convert_diffs_to_vectors``.
6. If a range data section is present, ``process_range_data_section`` overwrites
   the range of vectors 2..N in each section.

Supporting helpers worth knowing:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function
     - Purpose
   * - ``unpack_one_vector(bits, width, has_range)``
     - Unpacks a full (uncompressed) vector. Pads to a byte boundary,
       ``np.packbits``, then ``twos_complement`` per axis.
   * - ``twos_complement(value, bits)``
     - Sign-extends a big-endian byte array of arbitrary bit width.
   * - ``convert_diffs_to_vectors(first, diffs, count)``
     - Cumulative sum of deltas; sets every vector's range to the first
       vector's range (later overwritten by the range data section if present).
   * - ``process_range_data_section(range_bits, vectors)``
     - Two bits per vector, excluding the first. Raises if the length is wrong.
   * - ``_process_vector_section(...)``
     - Shared primary/secondary handling, including the uncompressed tail after
       a 60-bit overflow.

.. warning::

   ``process_compressed_vectors`` is the most intricate function in the MAG
   codebase and its boundary arithmetic (the ``+1``/``-8``/``+7 // 8 * 8``
   adjustments) is tuned against the validation cases. If you change it, run
   ``imap_processing/tests/mag/test_mag_validation.py::test_mag_l1a_validation``
   for all of T001-T008 before anything else.

Sequence gaps
-------------

**[DOC]** L1A files should carry a header giving the **first and last sequence
counter** in the file, and a header listing any gaps, remembering the counter is
a rolling 14-bit unsigned integer.

**[CODE]** Only the gap list is implemented, as the ``missing_sequences`` global
attribute, and it does **not** handle the 14-bit rollover: ``append_vectors``
does ``range(most_recent + 1, vector_sequence)``, which produces an empty range
when the counter wraps from 16383 to 0. First/last sequence counters are not
recorded at all. See :ref:`mag-implementation-status`.
