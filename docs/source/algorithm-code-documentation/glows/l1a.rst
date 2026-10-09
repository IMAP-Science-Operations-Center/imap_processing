.. _glows-l1a:

Level 1A - Unpacked Telemetry
=============================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**Goal:** turn the binary CCSDS stream into human-readable data structures.
Nothing is calibrated, nothing is masked, no SPICE is involved. The only real
algorithm here is **direct-event decompression**.

Entry point: ``glows_l1a(packet_filepath) -> list[xr.Dataset]`` in
``glows/l1a/glows_l1a.py``. One L0 file (one observational day) in, one or two
datasets out.

.. code-block:: python

   hist_l0, de_l0 = decom_packets(packet_filepath)

   if hist_l0:
       l1a_hists = [HistogramL1A(hist) for hist in hist_l0]
       output_datasets.append(generate_histogram_dataset(l1a_hists, glows_attrs))

   if de_l0:
       l1a_de = process_de_l0(de_l0)
       output_datasets.append(generate_de_dataset(l1a_de, glows_attrs))

Decommutation
-------------

**[CODE]** ``glows/l0/decom_glows.py``.

``decom_packets`` iterates ``packet_generator(packet_file_path, xtce_document)``
where the XTCE is ``glows/packet_definitions/GLX_COMBINED.xml``, and dispatches
on ``PKT_APID``:

* **1480** - ``separate_ccsds_header_userdata`` splits the packet, and a
  ``HistogramL0`` is built from ``(__version__, filename, CcsdsData(header),
  *userdata.values())``.
* **1481** - the first 7 fields are treated as the header and the remainder are
  taken as ``item.raw_value``. This raw-value path matters: ``DE_DATA`` must
  stay as uninterpreted bytes.

.. note::

   ``glows.__version__`` (``"v001"``) is passed in as ``ground_sw_version``. It
   is the GLOWS module's own version string, not the package version.

``DirectEventL0`` defines ``within_same_sequence(other)`` (currently checks
``SEC`` and ``LEN``) and ``__lt__`` on ``SEQ`` so packet groups can be sorted.

Histograms: L0 → L1A
--------------------

**[CODE]** ``HistogramL1A`` in ``glows/l1a/glows_l1a_data.py``. It is a
mechanical field copy plus four ``TimeTuple`` constructions:

.. code-block:: python

   self.imap_start_time  = TimeTuple(l0.SEC,      l0.SUBSEC)
   self.imap_time_offset = TimeTuple(l0.OFFSETSEC, l0.OFFSETSUBSEC)
   self.glows_start_time = TimeTuple(l0.GLXSEC,   l0.GLXSUBSEC)
   self.glows_time_offset= TimeTuple(l0.GLXOFFSEC, l0.GLXOFFSUBSEC)

   self.last_spin_id  = l0.STARTID + l0.ENDID     # ENDID is a difference
   self.first_spin_id = l0.STARTID

   self.flags = {"flags_set_onboard": l0.FLAGS, "is_generated_on_ground": False}

Two quirks worth knowing:

* **Odd bin counts lose a byte.** CCSDS packets must have an even number of
  bytes, so when ``NBINS`` is odd the payload carries a pad byte. The code
  drops the last element: ``if self.number_of_bins_per_histogram % 2 == 1:
  self.histogram = self.histogram[:-1]``.
* The ``ENDID`` vs. ``SPINS`` consistency check the document asks for exists
  only as a **commented-out block** (``glows_l1a_data.py`` line ~248) because
  the emulator did not set the values correctly. A length mismatch between
  ``NBINS`` and the decoded histogram logs a warning but does not raise.

``is_generated_on_ground`` is hard-coded ``False``: GLOWS can in principle build
histograms on the ground from direct events (document §8.3), but that path does
not exist here.

Direct events: L0 → L1A
-----------------------

This is the interesting part. Three problems have to be solved in order:
**reassembly**, **decompression** and **padding into a rectangular array**.

Reassembly across packets
^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §3.4.2]** One second of direct events may not fit in one CCSDS packet.
When it is split, the split happens **at the byte layer, not at the direct-event
layer** - a single event can straddle two packets. So you must concatenate
first, parse second.

**[CODE]** ``process_de_l0`` in ``glows_l1a.py``:

.. code-block:: python

   sorted_l0 = sorted(de_l0, key=lambda x: x.SEC)
   for sec, de in groupby(sorted_l0, lambda x: x.SEC):
       ...

For each second:

* One packet with ``LEN == 1`` → construct ``DirectEventL1A`` and parse
  immediately.
* One packet with ``LEN != 1`` → packets are missing off the end;
  ``finish_incomplete_packet()`` records the missing sequence numbers and
  populates ``status_data`` only, leaving ``direct_events`` unset.
* Several packets → sort by ``SEQ``. **If ``SEQ != 0`` is missing, the whole
  second is skipped with a warning**, because the ``data_every_second``
  structure lives only in the first packet. Otherwise each subsequent packet is
  appended via ``merge_de_packets``, which validates ordering and
  ``within_same_sequence``, and accumulates gaps into ``missing_seq``.

Finally records with no ``direct_events`` are filtered out entirely. Missing
sequence numbers are joined into the dataset global attribute
``missing_packets_sequence``.

.. code-block:: python

   l1a_output = [de for de in l1a_output if de.direct_events]

The ``data_every_second`` structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC Table 3.2]** / **[CODE]** ``StatusData``, built from the **first 40
bytes** (320 bits) of the reassembled payload. This is the onboard housekeeping
that the flight software's histogramming algorithm itself uses, so downlinking
it is what makes ground-side histogram regeneration possible in principle.

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Field
     - Bits
     - Notes
   * - ``imap_sclk_last_pps``
     - 32
     - IMAP seconds at the last PPS.
   * - ``glows_sclk_last_pps``
     - 32
     - GLOWS seconds at the last PPS.
   * - ``glows_ssclk_last_pps``
     - 32
     - GLOWS subseconds at the last PPS.
   * - ``imap_sclk_next_pps``
     - 32
     - IMAP seconds at the next PPS.
   * - ``catbed_heater_active``
     - 8
     - Flag. Thruster catbed heaters on → repointing imminent.
   * - ``spin_period_valid``
     - 8
     - Flag.
   * - ``spin_phase_at_next_pps_valid``
     - 8
     - Flag.
   * - ``spin_period_source``
     - 8
     - Flag.
   * - ``spin_period``
     - 16
     - Integer-encoded; decoded at L1B.
   * - ``spin_phase_at_next_pps``
     - 16
     - Integer-encoded; decoded at L1B.
   * - ``number_of_completed_spins``
     - 32
     - Provided to GLOWS by IMAP.
   * - ``filter_temperature``
     - 16
     - Integer-encoded.
   * - ``hv_voltage``
     - 16
     - Integer-encoded.
   * - ``glows_time_on_pps_valid``
     - 8
     - Flag.
   * - ``time_status_valid``
     - 8
     - Flag.
   * - ``housekeeping_valid``
     - 8
     - Flag.
   * - ``is_pps_autogenerated``
     - 8
     - Flag.
   * - ``hv_test_in_progress``
     - 8
     - Flag.
   * - ``pulse_test_in_progress``
     - 8
     - Flag.
   * - ``memory_error_detected``
     - 8
     - Flag.
   * - ``zero_padding``
     - 8
     - Pad to an even byte count. Present in the document's table; not carried
       as a field in ``StatusData``.

Direct-event decompression
^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §3.4.2 / §3.5.2, Tables 3.3 and 3.4]**. Events are stored as time
*differences* from the previous event, with the first event of each second
carried as a full timestamp. The **top two bits of the first byte** of every
record are a marker:

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Marker
     - Record size
     - Layout
   * - ``0x0``
     - 8 bytes
     - **Full timestamp.** 2-bit marker, 30-bit GLOWS seconds, 8-bit
       ``impulse_length`` (50 ns units), 1-bit ``multi_event``, 2 unused bits
       (not guaranteed zero), 21-bit GLOWS subseconds.
   * - ``0x2``
     - 2 bytes
     - **14-bit offset** (``16 - 2``) in GLOWS clock ticks relative to the
       previous event, plus 8-bit ``impulse_length``.
   * - ``0x3``
     - 3 bytes
     - **22-bit offset** (``24 - 2``), plus 8-bit ``impulse_length``.

An event is written as a full timestamp whenever the offset from the previous
event will not fit in 22 bits.

``multi_event`` is set by the FPGA. Per KPLabs it is **not currently used** by
the flight software and is always ``False``.

**[CODE]** ``DirectEventL1A._generate_direct_events`` implements the loop:

.. code-block:: python

   current_event = self._build_uncompressed_event(direct_events[:8])
   processed_events = [current_event]

   i = 8
   while i < len(direct_events) - 1:
       first_byte  = int(direct_events[i]); i += 1
       oldest_diff = first_byte & 0x3F        # low 6 bits
       marker      = first_byte >> 6

       if marker == 0x0:   # 7 more bytes; oldest_diff becomes the top byte
           part = bytearray([oldest_diff]); part.extend(direct_events[i:i+7]); i += 7
           current_event = self._build_uncompressed_event(part)
       elif marker == 0x2: # 2 more bytes
           current_event = self._build_compressed_event(
               direct_events[i:i+2], oldest_diff, current_event.timestamp); i += 2
       elif marker == 0x3: # 3 more bytes
           current_event = self._build_compressed_event(
               direct_events[i:i+3], oldest_diff, current_event.timestamp); i += 3
       else:
           raise IndexError(...)
       processed_events.append(current_event)

Offset reconstruction, ``_build_compressed_event``:

.. code-block:: python

   # 2-byte:  diff = oldest_diff << 8  | raw[0];        length = raw[1]
   # 3-byte:  diff = oldest_diff << 16 | int(raw[0:2]); length = raw[2]
   subseconds = previous_time.subseconds + diff
   seconds    = previous_time.seconds
   return DirectEvent(TimeTuple(seconds, subseconds), length, False)

Carry from subseconds into seconds is handled by ``TimeTuple.__post_init__``,
which folds anything ``>= 2 000 000`` into whole seconds. **This is the only
place the carry happens** - do not add manual carry logic.

Full timestamp reconstruction, ``_build_uncompressed_event``:

.. code-block:: python

   values         = struct.unpack(">II", raw)          # 8 bytes
   seconds        = values[0]
   subseconds     = values[1] & 0x1FFFFF                # low 21 bits
   impulse_length = (values[1] >> 24) & 0xFF            # top byte
   multi_event    = bool((values[1] >> 23) & 0b1)

.. warning::

   ``seconds = values[0]`` uses all 32 bits, but the document says the field is
   a 2-bit marker followed by **30** bits of seconds. For the first event of a
   second the marker is ``0x0`` so the top two bits are zero and the two agree;
   for a mid-stream full timestamp the marker bits have already been consumed
   into ``oldest_diff`` and re-prepended, so they are also zero. It works, but it
   works by construction rather than by masking.

Robustness
^^^^^^^^^^

* Payloads shorter than 8 bytes after the status block log a warning and yield
  an empty event list.
* Any unexpected marker or a truncated record raises ``IndexError``, which
  ``process_de_l0`` catches per packet, logs, and continues - **DE errors never
  stop processing.**

L1A output datasets
-------------------

``imap_glows_l1a_hist``
^^^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``generate_histogram_dataset``. Three filters run before anything is
written:

1. Histograms with ``number_of_bins_per_histogram == 0`` are dropped.
2. Histograms with ``imap_start_time.seconds == 0`` are dropped as invalid
   timing (warning logged).
3. **Deduplication** on the 4-tuple ``(imap_start_time.seconds,
   .subseconds, imap_time_offset.seconds, .subseconds)``, keeping the first
   occurrence (warning logged).

Coordinates: ``epoch``, ``bins`` (0-3599), ``bins_label``.

.. code-block:: python

   epoch_time = met_to_ttj2000ns(
       hist.imap_start_time.to_seconds() + hist.imap_time_offset.to_seconds() / 2
   )

i.e. **epoch is the block midpoint**, not the block start.

The histogram array is always allocated at ``GlowsConstants.STANDARD_BIN_COUNT``
(3600) and pre-filled with ``GlowsConstants.HISTOGRAM_FILLVAL`` (65535, i.e.
``uint16`` max); shorter histograms occupy the leading bins and the rest stay
fill. Every downstream stage must respect that fill value.

.. list-table::
   :header-rows: 1
   :widths: 40 14 46

   * - Variable
     - dtype
     - Notes
   * - ``histogram``
     - uint16
     - ``(epoch, bins)``. Counts. 65535 = unused bin.
   * - ``seq_count_in_pkts_file``
     - uint16
     - CCSDS ``SRC_SEQ_CTR``.
   * - ``first_spin_id`` / ``last_spin_id``
     - uint32
     - ``STARTID`` and ``STARTID + ENDID``.
   * - ``flags_set_onboard``
     - uint16
     - The raw 16-bit onboard flag word, undecoded.
   * - ``is_generated_on_ground``
     - uint8
     - Always 0.
   * - ``number_of_spins_per_block``
     - uint8
     - ``n_block``.
   * - ``number_of_bins_per_histogram``
     - uint16
     - ``n_bin``.
   * - ``number_of_events``
     - uint32
     - Total counts.
   * - ``filter_temperature_average`` / ``_variance``
     - uint32
     - **Still encoded.**
   * - ``hv_voltage_average`` / ``_variance``
     - uint32
     - Still encoded.
   * - ``spin_period_average`` / ``_variance``
     - uint32
     - Still encoded.
   * - ``pulse_length_average`` / ``_variance``
     - uint32
     - Still encoded.
   * - ``imap_start_time`` / ``imap_time_offset``
     - float64
     - Seconds with subseconds as decimals.
   * - ``glows_start_time`` / ``glows_time_offset``
     - float64
     - Seconds with subseconds as decimals.

Global attribute: ``flight_software_version``, taken from the first histogram.

.. note::

   The document (Table 3.5) keeps L1A times as **integer** second/subsecond
   pairs and defers the float conversion to L1B. **[CODE]** The L1A CDF already
   writes floats, because ``TimeTuple.to_seconds()`` is applied on the way out.
   The ``TimeTuple`` objects themselves are integer-valued inside
   ``HistogramL1A``.

``imap_glows_l1a_de``
^^^^^^^^^^^^^^^^^^^^^

**[CODE]** ``generate_de_dataset``. One epoch per **second** of direct events
(``met_to_ttj2000ns(de.l0.MET)``).

Coordinates: ``epoch``, ``within_the_second``, ``direct_event_components``.

.. list-table::
   :header-rows: 1
   :widths: 40 14 46

   * - Variable
     - dtype
     - Notes
   * - ``direct_events``
     - float64
     - ``(epoch, within_the_second, direct_event_components)``. The last axis is
       ``[seconds, subseconds, impulse_length, multi_event]``.
   * - ``seq_count_in_pkts_file``
     - uint16
     - CCSDS sequence counter.
   * - ``number_of_de_packets``
     - uint32
     - ``LEN``.
   * - 20 ``StatusData`` fields
     - uint32/uint8/float64
     - See the table above.

The ``within_the_second`` dimension is sized to the **longest** second in the
file; the array is grown with ``np.pad`` as longer seconds are encountered and
shorter seconds are **zero padded**. There is no fill value and no explicit
count of valid events per epoch, so downstream code cannot cheaply distinguish a
padded slot from a genuine event at GLOWS time zero. In practice
``TimeTuple(0, 0)`` never occurs in flight data, but it is a latent trap.

Global attribute: ``missing_packets_sequence``, a comma-joined list of the
``missing_seq`` lists.

Testing
-------

**[CODE]** ``imap_processing/tests/glows/``:

* ``test_glows_decom.py`` - packet counts (505 histogram + 1088 DE packets in
  the bundled test file), header fields, byte-array handling.
* ``test_glows_l1a_data.py`` - all three compression markers individually,
  sequential events, packet merging with and without gaps, ``StatusData``
  parsing, and comparison against ``glows_l1a_hist_validation.json``.
* ``test_glows_l1a_cdf.py`` - dataset generation, the empty/zero-time filters,
  and deduplication.

Tests using the larger in-flight packet file are marked
``@pytest.mark.external_test_data`` and are skipped by the default selection.
