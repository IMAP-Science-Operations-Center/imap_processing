.. _hit-l1a:

L0 to L1A: Frame Assembly, Decompression and Counts
===================================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Section 5 of the algorithm document.

* **Input**: CCSDS packets (APID 1251 housekeeping, 1252 science)
* **Processing requirement**: the decommutation table (the frame byte map)
* **Output**: L1A CDF files

L1A does no physics. It turns a byte stream into labelled integer arrays,
expands the onboard rate compression, and attaches Poisson uncertainties.

.. _hit-l1a-day-boundary:

Day boundaries and the input buffer
-----------------------------------

**[DOC]** A day's L0 request must include **all science packets for the day
plus a 5-minute buffer at the start (from 23:55 of the previous day) and a
15-minute buffer at the end (to 00:15 of the following day)**. The reason is
the 10-minute sectored block: without the buffer, a block straddling midnight
cannot be completed, and the previous block's livetime would be missing.

.. note::

   The document writes the leading buffer as "11:55 of the previous day",
   which is a 12-hour-clock slip for 23:55. The 10-minute figure quoted in the
   same paragraph is the sectored block length, not the buffer length.

**[CODE]** ``hit_l1a`` takes ``packet_date`` (``YYYYMMDD``, wired to
``self.start_date`` in the CLI) and raises ``ValueError`` if it is missing. Its
docstring says the L0 file has "a 20-minute buffer before and after the
processing day", which does not match the document's asymmetric 5/15 split.
The code does not depend on the buffer being any particular size - it just
trims - so this is a docstring inaccuracy rather than a bug, but it is worth
knowing which number to believe.

Trimming is done by ``filter_dataset_to_processing_day``:

* Convert ``epoch`` (TT2000 ns) to ``datetime64`` via
  ``et_to_datetime64(ttj2000ns_to_et(...))`` and keep indices whose date
  equals the processing day.
* Optionally (``sc_tick=True``, used for the standard counts product) do the
  same on ``sc_tick`` via ``met_to_datetime64``, because the CCSDS header
  fields live on a per-packet dimension rather than per-frame.
* For **sectored** data the filter is applied to an array of *mean epochs per
  10-minute set*, repeated 10 times, rather than to the per-frame epochs. That
  keeps a block that straddles midnight together, assigned to whichever day
  its centre falls in.

.. note::

   **[CODE]** There is a standing ``TODO`` in ``process_science`` about this:
   a frame whose mean epoch lands in the processing day may still contain
   packets from the previous day, and the ``sc_tick`` filter will drop those
   header rows. Nobody has decided whether that matters.

Assembling a science frame
--------------------------

**[CODE]** ``decom_hit.assemble_science_frames``.

A valid frame is **20 consecutive packets** whose CCSDS grouping flags match:

.. code-block:: text

   FLAG_PATTERN = [1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 2]
                   ^  first packet          18 middle       ^ last packet

``get_valid_starting_indices`` slides a 20-wide window over the flag array
(``np.lib.stride_tricks.sliding_window_view``), keeps the matches, and then
additionally requires the ``src_seq_ctr`` values within each match to be
**sequential modulo 16384** (``is_sequential``).

For each valid start index:

* Packets 0-5 are concatenated into ``count_rates_raw`` (1572 bytes).
* Packets 6-19 are concatenated into ``pha_raw``.
* ``epoch`` for the frame is the **mean of the first and last packet epoch**
  (``calculate_epoch_mean``) - the centre of the collection minute.

Before any of this, ``update_ccsds_header_dims`` swaps the dataset's dimension
from ``epoch`` to ``sc_tick``, because at this point ``epoch`` is per-packet and
is about to become per-frame.

.. warning::

   **[CODE]** **Incomplete frames are silently dropped.** There is no fill-value
   path and no quality flag. The code ``print``\ s (does not log) a note when
   packets at the start or end of the file belong to a neighbouring day's frame,
   and there is a ``TODO`` about handling those when processing multiple files.
   ``get_valid_starting_indices`` returning an empty array will also raise
   ``IndexError`` on ``starting_indices[0]`` rather than a useful message.
   See :ref:`hit-gap-incomplete-frames`.

Rate compression and decompression
----------------------------------

**[DOC]** Section 5.1. Counters are held in 24-bit (some 32-bit) onboard
registers and squeezed into 16 bits by a **modified biased-exponent,
hidden-one** scheme originally suggested by Don Reames for STEREO/LET:
**12 mantissa bits, 4 exponent bits**. Values up to :math:`2^{12}` are stored
exactly; values up to :math:`2^{13}` decompress with no error.

The document reproduces the heritage C verbatim (``pack_rate``, ``long_rate``,
``dbl_rate``). Only unpacking is needed on the ground.

**[CODE]** ``decom_hit.decompress_rates_16_to_32``, with ``MANTISSA_BITS = 12``
and ``EXPONENT_BITS = 4``:

.. code-block:: python

   power = packed >> 12                     # top 4 bits = exponent
   if power > 1:
       mantissa = packed & 0x0FFF           # bottom 12 bits
       value = (mantissa | 0x1000) << (power - 1)   # restore the hidden one
   else:
       value = packed                       # stored exactly

This matches the heritage ``long_rate()`` exactly, including the ``power > 1``
(not ``>= 1``) boundary.

.. important::

   **The one exception the document calls out is not implemented.** Section
   4.2.4 states that the **Front End Electronics livetime counter**
   (frame bytes 6-7) is *"scaled from 24 bits to 16 bits"* rather than
   compressed with this algorithm. The code runs every non-header field
   through ``decompress_rates_16_to_32``, including ``livetime_counter``.

   In practice this appears to be intentional and correct: the L1B livetime
   conversion (section 6.2, and :ref:`hit-l1b-livetime`) is a piecewise linear
   fit **defined on the decompressed value**, with explicit breakpoints at
   4101 and 16000 and an explicit warning about rollover. Do not "fix" the
   decompression without checking the L1B fit with the HIT team.

Fields that skip decompression are those whose name contains ``hdr``,
``spare`` or ``pha``.

The frame byte map
------------------

**[CODE]** ``COUNTS_DATA_STRUCTURE`` in ``hit/l0/constants.py`` is an ordered
dict of ``HITPacking(bit_length, section_length, shape)``.
``parse_count_rates`` walks it in order, slicing the binary string, so **the
order of that dict is the wire format**. Do not reorder it.

It has been checked byte-for-byte against algorithm document Tables 11-26 and
agrees exactly:

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 22 28

   * - Field(s)
     - Bytes
     - Shape
     - Doc table
     - Contents
   * - ``hdr_*``
     - 0-2
     - scalar
     - 11 (MISCBITS)
     - Unit/version, code-ok, heater duty cycle, leak conv, **dynamic
       threshold state**, minute counter.
   * - ``spare``
     - 3-5
     - --
     - 11
     - Dropped by ``decom_hit``.
   * - ``livetime_counter`` .. ``num_haz_acc_no_pha``
     - 6-23
     - scalar x9
     - 12 (ERATES)
     - FEE livetime, then NUMTRIG / NUMREJECT / NUMACCPHA / NUMACCNPHA and
       their ``.HAZ`` counterparts.
   * - ``sngrates``
     - 24-255
     - (2, 58)
     - 13 (SNGRATES)
     - 58 ADCs x high/low gain. **Interleaved on the wire** as
       (ADC0 high, ADC0 low, ADC1 high, ...); the code de-interleaves with
       ``data[::2]`` / ``data[1::2]`` into a ``gain`` dimension. Ordered by
       detector address, starting at ``L2A9``.
   * - ``nread`` .. ``nbadtags``
     - 256-289
     - scalar x17
     - 14 (EVPRATES)
     - Event-processing counters.
   * - ``coinrates``
     - 290-341
     - (26,)
     - 15 (COINRATES)
     - Coincidence rates: ``L12A``, ``L123A``, ``L12B``, ``L123B``, ``2TEL``,
       ``PENA``, ``PENA?``, ``PENB``, ``PENB?``, ``ILA``, ``IHA``, ``ILB``,
       ``IHB``, ``L142A``, ``L1423A``, ``L142B``, ``L1423B``, ``PEN4A``,
       ``PEN4B``, ``ERROR``, then the six singles-by-layer counters ``L1A``,
       ``L2A``, ``L3A``, ``L1B``, ``L2B``, ``L3B``.
   * - ``pbufrates``
     - 342-405
     - (32,)
     - 16 (PBUFRATES)
     - The 32 priority buffers. #29/#30 are the "clean"/"poor" livetime STIM
       buffers; #31 is the onboard processing error counter.
   * - ``l2fgrates``
     - 406-669
     - (132,)
     - 17
     - **Range 2 foreground.** Index = Particle ID.
   * - ``l2bgrates``
     - 670-693
     - (12,)
     - 18
     - Range 2 background. All Particle ID 255.
   * - ``l3fgrates``
     - 694-1027
     - (167,)
     - 19
     - **Range 3 foreground.** Index = Particle ID.
   * - ``l3bgrates``
     - 1028-1051
     - (12,)
     - 20
     - Range 3 background.
   * - ``penfgrates``
     - 1052-1117
     - (33,)
     - 21
     - **Range 4 foreground.** Index = Particle ID.
   * - ``penbgrates``
     - 1118-1147
     - (15,)
     - 22
     - Range 4 background.
   * - ``ialirtrates``
     - 1148-1187
     - (20,)
     - 23
     - The 20 I-ALiRT rates, also present in the 1 Hz I-ALiRT packet: 6
       ``L4Ai dE`` bins, 4 ``L4Ai vs L3A`` bins, 6 ``L4Bi dE`` bins, 4
       ``L4Bi vs L3B`` bins.
   * - ``sectorates``
     - 1188-1427
     - (8, 15)
     - 24 (SECTORRATES)
     - 120 look directions for **one** species/energy combination.
   * - ``l4fgrates``
     - 1428-1523
     - (48,)
     - 25
     - I-ALiRT-aperture ion foreground rates (ranges L1L4L2, L1L4L2L3,
       L1L4L2L3L3).
   * - ``l4bgrates``
     - 1524-1571
     - (24,)
     - 26
     - I-ALiRT-aperture ion background rates.
   * - (Event Buffer)
     - 1572-5239
     - --
     - 27
     - 2-byte event count header, then variable-length event records. Handled
       as ``pha_raw``.

Total fixed-format section: **12576 bits = 1572 bytes = exactly 6 packets**.

.. note::

   Section 4.2.1's "371 Science Rates ... for a total of 742 bytes" checks out
   exactly: 132 + 12 + 167 + 12 + 33 + 15 = 371 entries in the six FG/BG
   arrays, at 2 bytes each.

   One place where the prose does **not** match its own table: section 4.2.3
   says "space is allocated for 20 coincidence rates", while Table 15 lists
   **26**. The code follows the table.

Sectored-rate subcommutation
----------------------------

**[CODE]** ``hit_l1a.subcom_sectorates``.

Each frame's ``sectorates`` array holds 120 look directions for **one**
species/energy combination, selected by ``hdr_minute_cnt % 10`` via
``MOD_10_MAPPING`` (see :ref:`hit-overview`). The function:

#. Builds, for each of the 10 mod-10 slots, an array of shape
   ``(n_frames, 15, 8)`` filled with ``-9223372036854775808`` (the int64 fill
   value).
#. Writes each frame's ``sectorates`` into the slot its minute counter
   selects. Every other slot for that frame stays fill.
#. Regroups the 10 slots into 5 species (H gets 3 energy bins, 4He/CNO/NeMgSi
   get 2, Fe gets 1) and transposes to ``(epoch, energy_mean, azimuth,
   zenith)``.
#. Adds ``<species>_energy_mean``, ``_energy_delta_plus``,
   ``_energy_delta_minus`` via ``add_energy_variables``.

So the resulting arrays are **9/10 fill by construction** at 1-minute
resolution. They only become dense when regrouped into 10-minute records at
L2 (``transform_to_10_minute_chunks``, see :ref:`hit-l2`).

Selecting complete 10-minute sets
---------------------------------

**[CODE]** ``hit_l1a.subset_sectored_counts``, and its helpers.

#. ``update_livetime_coord`` attaches a **shadow coordinate**
   ``epoch_livetime`` to ``livetime_counter`` and swaps its dimension, so that
   subsequent filtering of ``epoch`` does not also filter livetime. This is the
   trick that lets the 10-minute livetime offset survive the trimming.
#. ``find_complete_mod10_sets`` slides a 10-wide window over
   ``hdr_minute_cnt % 10`` and returns the indices where it equals exactly
   ``[0,1,2,...,9]``.
#. Start indices **< 10 are discarded**, because the previous 10 minutes'
   livetime would not be available.
#. The dataset is subset to those 10-frame runs.
#. A mean epoch per set is computed and repeated 10 times, and the set is
   trimmed to the processing day on that array.
#. ``subset_livetime`` then slices ``epoch_livetime`` to the same length but
   **shifted 10 indices earlier**.

Failure modes, all ``ValueError``:

* No complete mod-10 set found at all.
* Empty epoch values after filtering.
* ``start_idx < 10`` in ``subset_livetime`` (dataset too small to shift).

.. warning::

   **[CODE]** This machinery assumes **one frame per minute with no gaps**.
   ``subset_livetime`` shifts by **10 array positions**, not by 10 minutes of
   wall-clock time. A dropped science frame inside or just before a sectored
   block will silently pair the counts with the wrong block's livetime. See
   :ref:`hit-gap-livetime-shift`.

Statistical uncertainties
-------------------------

**[DOC]** Section 5.5. Asymmetric Poisson at 1 sigma (0.8413), Gehrels 1986
with S = 1:

.. math::

   \lambda_u = n + \sqrt{n+1} + 1 \quad\Rightarrow\quad \delta_u = \sqrt{n+1} + 1

.. math::

   \lambda_l = n - \sqrt{n} \quad\Rightarrow\quad \delta_l = \sqrt{n}

``DELTA_PLUS`` and ``DELTA_MINUS`` in the CDF carry :math:`\delta_u` and
:math:`\delta_l`. **Uncertainties for fill values must themselves be fill
values.**

**[CODE]** ``hit_l1a.calculate_uncertainties``. It applies the formulas to
every data variable *except* an explicit ignore list (CCSDS header fields,
``hdr_*``, ``livetime_counter``, and the ``*_energy_delta_*`` variables), and
uses ``np.where(mask, ..., dataset[var])`` so that fill values pass through
unchanged. ``np.maximum(..., 0)`` guards the square root.

.. note::

   ``livetime_counter`` is deliberately excluded - it is a clock-cycle count,
   not a particle count, so Poisson statistics do not apply to it.

Housekeeping at L1A
-------------------

**[CODE]** ``hit_utils.process_housekeeping_data``, shared with L1B. It:

* Drops the CCSDS header fields and the five ``hskp_spare*`` fields.
* Collapses ``leak_i_00`` .. ``leak_i_63`` into a single 2-D ``leak_i`` on a
  new ``adc_channels`` (0-63) coordinate - ``concatenate_leak_variables``.
* Applies the CDF attributes, and manually sets ``DEPEND_0 = epoch`` on
  ``sc_tick`` (which is a coordinate in the counts product but a variable
  here).

L1A reads the packet with ``use_derived_value=False``, so all values are raw
DN. L1B reads the same packet with ``True``. See :ref:`hit-l1b-hk`.
