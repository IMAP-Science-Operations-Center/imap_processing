.. _idex-data-products:

Data Products and What Feeds What
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This is the "what goes into what" map. It is the page to read before touching
``cli.py``, adding a product, or changing a cadence.

Product inventory
-----------------

**[CODE]** Everything this repository produces for IDEX. The ``logical_source``
strings come from ``imap_processing/cdf/config/imap_idex_global_cdf_attrs.yaml``
and are what ``write_cdf()`` uses to name the output file.

.. list-table::
   :header-rows: 1
   :widths: 34 8 10 48

   * - ``logical_source``
     - Level
     - Cadence
     - Contents
   * - ``imap_idex_l1a_sci-10days``
     - L1A
     - 10 days
     - One record per dust event. Six raw-DN waveform arrays, the two waveform
       time axes, the full FPGA header telemetry as event metadata, and ten
       event/saturation flags.
   * - ``imap_idex_l1a_msg-10days``
     - L1A
     - 10 days
     - One record per instrument log entry. ``epoch``, ``elsec_evtpkt``,
       ``elssec_evtpkt`` and a rendered ``messages`` string.
   * - ``imap_idex_l1a_catlst-10days``
     - L1A
     - 10 days
     - Packet catalog summary (APID 1419), **raw** values, with ``epoch``
       added.
   * - ``imap_idex_l1b_sci-10days``
     - L1B
     - 10 days
     - Same events, waveforms in engineering units, 31 unpacked instrument
       settings, ``dead_time``, trigger mode / level / origin, and ten
       SPICE-derived geometry variables.
   * - ``imap_idex_l1b_msg-10days``
     - L1B
     - 10 days
     - Reduced state log: ``science_on`` and ``pulser_on`` only, and only at
       epochs where one of them actually changes.
   * - ``imap_idex_l1b_catlst-10days``
     - L1B
     - 10 days
     - Packet catalog summary, **derived** values. **[CODE]** Produced by the
       *L1A* job (``PacketParser``), not by ``idex_l1b()``. See below.
   * - ``imap_idex_l2a_sci-10days``
     - L2A
     - 10 days
     - Per-event fits for the three low-rate channels (fit parameters, impact
       charge, modelled waveform, chi-square), velocity and mass estimates, and
       the TOF mass-spectrum variables. **Many of these are deliberately NaN.**
   * - ``imap_idex_l2b_sci-1mo``
     - L2B
     - 1 month
     - Daily dust counts and uptime-corrected count rates, binned by spin-phase
       quadrant, optionally by impact charge or mass. **The mass- and
       charge-binned variables are deliberately fill-valued.**
   * - ``imap_idex_l2c_rectangular-map-1mo``
     - L2C
     - 1 month
     - The same daily counts and rates binned on a 6° rectangular sky map in
       ECLIPJ2000 instead of by spin phase. Same deliberate fill block.

The processing chain
--------------------

**[DOC]** Algorithm document Figure 4.1, annotated **[CODE]** with what the
repository actually does.

.. code-block:: text

                      CCSDS packets (imap_idex_l0_raw_YYYYMMDD_vNNN.pkts)
                                        |
                       XTCE: idex_science_packet_definition.xml
                             idex_housekeeping_packet_definition.xml
                                        |
                             idex_l0.decom_packets()
                       +----------------+----------------+
                       |                |                |
             science packets     APID 1418 (EVT)   APID 1419 (CATLST)
                  (APID 1424)           |                |
                       |                |                |
             +---------v--------+  +----v----------+  +--v-------------+
             | l1a_sci-10days   |  | l1a_msg-10days|  | l1a_catlst     |
             | fragment assembly|  | template      |  | l1b_catlst     |
             | Rice decompress  |  | rendering     |  | (passthrough)  |
             | event flags      |  |               |  +----------------+
             +---------+--------+  +----+----------+
                       |                |
   DN->EU CSV ---->    |                |
   SPICE kernels ->    |                |
   spin table ---->    |                |
             +---------v--------+  +----v----------+
             | l1b_sci-10days   |  | l1b_msg-10days|
             | pC / mA waveforms|  | science_on    |
             | dead_time        |  | pulser_on     |
             | trigger decode   |  +----+----------+
             | ephemeris, spin  |       |
             +---------+--------+       |
                       |                |
   t-rise cal CSV ->   |                |
   yield cal CSV ->    |                |
   atomic_masses.csv-> |                |
             +---------v--------+       |
             | l2a_sci-10days   |       |
             | target/IG fits   |       |
             | charge, v, mass  |       |
             | TOF mass scale   |       |
             +---------+--------+       |
                       |                |
                       +-------+--------+
                               |
                    idex_l2b()  (3 x l2a + msg)
                               |
                  +------------+------------+
                  |                         |
        +---------v--------+     +----------v-------------------+
        | l2b_sci-1mo      |     | l2c_rectangular-map-1mo      |
        | counts/rates vs  |     | counts/rates on a 6 deg      |
        | spin quadrant    |     | ECLIPJ2000 rectangular grid  |
        +------------------+     +------------------------------+

**[CODE]** Note the shape of the last step: ``idex_l2b()`` returns a
**two-element list**, ``[l2b_dataset, l2c_dataset]``. L2C is produced by the L2B
job because it needs no additional dependencies and is a rebinning of the same
counts. There is no ``l2c`` branch in ``cli.py``'s ``Idex.do_processing`` -
asking for ``--level l2c`` raises ``NotImplementedError`` even though ``"l2c"``
is listed in ``PROCESSING_LEVELS`` in ``imap_processing/__init__.py``.

The windowing model
-------------------

IDEX does not produce daily files. It produces **10-day** files up to L2A and
**monthly** files at L2B/L2C. This is a direct consequence of the event rate:
**[DOC]** roughly 16 dust events per day, so a daily product would frequently
contain a handful of events or none at all.

The 10-day windows
^^^^^^^^^^^^^^^^^^

**[CODE]** The windows are **not** computed. They come from a fixed lookup
table supplied by the IDEX team:
``imap_processing/idex/idex_10_day_CDF_names.csv``, 444 rows covering
2025-01-01 through 2037-01-01, with columns ``start_date``, ``end_date``,
``doy``.

.. code-block:: text

   start_date,end_date,doy
   20250101,20250110,1
   20250110,20250120,10
   ...
   20361225,20370101,360

Windows **restart at each new year**, so the last window of a year is 5-6 days
rather than 10. That is expected and is called out in a comment in
``idex_constants.py``.

``idex_utils.get_10_day_window_end_date(start_date)`` is the only reader. It
requires an **exact** match on ``start_date`` and raises ``ValueError``
otherwise. The practical consequence: **an IDEX L1A job can only be launched on
a start date that appears in that CSV.** The SDC scheduler, not this repository,
is responsible for knowing those dates.

Window boundaries are compared as TT-J2000 nanoseconds via
``str_yyyymmdd_to_ttj2000ns()``, half-open: ``start <= epoch < end``.

The monthly windows
^^^^^^^^^^^^^^^^^^^

**[CODE]** L2B/L2C take **three** 10-day L2A files (the CLI expects 3 or 4
dependencies: three science files plus at least one message file) and produce
one monthly product. There is no monthly lookup table - the month is whatever
the three inputs span. Internally L2B works **per day of year**, so the monthly
product's ``epoch`` dimension is one entry per day that had any event, with the
value set to the **mean epoch of that day's events** (not midnight).

.. warning::

   ``epoch_to_doy`` is used as the grouping key, so day-of-year, not a full
   date, is the identity of a daily bin. A monthly product that spans a New Year
   boundary relies on ``dict.fromkeys`` to preserve encounter order (so DOY 365
   sorts before DOY 1), but a product spanning **more than one year** would
   collide DOYs from different years into the same bin. At a one-month cadence
   this cannot happen in practice; do not extend the cadence without fixing it.

The L0 time-tagging problem
---------------------------

.. important::

   **An L0 file's date does not tell you which events are inside it.**

**[DOC]** Section 3.4 is explicit that the CCSDS secondary header time
(``SHCOARSE`` / ``SHFINE``) "describes when the packet was generated" and is
"distinct from the dust-impact event time, which is reconstructed from the FPGA
metadata header."

Because IDEX stores events onboard and transmits them later in transmit mode,
the packet-generation time is effectively a **downlink** time. L0 files are
organized by that time. So:

* Events from a single day can be spread across **several** L0 files.
* A single L0 file can contain events from **several different days**, possibly
  far apart.
* An event's correct 10-day window may have been "closed" days before the packet
  carrying it was ever generated.

**[CODE]** Every level deals with this the same way - over-query the inputs,
then filter on the reconstructed event epoch:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Level
     - How it copes
   * - L1A
     - ``idex_l1a(packet_files, window_start_date)`` accepts a **list** of L0
       files, parses all of them, concatenates per product type along ``epoch``,
       sorts, drops duplicate epochs, and *then* masks to
       ``[window_start, window_end)``. The input list is sorted first
       (``sorted(packet_files)``) so that ``drop_duplicates("epoch",
       keep="last")`` keeps the **highest file version** when the same event
       arrives in two files.
   * - L1B
     - ``cli.py`` notes: *"since there may be events that occur before the start
       date of the job, there is a buffer added to the upstream dependency
       query. This means that there may be multiple l1a science files that are
       returned but we only want to process the file with the same start date."*
       It therefore filters ``science_files`` down to the one whose **filename**
       contains ``self.start_date``, and raises ``ValueError`` if none matches.
   * - L2A
     - Takes ``science_files[0]`` - a single L1A file - plus two ancillary
       calibration files.
   * - L2B
     - Selects inputs by descriptor (``sci-10days`` and ``msg-10days``),
       de-duplicates the housekeeping files with ``set()``, and sorts **both**
       lists by first epoch before concatenating.

Do not simplify any of this to "one input file per job". The over-query is
deliberate and the filtering is what makes the products correct.

.. note::

   The *buffering* half of the fix lives outside this repository - it is the
   SDC's dependency query that decides how many L0 or L1A files to hand a job.
   This repository only implements the filtering half. If events start going
   missing from products, the first question is whether the upstream query
   buffer is wide enough, not whether ``idex_l1a()`` is filtering correctly.

CLI wiring and dependency counts
--------------------------------

**[CODE]** ``imap_processing/cli.py``, ``class Idex``. The dependency-count
checks are strict and are the first thing to fail when the SDC query changes.

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Level
     - Dependencies
     - Behavior
   * - ``l1a``
     - at most 2
     - ``ValueError`` if ``len(dependency_list) > 2``. Passes **all**
       ``source="idex"`` file paths plus ``self.start_date`` to ``idex_l1a()``,
       which returns a **list** of datasets (science, message, catlst - whichever
       were present and non-empty in the window).
   * - ``l1b``
     - 3 for ``sci-10days``, 1 otherwise
     - ``ValueError`` on a mismatch. The three for science are the L1A file, the
       SPICE kernels and the spin data. Selects the L1A file by start date, then
       calls ``idex_l1b(load_cdf(file), self.descriptor)``.
   * - ``l2a``
     - exactly 3
     - One L1B science file plus the two calibration CSVs. Ancillary files are
       keyed by ``path.stem.split("_")[2]``, i.e. the descriptor segment of
       ``imap_idex_l2a-calibration-curve-t-rise_20250101_v002.csv`` becomes
       ``l2a-calibration-curve-t-rise``.
   * - ``l2b``
     - 3 or 4
     - Three ``sci-10days`` L2A files and one or more ``msg-10days`` L1B files.
       Returns both the L2B and the L2C dataset.
   * - ``l2c``
     - --
     - **No branch.** Falls through to ``NotImplementedError``.

Example invocations::

    imap_cli --instrument idex --level l1a --start-date 20260101 \
        --descriptor sci-10days
    imap_cli --instrument idex --level l1b --start-date 20260101 \
        --descriptor sci-10days
    imap_cli --instrument idex --level l2a --start-date 20260101 \
        --descriptor sci-10days
    imap_cli --instrument idex --level l2b --start-date 20260101 \
        --descriptor sci-1mo

The catalog-list oddity
-----------------------

**[CODE]** ``PacketParser.__init__`` loops over ``{"l1a": raw_datasets_by_apid,
"l1b": derived_datasets_by_apid}``. Event messages are produced from the raw
dictionary only, guarded by ``level == "l1a"``. The catalog list has **no such
guard**, so a single L1A job emits *both* ``l1a_catlst-10days`` (raw DN) and
``l1b_catlst-10days`` (derived engineering units).

Consequences worth knowing:

* An L1B ``catlst`` product exists that ``idex_l1b()`` cannot produce -
  ``idex_l1b()`` raises ``ValueError`` for any descriptor that is not
  ``sci-10days`` or ``msg-10days``.
* The processing of ``catlst`` is a passthrough plus an ``epoch`` computed from
  ``shcoarse`` / ``shfine`` - i.e. tagged by **packet creation time**, which for
  a catalog summary is the right answer.
* The algorithm document does not mention the catalog-list product at all
  (section 4.6.2 states L1A produces only ``sci`` and ``msg``), although
  section 4.2 mentions "catalog-list products" in passing.
