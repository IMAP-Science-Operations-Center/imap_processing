.. _hit-ialirt:

I-ALiRT: The Real-Time Space Weather Product
============================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Sections 2.5.1, 2.5.3, 4.4 and 8 of the algorithm document.

I-ALiRT (IMAP Active Link for Real-Time) is the continuous low-rate broadcast
used for space weather monitoring. HIT's contribution is **12 rates at a
1-minute cadence**: 6 electron, 4 proton and 2 helium.

This product does **not** produce a CDF. It produces a ``list[dict]`` destined
for the I-ALiRT database, and it lives outside the ``hit`` package, in
``imap_processing/ialirt/l0/process_hit.py``.

Why HIT can measure electrons at all
------------------------------------

**[DOC]** Two of HIT's ten apertures (``A0`` and ``B0``) are **I-ALiRT
apertures**. Behind their L1 detectors sits a 1500 um **L4** detector split
into an inner region (``L4Ai``/``L4Bi``) and an outer annulus
(``L4Ao``/``L4Bo``).

Energetic ~0.5-1 MeV electrons are identified by requiring L1 **and** the outer
annulus of L4, **in anticoincidence with the central region of L4** - a
particle that also lights up the centre is a proton, not an electron. This is
the one place where HIT's detector stack is optimised for electrons rather than
ions.

The I-ALiRT apertures are also **never affected by the dynamic thresholds** -
they stay in their nominal configuration even during the largest SEP events,
which is precisely when space weather monitoring matters most.

The packet: a 60-slot subcommutation
------------------------------------

**[DOC]** Section 8.1. APID **1253**, 54 bytes, **one packet per second**.

Each second carries **6 bytes of rate data** arranged as three 2-byte fields:

.. code-block:: text

   FAST_RATE_1    2 bytes    cycles through 4 values (period 4)
   FAST_RATE_2    2 bytes    cycles through 4 values (period 4)
   SLOW_RATE      2 bytes    cycles through 60 values (period 60)

A **subcom counter** 0-59 says which slot this second carries. **A full I-ALiRT
set is 60 consecutive seconds = 1 minute**, which is why the public products
have a 1-minute cadence even though the packets arrive at 1 Hz.

The fast rates repeat every 4 slots, so each of them is sampled 15 times per
minute:

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - subcom mod 4
     - ``FAST_RATE_1``
     - ``FAST_RATE_2``
   * - 0
     - L1A Trigger
     - L1B Trigger
   * - 1
     - IAevent Trigger
     - IBevent Trigger
   * - 2
     - NFORMAT
     - Aevent Trigger
   * - 3
     - L3A Trigger
     - L3B Trigger

The slow rate is the interesting one - 60 distinct quantities per minute,
covering singles, the 20 dedicated I-ALiRT rates, event counters, the ERATES
livetime block, coincidence rates, and five duplicated H/He science rates.
The full mapping is algorithm document **Table 38** (PDF pages 145-147).

The 20 dedicated I-ALiRT rates
------------------------------

**[DOC]** Section 8.2 and Table 39. Slow-rate slots 16-35 carry
``I-ALiRT Rate 1`` through ``I-ALiRT Rate 20``. These are the **same** 20 rates
that appear in the science frame as ``ialirtrates`` (bytes 1148-1187, Table
23), so the two paths are cross-checkable.

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Rates
     - Quantity
     - Meaning
   * - 1-6
     - ``L4Ai`` dE bins 0-5
     - Six horizontal slices of energy deposit in the A-side inner L4.
       Rates 1-2 are **below** the minimum-ionising peak (< ~270 keV); 3-4
       are **within** it (~270 keV - 1 MeV); 5-6 are **above** it (> ~1 MeV).
   * - 7-10
     - ``L4Ai`` vs ``L3A`` bins 0-3
     - Four boxes in the (L4, L3) energy-loss plane. In order of increasing
       L3 deposit: (1) ~1-10 MeV solar electrons, (2) ~30-200 MeV solar
       protons, (3) >~500 MeV galactic protons, (4) everything else
       (background).
   * - 11-16
     - ``L4Bi`` dE bins 0-5
     - B-side mirror of 1-6.
   * - 17-20
     - ``L4Bi`` vs ``L3B`` bins 0-3
     - B-side mirror of 7-10.

Algorithm document Figure 15 shows the simulated L4 energy-loss distributions
that these boxes were drawn on.

The 12 public products
----------------------

**[DOC]** Section 8.2, and this is the whole algorithm:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Product
     - Formula
   * - HIT low energy e- A side
     - ``I-ALiRT Rate 1 + I-ALiRT Rate 2``
   * - HIT medium energy e- A side
     - ``I-ALiRT Rate 5 + I-ALiRT Rate 6``
   * - HIT high energy e- A side
     - ``I-ALiRT Rate 7``
   * - HIT low energy e- B side
     - ``I-ALiRT Rate 11 + I-ALiRT Rate 12``
   * - HIT medium energy e- B side
     - ``I-ALiRT Rate 15 + I-ALiRT Rate 16``
   * - HIT high energy e- B side
     - ``I-ALiRT Rate 17``
   * - HIT low energy H omni
     - ``H (6.0-8.0 MeV/nuc) L23FG``
   * - HIT medium energy H omni
     - ``H (12.0-15.0 MeV/nuc) L23FG``
   * - HIT high energy H A side
     - ``I-ALiRT Rate 8``
   * - HIT high energy H B side
     - ``I-ALiRT Rate 18``
   * - HIT low energy He omni
     - ``He-4 (6.0-8.0 MeV/nuc) L23FG``
   * - HIT high energy He omni
     - ``He-4 (15.0-70.0 MeV/nuc) L23FG``

**[DOC]** These are explicitly labelled *"simplified algorithms (TBD)"*.
Section 8.2 states: *"After launch, the algorithms will be updated to use the
high energy SEP proton box as a background subtraction for the high energy
electrons."* In other words, **rate 8 will eventually be subtracted from rate
7** (and 18 from 17) with some coefficient. That is not defined yet.

The supplemental (non-public) rates
-----------------------------------

**[DOC]** Section 4.4 lists what else rides along in the I-ALiRT stream, all at
60-second cadence, so that the algorithms can be improved in flight:

* 26 singles and trigger rates (13 per side)
* 5 engineering rates (the ERATES block: livetime, NUMTRIG, NUMREJECT,
  NUMACCPHA, NUMACCNPHA)
* 3 species-identified proton rates
* 2 species-identified helium rates
* 12 rates (6 per side) of 1-D cuts on inner-L4 energy deposit - these are
  I-ALiRT Rates 1-6 and 11-16
* 8 rates (4 per side) of events triggering inner L4 and the corresponding L3
  but **not** the opposite-side L3 - these are I-ALiRT Rates 7-10 and 17-20

The implementation
------------------

**[CODE]** ``imap_processing/ialirt/l0/process_hit.py``.

``HIT_PREFIX_TO_RATE_TYPE`` is a dict of three lists that name each
subcommutation slot. ``FAST_RATE_1`` and ``FAST_RATE_2`` are generated as 15
repetitions of a 4-element pattern; ``SLOW_RATE`` is written out as 60 explicit
names.

``process_hit`` then:

#. Computes MET from ``sc_sclk_sec`` and ``sc_sclk_sub_sec`` with
   ``calculate_time(..., 256)`` - **LSB = 1/256 s**, per the 7516-9054 GSW-FSW
   ICD.
#. Calls ``find_groups(xarray_data, (0, 59), "hit_subcom", "met")`` to collect
   each minute's 60 packets.
#. **Rejects any group whose ``hit_subcom`` values are not exactly
   ``np.arange(60)``** - no duplicates, no gaps, in order. Rejected groups are
   logged at INFO and skipped entirely; there is no partial-set path.
#. Logs a **warning** for any group containing a zero ``hit_status`` value, but
   still emits the record.
#. ``create_l1`` zips the three name lists against the three data arrays into a
   flat dict.
#. Emits one dict per group with the 12 products above, each as a
   ``Decimal`` formatted to 3 decimal places, plus ``instrument: "hit"`` and
   ``hit_epoch`` from ``met_to_ttj2000ns(hit_met)``.

The 12 output keys are:

.. code-block:: text

   hit_e_a_side_low_en     hit_e_a_side_med_en     hit_e_a_side_high_en
   hit_e_b_side_low_en     hit_e_b_side_med_en     hit_e_b_side_high_en
   hit_h_omni_low_en       hit_h_omni_med_en
   hit_h_a_side_high_en    hit_h_b_side_high_en
   hit_he_omni_low_en      hit_he_omni_high_en

which map one-to-one onto the document's 12 products.

.. warning::

   **[CODE]** The slow-rate slot names in
   ``HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"]`` **do not match Table 38 of
   document version 1.11.00** in three places. The document's revision history
   for 1.11.00 says *"JGM updated Table 38 to reflect what is actually in the
   packets"*, so the code is almost certainly working from the previous
   revision.

   .. list-table::
      :header-rows: 1
      :widths: 14 40 46

      * - Slot
        - Code name
        - Table 38 (v1.11.00)
      * - 8, 9, 10
        - ``SLOW_RATE_08``, ``SLOW_RATE_09``, ``SLOW_RATE_10``
        - ``L1B``, ``L2B``, ``L3B``
      * - 38, 39
        - ``NASIDE_IALRT``, ``NBSIDE_IALRT``
        - ``NFORMAT``, ``NASIDE``
      * - 47-54
        - ``L12B``, ``L123B``, ``PENB``, ``SLOW_RATE_51``..``SLOW_RATE_54``
        - ``L12B``, ``2TEL``, ``PENA``, ``PENA?``, ``L12B``, ``2TEL``,
          ``PENA``, ``PENA?``

   **None of the 12 public products read any of the affected slots**, so the
   output is currently correct. But the labels are wrong, and anyone adding a
   product that uses them will get the wrong quantity. Note also that Table 38
   itself repeats ``L12B``/``2TEL``/``PENA``/``PENA?`` at slots 47-50 and
   51-54, which looks like a copy-paste artefact in the document - confirm with
   the HIT team before changing the code. See :ref:`hit-gap-ialirt-slots`.

   The code comment also cites "Table 37 of the HIT Algorithm Document"; in
   v1.11.00 this is **Table 38** (Table 37 is now the summed-rate geometry
   factors).

Relationship to the science frame
---------------------------------

Worth keeping straight, because the same numbers appear twice:

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Quantity
     - In the science frame
     - In the I-ALiRT stream
   * - The 20 I-ALiRT rates
     - ``ialirtrates`` (APID 1252, bytes 1148-1187), decompressed at L1A,
       rate-converted at L1B, **not used at L2**
     - Slow-rate slots 16-35 (APID 1253), used to build the 12 public
       products
   * - The five H/He science rates
     - ``l3fgrates`` entries (Range 3 foreground)
     - Slow-rate slots 55-59, duplicated for real-time use
   * - The ERATES block
     - bytes 6-23
     - Slow-rate slots 40-44

The science-frame copies go through the full livetime and intensity chain; the
I-ALiRT copies do not. They are **raw counts** and are reported as such.
