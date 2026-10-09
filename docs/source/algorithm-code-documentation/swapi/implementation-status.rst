.. _swapi-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/swapi`` and
``imap_processing/ialirt/l0/process_swapi.py`` against algorithm document
version 07 (2026-06-03). If you change something material, update this page in
the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Level
     - State
     - Notes
   * - L0 / L1 science
     - **Mature, with a real gap**
     - Grouping, reordering, decompression and flags are complete and tested.
       **Three of the five document rejection criteria are missing**, including
       checksum verification. Uncertainty is Poisson-only by design (the
       compression term has not been delivered).
   * - L1A / L1B housekeeping
     - **Complete**
     - There is no algorithm - raw and derived decommutation of the same
       packet. All 102 fields.
   * - L2
     - **Complete for what it claims**
     - Rates and the energy solve both work and the solve is more general than
       the document. Missing the thruster flag and any energy uncertainty.
   * - I-ALiRT
     - **Working, several undocumented choices**
     - Produces records; the fit window, the averaging window, the
       :math:`R^2` threshold and the ``+8`` decompression offset all differ
       from or extend the document.
   * - L3a / L3b
     - **Out of scope, correctly**
     - The SWAPI team's Docker container. See :ref:`swapi-l3-scope`.
   * - Quick look
     - **Not started**
     - Section 15 specifies spectrograms. No SWAPI quicklook code exists here.

Hard failures in the code
-------------------------

Explicit exceptions, so you know what a bad input looks like:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Location
     - Condition
   * - ``cli.py`` ``Swapi.do_processing``
     - ``NotImplementedError`` for any data level other than l1/l1a/l2.
   * - ``cli.py`` ``Swapi.do_processing``
     - ``ValueError`` if the dependency count is not exactly 3 (l1 sci), 2
       (l1 hk) or 3 (l2).
   * - ``swapi_l1.swapi_l1``
     - ``ValueError`` if not exactly one L0 ``raw`` file, or (for ``sci``) not
       exactly one L1 ``hk`` file.
   * - ``swapi_l1.decompress_count``
     - ``ValueError`` if any count is negative (the field must be unsigned).
   * - ``swapi_l2.solve_full_sweep_energy``
     - ``ValueError`` if no ESA table entry exists for a sweep table id even at
       the earliest timestamp.
   * - ``swapi_l2.solve_full_sweep_energy``
     - ``ValueError`` if an ``esa_lvl5`` hex value is not in the LUT notes
       ``ESA DAC (Hex)`` column.
   * - ``process_swapi.process_swapi_ialirt``
     - ``ValueError`` if the ESA unit conversion table has no rows for the
       packet's ``swapi_version`` sweep id.

Missing L1 rejection criteria
-----------------------------

**This is the most substantive gap in the SWAPI pipeline.** Document section
10.1.2 lists five conditions under which L0 data must be marked ``NaN``.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Criterion
     - State
     - Detail
   * - ``SWP_SCI.MODE`` is not ``HVSCI``
     - **Done**
     - ``filter_good_data``, all 12 packets must be HVSCI.
   * - ``SWP_HK.CHKSUM`` is wrong
     - **Missing**
     - The field is parsed by the XTCE (``SWP_HK.CHKSUM`` and
       ``SWP_SCI.CHKSUM``) and never checked. The document specifies the
       algorithm precisely - "an 8-bit xor of all packet bytes (including the
       CCSDS header), except for the checksum byte" - and notes that it is
       computed on board *and* on the ground. Implementing it needs raw packet
       bytes, which ``packet_file_to_datasets`` does not return, so this is not
       a one-line fix.
   * - Saturation above 4.0 MHz
     - **Missing**
     - "If count rates exceed 4.0 MHz using ``SWP_SCI.PCEM_CNT0`` through
       ``PCEM_CNT5`` or ``SCEM_CNT0`` through ``SCEM_CNT5`` the sweep may later
       be discarded." At 0.145 s live time, 4.0 MHz is ~580,000 counts - i.e.
       compression region 2 or 3. Cheap to add once decompressed.
   * - ``SWP_HK.PCEM_RATE_ST == 1`` during the sweep
     - **Missing**
     - The field exists in the XTCE and in the HK products. It is neither a
       rejection criterion nor a bit in ``SWAPIFlags``. Note the near-miss:
       ``PCEM_CNT_ST`` **is** flagged (bit 7), and it means something
       different - "tripped but handled by FSW" versus "still exceeded despite
       FSW measures".
   * - ``SWP_HK.SCEM_RATE_ST == 1`` during the sweep
     - **Missing**
     - as above, and ``SCEM_CNT_ST`` is bit 8.

Consequence: a sweep taken while the detector was saturated, or one whose
packets failed their checksum, currently flows through L1 and L2 unmarked. L3
assumes L2 is already clean.

Missing flags
-------------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Flag
     - State
     - Detail
   * - ``SWP_SC_THRUSTER``
     - **Missing**
     - **[DOC]** Section 10.2 adds this to ``SWP_L1_FLAGS`` at L1-L2 to
       indicate data taken during spacecraft thruster activity, "namely, daily
       repointing maneuvers". ``swapi_l2.py`` has
       ``# TODO: add thruster firing flag``; ``SWAPIFlags`` has no bit for it.
       Needs the MOC thruster firing history as an ancillary input, which the
       SDC does not currently deliver to this pipeline.
   * - "other flags"
     - **Missing**
     - ``swapi_l2.py`` ``# TODO: add other flags``. Unspecified.
   * - ``PRELIMINARY_MAG``
     - N/A
     - L3a alpha only. Correctly absent.
   * - ``FIT_ERROR`` / ``BAD_FIT``
     - N/A
     - L3a only. Correctly absent.

Uncertainty gaps
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Item
     - State
     - Detail
   * - Poisson term
     - **Done**
     - :math:`\sqrt{N}` at L1, divided by :math:`t_{\mathrm{live}}` at L2.
   * - Compression term
     - **Missing**
     - **[DOC]** "The compression of counts also contributes to the
       uncertainty. This uncertainty is estimated in an empirical way ...
       generate some count rate numbers that span the range of higher count
       rates, compress and un-compress these count rates and compare the
       difference between the two cases. These differences are then used to
       construct an empirical expression to estimate the error." The empirical
       expression has never been delivered; the code carries an explicit
       ``TODO``. Note that :math:`\sqrt{N}` on decompressed counts
       **understates** the true uncertainty for every compressed sample - which
       is exactly the high-rate solar wind core.
   * - Asymmetric uncertainties
     - **Deferred, per the document**
     - ``_plus`` and ``_minus`` are identical arrays at both levels. The
       document explicitly allows this ("will be the same, except if we modify
       them to be asymmetric uncertainties in the future"), so the duplicated
       variables are placeholders for a decision not yet made, not a bug.
   * - Energy uncertainty :math:`\Delta(E/q)`
     - **Missing**
     - Required by the L3b flux uncertainty (equation 14). The passband edges
       are in the LUT notes table's unused ``Lower Energy`` / ``Upper Energy``
       columns.

Deviations from the document
----------------------------

These are design decisions or judgement calls, not bugs, but they will surprise
anyone reading the document first.

Product descriptors do not match the document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** the L1 and L2 descriptor is ``12s``.

**Code:** the descriptors are ``sci`` and ``hk``; logical sources are
``imap_swapi_l1_sci``, ``imap_swapi_l1a_hk``, ``imap_swapi_l1b_hk``,
``imap_swapi_l2_sci``. ``12s`` appears nowhere.

**Consequence:** cosmetic for processing, but the document's filename table is
not a reliable guide to what is on disk. Do **not** rename products to match -
the SDC file catalog and every downstream consumer key on the current strings.

The quality gate uses grouping criteria as rejection criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** constant ``PLAN_ID`` and ``SWEEP_TABLE`` across the 12 packets is
how you *group* packets into a sweep (10.1.1 step 3.b.ii). The rejection
criteria are the five in 10.1.2.

**Code:** ``filter_good_data`` groups by timestamp continuity and then
*rejects* sweeps whose ``plan_id`` or ``sweep_table`` varies. Same outcome for
well-formed data; different outcome at a plan change boundary, where the
document's reading would start a new sweep and the code's discards one.

Sweep grouping tests time, not sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** "the entire sweep is marked by ``SEQ_NUMBER`` 0-11; all packets
must be present to process a sweep."

**Code:** ``find_sweep_starts`` requires ``seq_number == 0`` at the start and
then eleven consecutive 1-second timestamp gaps. It does not verify that the
intervening sequence numbers are 1..11.

**Note:** the I-ALiRT code does the stricter check
(``np.array_equal(seq_values, np.arange(12))``). The two grouping
implementations are not equivalent, which is worth knowing if the two products
ever disagree about which sweeps exist.

The overflow sentinel is a finite number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** on overflow, "``actual_value = <some constant that indicates
overflow>``".

**Code:** ``np.iinfo(np.int32).max`` = 2147483647, cast to ``float32``.

**Consequence:** it survives into L2 as a count rate of ~1.5e10 Hz. It is not
``NaN``, not a CDF ``FILLVAL``, and not flagged. Any downstream statistic that
does not filter on ``VALIDMAX`` will be destroyed by a single overflow sample.
The compression flag bit *is* set for those samples, so they are detectable -
but only if you know to look.

Fine-step index flooring is silent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** the fine-step energies are at rows ``r_p16 - 4k``. It does not
say what to do if that goes negative.

**Code:** negative ladder indices are clamped to 0, per SWAPI instruction
("flooring"). Nothing records that it happened - no flag, no log line - so
several fine steps can silently share one energy.

L2 blanks negative rates but not their uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Code:** ``l2_dataset[var].where(l2_dataset[var] >= 0, np.nan)`` is applied to
the three rate variables only. A step can therefore have a ``NaN`` rate and a
finite uncertainty.

The ``k`` factor is never used
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** :math:`k_{L2} = 1.93` eV/V/e is "the ``k``-factor estimated
pre-launch from lab measurements and used to convert from ESA energy in the L2
CDF files to the actual ESA voltage of the instrument", and it differs from the
SIMION value :math:`k^{*} = 1.89` for reasons still under investigation.

**Code:** the ADP's ``K factor`` and ``Voltage`` columns are read into the
DataFrame and ignored; L2 takes ``Energy`` directly. This is correct arithmetic
but the L2 product does not record which ``k`` its energies were built with,
and the ADP's own value changed between deliveries (1.88 in the 2025-02-11
rows, 1.93 in the 2025-05-19 rows).

I-ALiRT: fit window is 2+1+2, not 3+1+2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** "three energy bins on the left and two on the right of
:math:`E_{\mathrm{peak}}`" - six points.

**Code:** ``range(max_index - 2, max_index + 2 + 1)`` - five points, symmetric.

**Consequence:** one fewer constraint, and asymmetric relative to the document,
which matters because the low-energy side of the proton peak is where the
shoulder lives. Worth asking the SWAPI team which is intended.

I-ALiRT: averaging window is trailing and geometric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** "5 sweeps averaged (2 sweeps before, 2 sweeps after, and 1
current)" - a **centred** window. The kind of average is not specified.

**Code:** ``swapi_met_list[-5:]`` - the current sweep and the 4 before it, a
**trailing** window - and a **geometric** mean in log space, with ``NaN``
sweeps excluded.

**Consequence:** the reported ``swapi_epoch`` is the mean MET of the window, so
the timestamp is honest, but the record is offset ~24 s later than a centred
average would place it. For a real-time stream, trailing is arguably the right
engineering choice (a centred window cannot be computed until 2 sweeps in the
future have arrived, costing 24 s of latency) - but it is a deviation and it is
undocumented in the code.

I-ALiRT: 63 energy steps, not 62
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** the I-ALiRT input is "coincidence count rates ... for 62 coarse
energy bins (1 sweep)".

**Code:** ``NUM_IALIRT_ENERGY_STEPS = 63`` and the arrays are truncated to
``[:, :63]``.

**Consequence:** step 0 - the ESA ramp-up step - is included. The science
pipeline explicitly ``NaN``\ s it; the I-ALiRT path does not. Since the fit
window is centred on the peak, which is far from step 0, this is unlikely to
change a result, but ``np.argmax`` over an array that includes a ramp-up
sample is not obviously safe.

I-ALiRT: undocumented decompression offset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Code:** ``raw_coin_count = raw_coin_count * 16 + 8``, unconditionally, with
the rationale in a comment ("add 8 to avoid having counts truncated to 0 and to
avoid counts being systematically too low"). The document says nothing about
I-ALiRT count compression at all. The mid-bin correction is defensible; it is
just not written down anywhere but the comment.

I-ALiRT: :math:`R^2` threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Code:** :math:`R^2 \ge 0.7` to accept a fit. The document specifies
:math:`R^2 \ge 0.9` for the **L3a** fits and gives no threshold for I-ALiRT.
The 0.7 is the code's own choice.

I-ALiRT: density factor placement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** correct the pseudo density by scaling it to :math:`1/e` times the
5-sweep-averaged fitted density.

**Code:** multiplies the *model* density by :math:`e` inside ``count_rate``,
which produces the same fitted result. The constant is named
``temporary_density_factor`` and the comment says it is "to be replaced once
SWAPI's L3 processing pipeline is finalized". Algebraically fine; flagged here
because it is applied before averaging rather than after, and because it is an
empirical fudge with a shelf life.

Not written at all
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Item
     - Notes
   * - Quick-look products
     - **[DOC]** Section 15: colour-coded spectrograms of count rate vs
       energy/charge vs time. Figure 25 shows the daily plot - COIN rate vs
       E/q (top), spectrogram (middle), MAG L1D :math:`B_R, B_T, B_N, |B|`
       (bottom), with the He\ :sup:`+` PUI cutoff energy marked. No SWAPI code
       in this repository produces it.
   * - ``SWP_LGSCI`` processing
     - "Large" science - all 6 ESA levels per sample instead of one, used for
       testing. Not in the XTCE, no APID enum, not processed.
   * - ``SWP_AUT`` processing
     - APID 1192 is in ``SWAPIAPID`` but there is no XTCE container and no
       code path. Autonomy power off/cycle flags.
   * - ``SWP_MG`` / ``SWP_MD`` processing
     - Event messages and memory dump. Not in the XTCE, not processed. Probably
       correctly - they are engineering-only.
   * - ``orbnum`` in the L1 product
     - **[DOC]** Table 2 lists it ("orbit number in which observation began,
       generated by MOC and stored at SDC"). Absent from the CDF.
   * - ETE-model validation data
     - **[DOC]** Section 12 defines the intended verification method: the SWAPI
       team runs the end-to-end model and supplies input/output pairs so the
       SDC can verify the implementation. No such pairs are in the repository.
       The only SWAPI-supplied validation data is a decommutation check on one
       pre-launch idle packet.
   * - Compression-uncertainty empirical expression
     - See above. Blocked on the SWAPI team.

Test coverage
-------------

**[CODE]** ``imap_processing/tests/swapi/`` and
``imap_processing/tests/ialirt/unit/test_process_swapi.py``.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Area
     - Coverage
     - Notes
   * - Decommutation
     - **Validated**
     - Field-by-field against SWAPI CSV exports for the first ``SWP_SCI`` and
       ``SWP_HK`` packet, plus packet counts.
   * - Sweep grouping
     - **Unit tested**
     - ``find_sweep_starts``, ``get_indices_of_full_sweep``,
       ``filter_good_data``.
   * - Decompression
     - **Unit tested**
     - ``decompress_count``.
   * - Reordering
     - **Unit tested**
     - via ``test_swapi_algorithm`` / ``test_process_swapi_science``.
   * - L1 CDF write
     - **Tested**
     - Round trip through ``write_cdf()``.
   * - L2 rates
     - **Tested**
     - Rate and uncertainty arrays checked against
       ``counts / SWAPI_LIVETIME``.
   * - L2 energy solve
     - **Tested**
     - ``test_solve_full_sweep_energy`` pins the 63 fixed energies and the 9
       fine energies for ``esa_lvl5 = 4663``, sweep table 0.
   * - L2 CDF attributes
     - **Heavily tested**
     - ISTP attributes, ``DEPEND_1`` rewiring, ``VALIDMIN``/``VALIDMAX``,
       ``DELTA_PLUS_VAR``/``DELTA_MINUS_VAR``.
   * - I-ALiRT
     - **Unit tested**
     - ``count_rate``, ``optimize_pseudo_parameters``, ``geometric_mean``,
       and the full ``process_swapi_ialirt`` output keys.
   * - Overflow sentinel handling
     - **Untested end to end**
     - ``decompress_count`` is tested; nothing checks what the sentinel does to
       L2 rates.
   * - Index flooring
     - **Untested**
     - The clamp in ``solve_full_sweep_energy`` has no test.
   * - Flight-representative science
     - **Not tested**
     - The only L0 test data is pre-launch idle data from 2024-09-24.

Where to start, if you are picking up work
------------------------------------------

Rough order of value, highest first:

1. **The three missing L1 rejection criteria** (saturation, ``PCEM_RATE_ST``,
   ``SCEM_RATE_ST``). Cheap, and they are the difference between "L2 is clean"
   being true and being assumed. Saturation needs a threshold check on
   decompressed counts; the two ``RATE_ST`` conditions need two more
   ``SWAPIFlags`` bits and two more entries in ``hk_flags_name``.
2. **Do something visible about the overflow sentinel** - at minimum a test
   showing what it produces at L2, ideally a fill value or a flag.
3. **The thruster flag.** Blocked on an ancillary input, so the real work is
   agreeing with the SDC on how thruster history reaches this pipeline.
4. **Checksum verification.** Real work, because it needs raw packet bytes that
   ``packet_file_to_datasets`` does not surface.
5. **Reconcile the I-ALiRT fit window** with the document, or record the
   deviation as intentional.
6. **Ask the SWAPI team for the compression-uncertainty expression** and for
   ETE input/output validation pairs. Both are blocking, neither is ours to
   invent.
