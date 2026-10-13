.. _glows-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/glows``,
``imap_processing/cli.py`` (class ``Glows``),
``imap_processing/ancillary/ancillary_dataset_combiner.py`` and
``imap_processing/quality_flags.py``. If you change something material, update
this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 12 24 64

   * - Level
     - State
     - Notes
   * - L0 / L1A
     - **Mature**
     - Both APIDs decommutated; all three direct-event compression markers
       implemented; multi-packet reassembly with gap tracking; dedup and
       invalid-time filters. Validated against the GLOWS team's JSON.
   * - L1B histograms
     - **Substantially complete, several stubs**
     - Decoding, SPICE geometry and sky masking all work. Two of the seventeen
       bad-time flags are hard-coded to "good", and the excluded-region radius
       ignores the configured value.
   * - L1B direct events
     - **Complete for what it is**
     - Times, housekeeping decode and the 11 DE flags are all there. Pulse
       lengths are not converted to µs and ``unique_identifier`` is missing.
       This is the end of the DE pipeline by design.
   * - L2
     - **Complete for the nominal path**
     - Co-adding, exposure, calibration, position-angle conversion and sky
       coordinates all work. Per-bin exclusion, the active-bad-angle mask, and
       the rejection counters are not implemented.
   * - L3A-L3E
     - **Out of scope**
     - Belongs to a separate repository closer to the science team.

There are **no** ``NotImplementedError`` raises anywhere in the GLOWS code apart
from the generic unknown-data-level branch in ``Glows.do_processing``.

Not implemented at all
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Detail
   * - **Ground-generated histograms from direct events**
     - **[DOC §8.3, §3.5.1, §12.5.2]** GLOWS can rebuild block histograms on the
       ground from downlinked DEs, using "Algorithm 2" (bin incrementing once
       per block, because the full GLOWS-vs-IMAP clock history is not
       downlinked). The resulting structures would be tagged
       ``is_generated_on_ground`` and fed into the normal pipeline. **None of
       this exists.** ``is_generated_on_ground`` is hard-coded ``False`` at L1A.
       The document calls this a contingency capability, not routine operations.
   * - **``is_beyond_daily_statistical_error`` (flag 11)**
     - **[DOC §12.7.1, Eqs. 45-46]** Reject blocks whose total counts fall
       outside ``C_block ± n_reject·√C_block``, with ``n_reject`` between 3 and 4
       and ``C_block`` determined **daily**. This is the primary
       particle-background rejection mechanism and arguably the most important
       single filter GLOWS has. **[CODE]** ``np.uint8(1)`` with the comment
       *"Placeholder until daily histogram is available in glows_l1b.py"* and
       *"TODO: this equation needs to be clarified"*. The thresholds
       (``n_sigma_threshold_lower``/``_upper``) are sitting unused in the
       settings file.
   * - **``is_spin_period_difference_beyond_threshold`` (flag 16)**
     - **[DOC Table 3.10 item 30.17]** Compare the onboard and ground spin
       periods and flag when they disagree. Both values are computed at L1B and
       the threshold (``relative_difference_threshold``) is in the settings file.
       **[CODE]** The slot is filled with ``np.uint8(1)`` under the name
       ``is_beyond_background_error`` - a *different* condition, which the
       document marks TBC. The comparison is never made.
   * - **``bad_time_flag_occurrences``**
     - **[DOC §3.9.1 item 1, Table 3.13 item 24]** Count how many blocks were
       rejected for each of the 17 flags, so an analyst can see why a day is
       thin. **[CODE]** ``np.zeros((1, FLAG_LENGTH))`` with ``# TODO fill this
       in``. The information is trivially available at the point where
       ``return_good_times`` runs.
   * - **Per-bin exclusion at L2**
     - **[DOC §3.9.1 item 5]** ``HistogramL2.filter_bad_bins`` exists but returns
       its input unchanged, with two TODOs. There are also
       ``# TODO: bad angle filter`` / ``# TODO: filter bad bins out`` in
       ``__init__``. Today all bins from all good blocks contribute to the sum
       regardless of their bad-angle flags. That matches the document's *"the
       signal values are kept at Level-2 as they are"* for the **flux**, but the
       document also wants the active-flag mask applied to the **flags**.
   * - **``active_bad_angle_flags`` mask**
     - **[DOC §3.9.1, bad-angle masking]** *"inactive flags have zeroes set in
       Level-2 even if they are not zeroed in Level-1B"*. **[CODE]**
       ``PipelineSettings.active_bad_angle_flags`` is parsed and never read.
   * - **HV-test pre-filter before ``is_night`` transition detection**
     - **[DOC §3.9.1 item 2]** Blocks with ``is_hv_test_in_progress`` raised must
       be excluded before locating ``is_night`` transitions, because the monthly
       gain test's time-tagged command loads produce **fake transitions**.
       **[CODE]** ``apply_is_night_offsets`` looks at the raw ``is_night`` column
       only. This will misbehave once a month unless ``is_hv_test_in_progress``
       is separately active in the good-time mask (which it is in the bundled
       settings, but is *not* in the test settings file).
   * - **``angular_radius_for_excl_regions__deg``**
     - **[DOC §12.7.6]** *"will be probably set to the half of the nominal radius
       of the GLOWS FOV"*, delivered in pipeline settings. **[CODE]** L1B
       hard-codes ``np.deg2rad(0.1 / 2)`` = 0.05°, i.e. half a **bin width**,
       roughly 30× smaller than the bundled 2.0° setting and 10× smaller than the
       validation file's 0.5°. Excluded regions are therefore masked far more
       narrowly than the instrument team intends. **This is probably the most
       consequential single deviation on this page.**
   * - **``unique_identifier`` for direct events**
     - **[DOC Table 3.12 item 2]** Commented out in ``DirectEventL1B`` with a
       note that strings belong in attributes, not the data section.
   * - **Direct-event pulse length in µs**
     - **[DOC Table 3.12 item 13]** ``direct_event_pulse_lengths`` is the raw
       encoded value copied straight from ``de[2]``, not converted with the
       ``pulse_length`` entry of the conversion table. The document marks the
       unit "µs TBC".
   * - **``multi_event`` beyond L1A**
     - Carried in the L1A ``direct_events`` array, dropped at L1B with
       ``# TODO: where does the multi-event flag go?``. Harmless today - KPLabs
       say the flight software never sets it.
   * - **Full ancillary time series product**
     - **[DOC §12.6.1]** muses that a companion product holding the full
       spacecraft time series (rather than block averages and spreads) would be
       useful. Not implemented, and not formally required.
   * - **Point-source / temperature / HV-sensitivity / flight-calibrated
       products**
     - **[DOC §15]** Five future data products are sketched in the concluding
       remarks. None are specified in enough detail to implement and none exist.
   * - **Thruster operation flag**
     - **[DOC §12.1.2 item 4]** GLOWS wants to know whether thrusters fired in a
       given second. Marked TBD in the document; no corresponding input exists.

Suspected bugs
--------------

These need confirmation with the GLOWS team or a test before being changed.
Ordered by how likely they are to affect released data.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Location
     - Issue
   * - ``glows_l2_data.py``, ``HistogramL2.__init__``
     - ``circstd(np.radians(spin_axis_data[:, 0]), low=0, high=360)`` - the data
       are converted to **radians** but the range is given in **degrees**. The
       adjacent ``circmean`` call correctly uses ``high=2*np.pi``. The result is
       then passed through ``np.degrees``. This will produce a wrong
       ``spin_axis_orientation_std_dev`` longitude. The equivalent code in
       ``HistogramL1B.update_spice_parameters`` uses ``high=2*np.pi``
       consistently, so L1B is fine and only L2 is affected.
   * - ``glows_l1b_data.py`` vs. ``glows_l2_data.py``, position angle
     - ``position_angle_offset_average`` is computed **two different ways**.
       L1B: ``360 - get_spin_angle(get_instrument_spin_phase(imap_start_time,
       instrument=IMAP_GLOWS), degrees=True) + spin_offset_correction``, i.e. a
       per-block SPICE spin-phase lookup, with **no** modulo. L2:
       ``(360 - get_instrument_mounting_az_el(IMAP_GLOWS)[0] +
       spin_offset_correction) % 360``, i.e. a static mounting azimuth. L2 uses
       its own value for the ψ→ψ\ :sub:`PA` conversion and ignores the L1B
       variable it is writing out. The document (§10.6) says the quantity is
       ``360° - ψ_GLOWS``, constant, which matches the L2 form. Both are written
       into products; they need not agree.
   * - ``glows_l1b.py``, ``create_l1b_hist_output``
     - The ``flags`` variable is emitted on a dimension called ``flag_dim`` (from
       ``output_dimension_mapping``) while the dataset declares a coordinate
       ``bad_time_flags``. Both are length 17. Likewise ``glows_l2.py`` writes
       ``spin_axis_orientation_*`` on ``latitudinal``, which is never declared as
       a coordinate. Confirm against a written CDF; ISTP compliance may be
       affected.
   * - ``glows_l2_data.py``, ``HistogramL2.__init__``
     - ``self.identifier = int(repointing.replace("repoint", ""))`` where
       ``repointing = l1b_dataset.attrs.get("Repointing")``. If the attribute is
       absent this raises ``AttributeError: 'NoneType'``. Since GLOWS is
       organised strictly per pointing this should never happen, but the failure
       mode is opaque.
   * - ``glows_l1a_data.py``, ``_build_uncompressed_event``
     - ``seconds = values[0]`` takes all 32 bits where the document specifies a
       2-bit marker plus 30 bits of seconds. It happens to be correct because
       the marker bits are zero in both paths that reach this function, but it is
       correct by accident rather than by masking.
   * - ``glows_l1a.py``, ``generate_de_dataset``
     - The ``within_the_second`` axis is **zero padded** with no fill value and
       no per-epoch valid count, so a padded slot is indistinguishable from a
       genuine event at GLOWS time ``(0, 0)`` with zero pulse length. L1B then
       converts every slot, padding included, into
       ``direct_event_glows_times``.
   * - ``glows_l1b_data.py``, ``get_threshold``
     - Returns ``None`` when no key matches; the caller then evaluates
       ``value <= None`` and raises ``TypeError``. A settings file missing one
       threshold crashes L1B rather than failing loudly with a useful message.
   * - ``glows_l2_data.py``, ``return_good_times``
     - Uses ``print()`` rather than the module logger when the active-flag mask
       length does not match, and then continues with a mismatched boolean index.
       Repository convention is ``logger``; ``print`` is also used in
       ``Glows.do_processing``.
   * - ``glows_l0_data.py``, ``within_same_sequence``
     - Compares only ``SEC`` and ``LEN``, with ``# TODO: What other fields need
       to match?``. Two genuinely different second-groups with the same second
       and packet count would merge silently.
   * - ``glows_l1a_data.py``, ``HistogramL1A.__post_init__``
     - The ``ENDID`` vs. ``SPINS`` cross-check the document asks for (§3.4.1
       item 13) is present only as commented-out code, disabled because the
       emulator did not populate the fields correctly. Worth re-enabling once
       flight data is available.

Deviations from the algorithm document
--------------------------------------

These are design decisions, not bugs, but they will surprise anyone reading the
document first.

Flag polarity is inverted
^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC Table 3.10]** says ``false`` = normal, ``true`` = problem. **[CODE]**
``HistogramL1B.compute_flags`` returns ``1 = good, 0 = bad``, and
``return_good_times`` selects rows where all active flags are ``1``. Every
comparison against the GLOWS team's JSON validation output has to invert. This
is consistent within the codebase and there is no reason to change it, but it
must be documented at every boundary.

The DE flags at L1B are **not** inverted - they are copied straight through.

ψ→ψ\ :sub:`PA` conversion happens at L2, not L1B
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §12.6.2 item 6]** says the L1B histogram is re-arranged to the position
angle. **[DOC §3.9.1 item 9 and §3.14 item 2]** say the conversion happens at
L2. The document states that §3 supersedes §12 where they conflict, and the code
follows §3. L1B carries ``imap_spin_angle_bin_cntr`` in raw ψ.

Calibration is a division, not a multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC Eq. 53]** ``I_m = S_m × α``. **[CODE]** ``photon_flux = (counts /
exposure) / calibration_factor``. Since ``α`` is in **cps per Rayleigh**, the
code is dimensionally correct and the equation as printed is not. Do not
"correct" the code.

Conversion table values differ from Table 11.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The document's Table 11.1 gives ``hv_voltage`` as 16 bits over 0-56012.82 V (a
consequence of promoting a 12-bit ADC value to a 16-bit field). The delivered
conversion table uses **12 bits over 0-3500 V**. The code uses whatever the
delivered file says, which is right. Table 11.1 also flags the filter
temperature relation as *probably nonlinear*, needing a lookup table; the
``p01``-``p04`` slots in the JSON appear to be reserved for that and are
currently all zero and unused.

Excluded-region masking radius
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Covered above under "not implemented", but repeated here because it is a silent
science-affecting difference rather than a missing feature: the code masks
within **half a bin width (0.05°)** of an excluded-region point, not within the
configured ``angular_radius_for_excl_regions__deg``.

Masking algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §3.7.1]** describes the GLOWS team's implementation using
``astropy.coordinates.search_around_sky``. **[CODE]** uses SPICE frame
transforms and dot products of unit vectors instead. This is a legitimate
reimplementation - no astropy dependency, and it reuses the repository's
standard geometry helpers - but the two will not be bit-identical.

Single-time geometry approximations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* L1B bad-angle masking transforms look vectors at the **block start time**
  only, for a block spanning ~2 minutes.
* L2 ecliptic coordinates are computed at the **midpoint block** of the day and
  applied to the whole day.

Both are defensible (the spin axis is nominally fixed within a pointing) but
neither is what a strict reading of the document implies.

L2 start/end times
^^^^^^^^^^^^^^^^^^

**[DOC Table 3.13 items 3-4]** wants the UTC start and end of the observational
day. **[CODE]** uses the first and last *good block* epoch, each of which is a
block **midpoint**. The reported day is therefore shorter than the real one by
up to a block at each end, plus whatever was culled at the edges.

Exposure is uniform across bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Correct today, since no bins are dropped. It becomes wrong the moment per-bin
exclusion is implemented. Whoever implements ``filter_bad_bins`` must also
convert ``exposure_times`` from a scalar broadcast to a genuine per-bin
accumulation (document Eq. 49 is already written per bin).

Housekeeping is out of scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**[DOC §3, item 3]** lists HK Full and HK Brief as GLOWS telemetry categories,
and §12.5.3 points at a separate engineering document for their structure. No
GLOWS housekeeping APID is decommutated here, and none is planned. Everything
science processing needs (filter temperature, HV, spin period, pulse length) is
carried inside the science packets.

Cross-cutting notes for anyone touching GLOWS
---------------------------------------------

Field order is load-bearing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``HistogramL1B``, ``DirectEventL1B`` and ``HistogramL2`` are unpacked positionally
by ``xr.apply_ufunc`` / ``dataclasses.asdict``. Inserting a field in the middle
of a dataclass silently shifts every following variable. Add at the end, and
update ``output_dimension_mapping`` in ``glows_l1b.py`` and the special-case
lists in ``create_l2_dataset``.

Fill values
^^^^^^^^^^^

``GlowsConstants.HISTOGRAM_FILLVAL = 65535`` propagates through L1A and L1B
untouched and is zeroed at L2 before summing. ``DailyLightcurve`` then chops all
bin arrays to ``number_of_bins`` and ``glows_l2.create_l2_dataset`` re-expands
them using each variable's CDF ``FILLVAL``. A new lightcurve variable without a
``FILLVAL`` in ``imap_glows_l2_variable_attrs.yaml`` will raise during padding.

Empty output is normal
^^^^^^^^^^^^^^^^^^^^^^

``glows_l2`` returning ``[]`` (no good-time blocks, or all-zero flux/exposure)
is an expected outcome, not a failure. Batch jobs must tolerate a pointing that
produces no L2 file.

Configuration dominates behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whether ``is_night`` blocks survive, whether HV-test blocks survive, how wide the
sky masks are, and what the calibration factor is are **all** decided by
ancillary files, not by code. When triaging a data anomaly, identify the exact
``pipeline-settings`` and ``l2-calibration`` versions that were used first.

Full TODO inventory
-------------------

**[CODE]** Every ``TODO``/placeholder marker currently in
``imap_processing/glows``:

.. list-table::
   :header-rows: 1
   :widths: 42 8 50

   * - File
     - Line
     - Comment
   * - ``l0/glows_l0_data.py``
     - 205
     - What other fields need to match? (``within_same_sequence``)
   * - ``l1a/glows_l1a.py``
     - 163
     - Block header per second, or global attribute?
   * - ``l1a/glows_l1a_data.py``
     - 248
     - Sanity check should exist in final code (``ENDID`` vs ``SPINS``)
   * - ``l1b/glows_l1b.py``
     - 387
     - The four spacecraft location/velocity values should each get their own
       dimension/attributes
   * - ``l1b/glows_l1b_data.py``
     - 460
     - ``number_of_de_packets`` missing from algorithm document
   * - ``l1b/glows_l1b_data.py``
     - 522
     - Is ``number_of_de_packets`` required in L1B?
   * - ``l1b/glows_l1b_data.py``
     - 547
     - First two values of DE are sec/subsec
   * - ``l1b/glows_l1b_data.py``
     - 551
     - Where does the multi-event flag go?
   * - ``l1b/glows_l1b_data.py``
     - 608-609
     - Double check ``unique_identifier`` time base; strings must go in
       attributes
   * - ``l1b/glows_l1b_data.py``
     - 761
     - ``flags_set_onboard`` should be renamed in L1B
   * - ``l1b/glows_l1b_data.py``
     - 800
     - Determine human-readable flag output; bad-angle algorithm using SPICE;
       move ancillary file to AWS
   * - ``l1b/glows_l1b_data.py``
     - 847-848
     - Ancillary should be an AWS file; pass ``AncillaryParameters`` in rather
       than reading here (note: the CLI *does* now pass it in)
   * - ``l1b/glows_l1b_data.py``
     - 1035-1036
     - ``is_beyond_daily_statistical_error`` placeholder; equation needs
       clarification
   * - ``l1b/glows_l1b_data.py``
     - 1053-1054
     - ``is_beyond_background_error`` listed as TBC in the document; placeholder
   * - ``l2/glows_l2.py``
     - 163
     - Create CDF attributes (epoch)
   * - ``l2/glows_l2_data.py``
     - 398-399
     - Bad angle filter; filter bad bins out
   * - ``l2/glows_l2_data.py``
     - 406
     - Fill in ``bad_time_flag_occurrences``
   * - ``l2/glows_l2_data.py``
     - 528-529
     - ``filter_bad_bins`` needs the exclusions ancillary file and a working
       ``unique_block_identifier``

The CDF attribute YAMLs additionally carry ``TODO: Remove unneeded attributes
once SAMMI is fixed`` (all three levels), ``TODO: I am not sure what the
FIELDNAM should be`` (L1B) and ``TODO: Update validmin and validmax`` (L2).

Test coverage notes
-------------------

* Validation against the GLOWS team's own JSON exists for L1A histograms
  (``glows_l1a_hist_validation.json``), L1B histograms
  (``imap_glows_l1b_hist_full_output.json``) and L1B direct events
  (``imap_glows_l1b_de_output.json``). **There is no equivalent L2 validation
  file in the repository** - L2 tests exercise the arithmetic against
  synthetic fixtures instead.
* Tests requiring SPICE kernels are marked ``@pytest.mark.external_kernel``;
  tests requiring the larger in-flight packet file are marked
  ``@pytest.mark.external_test_data``. Both are excluded by the repository's
  default ``-m "not external_kernel and not external_test_data"`` selection, so
  a green local run does **not** exercise the SPICE geometry paths.
* ``test_glows_l1b.py`` around line 467 carries the comment *"This needs to be
  added eventually, but is skipped for now."*
* The bundled packet file ``glows_test_packet_20110921_v01.pkts`` contains 505
  histogram and 1088 direct-event packets. The date in the name is not
  meaningful.
* **[DOC §3.13.1]** describes Validation Set 1
  (``data_products_cbk_implementation_2025-07-25_validation_set_1.zip``): five
  pointings, of which 1-3 are real EQM data taken under a deuterium lamp (real
  flight software v3.0.1, but **flat histograms** because the UV beam was not
  modulated with spin phase, plus monthly calibration tests visible as count-rate
  drops) and 4-5 are purely synthetic packets with realistic helioglow
  modulation from the WawHelioGlow model plus stars (**histograms only, no
  DEs**). Only a subset of this is checked into the repository.
