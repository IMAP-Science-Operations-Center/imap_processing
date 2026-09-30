.. _lo-implementation-status:

Implementation Status and Known Gaps
====================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/lo``. If you change
something material, update this page in the same commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - Level
     - State
     - Notes
   * - L0 / L1A
     - **Mature**
     - All 7 processed APIDs implemented, including segmented direct events and
       all three decompression tables. Well tested. Unlikely to change now that
       the spacecraft has launched.
   * - L1B
     - **Substantial, uneven**
     - Annotated DEs, all rate products, background/goodtimes and star sensor
       are real. Badtimes is mostly a placeholder. ~30 TODOs, several
       load-bearing.
   * - L1C
     - **Works, but orphaned**
     - Pointing sets are produced correctly but nothing downstream reads them.
   * - L2
     - **Real for hydrogen rectangular maps only**
     - All four corrections implemented, several beyond what the document
       specifies. Oxygen and HEALPix raise ``NotImplementedError``.
   * - L3
     - **Not started**
     - No survival probability code at all.
   * - ISN products
     - **Not started**
     - Only ISN *masking* of ENA maps exists.

Hard failures in the code
-------------------------

Explicit ``NotImplementedError``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Location
     - Condition
   * - ``lo_l2.py:149``
     - Any map species other than ``"h"``.
   * - ``lo_l2.py:156``
     - Any HEALPix (non-rectangular) map.
   * - ``cli.py`` ``Lo.do_processing``
     - Any data level other than l1a/l1b/l1c/l2.

Structural deviations from the algorithm document
-------------------------------------------------

These are design decisions, not bugs, but they will surprise anyone reading the
document first.

L2 does not use the pointing set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Document:** maps are formed by projecting L1C pointing set pixels
(0.1-degree bins) into map pixels.

**Code:** ``lo_l2.REQUIRED_PRODUCTS = ("goodtimes", "bgrates", "histrates")``,
all L1B, grouped by repointing. The L1C PSET is produced and then ignored.

**Consequence:** maps are built from the on-board 6-degree histograms, not from
direct events. Any map property that depends on 0.1-degree resolution or on
per-event species/coincidence selection is not achievable in the current
pipeline. This is the single largest architectural gap.

Product placement differs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Document
     - Document level
     - Code
   * - Instrument State Vector
     - L1B
     - ``imap_lo_l1b_instrument-status-summary``, produced by the **L1A**
       entry point.
   * - SweepTable
     - an L1B **product**
     - an L1B **ancillary input** (``sweep-table``, ``esa-mode-lut`` CSVs).
       Never written out.
   * - Goodtimes
     - L1C
     - produced at **L1B** (``imap_lo_l1b_goodtimes``); L1C emits a reference
       copy.
   * - Background rates
     - an ancillary file from the Lo team
     - **computed** at L1B (``imap_lo_l1b_bgrates``) from the anti-ram
       histogram signal.

Two different species definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **L1B** ``identify_species``: TOF2 in [13, 40] ns -> H, [75, 200] ns -> O,
  applied to every event with a TOF2. Matches the document.
* **L1C** ``get_h_species`` / ``get_o_species``: requires a **golden triple**
  and a 3-D box on ``TOF0 + 0.5*TOF3``, ``TOF1 - 0.5*TOF3`` and ``TOF2``, with
  boxes ``[20-70, 10-50, 10-40]`` for H and ``[100-270, 60-150, 60-150]`` for
  O. **Not in the document at all.**

These give different answers. Also note the document says species ID applies
only to triple coincidences, which the L1B implementation does not enforce.

Two different coincidence-type encodings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 4-bit ``ABSENT`` integer of the document and ``CASE_DECODER`` versus the
6-character bit strings (``"111111"``, ``"110100"``, ...) used in
``lo_l1c.py``. Translating between them is manual and error-prone.

Corrections that exceed the document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not gaps, but places where the code is the only specification:

* **Sputter correction** - a full (target, source) ESA matrix over 7 levels
  applied to *counts*, instead of the document's two coefficients
  :math:`f_5^O = 0.15`, :math:`f_6^O = 0.01` applied to intensities.
* **Bootstrap** - ``BOOTSTRAP_SCALE = 0.5`` halves the Appendix A coefficients,
  with 0.25 / 1.0 bracketing used as the systematic error. Plus neighborhood
  median spectral-index filling and a default index of 1.6.
* **ISN mask** - entirely absent from the document.
* **Systematic error** - asymmetric, from geometric factor bounds, rather than
  the document's single :math:`\sigma_G / G`.
* **Goodtimes** - a full automated algorithm where the document says "TBD".
* **Star sensor processing** - real filtering and profile construction where
  the document says "TBD".

Not implemented at all
----------------------

From the document's product list:

* **ISN Rates** (12.4), **ISN 1AU Maps** (12.5), **ISN Pointing Event Lists**
  (12.6). ISN is roughly half the instrument's science.
* **ENA Spectra in Selected Directions**.
* **ENA fluxes in the S/C frame as a standalone "strip of sky" product**
  (12.1) - the map product subsumes some of this but is not the same thing.
* **L3 Survival Probability Corrected Fluxes** (section 13). Requires
  survival probability maps, likely from GLOWS, and at least two independent
  models.
* **L3 off-nominal pivot-angle maps**.
* **Boot housekeeping** (APID 673) is decommutable but not in the pipeline.
* **Oxygen maps**. The ``ELEMS = ("H", "O")`` plumbing exists through L1B and
  the geometric factors are present, but L2 refuses.
* **Off-diagonal geometric factors.** The v004 ancillary only carries
  ``incident_E-Step == Observed_E-Step``; the document and the v001 file
  describe the full :math:`k \le i` response matrix.

Badtimes is a placeholder
-------------------------

``create_badtimes_dataset`` implements roughly one of the document's five
criteria. It flags spins with ``thruster_firing`` from the spin data, and
returns an empty (correctly shaped) dataset if no spin data is available. There
is a bare ``# TODO: Add badtimes`` at ``lo_l1b.py:1826``.

Not implemented:

1. Data gaps (no histogram and/or DE data for the window).
2. Off-nominal instrument state (gain/threshold testing, off-nominal HK).
3. Pivot platform motion (thruster firing is covered; platform motion is not).
4. Pointing-accuracy loss: star sensor versus star tracker disagreement,
   irregular spin quaternions.
5. Spin packets not containing 28 spins.

Because goodtimes are computed independently at L1B from background rates,
the pipeline is not currently blocked on this - but the two mechanisms
overlap and nobody has reconciled them.

Load-bearing TODOs
------------------

The ones that can change numerical results.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Location
     - Issue
   * - ``lo_l1b.py:604``
     - ``set_spin_cycle`` assumes 28 spins per ASC. Explicitly flagged as
       wrong for direct events.
   * - ``lo_l1b.py:950``
     - Left checksum boundary hardcoded to ``-21``. Should come from a LUT.
       Affects golden-triple TOF recovery.
   * - ``lo_l1b.py:1147``
     - Unresolved question of what to do when an event's off-angle latitude
       falls outside +/-2 degrees. Currently only warns.
   * - ``lo_l1b.py:1326``, ``:1334``
     - Marked ``TEMPORARY``, needs real L1A data.
   * - ``lo_l1b.py:1934``
     - Suspicion that a value being computed is also available in the sweep
       table; potential inconsistency.
   * - ``lo_l1b.py:2323``
     - Workaround for a broken L1A ``shcoarse`` ``DEPEND_0``.
   * - ``lo_l1b.py:2337``
     - A value that should be read from an ancillary config file is hardcoded.
   * - ``lo_l1c.py:571``, ``:651``, ``:663``
     - Marked ``TEMPORARY``; depend on L1B DE processing using the spin packet.
   * - ``tof_conversions.py:8``
     - TOF DN-to-ns coefficients are not in the algorithm document. They came
       from a Word document and an email. No versioned provenance.

Metadata TODOs
--------------

A long tail of ``# TODO: Add <field> to YAML file`` in ``lo_l1b.py``, where a
variable is written to the dataset with its ``attrs=`` line commented out.
Affected: ``pos``, ``mode``, ``absent``, ``esa_step``, ``shcoarse``,
``esa_mode``, ``spin cycle``, ``coincidence_type``, ``direction_lon``,
``direction_lat``, ``hae_x/y/z``, ``pointing_bin_lon``, ``pointing_bin_lat``.

These variables therefore ship without CATDESC, FIELDNAM, units or fill values.
They are usable inside the pipeline but not properly self-describing in the
CDF. This is a real archive-quality problem, and it is mechanical to fix: add
entries to ``imap_processing/cdf/config/imap_lo_l1b_variable_attrs.yaml`` and
uncomment the lines.

Naming hazards
--------------

Things that have bitten people:

* ``N_SPIN_ANGLE_BINS`` is **60** in ``lo/constants.py`` and **3600** in
  ``lo/l1c/lo_l1c.py``.
* "6 degree bins" means 60 bins; "60 degree bins" means 6 bins.
* ``imap_lo_l1b_nhk``, ``imap_lo_l1b_shk`` and
  ``imap_lo_l1b_instrument-status-summary`` are named L1B but produced by the
  L1A entry point.
* ``BADTIME`` in document outputs is ``1 = bad, 2 = good``, not a boolean.
* ESA levels are 1-indexed in science products and 0-indexed in arrays.
  ``RAM_ESA_LEVELS = (6, 7)`` is 1-indexed and converted at the point of use.

Test coverage
-------------

``imap_processing/tests/lo/`` has modules for every substantive source file:
``test_lo_l1a.py``, ``test_lo_l1b.py``, ``test_lo_l1c.py``, ``test_lo_l2.py``,
``test_lo_science.py``, ``test_star_sensor.py``, ``test_bit_decompression.py``,
``test_binary_string.py``, ``test_lo_ancillary.py``,
``test_lo_housekeeping.py``, ``test_science_counts.py``, plus ``test_pkts/``,
``test_cdfs/``, ``test_anc/`` and ``validation_data/`` fixtures.

Not directly tested: ``lo_apid.py``, ``constants.py``,
``decompression_tables.py``, and the ``cli.Lo`` wiring (including
``pre_processing``, which contains real filtering logic).

Documentation provenance
------------------------

These pages are a **derived summary**. The algorithm document itself is not in
this repository and must not be added to it - see :ref:`lo-source-documents`.

If you update the code in a way that changes behavior described here, update
these pages in the same commit. If you find that the document says something
these pages do not, add it as a ``[DOC]`` statement rather than assuming the
code is right. If you find the reverse, add it as ``[CODE]`` and raise it with
the instrument team, because it means the document no longer describes what the
SDC produces.
