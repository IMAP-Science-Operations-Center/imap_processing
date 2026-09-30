.. _lo-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

The algorithm document is dominated by large field-definition tables. They are
**deliberately not reproduced** in these pages: they are long, they go stale,
and in almost every case a machine-readable version already exists in the
repository that the code actually reads.

The document itself is **not in this repository** - see
:ref:`lo-source-documents`. Almost nothing depends on having it, because the
tables that matter are vendored as data files. This page tells you which.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - If you need...
     - Go to
   * - A packet field's name, bit offset, width or type
     - ``imap_processing/lo/packet_definitions/lo_xtce.xml``
   * - A compression lookup value
     - ``imap_processing/lo/l0/decompression_tables/*.csv``
   * - A direct event's field layout for a case
     - ``lo/l0/decompression_tables/decompression_tables.py`` (``CASE_DECODER``)
   * - A geometric factor, sputter, bootstrap or ISN mask number
     - ``imap_processing/lo/ancillary_data/*.csv``
   * - A CDF variable's units, fill value or description
     - ``imap_processing/cdf/config/imap_lo_l1{a,b,c}_variable_attrs.yaml``
   * - A tunable constant
     - ``imap_processing/lo/constants.py``
   * - An equation from the L1B or L2 algorithms
     - :ref:`lo-l1b` / :ref:`lo-l2` - the load-bearing ones are transcribed
   * - Anything else
     - the algorithm document, using the page index below

Machine-readable tables in the repository
-----------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

``imap_processing/lo/packet_definitions/lo_xtce.xml`` (~7200 lines) is the
authoritative field definition for every processed APID. It supersedes the
document's housekeeping tables (document pages 55-87, 90-105) and, unlike them,
cannot be out of date with respect to processing - it is what the code parses.
Query it directly; for example:

.. code-block:: bash

   grep -o 'name="ILO_APP_NHK[^"]*"' imap_processing/lo/packet_definitions/lo_xtce.xml

Decompression tables
^^^^^^^^^^^^^^^^^^^^

``imap_processing/lo/l0/decompression_tables/``:

* ``log10_8_to_12_bit_uncompress.csv`` (257 rows)
* ``log10_8_to_16_bit_uncompress.csv`` (257 rows)
* ``log10_12_to_16_bit_uncompress.csv`` (4097 rows)

Each row is one compressed value; the code uses **column index 2, the bin
mean**. Columns 1 and 3 are the bin bounds. These are the vendored form of the
document's ``IMAP-Lo_Compression_Tables.xlsx``.

Direct event case table
^^^^^^^^^^^^^^^^^^^^^^^

``lo/l0/decompression_tables/decompression_tables.py`` holds
``PACKET_FIELD_BITS``, ``FIXED_FIELD_BITS``, ``VARIABLE_FIELD_BITS``,
``CASE_DECODER`` (19 ``(case, mode)`` entries) and ``DE_BIT_SHIFT``. This is
the code's version of the document's event packing table and is what actually
runs. It is small enough to read directly; it is summarized in :ref:`lo-l1a`.

Calibration CSVs
^^^^^^^^^^^^^^^^

``imap_processing/lo/ancillary_data/`` - all small enough to ``cat``. See
:ref:`lo-ancillary` for column meanings.

CDF metadata
^^^^^^^^^^^^

``imap_processing/cdf/config/``:

* ``imap_lo_global_cdf_attrs.yaml``
* ``imap_lo_l1a_variable_attrs.yaml``
* ``imap_lo_l1b_variable_attrs.yaml``
* ``imap_lo_l1c_variable_attrs.yaml``

L2 maps use the shared ENA map attribute configuration rather than a Lo-specific
file. Note that a number of L1B variables are still missing entries here; see
:ref:`lo-implementation-status`.

What is only in the external document
-------------------------------------

The short list of things you cannot get from this repository. If you are
looking for anything *not* on this list, it is in the repo somewhere and you do
not need the document.

* **Narrative rationale** - why a correction exists, what physical effect it
  models, what was tried and rejected. The equations are transcribed here; the
  reasoning behind them is not, beyond a sentence each.
* **The full Appendix A derivations** - equations 1-64 and the A1-A10
  predictor-corrector are transcribed in :ref:`lo-l2`, but their derivations
  are not. Much of this is also in the Schwadron et al. paper, which is the
  better citation once published.
* **The eta_ESA polynomial coefficients (Table 4)** - the code reads these from
  the ``esa-eta-fit-factors`` ancillary at runtime, so they are not in the repo
  either way. The document's copy is useful only for cross-checking a delivered
  ancillary.
* **Off-diagonal geometric factors** - the (incident, observed) ESA step
  response matrix. Partly present in the legacy ``_v001`` ancillary; the
  document describes the full structure.
* **Validation sections** - almost all are marked TBD in the document, so there
  is little to lose.
* **Quicklook plot limits** (pages 96-105) - not used by this repository.

Page index into the algorithm document
--------------------------------------

UNH-IMAP-Lo-27850-6002 rev. 10 with appendices, 230 pages. Page numbers below
are PDF page numbers, which match the printed page numbers.

This index is here so that someone who *has* the document can jump straight to
the right section instead of re-reading it. If you have a copy, put it in
``docs/reference/`` - that directory is gitignored and will never be committed:

.. code-block:: python

   from pathlib import Path
   from pypdf import PdfReader

   doc = next(Path("docs/reference").glob("*Data-Product-Algorithms*.pdf"))
   reader = PdfReader(doc)
   # Annotated Direct Events, pages 132-137 (1-based) -> indices 131-136
   text = "\n".join(reader.pages[i].extract_text() for i in range(131, 137))

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - Pages
     - Content
     - Summarized in
   * - 1-14
     - Title, approvals, revision history, table of contents
     - -
   * - 15-21
     - Conventions: file naming, CDF/JSON/CSV formats, time period lists
     - :ref:`lo-data-products`
   * - 22-28
     - Instrument overview; subsystems; electronics; IBEX-Lo differences
     - :ref:`lo-overview`
   * - 29-37
     - Data product level definitions; **the full data product list**;
       processing pipeline; audit trail
     - :ref:`lo-data-products`
   * - 38-54
     - Telemetry: ITF, CCSDS SPP, packet reading algorithm, user data
       transformations
     - :ref:`lo-l1a`
   * - 55-87
     - **L0 packet field tables per APID** (large; superseded by the XTCE)
     - not summarized - use the XTCE
   * - 88-89
     - L1A: SPP to CDF
     - :ref:`lo-l1a`
   * - 90-91
     - Static housekeeping field table (~27 fields)
     - not summarized - use the XTCE
   * - 92-105
     - Nominal housekeeping field table (~157 fields) and quicklook plot limits
     - not summarized - use the XTCE
   * - 106-110
     - Science counts: monitor rates and histograms, compression schemes
     - :ref:`lo-l1a`
   * - 111-116
     - Science direct events: packing, case table, unpacking pseudocode
     - :ref:`lo-l1a`
   * - 117-122
     - Star sensor, spin data, pivot platform, spin pointing data
     - :ref:`lo-l1a`
   * - 123-127
     - L1B: spacecraft state; **Instrument State Vector** inputs/outputs and
       the EU range-transform table
     - :ref:`lo-l1b`
   * - 128-129
     - Badtimes list: criteria and output columns
     - :ref:`lo-l1b`
   * - 130-131
     - SweepTable and Aggregated Science Cycles
     - :ref:`lo-l1b`
   * - 132-137
     - **Annotated Direct Events** - the densest L1B algorithm
     - :ref:`lo-l1b`
   * - 138-143
     - Monitor rates: input/output tables, exposure and rate formulas
     - :ref:`lo-l1b`
   * - 144-147
     - Histogram rates
     - :ref:`lo-l1b`
   * - 147-151
     - Direct event rates
     - :ref:`lo-l1b`
   * - 152-153
     - Processed star sensor data (mostly TBD)
     - :ref:`lo-l1b`
   * - 154-160
     - **L1C pointing sets**: grid, exposure, resweep, binning
     - :ref:`lo-l1c`
   * - 161-169
     - Ancillary files: goodtimes, efficiency factor, backgrounds,
       map-pointing files, geometric factors
     - :ref:`lo-ancillary`
   * - 170-173
     - L2 ENA fluxes in the S/C frame
     - :ref:`lo-l2`
   * - 174-188
     - L2 corrected ENA fluxes in the Sun rest frame: sputter, bootstrap,
       solar frame transform, Compton-Getting
     - :ref:`lo-l2`
   * - 185-186
     - **Table 12.1** - eta_ESA polynomial fit coefficients M0-M5 per level
     - superseded by the ``esa-eta-fit-factors`` ancillary
   * - 188-199
     - L2 1 AU ENA flux maps: pixel binning, re-binning, CG at pixel level
     - :ref:`lo-l2`
   * - 200-203
     - **ISN Rates** (not implemented)
     - :ref:`lo-l2`
   * - 203-206
     - **ISN 1AU Maps** (not implemented)
     - :ref:`lo-l2`
   * - 206-207
     - **ISN Pointing Event Lists** (not implemented)
     - :ref:`lo-l2`
   * - 207-208
     - Level 3 survival probability fluxes (stub in the document)
     - :ref:`lo-l2`
   * - 209-226
     - **Appendix A - "The IMAP-Lo Mapping Algorithms" v6** (Schwadron et al.,
       submitted to ApJ). *This is the authoritative algorithm text and
       supersedes section 12.* Contains eqs. 1-64 plus the predictor-corrector
       scheme A1-A10, Table 2 (sputter), Table 3 (bootstrap h matrix),
       Table 4 (eta_ESA fit factors).
     - :ref:`lo-l2`
   * - 227-228
     - Appendix B - ``ra_dec_pset`` function: constructing the (3600, 40)
       RA/DEC pointing matrices in ECLIPJ2000
     - :ref:`lo-l1c`
   * - 229-230
     - Appendix C - L2 and L3 mapping deliverables lists
     - :ref:`lo-l2`

If you need a table that is not here
------------------------------------

Do **not** paste it out of the algorithm document into a prose page. That
creates a second copy that drifts from whatever the code reads, and it puts
unreleased mission material into an open-source repository.

In order of preference:

1. **Generate it from what the code already reads.** The XTCE gives you every
   packet field; the ancillary CSVs give you every calibration number. A short
   script beats a transcription and cannot go stale.
2. **Add it as a versioned data file** under
   ``imap_processing/lo/ancillary_data/``, so the code and the documentation
   share one source. This is the right move for anything numeric that
   processing depends on.
3. **Transcribe it into these pages** only if it is small, static, and needed
   to understand the code rather than to run it - the way the direct event case
   table and the TOF conversion coefficients are. Cite the document section it
   came from, and check first that it is cleared for public release.

Before adding anything from the algorithm document to these pages, confirm with
the IMAP-Lo team that it can be published. The existing content is a derived
summary, but individual tables and figures are a different question.
