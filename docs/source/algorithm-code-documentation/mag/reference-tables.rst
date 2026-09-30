.. _mag-reference-tables:

Reference Tables - Where to Look Them Up
========================================

The algorithm document's large tables are **deliberately not reproduced** here:
they go stale, and in almost every case a machine-readable version already
exists in the repository that the code actually reads.

The document itself is **not in this repository** - see
:ref:`mag-source-documents`. MAG is unusually self-contained: the packet
definition, the compression tables (there are none - the algorithm is closed
form), and every calibration number arrive either as XTCE or as delivered CDFs.
The only things you genuinely need the document for are listed at the bottom of
this page.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - If you need...
     - Go to
   * - A packet field's name, bit offset, width or type
     - ``imap_processing/mag/packet_definitions/MAG_SCI_COMBINED.xml``
   * - The I-ALiRT packet layout
     - ``imap_processing/ialirt/packet_definitions/ialirt_mag.xml`` and the
       ``Packet0``-``Packet3`` classes in
       ``imap_processing/ialirt/l0/mag_l0_ialirt_data.py``
   * - The Fibonacci sequence or any tunable constant
     - ``imap_processing/mag/constants.py``
   * - Which interpolation method or sensor is configured
     - ``imap_processing/mag/imap_mag_sdc_configuration_v001.py``
   * - A CDF variable's units, fill value, valid range or description
     - ``imap_processing/cdf/config/imap_mag_l1{a,b,c}_variable_attrs.yaml`` and
       ``imap_mag_l2_variable_attrs.yaml``
   * - The complete list of MAG products
     - ``imap_processing/cdf/config/imap_mag_global_cdf_attrs.yaml``
   * - A calibration variable name or shape
     - :ref:`mag-ancillary`, or ``cdflib.CDF(path).cdf_info()`` on a file in
       ``imap_processing/tests/mag/validation/calibration/``
   * - The decompression algorithm
     - :ref:`mag-compression` - fully transcribed, including the worked example
   * - The gradiometry or spin-averaging equations
     - :ref:`mag-l1d` - transcribed
   * - The clock-angle formulae
     - :ref:`mag-ialirt` - transcribed
   * - How the L2 offsets and matrices were derived, or the quality bitmask bits
     - :ref:`mag-cmad`, summarising the public CMAD (not SW-009)
   * - Anything else
     - the algorithm document, using the section index below

Machine-readable tables in the repository
------------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

``imap_processing/mag/packet_definitions/MAG_SCI_COMBINED.xml`` is the
authoritative field definition for APIDs 1052 and 1068. It supersedes
``TLM_MAG`` ([RD01]) for anything the pipeline does, and unlike the spreadsheet
it cannot be out of date with respect to processing - it is what
``packet_generator`` parses.

Both APIDs are in the one file; ``decom_mag.decom_packets`` filters on
``PKT_APID``.

.. code-block:: bash

   grep -o 'name="[^"]*"' imap_processing/mag/packet_definitions/MAG_SCI_COMBINED.xml | sort -u

CDF metadata
^^^^^^^^^^^^

``imap_processing/cdf/config/``:

* ``imap_mag_global_cdf_attrs.yaml`` - **the definitive product list.** Every
  ``Logical_source`` MAG can emit has an entry here; if a string is not in this
  file, ``get_global_attributes`` will raise.
* ``imap_mag_l1a_variable_attrs.yaml``
* ``imap_mag_l1b_variable_attrs.yaml``
* ``imap_mag_l1c_variable_attrs.yaml``
* ``imap_mag_l2_variable_attrs.yaml`` - **also used by L1D**;
  ``mag_l1d`` calls ``add_instrument_variable_attrs("mag", "l2")``.

.. note::

   There is no ``imap_mag_l1d_variable_attrs.yaml``. L1D and L2 share the L2
   variable attributes because they produce structurally identical files. The
   L1D ancillary outputs (spin offsets, gradiometry offsets) have **no**
   attribute definitions at all - they are written with ``istp=False``.

Delivered calibration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real examples live under ``imap_processing/tests/mag/validation/``:

.. code-block:: text

   calibration/imap_mag_l1b-calibration_20240229_v001.cdf
   calibration/imap_mag_l1b-calibration_20240229_v002.cdf
   calibration/imap_mag_l1d-calibration_20000101_v003.cdf
   calibration/imap_mag_l2-calibration_20251017_v004.cdf
   calibration/imap_mag_l2-norm-offsets_20251017_20251017_v001.cdf
   L2/T021/imap_mag_l2-calibration-matrices_20250506_v004.cdf
   L2/T021/imap_mag_l2-calibration-matrices_20250506_v005.cdf
   L2/T022/imap_mag_l2_norm-offsets_20250506_v006.cdf

A copy of the L1B calibration is also bundled at
``imap_processing/mag/l1b/imap_mag_l1b-calibration_20240229_v002.cdf`` as a test
fallback.

Validation data
^^^^^^^^^^^^^^^

``imap_processing/tests/mag/validation/`` is organised by level and test number.
Each case pairs an input (binary packets or CSV) with MAG-team reference output
CSVs:

.. code-block:: text

   L1a/T001..T008/   mag-l0-l1a-tNNN-in.bin,  mag-l0-l1a-tNNN-out.csv
   L1b/T009..T012/   mag-l1a-l1b-tNNN-in.csv, ...-mago-out.csv, ...-magi-out.csv
   L1c/T013..T016,T024/  mag-l1b-l1c-tNNN-{mago,magi}-{normal,burst}-{in,out}.csv
   L2/T021,T022/     calibration + offsets CDFs, expected-output CSV

The descriptive text files alongside them (``field_like.txt``,
``all_p_ones.txt``, ``hdr_field_and_range_change.txt``, ...) name what each case
is exercising, which is the fastest way to find the case that covers a
behaviour you are changing.

Document section index
----------------------

For when you do have a copy of IMAP-MAG-SW-009-01B in ``docs/reference/``.
44 pages, Issue 5 Revision 2.

SW-009 is not part of the public CMAD. For where MAG material *does* appear in
the CMAD, see the location table in :ref:`mag-cmad`.

.. list-table::
   :header-rows: 1
   :widths: 14 16 70

   * - Section
     - Pages
     - Contents
   * - 4.1-4.3
     - 7-8
     - Hardware description, science objectives, boot/application software.
   * - 4.4
     - 8-9
     - **Ranging.** Table 4-1 (sensor ranges), autoranging behaviour.
   * - 4.4.1
     - 9-10
     - **Cadence.** Table 4-2 (the six science modes), NM/BM transition
       overlap.
   * - 4.5
     - 10-11
     - Table 4-3 (operating modes: Standby, Config, Normal, Burst).
   * - 4.6
     - 11-12
     - **Science telemetry layout.** APIDs, primary/secondary labelling, the
       50-bit sample structure, timestamp derivation.
   * - 4.7
     - 12-13
     - **Calibration.** Ground (Magnetsrode) and in-flight (Imperial College),
       including the Leinweber offset method. *The CMAD gives the as-flown
       method; see* :ref:`mag-cmad`.
   * - 5
     - 13-15
     - Product overview and the **product summary table** (inputs, outputs,
       frames, modes per level).
   * - 6
     - 15
     - Background - Solar Orbiter heritage code.
   * - 7.1
     - 16
     - General principle and the note on timing.
   * - 7.2
     - 16-18
     - **Calibration file contents** and the **quality flag / bitmask table**.
       *The bitmask table is superseded by CMAD section 5.4.5; see*
       :ref:`mag-quality-flags`.
   * - 7.2.1
     - 18-19
     - **Gradiometer mode** and the kappa equation.
   * - 7.3.1
     - 20
     - Figure 1, the full pipeline workflow diagram. *L2P in the diagram is now
       L1D.*
   * - 7.3.2
     - 21-23
     - **L0 to L1A**, step by step.
   * - 7.3.2.1
     - 23-27
     - **Decompression**, including the compressed block layout table, the
       Python decode functions, and the worked binary example.
   * - 7.3.3
     - 27-29
     - **L1A to L1B.**
   * - 7.3.4
     - 29-32
     - **L1B to L1C**, including all six interpolation methods and the CIC
       filter code template.
   * - 7.3.5
     - 32-35
     - **L1B/C to L1D.** The longest procedure - 14 steps.
   * - 7.3.6
     - 35-37
     - **L1B/C to L2.**
   * - 7.4
     - 37-42
     - **I-ALiRT**, including Table 7-1 (packet structure), Table 7-2 (STATUS
       field over four packets), Table 7-3 (decommutated structure) and the
       clock-angle formulae.
   * - 8
     - 42-44
     - Appendix 1 - the compression algorithm rationale, zig-zag encoding, and
       Fibonacci encoding with Zeckendorf's theorem.

What is only in the external document
--------------------------------------

The short list of things you cannot get from this repository:

* **Figure 1**, the pipeline workflow diagram. :ref:`mag-data-products` has an
  ASCII equivalent, but the original shows the reference-frame regions as shaded
  bands, which is a useful mental model.
* **Table 7-2**, the exact I-ALiRT STATUS bit layout. The code's
  ``Packet0``-``Packet3`` classes are the operative version and their docstrings
  record the bit positions, but the document is the source of truth if they ever
  disagree.
* **Narrative rationale** - why the Leinweber method, why a CIC filter, what the
  Solar Orbiter heritage code did. The equations and procedures are transcribed
  here; the reasoning is summarised in a sentence at most.
* **The change log and change record**, useful for understanding why something
  looks odd (for example, kappa becoming a matrix in Issue 5 Revision 1, and
  clock angles being added to I-ALiRT in Issue 5 Revision 2).
