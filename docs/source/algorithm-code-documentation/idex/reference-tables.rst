.. _idex-reference-tables:

Reference Tables - Where to Look Them Up
========================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

The algorithm document's large tables are **deliberately not reproduced** in
these pages: they are long, they go stale, and in almost every case a
machine-readable version already exists in the repository that the code actually
reads. The document itself says the same thing about its own packet tables
(section 3.6): "This document avoids reproducing the full XML tables because
doing so would duplicate information already maintained by the packet-definition
files."

The document is **not in this repository** - see :ref:`idex-source-documents`.
This page tells you where to look instead, and gives a page index for the parts
that only exist in the PDF.

Rule of thumb
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - If you need...
     - Go to
   * - A science packet field's name, bit offset, width, type or enumeration
     - ``imap_processing/idex/packet_definitions/idex_science_packet_definition.xml``
   * - An event-message or catalog-list packet field
     - ``imap_processing/idex/packet_definitions/idex_housekeeping_packet_definition.xml``
   * - Which 10-day window a date belongs to
     - ``imap_processing/idex/idex_10_day_CDF_names.csv``
   * - An instrument setting's bit layout or EU polynomial
     - ``imap_processing/idex/idex_variable_unpacking_and_eu_conversion.csv``
   * - An event-message template or value dictionary
     - ``imap_processing/idex/idex_evt_msg_parsing_dictionaries.json``
   * - The reference ion masses used by the TOF mass scale
     - ``imap_processing/idex/atomic_masses.csv``
   * - A rise-time or charge-yield calibration coefficient
     - the two ``imap_idex_l2a-calibration-curve-*`` ancillary CSVs
       (copies in ``imap_processing/tests/idex/test_data/``)
   * - A DN-to-engineering-unit waveform factor
     - ``imap_processing/idex/idex_constants.py``, ``class ConversionFactors``
       - **not** an ancillary file, and **not** document Table 4.2
   * - An APID
     - ``imap_processing/idex/idex_constants.py``, ``class IDEXAPID``
   * - A science packet type value
     - ``imap_processing/idex/idex_l1a.py``, ``class Scitype``
   * - A trigger mode or origin label
     - ``imap_processing/idex/idex_l1b.py``, ``class TriggerMode``,
       ``class TriggerOrigin``, ``TRIGGER_LABELS``
   * - An event-message string the L1B reduction matches on
     - ``imap_processing/idex/idex_l1b.py``, ``class EventMessage``
   * - An event or saturation flag name, threshold or rule
     - ``imap_processing/idex/idex_event_flags.py``
   * - An L2B bin edge (mass, charge, spin phase, sky grid)
     - ``imap_processing/idex/idex_l2b.py``, module level
   * - A CDF variable's units, fill value, valid range or ``DEPEND_n``
     - ``imap_processing/cdf/config/imap_idex_l{1a,1b,2a,2b,2c}_variable_attrs.yaml``
   * - A ``logical_source`` string or global attribute
     - ``imap_processing/cdf/config/imap_idex_global_cdf_attrs.yaml``
   * - A SPICE frame id, boresight or spin offset
     - ``imap_processing/spice/geometry.py``
   * - An equation from L1A, L1B, L2A or L2B
     - :ref:`idex-l1` / :ref:`idex-l2` - the load-bearing ones are transcribed
   * - Anything else
     - the algorithm document, using the page index below

Machine-readable tables in the repository
-----------------------------------------

Packet definitions
^^^^^^^^^^^^^^^^^^

The two XTCE XML files are the authoritative field definitions and, unlike the
document, cannot be out of date with respect to processing - they are what the
code parses. The document's Table 3.2 lists a *representative subset* of fields
(``SHCOARSE``, ``SHFINE``, ``IDX__SCI0TYPE``, ``IDX__SCI0FRAGOFF``,
``IDX__SCI0RAW``, ``IDX__TXHDRTIMESEC1/2``, ``IDX__TXHDRTIMESUBS``,
``IDX__TXHDRTRIGID``, ``IDX__TXHDRBLOCKS``, ``IDX__TXHDRSAMPDELAY``,
``ELSEC_EVTPKT``, ``ELSSEC_EVTPKT``, ``ELID_EVTPKT``,
``EL1PAR_EVTPKT``-``EL4PAR_EVTPKT``) and explicitly declines to reproduce the
rest.

Useful greps::

    # Every science field name
    grep -o 'name="IDX__[A-Z0-9]*"' \
      imap_processing/idex/packet_definitions/idex_science_packet_definition.xml \
      | sort -u

    # Every catalog-list field
    grep -o 'name="IDX_CATLST\.[A-Z0-9]*"' \
      imap_processing/idex/packet_definitions/idex_housekeeping_packet_definition.xml

    # Every event-message field
    grep -o 'name="EL[0-9A-Z]*_EVTPKT"' \
      imap_processing/idex/packet_definitions/idex_housekeeping_packet_definition.xml

Packed-field bit layouts
^^^^^^^^^^^^^^^^^^^^^^^^

Several header fields carry more than one quantity. These are the ones the
pipeline unpacks, gathered in one place because they are scattered across three
modules in the code:

.. list-table::
   :header-rows: 1
   :widths: 30 16 22 32

   * - Field
     - Bits
     - Quantity
     - Unpacked in
   * - ``IDX__TXHDRBLOCKS``
     - 6-11
     - low-rate pre-trigger blocks
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRBLOCKS``
     - 16-19
     - high-rate pre-trigger blocks
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRBLOCKS``
     - 20-23
     - dead-time shift
     - ``idex_l1b.get_event_dead_time``
   * - ``IDX__TXHDRBLOCKS``
     - 24-29
     - dead-time base
     - ``idex_l1b.get_event_dead_time``
   * - ``IDX__TXHDRSAMPDELAY``
     - 0-9
     - high-gain TOF delay
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRSAMPDELAY``
     - 10-19
     - mid-gain TOF delay
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRSAMPDELAY``
     - 20-29
     - low-gain TOF delay
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRTRIGID``
     - 0-2
     - delay channel select (HG/LG/MG priority)
     - ``idex_l1a._set_sample_trigger_times``
   * - ``IDX__TXHDRTRIGID``
     - 0-5
     - trigger origin bit set
     - ``idex_l1b.get_trigger_origin``
   * - ``IDX__TXHDRTRIGID``
     - 0-3, 4-5
     - active channels; software/external trigger
     - ``idex_event_flags.classify_event_flags``
   * - ``IDX__TXHDR{HG,MG,LG}TRIGCTRL1``
     - 22-31
     - trigger threshold level
     - ``idex_l1b.get_trigger_mode_and_level``,
       ``idex_event_flags.classify_event_flags``
   * - ``idx__txhdr{prochk,hvpshk,lvhk0,lvhk1}ch*``
     - varies
     - two instrument settings each
     - ``idex_l1b.unpack_instrument_settings`` (driven by the EU CSV)

Constants at a glance
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Constant
     - Value
     - Where
   * - High-rate sample period
     - 1/260 µs
     - ``idex_l1a.RawDustEvent.HIGH_SAMPLE_RATE``
   * - Low-rate sample period
     - 1/4.0625 µs
     - ``idex_l1a.RawDustEvent.LOW_SAMPLE_RATE``
   * - FM quartz period (seconds)
     - 3.8466235767e-9
     - ``idex_constants.FM_SAMPLING_RATE``
   * - Samples per low-rate block
     - 8
     - ``idex_constants.SAMPLES_PER_BLOCK``
   * - Samples per high-rate block
     - 512
     - ``idex_l1a.RawDustEvent.NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK``
   * - Max blocks (high / low)
     - 16 / 64
     - ``idex_l1a.RawDustEvent.MAX_{HIGH,LOW}_BLOCKS``
   * - Block duration
     - ≈1.96923 µs
     - ``idex_constants.DT_BLOCK``
   * - Rice subframe size
     - 64
     - ``decode.SUB_FRAME_SIZE``
   * - Saturation fraction
     - 0.95
     - ``idex_event_flags._SATURATION_FRACTION``
   * - TOF / low-rate full scale
     - 1023 / 4095 DN
     - ``idex_event_flags._TOF_MAX_DN``, ``_LOW_RATE_MAX_DN``
   * - Pulser threshold
     - 1000 DN
     - ``idex_event_flags._PULSER_THRESHOLD_DN``
   * - Dust-hit peak threshold
     - 7 σ
     - ``idex_event_flags._PEAK_THRESHOLD_SIGMA``
   * - Dust-hit minimum FWHM
     - 20 ns
     - ``idex_event_flags._MIN_PEAK_WIDTH_US``
   * - Fit initial rise / decay
     - 0.371 / 37.1 µs
     - ``idex_l2a.estimate_dust_mass``
   * - Velocity inversion bracket
     - 0.1-100 km/s
     - ``idex_l2a.invert_rise_time_to_velocity``
   * - TOF SNR baseline window
     - −7 to −5 µs
     - ``idex_l2a.BaselineNoiseTime``
   * - Mass-scale stretch search
     - 1400-1500 ns, 10 steps
     - ``idex_l2a.time_to_mass``
   * - Ion-grid V(R) constants
     - c=55, p=−3.2, v₀=1.5
     - ``idex_constants.ION_GRID_VELOCITY_*``
   * - Sky map spacing
     - 6°
     - ``idex_constants.IDEX_SPACING_DEG``
   * - Event reference frame
     - ECLIPJ2000
     - ``idex_constants.IDEX_EVENT_REFERENCE_FRAME``

Document page index
-------------------

For the parts that genuinely only exist in the PDF. Page numbers are **PDF
pages** (what your reader shows), which run two ahead of the document's own
printed page numbers.

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - PDF pages
     - Section
     - Worth opening for
   * - 5-6
     - 0. Nomenclature
     - Acronym list and definitions. Note that L3 is not among them.
   * - 8-10
     - 2. Instrument description
     - **Figure 2.1** (instrument outline with the three signal types labelled),
       **Figure 2.2** (sensor head), **Figure 2.3** (electronics block
       diagram). The figures are the reason to open this chapter; the prose is
       summarised in :ref:`idex-overview`.
   * - 11-14
     - 3.1-3.3 Modes and measurement
     - **Figure 3.1** (mode state diagram), **Figure 3.2** (high/low-rate
       packing), **Figure 3.3** (instrument timeline around a trigger).
       Figure 3.3 is the clearest explanation of pre-trigger blocks anywhere.
   * - 15-17
     - 3.4-3.6 Telemetry
     - **Table 3.1** (science packet types - reproduced in :ref:`idex-l1`),
       **Table 3.2** (representative XTCE fields). Both are short; the XML is
       authoritative.
   * - 19
     - 4.1 Overview
     - **Figure 4.1**, the full pipeline flowchart with ancillary inputs on
       both sides. The single most useful page in the document. Reproduced in
       text form in :ref:`idex-data-products`.
   * - 22-23
     - 4.4 Data volume
     - **Figure 4.2** (expected ISD/IDP counts over the mission) and
       **Table 4.1** (weekly data volume by product level). The source of the
       "~16 events per day" figure that justifies the 10-day cadence. Not
       reproduced here - neither is read by any code.
   * - 23-27
     - 4.5-4.6 L0 and L1A
     - Matches the code closely. See :ref:`idex-l1`.
   * - 27-31
     - 4.7.1-4.7.2 L1B
     - **Table 4.2** (waveform conversion factors) - **superseded by the code**,
       see :ref:`idex-l1`.
   * - 31-34
     - 4.7.3-4.7.6 L2A
     - **Table 4.3** (calibration parameters, reproduced in :ref:`idex-l2`) and
       the fit, inversion and mass-scale equations.
   * - 35
     - 4.7.7-4.7.10
     - The NaN placeholder rationale and the two future-work sections. Short
       and worth reading verbatim before touching L2A.
   * - 36-37
     - 4.8-4.9
     - Calibration maintenance policy and the testing requirement. The
       calibration policy in 4.8 is the operational contract for ancillary file
       versioning; see :ref:`idex-ancillary`.
   * - 38
     - Appendix A
     - A code fragment of ``parser.py`` from the ``space_packet_parser``
       library. Nothing IDEX-specific; ignore.
   * - 39-41
     - Appendix B
     - The first 100 lines of the science XTCE. Read the XML instead.

Notable absences from the document
----------------------------------

If you go looking for these in the PDF, save yourself the time - they are not
there:

* Event classification and saturation flags (``idex_event_flags.py``).
* L2B and L2C algorithms beyond a flowchart box - binning, uptime model, rate
  quality flags, the sky map.
* The catalog-list (``catlst``) products.
* Any IDEX L3.
* Quicklook products.
* Dead-time correction of count rates (dead time is *computed* per 4.7.1, but
  nothing says what consumes it).
* Checksum or CRC verification.
