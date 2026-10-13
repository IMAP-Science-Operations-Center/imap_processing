.. _idex-ancillary:

Ancillary Inputs and Dependencies
=================================

.. include:: /algorithm-code-documentation/_ai_generated_notice.inc

**[DOC]** Section 4.3 organises dependencies by the first level that needs them.
This page does the same, and adds **[CODE]** the exact paths, readers and
failure modes.

Two kinds of ancillary file
---------------------------

IDEX ancillary inputs split cleanly into two groups, and the distinction matters
operationally:

**In-repository, versioned with the code.** The XTCE definitions, the 10-day
window table, the EU conversion table, the event-message dictionaries and the
atomic mass table all live under ``imap_processing/idex/`` and ship with the
package. Changing one is a code change and goes through review and CI.

**Delivered as SDC ancillary files.** The two L2A calibration curves arrive as
versioned ``imap_idex_l2a-calibration-curve-*_YYYYMMDD_vNNN.csv`` files through
the normal dependency mechanism. **[DOC]** Section 4.8: "Calibration products
are external ancillary files, not hard-coded algorithm constants. New
calibration files will receive a new versioned filename and the processing
configuration should point to the newest file version."

.. note::

   The boundary is not where you might expect it. The waveform DN-to-engineering
   -unit factors (``ConversionFactors``) are **hard-coded constants in
   ``idex_constants.py``**, not an ancillary file - even though section 4.8
   explicitly says "L1B products are affected by DN-to-pC and engineering-unit
   conversion updates" and lists them alongside the calibration curves.
   Updating them is therefore a code release, not an ancillary delivery. This is
   the most likely place for a future refactor.

Product window and metadata
---------------------------

``idex_10_day_CDF_names.csv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 24 76

   * - Path
     - ``imap_processing/idex/idex_10_day_CDF_names.csv``
       (``idex_constants.IDEX_10_DAY_RANGES_PATH``)
   * - Provided by
     - The IDEX team.
   * - Read by
     - ``idex_utils.get_10_day_window_end_date()``, called from
       ``idex_l1a()``.
   * - Columns
     - ``start_date``, ``end_date``, ``doy`` - all as strings
       (``dtype=str``), dates as ``YYYYMMDD``.
   * - Rows
     - 444, covering 2025-01-01 through 2037-01-01.
   * - Behavior
     - Requires an **exact** ``start_date`` match. ``ValueError`` if no row
       matches, and a second ``ValueError`` guard if more than one does.
       Windows restart each 1 January, so the final window of each year is
       5-6 days rather than 10.

This file is the single source of truth for "what is a valid IDEX L1A start
date". If the instrument team changes the product cadence, this is the first
file to change - and see the warning in :ref:`idex` about the other three places
a cadence lives.

A 3-row test copy exists at
``imap_processing/tests/idex/test_data/test_idex_10_day_window.csv``.

CDF attribute definitions
^^^^^^^^^^^^^^^^^^^^^^^^^

Six YAML files under ``imap_processing/cdf/config/``:
``imap_idex_global_cdf_attrs.yaml`` plus one
``imap_idex_l{1a,1b,2a,2b,2c}_variable_attrs.yaml``. Loaded through
``idex_utils.get_idex_attrs(data_level)``, which wraps
``ImapCdfAttributes.add_instrument_global_attrs("idex")`` and
``.add_instrument_variable_attrs("idex", data_level)``.

These are where ``logical_source``, units, fill values, valid ranges and
``DEPEND_n`` relationships live. A missing entry surfaces as a ``KeyError`` from
``get_variable_attributes()`` at build time, or as a ``cdflib`` ``ISTPError`` at
write time.

L0 dependencies
---------------

XTCE packet definitions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 14 46

   * - File
     - Size
     - Defines
   * - ``packet_definitions/idex_science_packet_definition.xml``
     - ~148 KB
     - APID 1424. CCSDS headers, ``SHCOARSE``/``SHFINE``, the science routing
       fields (``IDX__SCI0TYPE``, ``IDX__SCI0FRAGOFF``, ``IDX__SCI0EVTNUM``,
       ``IDX__SCI0COMP``, ``IDX__SCI0PACK``, ``IDX__SCI0FRAG``,
       ``IDX__SCI0AID``, ``IDX__SCI0CAT``), the whole ``IDX__TXHDR*`` metadata
       header, the ``IDX__SCI0RAW`` waveform payload, sync and CRC fields.
       **Contains branching logic**, which is why IDEX science cannot use
       ``packet_file_to_datasets()``.
   * - ``packet_definitions/idex_housekeeping_packet_definition.xml``
     - ~612 KB
     - Housekeeping plus the event-message packet (APID 1418:
       ``ELSEC_EVTPKT``, ``ELSSEC_EVTPKT``, ``ELID_EVTPKT``,
       ``EL1PAR_EVTPKT`` … ``EL4PAR_EVTPKT``) and the catalog list (APID 1419:
       ``IDX_CATLST.*``).

**[DOC]** Both chapter 3 and section 3.6 are emphatic that these XML files are
**the authoritative source** for field names, bit widths, encodings, aliases and
enumerations, and that the document deliberately does not reproduce them field
by field. Appendix B shows only the first 100 lines of the science XML. Take the
document at its word here: if a field's definition is in question, read the XML.

L1A dependencies
----------------

``idex_evt_msg_parsing_dictionaries.json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 24 76

   * - Path
     - ``imap_processing/idex/idex_evt_msg_parsing_dictionaries.json``
   * - Read by
     - ``PacketParser._create_evt_msg_data()``, then
       ``evt_msg_decode_utils.render_event_template()``.
   * - Structure
     - A dict of named dictionaries. Two are looked up by name:
       ``eventMsgDictionary`` (event id → message template) and
       ``logEntryIdDictionary`` (event id → short log-entry name). The rest are
       value-lookup dictionaries referenced **by name from inside the
       templates**, e.g. ``sciState16Dictionary``, ``opCodeLCDictionary``.
   * - Gotcha
     - JSON stringifies all object keys. The reader converts every key back to
       ``int`` before use. If you hand-edit this file, keep keys numeric.

The template grammar and the ``dictName(value)`` output wrapper are described in
:ref:`idex-l1`. Remember that ``idex_l1b.EventMessage`` compares **whole
rendered strings** for equality, so this file and that enum are coupled.

L1B dependencies
----------------

``idex_variable_unpacking_and_eu_conversion.csv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 24 76

   * - Path
     - ``imap_processing/idex/idex_variable_unpacking_and_eu_conversion.csv``
   * - Read by
     - Twice: ``idex_l1b.unpack_instrument_settings()`` for the bit fields, and
       the shared ``imap_processing.utils.convert_raw_to_eu()`` for the
       polynomial coefficients (with ``packet_name="IDEX_SCI"``).
   * - Columns
     - ``index``, ``mnemonic``, ``var_name``, ``starting_bit``,
       ``nbits_padding_before``, ``unsigned_nbits``, ``unit``, ``c0``-``c7``,
       ``convertAs``, ``packetName``.
   * - Rows
     - 31 settings. Rows are de-duplicated on ``mnemonic`` for the unpacking
       pass, because a segmented-polynomial setting occupies several rows.

The 31 settings, grouped by the packed telemetry word they come from - note that
each ADC readout word carries **two** settings:

.. list-table::
   :header-rows: 1
   :widths: 34 44 22

   * - Source variable
     - Mnemonics
     - Units
   * - ``idx__txhdrprochkch01``
     - ``current_1v_pol``, ``current_1p9v_pol``
     - mA
   * - ``idx__txhdrprochkch23``
     - ``temperature_1``, ``temperature_2``
     - °C
   * - ``idx__txhdrprochkch45``
     - ``voltage_1v_bus``, ``fpga_temperature``
     - V, °C
   * - ``idx__txhdrprochkch67``
     - ``voltage_1p9v_bus``, ``voltage_3p3v_bus``
     - V
   * - ``idx__txhdrhvpshkch01``
     - ``detector_voltage``, ``sensor_voltage``
     - V
   * - ``idx__txhdrhvpshkch23``
     - ``target_voltage``, ``rejection_voltage``
     - V
   * - ``idx__txhdrhvpshkch45``
     - ``reflectron_voltage``, ``current_hvps_sensor``
     - V, mA
   * - ``idx__txhdrhvpshkch67``
     - ``positive_current_hvps``, ``negative_current_hvps``
     - mA
   * - ``idx__txhdrlvhk0ch01``
     - ``voltage_3p3_ref``, ``voltage_3p3_op_ref``
     - V
   * - ``idx__txhdrlvhk0ch23``
     - ``voltage_neg6v_bus``, ``voltage_pos6v_bus``
     - V
   * - ``idx__txhdrlvhk0ch45``
     - ``voltage_pos16v_bus``, ``voltage_pos3p3v_bus``
     - V
   * - ``idx__txhdrlvhk0ch67``
     - ``voltage_neg5v_bus``, ``voltage_pos5v_bus``
     - V
   * - ``idx__txhdrlvhk1ch01``
     - ``current_3p3v_bus``, ``current_16v_bus``
     - A
   * - ``idx__txhdrlvhk1ch23``
     - ``current_6v_bus``, ``current_neg6v_bus``
     - A
   * - ``idx__txhdrlvhk1ch45``
     - ``current_5v_bus``, ``current_neg5v_bus``
     - A
   * - ``idx__txhdrlvhk1ch67``
     - ``current_2p5v_bus``, ``current_neg2p5v_bus``
     - A

``target_voltage`` is the one to watch scientifically - it is the +3 kV target
bias that sets the ion acceleration, and therefore the TOF scale.

SPICE, spin and ephemeris
^^^^^^^^^^^^^^^^^^^^^^^^^

**[CODE]** All accessed through ``imap_processing/spice/``, never directly:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Need
     - Helper
   * - Spacecraft state relative to the Sun
     - ``spice.geometry.imap_state(et, observer=SpiceBody.SUN)``
   * - IDEX boresight in a target frame
     - ``spice.geometry.instrument_pointing(et, SpiceFrame.IMAP_IDEX,
       IDEX_EVENT_REFERENCE_FRAME, cartesian=True)``
   * - Cartesian → spherical
     - ``spice.geometry.cartesian_to_spherical()``
   * - Solar longitude
     - ``spice.geometry.solar_longitude(et, degrees=True)``
   * - Spin phase at an event
     - ``spice.spin.get_spacecraft_spin_phase(query_met_times=met)`` then
       ``spice.spin.get_spin_angle(..., degrees=True)``
   * - Time conversions
     - ``spice.time.ttj2000ns_to_et``, ``et_to_met``, ``met_to_ttj2000ns``,
       ``str_yyyymmdd_to_ttj2000ns``, ``epoch_to_doy``, ``et_to_datetime64``

Frame constants: ``SpiceFrame.IMAP_IDEX`` is NAIF id ``-43700``; boresight
``[0, 1, 0]``; spin-phase offset ``179.9229/360``. The configured event
reference frame is ``idex_constants.IDEX_EVENT_REFERENCE_FRAME =
SpiceFrame.ECLIPJ2000``, and it propagates all the way to the L2C map's
``Spice_reference_frame`` global attribute - change it in one place and the map
frame changes.

The **spin table** is a separate mission product from the kernels, and is a
separate CLI dependency (hence 3 dependencies for L1B science: the L1A file,
the kernels, the spin data).

.. note::

   Tests mock ``idex_l1b.get_spice_data`` wholesale (see
   ``imap_processing/tests/idex/conftest.py``), substituting ones for the
   ephemeris arrays and uniform random values for ``spin_phase``, ``longitude``
   and ``latitude``. The L1B validation comparison explicitly skips variables
   "known to differ because of time-system conventions or mocked SPICE
   quantities". **There is no test that exercises real SPICE geometry for
   IDEX.** Tests needing real kernels carry the ``external_kernel`` marker and
   are excluded from the default selection.

L2A dependencies
----------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Contents
   * - ``imap_idex_l2a-calibration-curve-t-rise_20250101_v002.csv``
     - Eight smooth-power-law parameters relating target rise time to impact
       speed, plus an unread ninth error factor. Inverted by
       ``invert_rise_time_to_velocity()``.
   * - ``imap_idex_l2a-calibration-curve-yield-params_20250101_v001.csv``
     - Eight smooth-power-law parameters for charge yield (C/kg) vs impact
       speed, plus an unread ninth error factor. Used by
       ``calculate_mass_from_velocity()``.
   * - ``imap_processing/idex/atomic_masses.csv``
     - 21 reference ion masses with names: H, H₂, C, O, Na, Mg24/25/26,
       Si28/29/30, K39/41, Ca40/42/44, Fe54/56/57/58, Au197. Read by
       ``time_to_mass()``. **In-repository**, not an SDC ancillary file.

Both calibration CSVs are keyed in the ``ancillary_files`` dict by the
descriptor segment of their filename - ``cli.py`` does
``path.stem.split("_")[2]``, giving ``l2a-calibration-curve-t-rise`` and
``l2a-calibration-curve-yield-params``. A filename that does not follow the
``imap_idex_<descriptor>_<date>_<version>`` convention will produce the wrong
dict key and a ``KeyError`` in ``load_calibration_files()``.

.. warning::

   The ``atomic_masses.csv`` mass column has apparent off-by-one problems
   relative to the isotope names it labels: ``22,Na`` (sodium-23),
   ``23,Mg24``, ``24,Mg25``, ``25,Mg26``, ``27,Si28``, ``39,Ca40``,
   ``53,Fe54``, ``196,Au197``. Either the masses are indices into some other
   scale or the file is misaligned by one row. Because the whole TOF mass path
   is currently NaN-filled this has no effect on published data, but it must be
   settled with the IDEX team before the mass scale is released - every
   ``time_to_mass()`` stretch factor is fitted against these numbers.

Calibration maintenance
-----------------------

**[DOC]** Section 4.8, worth reproducing because it defines the operational
contract:

* New calibration files get a **new versioned filename**; the processing
  configuration points at the newest version. Nothing is edited in place.
* If a calibration file is updated, existing products can be **regenerated by
  rerunning the affected levels**. L2A products are affected by rise-time and
  yield updates; L1B products by DN-to-pC and EU conversion updates.
* Validation tests are rerun after any calibration update.
* The instrument carries **on-board pulsers** that inject programmable known
  charges into each CSA to test the engineering-unit conversions. Those
  injections are reviewed manually and periodically to monitor each channel's
  DN-to-pC conversion.

**[CODE]** The pulser injections are visible to the pipeline in two places: the
``pulser_on`` state variable in the L1B message product, and the
``pulser_flag`` event classification at L1A (see
:ref:`idex-event-classification`). Nothing in this repository analyses them or
derives a conversion factor from them - that is the manual review the document
describes.
