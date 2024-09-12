.. _xtce_generator:

XML Telemetric and Command Exchange (XTCE)
==========================================

The `XTCE green book <https://public.ccsds.org/Pubs/660x2g2.pdf>`_.
introduces the main concepts and specifications of XTCE.
The XTCE format is a specification for spacecraft monitoring and control data transfer.
It can be used to define how packets can be sent and received from the spacecraft,
which we then unpack on the ground using the XTCE format.


Before you Start
----------------

Generating XTCEs is only done whenever packet definitions get updated, and thus it
is not regularly run as a part of processing. To use it there are a few extra
dependencies (like ``openpyxl`` for reading excel spreadsheets) that you
can install with the tools extra.

.. code::

    # with poetry
    poetry install --extras tools
    # or with pip
    pip install imap_processing[tools]

How to Use
----------

There is a command line utility ``imap_xtce`` that can be used to generate XTCE files
that is installed with the ``imap_processing`` package.
The utility takes in an excel file and generates XTCE files for each packet definition
in the excel file. If you don't provide an output file, it will generate the XTCE file
with the same name as the input Excel file but with the extension changed to ``.xml``.

.. code::

    imap_xtce path/to/excel_packet_file.xlsx --output path/to/output_packet_definition.xml


Spreadsheet definitions
-----------------------

The XTCE generator uses an excel spreadsheet to define the packet structure.
This is a commonly used spreadsheet format at the Laboratory for Atmospheric and Space Physics (LASP).
The required tabs are ``Subsystem``, ``Packets``, and whatever packet names you have.

Subsystem tab
~~~~~~~~~~~~~

The ``Subsystem`` tab is used to define the instrument name and last updated date of the packet data.

.. list-table:: Subsystem
   :header-rows: 1

   * - infoField
     - infoValue
   * - subsystem
     - MY_INSTRUMENT
   * - sheetReleaseDate
     - 01/01/2010
   * - sheetReleaseRev
     - 1.2.3

Packets tab
~~~~~~~~~~~

The packets tab contains the list of packets that you want to include within your XTCE
packet definition. You can remove rows from this to control which individual packet tabs
are read in later. The ``packetName`` column defines which other tabs to read in. So in
the following table, the generator will read in the ``MY_INSTRUMENT_HK`` and
``MY_INSTRUMENT_SCI`` tabs that contain the packet definitions.

.. note::
    The generator will also work with tabs prefixed with ``P_``, so ``P_MY_INSTRUMENT_HK`` and
    ``P_MY_INSTRUMENT_SCI`` tab names would also work.

.. list-table:: Packets
   :header-rows: 1

   * - packetName
     - apId
   * - MY_INSTRUMENT_HK
     - 123
   * - MY_INSTRUMENT_SCI
     - 124

Individual packet tabs
~~~~~~~~~~~~~~~~~~~~~~

Each packet tab contains the contents that will create the XTCE packet definition.
The required columns are ``packetName``, ``mnemonic``, ``lengthInBits``, ``dataType``,
``convertAs``, with optional ``shortDescription`` and ``longDescription`` columns.

Within the XTCE definition, the variable names will be ``packetName.mnemonic`` separated
with a period for easier distinguishing between packets and variables. For example,
the table below would have this XTCE parameter definition ``MY_INSTRUMENT_HK.VARIABLE1_UINT``
for the first variable. If an analog conversion is required, the ``convertAs`` column
should be set to ``ANALOG``, which will then look at the ``AnalogConversions`` tab for
the conversion details.

.. list-table:: MY_INSTRUMENT_HK
   :header-rows: 1

   * - packetName
     - mnemonic
     - lengthInBits
     - dataType
     - convertAs
     - shortDescription
     - longDescription
   * - MY_INSTRUMENT_HK
     - VARIABLE1_UINT
     - 3
     - UINT
     - NONE
     - My short variable description
     - My verbose variable description
   * - MY_INSTRUMENT_HK
     - VARIABLE2_CONVERTED
     - 3
     - UINT
     - ANALOG
     - Apply an analog conversion
     -
   * - MY_INSTRUMENT_HK
     - VARIABLE_ENUMERATED
     - 1
     - UINT
     - STATE
     - Apply an enumeration state
     -
   * - MY_INSTRUMENT_HK
     - VARIABLE_LENGTH_BINARY_SCIENCE
     - 100
     - BYTE
     - NONE
     -
     - This variable size will be dynamic and based on the packet size

AnalogConversions tab (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packet parsing can also apply analog conversions to the data being read in.
For example, to change from a raw unsigned integer value to a temperature in Kelvin.
The ``AnalogConversions`` tab is used to define these conversions.
It currently only supports unsegmented polynomial conversions, and looks for the
coefficients defined from ``c0`` to ``c7`` to define the order of the polynomial.

.. list-table:: AnalogConversions
   :header-rows: 1

   * - packetName
     - mnemonic
     - c0
     - c1
     - c2
     - c3
     - c4
     - c5
     - c6
     - c7
   * - MY_INSTRUMENT_HK
     - VARIABLE2_CONVERTED
     - 123.456
     - 0.234
     -
     -
     -
     -
     -
     -

States tab (optional)
~~~~~~~~~~~~~~~~~~~~~

Packet parsing can also apply enumeration/state conversions to the data being read in.
For example, to change from a raw unsigned integer value to a "VALID" / "INVALID" string.
The ``States`` tab is used to define these enumerations.

.. list-table:: States
   :header-rows: 1

   * - packetName
     - mnemonic
     - value
     - state
   * - MY_INSTRUMENT_HK
     - VARIABLE_ENUMERATED
     - 0
     - INVALID
   * - MY_INSTRUMENT_HK
     - VARIABLE_ENUMERATED
     - 1
     - VALID
