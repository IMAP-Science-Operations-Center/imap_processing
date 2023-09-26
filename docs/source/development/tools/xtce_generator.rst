.. _xtce_generator:

Generating Telemetry XML with Python Script
===========================================

Here  is some  info on `XTCE <https://public.ccsds.org/Pubs/660x2g2.pdf/>`_. This Green
Book introduces the main concepts of XML Telemetric and Command Exchange (XTCE), a
telemetry and telecommand database format for spacecraft monitoring
and control.

General
-------

This document provides steps and information on how to use
`xtce_generator_template.py` script as a base for users to generate
telemetry XML files. The script is designed to simplify the process of creating
telemetry definitions for various packet types.

The script is located in the `tools/xtce_generation` directory. The script is called
`xtce_generator_template.py`. The script is a ``template`` that can be modified to
generate telemetry XML files for different packet types. Your new file should be
called `xtce_generator_yourinstrument.py`.
An example of how to use the script is `xtce_generator_codice.py` which is also
located in the `tools/xtce_generation` directory.

Before you Start
----------------

Generating XTCEs is only done whenever packet definitions get updated, and thus it
is not a part of the main processing package. To use it there are a few extra
dependencies like ``pandas`` that you can install with

.. code::

    poetry install --extras tools

How to Use
----------

Define the instrument name in the `main()` function by setting the `instrument_name`
variable to the name of your instrument.

.. code::

        instrument_name = "your_instrument_name"

In the code, file paths are being configured. Make sure to change the file paths to
match your instrument's file structure.

.. code::

    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    # This is the path of the output directory
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # This is the path to the excel file that contains the telemetry definitions
    path_to_excel_file = f"{current_directory}/your_packet.xlsx"

Define packet names and `Application Process Identifiers (APIDs)
<https://sanaregistry.org/r/space_packet_protocol_application_process_id/>`_.
The packet names are **case sensitive** meaning the the packet names need to be exactly
what the tabs of the spreadsheet are. APID's must match the names and apIds in the
packet definition file. You can use as many packet names and apIds as you want.
The APID should be an integer (not hexadecimal).
Follow the format below.

.. code::

    packets = {
        # Define packet names and associated Application IDs (apId)
        "your_packet_A": ####,
        "your_packet_B": ####,
        # ... (other packet definitions)
    }

Generating Telemetry XML Files
-------------------------------

Once you have your xtce processing file defined, you can run it with the
following command:

.. code::

    python xtce_generator_instrument_name.py
