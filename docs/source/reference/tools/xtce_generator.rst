.. _xtce_generator:

Generating Telemetry XML with Python Script
===========================================

This document provides steps and information on how to use
`xtce_generator_template.py` script as a base for users to generate
telemetry XML files using the TelemetryGenerator class. The script is designed to
simplify the process of creating telemetry definitions for various packet types.

The script is located in the `tools/xtce_generation` directory. The script is called
`xtce_generator_template.py`. The script is a ``template`` that can be modified to
generate telemetry XML files for different packet types. Your new file should be
called `xtce_generator_yourinstrument.py`.
An example of how to use the script is `xtce_generator_codice.py` which is also
located in the `tools/xtce_generation` directory.

Prerequisites
-------------

Before you begin, ensure you have the following:

- Python installed (version 3.9 or higher)
- The required Python packages installed:
    - pathlib
    - tools.xtce_generation.telemetry_generator

How to Use
----------

Define the instrument name in the `main()` function by setting the `instrument_name`
variable to the name of your instrument.

::

    instrument_name = "your_instrument_name"


In the code, file paths are being configured. Make sure to change the file paths to
match your instrument's file structure.

::

    current_directory = Path(__file__).parent

    module_path = f"{current_directory}/../../imap_processing"

    # This is the path of the output directory
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"

    # This is the path to the excel file that contains the telemetry definitions
    path_to_excel_file = f"{current_directory}/your_packet.xlsx"

Define packet names and apIds. This is case sensitive. The packet names and apIds
must match the names and apIds in the excel file. You can use as many packet names
and apIds as you want. The apId should not be hexagon format. It should be an integer.
Follow the format below.

::

    packets = {
        # Define packet names and associated Application IDs (apId)
        "your_packet_A": ####,
        "your_packet_B": ####,
        # ... (other packet definitions)
    }


Generating Telemetry XML Files
-------------------------------

This code block generates telemetry XML files for the packets defined in the
`packets` dictionary. The telemetry XML files are saved in the directory specified
by the `packet_definition_path` variable.







