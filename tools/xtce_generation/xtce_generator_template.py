"""
Module for generating XTCE files for telemetry packets.

This module provides functionality to generate XTCE files for
telemetry packets of a specific instrument.
It includes a `TelemetryGenerator` class for creating XTCE
files based on packet definitions stored in
an Excel file.
"""

from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    """Function used by instrument to generate XTCE. Change values where TODO is."""

    # TODO: change instrument name
    instrument_name = "<instrument_name>"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # TODO: Copy packet definition to tools/xtce_generation/ folder
    path_to_excel_file = f"{current_directory}/<excel_file_name>"

    # TODO: update packets dictionary with packet name and appId
    # Eg.
    # packets = {
    #     "P_COD_HI_PHA": 1169,
    #     "P_COD_LO_PHA": 1153,
    #     "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
    #     "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
    # }

    packets = {
        "<packet_name>": "<app_id_number>",
    }

    for packet_name, app_id in packets.items():
        print(packet_name)
        telemetry_generator = TelemetryGenerator(
            packet_name=packet_name, path_to_excel_file=path_to_excel_file, apid=app_id
        )
        telemetry_generator.generate_telemetry_xml(
            f"{packet_definition_path}/{packet_name}.xml", packet_name
        )


if __name__ == "__main__":
    main()
