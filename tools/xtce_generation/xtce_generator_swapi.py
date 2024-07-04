"""
Module for generating XTCE files for telemetry packets for swapi.

This module provides functionality to generate XTCE files for telemetry packets
for swapi. It includes a `TelemetryGenerator` class for creating XTCE files
based on packet definitions stored in an Excel file.
"""

from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main() -> None:
    """Function used by swapi to generate XTCE."""

    instrument_name = "swapi"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # NOTE: Copy packet definition to tools/xtce_generation/ folder if hasn't already
    path_to_excel_file = f"{current_directory}/TLM_SWP_20231006-121021.xlsx"

    packets = {
        "P_SWP_HK": 1184,
        "P_SWP_SCI": 1188,
        "P_SWP_AUT": 1192,
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
