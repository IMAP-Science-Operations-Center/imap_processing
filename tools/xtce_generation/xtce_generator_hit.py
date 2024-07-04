"""
Module for generating XTCE files for telemetry packets for hit.

This module provides functionality to generate XTCE files for telemetry packets
for hit. It includes a `TelemetryGenerator` class for creating XTCE files
based on packet definitions stored in an Excel file.
"""

from pathlib import Path

from telemetry_generator import TelemetryGenerator

from imap_processing import imap_module_directory


def main() -> None:
    """Function used by hit to generate XTCE."""
    instrument_name = "hit"
    current_directory = Path(__file__).parent
    packet_definition_path = (
        f"{imap_module_directory}/{instrument_name}/packet_definitions"
    )
    path_to_excel_file = f"{current_directory}/TLM_HIT_modified.xlsx"

    # Lo packets
    packets = {
        "P_HIT_AUT": 1250,
        "P_HIT_HSKP": 1251,
        "P_HIT_SCIENCE": 1252,
        "P_HIT_MSGLOG": 1254,
        "P_HIT_MEMDUMP": 1255,
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
