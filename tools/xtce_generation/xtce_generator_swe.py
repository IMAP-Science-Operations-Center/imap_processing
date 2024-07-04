"""
Module for generating XTCE files for telemetry packets for swe.

This module provides functionality to generate XTCE files for telemetry packets
for swe. It includes a `TelemetryGenerator` class for creating XTCE files
based on packet definitions stored in an Excel file.
"""

from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main() -> None:
    """Function used by swe to generate XTCE."""

    instrument_name = "swe"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"

    path_to_excel_file = f"{current_directory}/TLM_SWE_20230904.xlsx"

    # SWE packets
    packets = {
        "P_SWE_APP_HK": 1330,
        "P_SWE_EVTMSG": 1317,
        "P_SWE_CEM_RAW": 1334,
        "P_SWE_SCIENCE": 1344,
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
