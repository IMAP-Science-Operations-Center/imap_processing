from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    """IMAP-Hi XTCE generator tool."""

    instrument_name = "hi"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # NOTE: Copy packet definition to tools/xtce_generation/ folder
    path_to_excel_file = f"{current_directory}/26850.02-TLMDEF-04.xlsx"

    packets = {"H45_APP_NHK": 754, "H45_SCI_CNT": 769, "H45_SCI_DE": 770}

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
