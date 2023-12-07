from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    """This function can be used by any instrument to generate XTCE
    for certain number of packets. Change values where TODO is
    """

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
