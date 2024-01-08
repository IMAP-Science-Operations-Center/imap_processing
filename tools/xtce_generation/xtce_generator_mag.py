from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    """This function can be used by any instrument to generate XTCE
    for certain number of packets. Change values where TODO is
    """

    instrument_name = "mag"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    path_to_excel_file = f"{current_directory}/TLM_MAG_SCI.xls"

    # Eg.
    # packets = {
    #     "P_COD_HI_PHA": 1169,
    #     "P_COD_LO_PHA": 1153,
    #     "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
    #     "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
    # }

    packets = {
        "P_MAG_SCI_BURST": "1068",
        "P_MAG_SCI_NORM": "1052",
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
