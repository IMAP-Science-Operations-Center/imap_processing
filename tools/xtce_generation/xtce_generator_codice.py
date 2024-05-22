from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    instrument_name = "codice"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    path_to_excel_file = f"{current_directory}/TLM_COD.xlsx"

    # CoDICE packets
    packets = {
        "P_COD_NHK": 1136,
        "P_COD_LO_PHA": 1153,
        "P_COD_LO_INSTRUMENT_COUNTERS": 1154,
        "P_COD_LO_SW_PRIORITY_COUNTS": 1155,
        "P_COD_LO_SW_SPECIES_COUNTS": 1156,
        "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
        "P_COD_LO_SW_ANGULAR_COUNTS": 1158,
        "P_COD_LO_NSW_ANGULAR_COUNTS": 1159,
        "P_COD_LO_NSW_PRIORITY_COUNTS": 1160,
        "P_COD_HI_PHA": 1169,
        "P_COD_HI_INSTRUMENT_COUNTERS": 1170,
        "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
        "P_COD_HI_SECT_SPECIES_COUNTS": 1173,
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
