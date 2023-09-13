from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    """This function can be used by any instrument to generate XTCE
    for certain number of packets. Change values where TODO is
    """

    # TODO: change instrument name
    instrument_name = "codice"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # TODO: Copy packet definition to tools/xtce_generation/ folder
    # path_to_excel_file = f"{current_directory}/TLM_SWE_20230904.xlsx"
    path_to_excel_file = f"{current_directory}/TLM_COD_20230629-110638.xlsx"

    # TODO: update packets dictionary with packet name and appId

    # SWE packets
    # packets = {
    #     "P_SWE_APP_HK": 1330,
    #     "P_SWE_EVTMSG": 1317,
    #     "P_SWE_CEM_RAW": 1334,
    #     "P_SWE_SCIENCE": 1344,
    # }

    # CoDICE packets
    packets = {
        "P_COD_HI_PHA": 1169,
        "P_COD_LO_PHA": 1153,
        "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
        "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
        "P_COD_LO_SW_SPECIES_COUNTS": 1156,
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
