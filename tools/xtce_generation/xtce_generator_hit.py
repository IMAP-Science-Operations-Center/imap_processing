from pathlib import Path

from telemetry_generator import TelemetryGenerator


def main():
    """This function can be used by any instrument to generate XTCE
    for certain number of packets. Change values where TODO is
    """

    # TODO: change instrument name
    instrument_name = "hit"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # TODO: Copy packet definition to tools/xtce_generation/ folder
    # path_to_excel_file = f"{current_directory}/TLM_SWE_20230904.xlsx"
    path_to_excel_file = f"{current_directory}/TLM_HIT_v20220524.xls"

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
        "P_HIT_AUT": 1250,
        "P_HIT_HSKP": 1251,
        "P_HIT_IALRT": 1252,
        "P_HIT_SCIENCE": 1253,
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
