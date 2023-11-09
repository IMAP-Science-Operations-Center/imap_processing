from pathlib import Path

from telemetry_generator import TelemetryGenerator


def main():
    """This function can be used by any instrument to generate XTCE
    for certain number of packets. Change values where TODO is
    """

    # TODO: change instrument name
    instrument_name = "lo"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    path_to_excel_file = f"{current_directory}/telem_def_lo_modified.xls"

    # TODO: update packets dictionary with packet name and appId

    # Lo packets
    packets = {
        "P_ILO_APP_NHK": 677,
        "P_ILO_APP_SHK": 676,
        "P_ILO_AUTO": 672,
        "P_ILO_BOOT_HK": 673,
        "P_ILO_BOOT_MEMDMP": 674,
        "P_ILO_DIAG_BULK_HVPS": 724,
        "P_ILO_DIAG_CDH": 721,
        "P_ILO_DIAG_IFB": 722,
        "P_ILO_DIAG_PCC": 725,
        "P_ILO_DIAG_TOF_BD": 723,
        "P_ILO_EVTMSG": 678,
        "P_ILO_MEMDMP": 679,
        "P_ILO_RAW_CNT": 689,
        "P_ILO_RAW_DE": 690,
        "P_ILO_RAW_STAR": 691,
        "P_ILO_SCI_CNT": 705,
        "P_ILO_SCI_DE": 706,
        "P_ILO_SPIN": 708,
        "P_ILO_STAR": 707,
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
