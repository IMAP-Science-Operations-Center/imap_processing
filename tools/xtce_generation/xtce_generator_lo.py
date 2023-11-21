from pathlib import Path

from telemetry_generator import TelemetryGenerator

# Following the creation of the XTCE files, manual updates need to be
# made to the following packets to make their binary field a variable
# length value in XTCE:
# P_ILO_BOOT_MEMDMP
# P_ILO_MEMDMP
# P_ILO_SCI_DE


def main():
    """IMAP Lo XTCE generator"""
    instrument_name = "lo"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # In the telemetry definition sheets, modifications need to be made to the
    # P_ILO_RAW_DE and P_ILO_SCI_CNT.
    #
    # P_ILO_RAW_DE: the rows for bits 112 to 20591 were collapsed into a
    # single binary row called RAW_DE. The reason for this is the number
    # of fields in the packet caused the XTCE file to exceed the repo
    # file size limit. Because of this modification, the binary field
    # will need to be parsed in the python code.
    #
    # P_ILO_SCI_CNT: the rows for bits 80 to 26959 were collapsed into a
    # single binary row called SCI_CNT. The reason for this is the number
    # of fields in the packet caused the XTCE file to exceed the repo
    # file size limit. Because of this modification, the binary field
    # will need to be parsed in the python code.
    path_to_excel_file = f"{current_directory}/telem_def_lo_modified.xls"

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
