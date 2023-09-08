"""This is the xtce generator for the P_COD_HI_PHA packet."""

import pandas as pd

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main():
    packet_name = "P_COD_HI_PHA"
    path_to_excel_file = (
        "/Users/gamo6782/Desktop/IMAP/TLM_COD_20230629-110638(update).xlsx"
    )
    apid = "1169"
    sci_byte = 276480

    xls = pd.ExcelFile(path_to_excel_file)
    pkt = xls.parse(packet_name)

    # Specify the desired output XML path
    output_xml_path = "../../packet_definitions/codice/hi_pha.xml"

    # Create an instance of the TelemetryGenerator class
    generator = TelemetryGenerator(packet_name, path_to_excel_file, apid, sci_byte, pkt)

    # Generate the telemetry XML with the provided output path
    generator.generate_telemetry_xml(output_xml_path)


if __name__ == "__main__":
    main()
