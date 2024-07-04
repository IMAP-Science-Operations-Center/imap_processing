"""
Module for generating XTCE files for telemetry packets for glows.

This module provides functionality to generate XTCE files for telemetry packets
for glows. It includes a `TelemetryGenerator` class for creating XTCE files
based on packet definitions stored in an Excel file.
"""

from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


def main() -> None:
    """
    Function is used to generate the GLOWS XTCE files for packet processing.

    This will create two XML files, P_GLX_TMSCDE.xml and P_GLX_TMSCHIST.xml. For
    processing, these need to be manually combined into one file: GLX_COMBINED.xml.

    To do this, first duplicate P_GLX_TMSCHIST into a new file GLX_COMBINED.xml.

    Copy the DE specific parameter entries out of P_GLX_TMSCDE.xml. ALl the non-CCSDS
    entries need to be copied into the combined XML file. This should go under the
    combined ParameterSet tag. Then, remove the "SEC" entry, as this is a duplicate.

    Finally, copy the SequenceContainer named "P_GLX_TMSCDE" out of P_GLX_TMSCDE.xml
    into the ContainerSet in GLX_COMBINED. Then, delete the "SEC" entry. This is a
    duplicate and will cause an error.
    """

    instrument_name = "glows"
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    path_to_excel_file = f"{current_directory}/tlm_glx_2023_06_22-edited.xlsx"

    # Eg.
    # packets = {
    #     "P_COD_HI_PHA": 1169,
    #     "P_COD_LO_PHA": 1153,
    #     "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
    #     "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
    # }

    packets = {"P_GLX_TMSCHIST": 1480, "P_GLX_TMSCDE": 1481}

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
