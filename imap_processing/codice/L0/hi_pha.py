"""This is the xtce generator for the P_COD_HI_PHA packet."""

import xml.etree.ElementTree as Et

import pandas as pd

from tools.xtce_generation.ccsds_header_xtce_generator import CCSDSParameters

"""
Important Steps:
- packet_name: This is the name of the packet (sheet) in the Excel file. This is
case-sensitive.
- path_to_excel_file: This is the path to the Excel file with packet definitions
- ccsds_parameters: This is a list of dictionaries containing the CCSDS parameters
- APID: This is the APID of the packet. This is used to filter the parameters
in the CoDICE Science Packet.
- sci_byte (optional): This is the BYTE number of lenghtInBits in the "Data" mnemonic
in the CoDICE Science Packet.

"""

packet_name = "P_COD_HI_PHA"

path_to_excel_file = "/Users/gamo6782/Desktop/IMAP/TLM_COD_20230629-110638(update).xlsx"

# Load CCSDS parameters from the CCSDSParameters class
ccsds_parameters = CCSDSParameters().parameters

if __name__ == "__main__":
    # ET.register_namespace is important!
    # This is the namespace for the IMAP xtce files currently.
    Et.register_namespace(
        "xtce", "http://www.omg.org/space"
    )  # Update the namespace here

    # Load data from Excel file
    xls = pd.ExcelFile(path_to_excel_file)
    pkt = xls.parse(packet_name)

    # Input the correct APID here
    APID = "1169"

    # sci_byte is the BYTE number of lenghtInBits in the "Data" mnemonic
    sci_byte = 276480

    # Fill missing values with '-*-'
    pkt.fillna("", inplace=True)

    # Create the root element and add namespaces
    root = Et.Element("{http://www.omg.org/space}SpaceSystem")

    root.attrib["name"] = packet_name

    # Create the Header element with attributes 'date', 'version', and 'author'
    # Versioning is used to keep track of changes to the XML file.
    header = Et.SubElement(root, "{http://www.omg.org/space}Header")
    header.attrib["date"] = "2023-09"
    header.attrib["version"] = "1.0"
    header.attrib["author"] = "IMAP SDC"

    """
    These lines of code create a structured XML hierarchy to represent data according
    to a defined XML schema:
    - The `TelemetryMetaData` element is added under the 'root' element,
    providing a namespace.
    - Within `TelemetryMetaData`, the `ParameterTypeSet` element is created
    to define parameter types.
    - Similarly, the `ParameterSet` element is added to `TelemetryMetaData`
    to define specific parameters.
    These elements and their organization help organize and standardize
    data representation in XML format.
    """

    # Create the TelemetryMetaData element
    telemetry_metadata = Et.SubElement(
        root, "{http://www.omg.org/space}TelemetryMetaData"
    )

    # Create the ParameterTypeSet element
    parameter_type_set = Et.SubElement(telemetry_metadata, "xtce:ParameterTypeSet")

    # Create the ParameterSet element
    parameter_set = Et.SubElement(telemetry_metadata, "xtce:ParameterSet")

    # Extract unique values from the 'lengthInBits' column
    unique_lengths = pkt["lengthInBits"].unique()
    # Handle the "Data" mnemonic separately
    data_mnemonic = "Event_Data"
    data_type = pkt[pkt["mnemonic"] == data_mnemonic]["dataType"].values[0]

    # Create parameter types based on 'dataType' for the unique 'lengthInBits' values
    for size in unique_lengths:
        if size == sci_byte and data_type == "BYTE":
            continue  # Skip creating "IntegerParameterType" for "Data" with "BYTE" type

        parameter_type = Et.SubElement(parameter_type_set, "xtce:IntegerParameterType")
        parameter_type.attrib["name"] = f"uint{size}"
        parameter_type.attrib["signed"] = "false"

        encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
        encoding.attrib["sizeInBits"] = str(size)
        encoding.attrib["encoding"] = "unsigned"

    if data_type == "BYTE":
        parameter_type = Et.SubElement(parameter_type_set, "xtce:ArrayParameterType")
        parameter_type.attrib[
            "name"
        ] = "BYTE"  # Set the name to "BYTE" for "Data" mnemonic
        parameter_type.attrib["signed"] = "false"

        encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
        encoding.attrib["sizeInBits"] = str(sci_byte)  # Specific to "Data" mnemonic
        encoding.attrib["encoding"] = "unsigned"

    """
    This loop generates XML elements to define CCSDS packet parameters:
    - It iterates over a list of `ccsds_parameters`, each containing parameter data.
    - For each parameter, an `xtce:Parameter` element is added to the 'parameter_set'.
    - Attributes like `name` and `parameterTypeRef` are set using the parameter data.
    - A `LongDescription` element is created with the description text.
    This loop effectively creates XML elements to define CCSDS packet parameters
    based on the given data.
    """

    for parameter_data in ccsds_parameters:
        parameter = Et.SubElement(parameter_set, "xtce:Parameter")
        parameter.attrib["name"] = parameter_data["name"]
        parameter.attrib["parameterTypeRef"] = parameter_data["parameterTypeRef"]

        description = Et.SubElement(parameter, "xtce:LongDescription")
        description.text = parameter_data["description"]

    # Create ContainerSet element
    container_set = Et.SubElement(telemetry_metadata, "xtce:ContainerSet")

    # Create CCSDSPacket SequenceContainer
    ccsds_packet_container = Et.SubElement(container_set, "xtce:SequenceContainer")
    ccsds_packet_container.attrib["name"] = "CCSDSPacket"
    ccsds_packet_entry_list = Et.SubElement(ccsds_packet_container, "xtce:EntryList")

    for parameter_data in ccsds_parameters:
        parameter_ref_entry = Et.SubElement(
            ccsds_packet_entry_list, "xtce:ParameterRefEntry"
        )
        parameter_ref_entry.attrib["parameterRef"] = parameter_data["name"]

    # Create CoDICESciencePacket SequenceContainer
    codice_science_container = Et.SubElement(container_set, "xtce:SequenceContainer")
    codice_science_container.attrib["name"] = "CoDICESciencePacket"

    base_container = Et.SubElement(codice_science_container, "xtce:BaseContainer")
    base_container.attrib["containerRef"] = "CCSDSPacket"

    restriction_criteria = Et.SubElement(base_container, "xtce:RestrictionCriteria")
    comparison = Et.SubElement(restriction_criteria, "xtce:Comparison")
    comparison.attrib["parameterRef"] = "PKT_APID"
    comparison.attrib["value"] = APID
    comparison.attrib["useCalibratedValue"] = "false"

    codice_science_entry_list = Et.SubElement(
        codice_science_container, "xtce:EntryList"
    )

    """
    Add ParameterRefEntry elements for CoDICESciencePacket. This will be the list of
    parameters that will be included in the CoDICE Science Packet after the CCSDS
    header.
    """
    # Get the 'mnemonic' values starting from row 10
    parameter_refs = pkt.loc[9:, "mnemonic"].tolist()

    for parameter_ref in parameter_refs:
        parameter_ref_entry = Et.SubElement(
            codice_science_entry_list, "xtce:ParameterRefEntry"
        )
        parameter_ref_entry.attrib["parameterRef"] = parameter_ref

    """
    This loop processes rows from the DataFrame starting from the 10th row onwards:
    - It iterates over the DataFrame rows using the `iterrows()` function.
    - For each row, it checks if the row index is less than 9.
    - If the condition is met, the loop continues to the next iteration,
        skipping these rows.
    - Otherwise, it creates an `xtce:Parameter` element and adds it to
        the `parameter_set`.
    - Attributes like `name` and `parameterTypeRef` are set based on row data.
    - A `shortDescription` element is created with the description text from the row.
    This loop effectively generates XML elements for parameters starting from the
        10th row of the DataFrame.
    """

    # Process rows from 10 until the last available row in the DataFrame
    for index, row in pkt.iterrows():
        if index < 9:
            continue  # Skip rows before row 10

        parameter = Et.SubElement(parameter_set, "{http://www.omg.org/space}Parameter")
        parameter.attrib["name"] = row["mnemonic"]
        parameter_type_ref = ""

        if row["mnemonic"] == "Event_Data":
            parameter_type_ref = "BYTE"  # Use BYTE type for "Data" mnemonic
        else:
            parameter_type_ref = f"uint{row['lengthInBits']}"  # Use UINT for others

        parameter.attrib["parameterTypeRef"] = parameter_type_ref

        description = Et.SubElement(
            parameter, "{http://www.omg.org/space}LongDescription"
        )
        description.text = row["shortDescription"]

    # Create the XML tree
    tree = Et.ElementTree(root)
    Et.indent(tree, space="\t", level=0)

    # Save the XML document to a file in the current working directory
    # imap_processing/packet_definitions/codice/
    output_xml_path = "../../packet_definitions/codice/hi_pha.xml"
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
