""" The module will take an Excel file and convert it into an XTCE formatted XML file.
This is specific for CODICE L0, but can be modified for other missions. This is the
start of CODICE L0 data processing.
"""
import xml.etree.ElementTree as Et

import pandas as pd

from imap_processing.codice.ccsds_header_xtce_generator import CCSDSParameters

# Make sure the "sheet" name is correct. In an Excel file
# There might be several "packets", which are "sheets" within the file.
# This is case-sensitive.
packet_name = "P_COD_AUT"  # This is the name of the packet (sheet) in the Excel file

# This is the path to the Excel file you want to convert to XML
path_to_excel_file = "/Users/gamo6782/Desktop/IMAP/TLM_COD_20230629-110638(update).xlsx"

ccsds_parameters = CCSDSParameters().parameters

if __name__ == "__main__":
    # ET.register_namespace is important!
    # Make sure you use the correct namespace for the xtce file you are using.
    # This is the namespace for the IMAP xtce files currently.

    Et.register_namespace(
        "xtce", "http://www.omg.org/space"
    )  # Update the namespace here

    # Load data from Excel file
    xls = pd.ExcelFile(path_to_excel_file)
    pkt = xls.parse(packet_name)

    # Fill missing values with '-*-'
    pkt.fillna("", inplace=True)

    # Create the root element and add namespaces
    root = Et.Element(
        "{http://www.omg.org/space}SpaceSystem",
        nsmap={"xtce": "http://www.omg.org/space/xtce"},
    )

    root.attrib["name"] = "packet_name"

    # Create the Header element with attributes 'date', 'version', and 'author'
    # Versioning is used to keep track of changes to the XML file.
    header = Et.SubElement(root, "{http://www.omg.org/space}Header")
    header.attrib["date"] = "2023-08-21"
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

    """
    The following loop creates a series of XML elements to
    define integer parameter types:
    - The loop iterates from 0 - 32, creating an 'IntegerParameterType' element for
    each number.
    - In `IntegerParameterType` element, attributes like `name` and `signed` are set.
    - An `IntegerDataEncoding` is added with attributes `sizeInBits` and `encoding`.
    - Lastly, a `UnitSet` element is added within each `IntegerParameterType`.
    This loop efficiently defines integer parameter types for a range of
    values in the XML structure.
    """

    # Create integer parameter types for all numbers between 0-32
    for size in range(33):  # Range goes up to 33 to include 0-32
        parameter_type = Et.SubElement(parameter_type_set, "xtce:IntegerParameterType")
        parameter_type.attrib["name"] = f"uint{size}"
        parameter_type.attrib["signed"] = "false"

        encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
        encoding.attrib["sizeInBits"] = str(size)
        encoding.attrib["encoding"] = "unsigned"
        # unit_set is used to define the units for the parameter.
        # This is not needed for CODICE L0.
        # UnitSet will be used for CODICE L1. It can be used for other missions as well.
        unit_set = Et.SubElement(parameter_type, "xtce:UnitSet")

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
    comparison.attrib["value"] = "1120"
    comparison.attrib["useCalibratedValue"] = "false"

    codice_science_entry_list = Et.SubElement(
        codice_science_container, "xtce:EntryList"
    )

    # Add ParameterRefEntry elements for CoDICESciencePacket
    # ******************************* NEED TO LOOK AT THIS: To pkt specific
    # This will be the list of parameters that will be included in the
    # CoDICE Science Packet after the CCSDS header'''
    parameter_refs = [
        "Spare",
        "Power_Cycle_Rq",
        "Power_Off_Rq",
        "Heater_Control_Enabled",
        "Heater_1_State",
        "Heater_2_State",
        "Spare2",
    ]

    for parameter_ref in parameter_refs:
        parameter_ref_entry = Et.SubElement(
            codice_science_entry_list, "xtce:ParameterRefEntry"
        )
        parameter_ref_entry.attrib["parameterRef"] = parameter_ref

    """
    This loop processes rows from the DataFrame starting from the 9th row onwards:
    - It iterates over the DataFrame rows using the `iterrows()` function.
    - For each row, it checks if the row index is less than 8 (rows before the 9th row).
    - If the condition is met, the loop continues to the next iteration,
        skipping these rows.
    - Otherwise, it creates an `xtce:Parameter` element and adds it to
        the `parameter_set`.
    - Attributes like `name` and `parameterTypeRef` are set based on row data.
    - A `LongDescription` element is created with the description text from the row.
    This loop effectively generates XML elements for parameters starting from the
        9th row of the DataFrame.
    """

    # Process rows from 9 until the last available row in the DataFrame
    for index, row in pkt.iterrows():
        if index < 8:
            continue  # Skip rows before row 9

        parameter = Et.SubElement(parameter_set, "{http://www.omg.org/space}Parameter")
        parameter.attrib["name"] = row["mnemonic"]
        parameter_type_ref = f"uint{row['lengthInBits']}"
        parameter.attrib["parameterTypeRef"] = parameter_type_ref

        description = Et.SubElement(
            parameter, "{http://www.omg.org/space}LongDescription"
        )
        description.text = row["longDescription"]

    """
    This section creates the final XML structure, indents it, then saves it to a file:
    - The 'Et.ElementTree(root)' creates an ElementTree with the root element 'root'.
    - 'Et.indent(tree, space="\t", level=0)' adds indentation for readability.
      - 'tree' is the ElementTree object.
      - 'space="\t"' specifies the character used for indentation (a tab in this case).
      - 'level=0' indicates the starting level of indentation.
    - 'tree.write("p_cod_aut_test.xml", encoding="utf-8", xml_declaration=True)'
        writes the XML content to a file named "p_cod_aut_test.xml".
      - 'encoding="utf-8"' specifies the character encoding for the file.
      - 'xml_declaration=True' adds an XML declaration at the beginning of the file.
    This section completes the XML generation process by creating a structured XML tree
    formatting it with indentation, and saving it to a file.
    """

    # Create the XML tree
    tree = Et.ElementTree(root)
    Et.indent(tree, space="\t", level=0)

    # Save the XML document to a file
    output_xml_path = "L0/p_cod_aut_test.xml"
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    # Read and modify the XML file contents
    with open(output_xml_path) as file:
        contents = file.read()

    modified_content = contents.replace(
        'xmlns:xtce="http://www.omg.org/space/"',
        'xmlns:xtce="http://www.omg.org/space/xtce"',
    )

    # Write the modified content back to the file
    with open(output_xml_path, "w") as file:
        file.write(modified_content)
