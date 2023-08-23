""" All L0 data is written with a CCSDS header. This script creates the CCSDS header
    parameters and writes them to an XML file. The XML file is then used to create
    the CCSDS header parameters in the XTCE file.
    """

import xml.etree.ElementTree as ET

ET.register_namespace("xtce", "http://www.omg.org/space")

# Create the root element

root = ET.Element("{http://www.omg.org/space}SpaceSystem")
root.attrib["name"] = "CCSDS_Headers"

# Create the Header element and its attributes
header = ET.SubElement(root, "{http://www.omg.org/space}Header")
header.attrib["date"] = "2023"
header.attrib["version"] = "1.0"
header.attrib["author"] = "IMAP SDC"

# Create the TelemetryMetaData element
telemetry_metadata = ET.SubElement(root, "{http://www.omg.org/space}TelemetryMetaData")

# Create the ParameterTypeSet element
parameter_type_set = ET.SubElement(
    telemetry_metadata, "{http://www.omg.org/space}ParameterTypeSet"
)

# Create integer parameter types
integer_sizes = [1, 2, 3, 11, 14, 16, 32]
for size in integer_sizes:
    parameter_type = ET.SubElement(
        parameter_type_set, "{http://www.omg.org/space}IntegerParameterType"
    )
    parameter_type.attrib["name"] = f"uint{size}"
    parameter_type.attrib["signed"] = "false"

    encoding = ET.SubElement(
        parameter_type, "{http://www.omg.org/space}IntegerDataEncoding"
    )
    encoding.attrib["sizeInBits"] = str(size)
    encoding.attrib["encoding"] = "unsigned"

    unit_set = ET.SubElement(parameter_type, "{http://www.omg.org/space}UnitSet")

# Create the ParameterSet element
parameter_set = ET.SubElement(
    telemetry_metadata, "{http://www.omg.org/space}ParameterSet"
)

# Create CCSDS Header parameters
ccsds_parameters = [
    {
        "name": "VERSION",
        "parameterTypeRef": "uint3",
        "description": "CCSDS Packet Version Number (always 0)",
    },
    {
        "name": "TYPE",
        "parameterTypeRef": "uint1",
        "description": "CCSDS Packet Type Indicator (0=telemetry)",
    },
    {
        "name": "SEC_HDR_FLG",
        "parameterTypeRef": "uint1",
        "description": "CCSDS Packet Secondary Header Flag (always 1)",
    },
    {
        "name": "PKT_APID",
        "parameterTypeRef": "uint11",
        "description": "CCSDS Packet Application Process ID",
    },
    {
        "name": "SEG_FLGS",
        "parameterTypeRef": "uint2",
        "description": "CCSDS Packet Grouping Flags (3=not part of group)",
    },
    {
        "name": "SRC_SEQ_CTR",
        "parameterTypeRef": "uint14",
        "description": "CCSDS Packet Sequence Count (increments with each new packet)",
    },
    {
        "name": "PKT_LEN",
        "parameterTypeRef": "uint16",
        "description": "CCSDS Packet Length (number of bytes after Packet length minus 1)",
    },
]

for parameter_data in ccsds_parameters:
    parameter = ET.SubElement(parameter_set, "{http://www.omg.org/space}Parameter")
    parameter.attrib["name"] = parameter_data["name"]
    parameter.attrib["parameterTypeRef"] = parameter_data["parameterTypeRef"]

    description = ET.SubElement(parameter, "{http://www.omg.org/space}LongDescription")
    description.text = parameter_data["description"]

# Create the ContainerSet element
container_set = ET.SubElement(
    telemetry_metadata, "{http://www.omg.org/space}ContainerSet"
)

# Create the SequenceContainer element
sequence_container = ET.SubElement(
    container_set, "{http://www.omg.org/space}SequenceContainer"
)
sequence_container.attrib["name"] = "CCSDSPacket"

# Create the EntryList element and add ParameterRefEntry elements
entry_list = ET.SubElement(sequence_container, "{http://www.omg.org/space}EntryList")
for parameter_data in ccsds_parameters:
    parameter_ref_entry = ET.SubElement(
        entry_list, "{http://www.omg.org/space}ParameterRefEntry"
    )
    parameter_ref_entry.attrib["parameterRef"] = parameter_data["name"]

# Create the XML tree
tree = ET.ElementTree(root)
ET.indent(tree, space="\t", level=0)
# Save the XML document to a file
tree.write("L0/ccsds-header.xml", encoding="utf-8", xml_declaration=True)
