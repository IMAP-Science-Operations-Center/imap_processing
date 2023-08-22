import pandas as pd
import xml.etree.ElementTree as ET

'''  ET.register_namespace is important! Make sure you use the correct namespace for the xtce file you are using.This
is the namespace for the IMAP xtce files currently.'''
ET.register_namespace('xtce', "http://www.omg.org/space")

# Load data from Excel file
''' This is important! Use the xls PATH to the file you want to convert to XML.
Also, make sure the "sheet" name is correct. Case sensitive.'''

xls = pd.ExcelFile('/Users/gamo6782/Desktop/IMAP/TLM_COD_20230629-110638(update).xlsx')
sheet_name = "P_COD_AUT"  # Assuming the sheet name is correct
df = xls.parse(sheet_name)
''' 
Returns
    -------
    DataFrame
        DataFrame from the passed in Excel file.
        '''

# Fill missing values with '-*-'
df.fillna('', inplace=True)

# Create the root element and add namespaces
root = ET.Element("{http://www.omg.org/space}SpaceSystem", nsmap={"xtce": "http://www.omg.org/space"})
root.attrib["name"] = "P_COD_AUT"

# Create the Header element
header = ET.SubElement(root, "{http://www.omg.org/space}Header")
header.attrib["date"] = "2023-08-21"
header.attrib["version"] = "1.0"
header.attrib["author"] = "Gabriel Moraga"

"""
These lines of code create a structured XML hierarchy to represent data according to a defined XML schema:

- The 'TelemetryMetaData' element is added under the 'root' element, providing a namespace.
- Within 'TelemetryMetaData', the 'ParameterTypeSet' element is created to define parameter types.
- Similarly, the 'ParameterSet' element is added to 'TelemetryMetaData' to define specific parameters.

These elements and their organization help organize and standardize data representation in XML format.
"""

# Create the TelemetryMetaData element
telemetry_metadata = ET.SubElement(root, "{http://www.omg.org/space}TelemetryMetaData")

# Create the ParameterTypeSet element
parameter_type_set = ET.SubElement(telemetry_metadata, "xtce:ParameterTypeSet")

# Create the ParameterSet element
parameter_set = ET.SubElement(telemetry_metadata, "xtce:ParameterSet")

"""
The following loop creates a series of XML elements to define integer parameter types:

- The loop iterates from 0 to 32, creating an 'IntegerParameterType' element for each number.
- Within each 'IntegerParameterType' element, attributes like 'name' and 'signed' are set.
- An 'IntegerDataEncoding' element is added with attributes 'sizeInBits' and 'encoding'.
- Lastly, a 'UnitSet' element is added within each 'IntegerParameterType'.

This loop efficiently defines integer parameter types for a range of values in the XML structure.
"""

# Create integer parameter types for all numbers between 0-32
for size in range(33):  # Range goes up to 33 to include 0-32
    parameter_type = ET.SubElement(parameter_type_set, "xtce:IntegerParameterType")
    parameter_type.attrib["name"] = f"uint{size}"
    parameter_type.attrib["signed"] = "false"

    encoding = ET.SubElement(parameter_type, "xtce:IntegerDataEncoding")
    encoding.attrib["sizeInBits"] = str(size)
    encoding.attrib["encoding"] = "unsigned"

    unit_set = ET.SubElement(parameter_type, "xtce:UnitSet")

# Create CCSDS Header parameters. This should be consistent for all CCSDS packets.
ccsds_parameters = [
    {"name": "VERSION", "parameterTypeRef": "uint3", "description": "CCSDS Packet Version Number (always 0)"},
    {"name": "TYPE", "parameterTypeRef": "uint1", "description": "CCSDS Packet Type Indicator (0=telemetry)"},
    {"name": "SEC_HDR_FLG", "parameterTypeRef": "uint1",
     "description": "CCSDS Packet Secondary Header Flag (always 1)"},
    {"name": "PKT_APID", "parameterTypeRef": "uint11", "description": "CCSDS Packet Application Process ID"},
    {"name": "SEG_FLGS", "parameterTypeRef": "uint2",
     "description": "CCSDS Packet Grouping Flags (3=not part of group)"},
    {"name": "SRC_SEQ_CTR", "parameterTypeRef": "uint14",
     "description": "CCSDS Packet Sequence Count (increments with each new packet)"},
    {"name": "PKT_LEN", "parameterTypeRef": "uint16",
     "description": "CCSDS Packet Length (number of bytes after Packet length minus 1)"},
    {"name": "SHCOARSE", "parameterTypeRef": "uint32", "description": "CCSDS Packet Time Stamp (coarse time)"}
]

"""
This loop generates XML elements to define CCSDS packet parameters:

- It iterates over a list of 'ccsds_parameters', each containing parameter data.
- For each parameter, an 'xtce:Parameter' element is added to the 'parameter_set'.
- Attributes like 'name' and 'parameterTypeRef' are set using the parameter data.
- A 'LongDescription' element is created with the description text.

This loop effectively creates XML elements to define CCSDS packet parameters based on the given data.
"""

for parameter_data in ccsds_parameters:
    parameter = ET.SubElement(parameter_set, "xtce:Parameter")
    parameter.attrib["name"] = parameter_data["name"]
    parameter.attrib["parameterTypeRef"] = parameter_data["parameterTypeRef"]

    description = ET.SubElement(parameter, "xtce:LongDescription")
    description.text = parameter_data["description"]

# Create ContainerSet element
container_set = ET.SubElement(telemetry_metadata, "xtce:ContainerSet")

# Create CCSDSPacket SequenceContainer
ccsds_packet_container = ET.SubElement(container_set, "xtce:SequenceContainer")
ccsds_packet_container.attrib["name"] = "CCSDSPacket"
ccsds_packet_entry_list = ET.SubElement(ccsds_packet_container, "xtce:EntryList")

for parameter_data in ccsds_parameters:
    parameter_ref_entry = ET.SubElement(ccsds_packet_entry_list, "xtce:ParameterRefEntry")
    parameter_ref_entry.attrib["parameterRef"] = parameter_data["name"]

# Create CoDICESciencePacket SequenceContainer
codice_science_container = ET.SubElement(container_set, "xtce:SequenceContainer")
codice_science_container.attrib["name"] = "CoDICESciencePacket"

base_container = ET.SubElement(codice_science_container, "xtce:BaseContainer")
base_container.attrib["containerRef"] = "CCSDSPacket"

restriction_criteria = ET.SubElement(base_container, "xtce:RestrictionCriteria")
comparison = ET.SubElement(restriction_criteria, "xtce:Comparison")
comparison.attrib["parameterRef"] = "PKT_APID"
comparison.attrib["value"] = "1120"
comparison.attrib["useCalibratedValue"] = "false"

codice_science_entry_list = ET.SubElement(codice_science_container, "xtce:EntryList")

# Add ParameterRefEntry elements for CoDICESciencePacket
# ******************************* NEED TO LOOK AT THIS: To pkt specific
''' This will be the list of parameters that will be included in the CoDICE Science Packet after the CCSDS header'''

parameter_refs = ["Spare", "Power_Cycle_Rq", "Power_Off_Rq",
                  "Heater_Control_Enabled", "Heater_1_State", "Heater_2_State", "Spare2"]

for parameter_ref in parameter_refs:
    parameter_ref_entry = ET.SubElement(codice_science_entry_list, "xtce:ParameterRefEntry")
    parameter_ref_entry.attrib["parameterRef"] = parameter_ref

"""
This loop processes rows from the DataFrame starting from the 9th row onwards:

- It iterates over the DataFrame rows using the 'iterrows()' function.
- For each row, it checks if the row index is less than 8 (rows before the 9th row).
- If the condition is met, the loop continues to the next iteration, skipping these rows.
- Otherwise, it creates an 'xtce:Parameter' element and adds it to the 'parameter_set'.
- Attributes like 'name' and 'parameterTypeRef' are set based on row data.
- A 'LongDescription' element is created with the description text from the row.

This loop effectively generates XML elements for parameters starting from the 9th row of the DataFrame.
"""

# Process rows from 9 until the last available row in the DataFrame
for index, row in df.iterrows():
    if index < 8:
        continue  # Skip rows before row 9

    parameter = ET.SubElement(parameter_set, "{http://www.omg.org/space}Parameter")
    parameter.attrib["name"] = row["mnemonic"]
    parameter_type_ref = f"uint{row['lengthInBits']}"
    parameter.attrib["parameterTypeRef"] = parameter_type_ref

    description = ET.SubElement(parameter, "{http://www.omg.org/space}LongDescription")
    description.text = row["longDescription"]

"""
This section creates the final XML structure, indents it, and then saves it to a file:

- The 'ET.ElementTree(root)' creates an ElementTree with the root element 'root'.
- 'ET.indent(tree, space="\t", level=0)' adds indentation to the XML structure for readability.
  - 'tree' is the ElementTree object.
  - 'space="\t"' specifies the character used for indentation (a tab in this case).
  - 'level=0' indicates the starting level of indentation.
- 'tree.write("p_cod_aut_test.xml", encoding="utf-8", xml_declaration=True)' writes the XML content to a file named "p_cod_aut_test.xml".
  - 'encoding="utf-8"' specifies the character encoding for the file.
  - 'xml_declaration=True' adds an XML declaration at the beginning of the file.

This section completes the XML generation process by creating a structured XML tree, formatting it with indentation, and saving it to a file.
"""

# Create the XML tree
tree = ET.ElementTree(root)
ET.indent(tree, space="\t", level=0)

# Save the XML document to a file
''' Important! tree.write will write the file, so make sure you use the correct name for the file you want to write.'''
tree.write("p_cod_aut_test.xml", encoding="utf-8", xml_declaration=True)
