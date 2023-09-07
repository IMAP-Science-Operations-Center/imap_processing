import xml.etree.ElementTree as Et

import pandas as pd

from tools.xtce_generation.ccsds_header_xtce_generator import CCSDSParameters


class TelemetryGenerator:
    def __init__(self, packet_name, path_to_excel_file, apid, sci_byte=None, pkt=None):
        self.packet_name = packet_name
        self.path_to_excel_file = path_to_excel_file
        self.apid = apid
        self.sci_byte = sci_byte
        self.pkt = pkt

    def create_telemetry_xml(self):
        """
        Create an XML representation of telemetry data based on input parameters.

        Parameters:
        - packet_name: The name of the packet (sheet) in the Excel file.
        - path_to_excel_file: The path to the Excel file with packet definitions.
        - APID: The APID of the packet used to filter the parameters.
        - sci_byte (optional): The BYTE number of lengthInBits in the "Data" mnemonic
          in the CoDICE Science Packet.
        - pkt: The DataFrame containing packet data.

        Returns:
        - root: The root element of the generated XML tree.
        - parameter_type_set: The ParameterTypeSet element.
        - parameter_set: The ParameterSet element.
        - telemetry_metadata: The TelemetryMetaData element.
        """
        # Register the XML namespace
        Et.register_namespace("xtce", "http://www.omg.org/space")

        # Load data from Excel file
        xls = pd.ExcelFile(self.path_to_excel_file)
        pkt = xls.parse(self.packet_name)

        # Fill missing values with '-*-'
        pkt.fillna("", inplace=True)

        # Create the root element and add namespaces
        root = Et.Element("{http://www.omg.org/space}SpaceSystem")
        root.attrib["name"] = self.packet_name

        # Create the Header element with attributes 'date', 'version', and 'author'
        # Versioning is used to keep track of changes to the XML file.
        header = Et.SubElement(root, "{http://www.omg.org/space}Header")
        header.attrib["date"] = "2023-09"
        header.attrib["version"] = "1.0"
        header.attrib["author"] = "IMAP SDC"

        # Create the TelemetryMetaData element
        telemetry_metadata = Et.SubElement(
            root, "{http://www.omg.org/space}TelemetryMetaData"
        )

        # Create the ParameterTypeSet element
        parameter_type_set = Et.SubElement(telemetry_metadata, "xtce:ParameterTypeSet")

        # Create the ParameterSet element
        parameter_set = Et.SubElement(telemetry_metadata, "xtce:ParameterSet")

        return root, parameter_type_set, parameter_set, telemetry_metadata

    def extract_data_info(self):
        """
        Extract unique lengths from the 'lengthInBits' column and
        handle the "Data" mnemonic separately.

        Parameters:
        - pkt: The DataFrame containing packet data.

        Returns:
        - unique_lengths: Unique values from the 'lengthInBits' column.
        - data_type: The data type for the "Data" mnemonic.
        """
        # Extract unique values from the 'lengthInBits' column
        unique_lengths = self.pkt["lengthInBits"].unique()

        # Handle the "Data" mnemonic separately
        data_mnemonic = "Data"
        data_type = self.pkt[self.pkt["mnemonic"] == data_mnemonic]["dataType"].values[
            0
        ]

        return unique_lengths, data_type

    def create_parameter_types(
        self, parameter_type_set, unique_lengths, data_type, sci_byte
    ):
        """
        Create parameter types based on 'dataType' for the unique 'lengthInBits' values.

        Parameters:
        - parameter_type_set: The ParameterTypeSet element where parameter types are.
        - unique_lengths: Unique values from the 'lengthInBits' column.
        - data_type: The data type for the "Data" mnemonic.
        - sci_byte: The BYTE number of lengthInBits in the "Data" mnemonic

        Returns:
        - parameter_type_set: The updated ParameterTypeSet element.
        """
        for size in unique_lengths:
            if size == sci_byte and data_type == "BYTE":
                continue  # Skip create "IntegerParameterType" for "Data" with "BYTE"

            parameter_type = Et.SubElement(
                parameter_type_set, "xtce:IntegerParameterType"
            )
            parameter_type.attrib["name"] = f"uint{size}"
            parameter_type.attrib["signed"] = "false"

            encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
            encoding.attrib["sizeInBits"] = str(size)
            encoding.attrib["encoding"] = "unsigned"

        if data_type == "BYTE":
            parameter_type = Et.SubElement(
                parameter_type_set, "xtce:ArrayParameterType"
            )
            parameter_type.attrib[
                "name"
            ] = "BYTE"  # Set the name to "BYTE" for "Data" mnemonic
            parameter_type.attrib["signed"] = "false"

            encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
            encoding.attrib["sizeInBits"] = str(sci_byte)  # Specific to "Data" mnemonic
            encoding.attrib["encoding"] = "unsigned"

        return parameter_type_set

    def create_ccsds_packet_parameters(self, parameter_set, ccsds_parameters):
        """
        Create XML elements to define CCSDS packet parameters based on the given data.

        Parameters:
        - parameter_set: The ParameterSet element where parameters will be added.
        - ccsds_parameters: A list of dictionaries containing CCSDS parameter data.

        Returns:
        - parameter_set: The updated ParameterSet element.
        """
        for parameter_data in ccsds_parameters:
            parameter = Et.SubElement(parameter_set, "xtce:Parameter")
            parameter.attrib["name"] = parameter_data["name"]
            parameter.attrib["parameterTypeRef"] = parameter_data["parameterTypeRef"]

            description = Et.SubElement(parameter, "xtce:LongDescription")
            description.text = parameter_data["description"]

        return parameter_set

    def create_container_set(self, telemetry_metadata, ccsds_parameters, apid):
        """
        Create XML elements for ContainerSet, CCSDSPacket SequenceContainer,
        and CoDICESciencePacket SequenceContainer.

        Parameters:
        - telemetry_metadata: The TelemetryMetaData element where containers are.
        - ccsds_parameters: A list of dictionaries containing CCSDS parameter data.
        - APID: The APID value used for comparison in CoDICESciencePacket.

        Returns:
        - telemetry_metadata: The updated TelemetryMetaData element.
        """
        # Create ContainerSet element
        container_set = Et.SubElement(telemetry_metadata, "xtce:ContainerSet")

        # Create CCSDSPacket SequenceContainer
        ccsds_packet_container = Et.SubElement(container_set, "xtce:SequenceContainer")
        ccsds_packet_container.attrib["name"] = "CCSDSPacket"
        ccsds_packet_entry_list = Et.SubElement(
            ccsds_packet_container, "xtce:EntryList"
        )

        for parameter_data in ccsds_parameters:
            parameter_ref_entry = Et.SubElement(
                ccsds_packet_entry_list, "xtce:ParameterRefEntry"
            )
            parameter_ref_entry.attrib["parameterRef"] = parameter_data["name"]

        # Create CoDICESciencePacket SequenceContainer
        codice_science_container = Et.SubElement(
            container_set, "xtce:SequenceContainer"
        )
        codice_science_container.attrib["name"] = "CoDICESciencePacket"

        base_container = Et.SubElement(codice_science_container, "xtce:BaseContainer")
        base_container.attrib["containerRef"] = "CCSDSPacket"

        restriction_criteria = Et.SubElement(base_container, "xtce:RestrictionCriteria")
        comparison = Et.SubElement(restriction_criteria, "xtce:Comparison")
        comparison.attrib["parameterRef"] = "PKT_APID"
        comparison.attrib["value"] = apid
        comparison.attrib["useCalibratedValue"] = "false"

        Et.SubElement(codice_science_container, "xtce:EntryList")

        return telemetry_metadata

    def add_codice_science_parameters(self, codice_science_container, pkt):
        """
        Add ParameterRefEntry elements for CoDICESciencePacket.

        Parameters:
        - codice_science_container: The CoDICESciencePacket SequenceContainer element.
        - pkt: The DataFrame containing packet data.

        Returns:
        - codice_science_container: The updated CoDICESciencePacket
        SequenceContainer element.
        """
        # Get the 'mnemonic' values starting from row 10
        parameter_refs = pkt.loc[9:, "mnemonic"].tolist()

        for parameter_ref in parameter_refs:
            parameter_ref_entry = Et.SubElement(
                codice_science_container, "xtce:ParameterRefEntry"
            )
            parameter_ref_entry.attrib["parameterRef"] = parameter_ref

        return codice_science_container

    def create_parameters_from_dataframe(self, parameter_set, pkt):
        """
        Create XML elements for parameters based on DataFrame rows starting from row 10.

        Parameters:
        - parameter_set: The ParameterSet element where parameters will be added.
        - pkt: The DataFrame containing packet data.

        Returns:
        - parameter_set: The updated ParameterSet element.
        """
        # Process rows from 10 until the last available row in the DataFrame
        for index, row in pkt.iterrows():
            if index < 9:
                continue  # Skip rows before row 10

            parameter = Et.SubElement(
                parameter_set, "{http://www.omg.org/space}Parameter"
            )
            parameter.attrib["name"] = row["mnemonic"]
            parameter_type_ref = ""

            if row["mnemonic"] == "Data":
                parameter_type_ref = "BYTE"  # Use BYTE type for "Data" mnemonic
            else:
                parameter_type_ref = f"uint{row['lengthInBits']}"  # Use UINT for others

            parameter.attrib["parameterTypeRef"] = parameter_type_ref

            description = Et.SubElement(
                parameter, "{http://www.omg.org/space}LongDescription"
            )
            description.text = row["shortDescription"]

        return parameter_set

    def generate_telemetry_xml(self, output_xml_path):
        # Call the functions in order to generate the XML
        (
            telemetry_xml_root,
            parameter_type_set,
            parameter_set,
            telemetry_metadata,
        ) = self.create_telemetry_xml()
        unique_lengths, data_type = self.extract_data_info()
        parameter_type_set = self.create_parameter_types(
            parameter_type_set, unique_lengths, data_type, self.sci_byte
        )
        parameter_set = self.create_ccsds_packet_parameters(
            parameter_set, CCSDSParameters().parameters
        )
        telemetry_metadata = self.create_container_set(
            telemetry_metadata, CCSDSParameters().parameters, self.apid
        )
        codice_science_container = self.create_container_set(
            telemetry_metadata, CCSDSParameters().parameters, self.apid
        )
        codice_science_container = self.add_codice_science_parameters(
            codice_science_container, self.pkt
        )
        parameter_set = self.create_parameters_from_dataframe(parameter_set, self.pkt)

        # Create the XML tree and save the document
        tree = Et.ElementTree(telemetry_xml_root)
        Et.indent(tree, space="\t", level=0)

        # Use the provided output_xml_path
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
