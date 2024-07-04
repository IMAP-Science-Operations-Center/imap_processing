"""Class Creation for TelemetryGenerator."""

import xml.etree.ElementTree as Et
from datetime import datetime
from typing import Optional

import pandas as pd

from tools.xtce_generation.ccsds_header_xtce_generator import CCSDSParameters


class TelemetryGenerator:
    """
    This class will automatically generate XTCE files from excel definition files.

    The excel file should have the following columns: mnemonic, sequence, lengthInBits,
    startBit, dataType, convertAs, units, source, and either shortDescription or
    longDescription.

    This class will correctly generate a CCSDS header and the provided data types.
    It doesn't handle the following cases:
    - Enum generation
    - Variable packet lengths
    - Multiple APIDs or complex APID comparison

    It is intended for use as a first pass of XTCE generation, and most cases, the
    packet definitions will require manual updates.

    Use generate_telemetry_xml to create XML files.

    Parameters
    ----------
    packet_name : str
        Name of packet.
    path_to_excel_file : dict todo check
        Path to the excel file.
    apid : int todo check
        The application ID associated with the telemetry packet.
    pkt : str todo check
        Default set to None.
    """

    def __init__(
        self,
        packet_name: str,
        path_to_excel_file: str,
        apid: int,
        pkt: Optional[str] = None,
    ) -> None:
        """Initialize TelemetryGenerator."""
        self.packet_name = packet_name
        self.apid = apid
        self.source_link = "http://www.omg.org/space/xtce"

        if pkt is None:
            # Read excel sheet
            xls = pd.ExcelFile(path_to_excel_file)
            self.pkt = xls.parse(packet_name)
        else:
            self.pkt = pkt

    def create_telemetry_xml(self) -> tuple:
        """
        Create an XML representation of telemetry data based on input parameters.

        Returns
        -------
        root
            The root element of the generated XML tree.
        parameter_type_set
            The ParameterTypeSet element.
        parameter_set
            The ParameterSet element.
        telemetry_metadata
            The TelemetryMetaData element.
        """
        # Register the XML namespace
        Et.register_namespace("xtce", self.source_link)

        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m")

        # Create the root element and add namespaces
        root = Et.Element("xtce:SpaceSystem")
        root.attrib["xmlns:xtce"] = f"{self.source_link}"
        root.attrib["name"] = self.packet_name
        # xmlns:xtce="http://www.omg.org/space/xtce"

        # Create the Header element with attributes 'date', 'version', and 'author'
        # Versioning is used to keep track of changes to the XML file.
        header = Et.SubElement(root, "xtce:Header")
        header.attrib["date"] = current_date
        header.attrib["version"] = "1.0"
        header.attrib["author"] = "IMAP SDC"

        # Create the TelemetryMetaData element
        telemetry_metadata = Et.SubElement(root, "xtce:TelemetryMetaData")

        # Create the ParameterTypeSet element
        parameter_type_set = Et.SubElement(telemetry_metadata, "xtce:ParameterTypeSet")

        # Create the ParameterSet element
        parameter_set = Et.SubElement(telemetry_metadata, "xtce:ParameterSet")

        return root, parameter_type_set, parameter_set, telemetry_metadata

    def get_unique_bits_length(self) -> dict:
        """
        Create dictionary.

        Get unique values from the 'lengthInBits' column and create dictionary
        with key and value using the dataType and lengthInBits.

        Returns
        -------
        unique_lengths : dict
            Dictionary containing all unique bits lengths.

        Examples
        --------
        {
            'UINT16': 16,
            'UINT32': 32,
            'BYTE13000': 13000,
        }
        """
        # Extract unique values from the 'lengthInBits' column
        length_in_bits = self.pkt["lengthInBits"]
        data_types = self.pkt["dataType"]
        unique_lengths = {}
        for index in range(len(length_in_bits)):
            # Here, we are creating a dictionary like this:
            # {
            #     'UINT16': 16,
            #     'UINT32': 32,
            #     'BYTE13000': 13000,
            # }
            unique_lengths[f"{data_types[index]}{length_in_bits[index]}"] = (
                length_in_bits[index]
            )
        # Sort the dictionary by the value (lengthInBits)
        unique_lengths = dict(sorted(unique_lengths.items(), key=lambda item: item[1]))
        return unique_lengths

    def create_parameter_types(
        self,
        parameter_type_set: Et.Element,
        unique_lengths: dict,
    ) -> Et.Element:
        """
        Create parameter types based on 'dataType' for the unique 'lengthInBits' values.

        This will loop through the unique lengths and create a ParameterType element
        for each length representing a data type.

        Parameters
        ----------
        parameter_type_set : Et.Element
            The ParameterTypeSet element where parameter types are.
        unique_lengths : dict
            Unique values from the 'lengthInBits' column.

        Returns
        -------
        parameter_type_set : Et.Element
            The updated ParameterTypeSet element.
        """
        for parameter_type_ref_name, size in unique_lengths.items():
            if "UINT" in parameter_type_ref_name:
                parameter_type = Et.SubElement(
                    parameter_type_set,
                    "xtce:IntegerParameterType",
                )
                parameter_type.attrib["name"] = parameter_type_ref_name
                parameter_type.attrib["signed"] = "false"

                encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
                encoding.attrib["sizeInBits"] = str(size)
                encoding.attrib["encoding"] = "unsigned"

            elif any(x in parameter_type_ref_name for x in ["SINT", "INT"]):
                parameter_type = Et.SubElement(
                    parameter_type_set,
                    "xtce:IntegerParameterType",
                )
                parameter_type.attrib["name"] = parameter_type_ref_name
                parameter_type.attrib["signed"] = "true"
                encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
                encoding.attrib["sizeInBits"] = str(size)
                encoding.attrib["encoding"] = "signed"

            elif "BYTE" in parameter_type_ref_name:
                binary_parameter_type = Et.SubElement(
                    parameter_type_set,
                    "xtce:BinaryParameterType",
                )
                binary_parameter_type.attrib["name"] = parameter_type_ref_name

                Et.SubElement(binary_parameter_type, "xtce:UnitSet")

                binary_data_encoding = Et.SubElement(
                    binary_parameter_type, "xtce:BinaryDataEncoding"
                )
                binary_data_encoding.attrib["bitOrder"] = "mostSignificantBitFirst"

                size_in_bits = Et.SubElement(binary_data_encoding, "xtce:SizeInBits")
                fixed_value = Et.SubElement(size_in_bits, "xtce:FixedValue")
                fixed_value.text = str(size)

        return parameter_type_set

    def create_ccsds_packet_parameters(
        self, parameter_set: Et.Element, ccsds_parameters: list
    ) -> Et.Element:
        """
        Create XML elements to define CCSDS packet parameters based on the given data.

        Parameters
        ----------
        parameter_set : Et.Element
            The ParameterSet element where parameters will be added.
        ccsds_parameters : list
            A list of dictionaries containing CCSDS parameter data.

        Returns
        -------
        parameter_set : Et.Element
            The updated ParameterSet element.
        """
        for parameter_data in ccsds_parameters:
            parameter = Et.SubElement(parameter_set, "xtce:Parameter")
            parameter.attrib["name"] = parameter_data["name"]
            parameter.attrib["parameterTypeRef"] = parameter_data["parameterTypeRef"]

            description = Et.SubElement(parameter, "xtce:LongDescription")
            description.text = parameter_data["description"]

        return parameter_set

    def create_container_set(
        self,
        telemetry_metadata: Et.Element,
        ccsds_parameters: list,
        container_name: str,
    ) -> Et.Element:
        """
        Create XML elements.

        These elements are for ContainerSet, CCSDSPacket SequenceContainer,
        and Packet SequenceContainer.

        Parameters
        ----------
        telemetry_metadata : Et.Element
            The TelemetryMetaData element where containers are.
        ccsds_parameters : list
            A list of dictionaries containing CCSDS parameter data.
        container_name : str
            The name of sequence container.

        Returns
        -------
        telemetry_metadata : Et.Element
            The updated TelemetryMetaData element.
        """
        # Create ContainerSet element
        container_set = Et.SubElement(telemetry_metadata, "xtce:ContainerSet")
        # Create CCSDSPacket SequenceContainer
        ccsds_packet_container = Et.SubElement(container_set, "xtce:SequenceContainer")
        ccsds_packet_container.attrib["name"] = "CCSDSPacket"
        ccsds_packet_entry_list = Et.SubElement(
            ccsds_packet_container, "xtce:EntryList"
        )

        # Populate EntryList for CCSDSPacket SequenceContainer
        for parameter_data in ccsds_parameters:
            parameter_ref_entry = Et.SubElement(
                ccsds_packet_entry_list, "xtce:ParameterRefEntry"
            )
            parameter_ref_entry.attrib["parameterRef"] = parameter_data["name"]

        # Create Packet SequenceContainer that use CCSDSPacket SequenceContainer
        # as base container
        science_container = Et.SubElement(container_set, "xtce:SequenceContainer")
        science_container.attrib["name"] = container_name

        base_container = Et.SubElement(science_container, "xtce:BaseContainer")
        base_container.attrib["containerRef"] = "CCSDSPacket"

        # Add RestrictionCriteria element to use the given APID for comparison
        restriction_criteria = Et.SubElement(base_container, "xtce:RestrictionCriteria")
        comparison = Et.SubElement(restriction_criteria, "xtce:Comparison")
        comparison.attrib["parameterRef"] = "PKT_APID"
        comparison.attrib["value"] = f"{self.apid}"
        comparison.attrib["useCalibratedValue"] = "false"

        # Populate EntryList for packet SequenceContainer
        packet_entry_list = Et.SubElement(science_container, "xtce:EntryList")
        parameter_refs = self.pkt.loc[7:, "mnemonic"].tolist()

        for parameter_ref in parameter_refs:
            parameter_ref_entry = Et.SubElement(
                packet_entry_list, "xtce:ParameterRefEntry"
            )
            parameter_ref_entry.attrib["parameterRef"] = parameter_ref

        return telemetry_metadata

    def create_remaining_parameters(self, parameter_set: Et.Element) -> Et.Element:
        """
        Create XML elements for parameters.

        These are based on DataFrame rows starting from SHCOARSE (also known as MET).

        Parameters
        ----------
        parameter_set : Et.Element
            The ParameterSet element where parameters will be added.

        Returns
        -------
        parameter_set : Et.Element
            The updated ParameterSet element.
        """
        # Process rows from SHCOARSE until the last available row in the DataFrame
        for index, row in self.pkt.iterrows():
            if index < 7:
                # Skip rows until SHCOARSE(also known as MET) which are
                # not part of CCSDS header
                continue

            parameter = Et.SubElement(parameter_set, "xtce:Parameter")
            parameter.attrib["name"] = row["mnemonic"]
            parameter_type_ref = f"{row['dataType']}{row['lengthInBits']}"

            parameter.attrib["parameterTypeRef"] = parameter_type_ref

            # Add descriptions if they exist
            if pd.notna(row.get("shortDescription")):
                parameter.attrib["shortDescription"] = row.get("shortDescription")
            if pd.notna(row.get("longDescription")):
                description = Et.SubElement(parameter, "xtce:LongDescription")
                description.text = row.get("longDescription")

        return parameter_set

    def generate_telemetry_xml(self, output_xml_path: str, container_name: str) -> None:
        """
        Create and output an XTCE file based on the data within the class.

        Parameters
        ----------
        output_xml_path : str
            The path for the final xml file.
        container_name : str
            The name of the sequence container in the output xml.
        """

        unique_bits_lengths_data = self.get_unique_bits_length()

        # Here, we create the XML components so that we can add data in following steps
        (
            telemetry_xml_root,
            parameter_type_set,
            parameter_set,
            telemetry_metadata,
        ) = self.create_telemetry_xml()

        parameter_type_set = self.create_parameter_types(
            parameter_type_set, unique_bits_lengths_data
        )

        # Create CCSDSPacket parameters and add them to the ParameterSet element
        parameter_set = self.create_ccsds_packet_parameters(
            parameter_set, CCSDSParameters().parameters
        )
        # Add remaining parameters to the ParameterSet element
        parameter_set = self.create_remaining_parameters(parameter_set)

        # Create ContainerSet with CCSDSPacket SequenceContainer and packet
        # SequenceContainer
        telemetry_metadata = self.create_container_set(
            telemetry_metadata,
            CCSDSParameters().parameters,
            container_name=container_name,
        )

        # Create the XML tree and save the document
        tree = Et.ElementTree(telemetry_xml_root)
        Et.indent(tree, space="\t", level=0)

        # Use the provided output_xml_path
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
