import xml.etree.ElementTree as Et

import pandas as pd

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


class GlowsTelemetryGenerator(TelemetryGenerator):
    # def extract_data_info(self):
    #     """
    #     Extract unique lengths from the 'lengthInBits' column and
    #     handle the "Data" mnemonic separately.
    #
    #     Parameters:
    #     - pkt: The DataFrame containing packet data.
    #
    #     Returns:
    #     - unique_lengths: Unique values from the 'lengthInBits' column.
    #     - data_type: The data type for the "Data" mnemonic.
    #     """
    #     # Extract unique values from the 'lengthInBits' column
    #     unique_lengths = self.pkt["lengthInBits"].unique().astype(int)
    #
    #     # Handle the "Data" mnemonic separately
    #     data_mnemonic = "BIN"
    #
    #     data_type = self.pkt[self.pkt["mnemonic"].str.contains(data_mnemonic)][
    #         "dataType"
    #     ].values[0]
    #
    #     return unique_lengths, data_type, None

    def create_parameter_types(
        self,
        parameter_type_set,
        unique_lengths,
        data_types,
        sci_byte=None,
    ):
        """
        Create parameter types based on 'dataType' for the unique 'lengthInBits' values.
        This will loop through the unique lengths and create a ParameterType element
        for each length representing a UINT type.

        Parameters:
        - parameter_type_set: The ParameterTypeSet element where parameter types are.
        - unique_lengths: Unique values from the 'lengthInBits' column.
        - data_type_data: The data type for the "Data" mnemonic.
        - data_type_event_data: The data type for the "Event_Data" mnemonic.
        - sci_byte: The BYTE number of lengthInBits in both "Data" and "Event_Data"

        Returns:
        - parameter_type_set: The updated ParameterTypeSet element.
        """
        for size in unique_lengths:
            # All bins equal?
            # Are bins the only thing that matters? -> check glows code for this
            parameter_type_ref = data_types["BIN"]  # Use UINT for other sizes
            print(parameter_type_ref)
            parameter_type = Et.SubElement(parameter_type_set, "xtce:ParameterType")
            parameter_type.attrib["name"] = f"uint{size}"
            parameter_type.attrib["signed"] = "false"

            encoding = Et.SubElement(parameter_type, "xtce:IntegerDataEncoding")
            encoding.attrib["sizeInBits"] = str(size)
            encoding.attrib["encoding"] = "unsigned"

            # Create BinaryParameterType if parameter_type_ref is "BYTE"
            if parameter_type_ref == "BYTE":
                binary_parameter_type = Et.SubElement(
                    parameter_type_set, "xtce:BinaryParameterType"
                )
                binary_parameter_type.attrib["name"] = parameter_type_ref

                Et.SubElement(binary_parameter_type, "xtce:UnitSet")

                binary_data_encoding = Et.SubElement(
                    binary_parameter_type, "xtce:BinaryDataEncoding"
                )
                binary_data_encoding.attrib["bitOrder"] = "mostSignificantBitFirst"

                size_in_bits = Et.SubElement(binary_data_encoding, "xtce:SizeInBits")
                fixed_value = Et.SubElement(size_in_bits, "xtce:FixedValue")
                fixed_value.text = str(sci_byte)  # Set the size in bits to sci_byte

        return parameter_type_set


if __name__ == "__main__":
    # Histogram science packet
    packet_name = "P_GLX_TMSCHIST"
    excel_file = "../../../../TLM_GLX_2023_06_22.xls"

    # Histogram
    apid = "1480"

    sci_byte = 0

    xls = pd.ExcelFile(excel_file)
    packet_df = xls.parse(packet_name)

    # Clean up packet
    packet_df.drop(
        labels=packet_df.columns[packet_df.columns.str.contains("Unnamed")],
        axis=1,
        inplace=True,
    )
    packet_df.dropna(subset="packetName", inplace=True)

    generator = GlowsTelemetryGenerator(packet_name, excel_file, apid, pkt=packet_df)

    generator.generate_telemetry_xml(
        "../packet_definitions/output2.xml", ["BIN"], "GLOWSSciencePacket"
    )
