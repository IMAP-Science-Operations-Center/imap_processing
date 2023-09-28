from pathlib import Path

from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory

# Define paths
packet_file = Path("housekeeping_data.bin")
xtce_document = Path(f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml")


def decom_packets(packet_file: str, xtce_packet_definition: str):
    """Unpack CoDICE raw housekeeping data packet. In this function, we unpack and.

        return data as it is. Data modification will not be done at this step.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename
    xtce_packet_definition : str
        Path to XTCE file with filename

    Returns
    -------
    list :
        List of all the unpacked data
    """
    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_parser.generator(binary_data)
        return list(packet_generator)


# Decompose packets
decomposed_packets = decom_packets(packet_file, xtce_document)

# Print decomposed packets
for packet in decomposed_packets:
    if packet.header["PKT_APID"].raw_value == 1136:
        print(packet.data)
