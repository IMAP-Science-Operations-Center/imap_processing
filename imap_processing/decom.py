from space_packet_parser import parser, xtcedef


def decom_packets(packet_file: str, xtce_packet_definition: str):
    """Unpack CCSDS data packet. In this function, we unpack and return data
    as it is. Data modification will not be done at this step.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename
    xtce_packet_definition : str
        Path to XTCE file with filename

    Returns
    -------
    List
        List of all the unpacked data
    """
    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    parser = parser.PacketParser(packet_definition)

    with packet_file.open(mode="rb") as binary_data:
        packet_generator = parser.generator(binary_data)
        return list(packet_generator)
