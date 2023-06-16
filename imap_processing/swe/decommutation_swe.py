from space_packet_parser import parser, xtcedef


def decom_packet(packet_file: str, xtce_packet_definition: str):
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
    my_parser = parser.PacketParser(packet_definition)
    packet_list = []

    with packet_file.open(mode="rb") as binary_data:
        packet_generator = my_parser.generator(binary_data)
        for packet in packet_generator:
            # Add packet to list
            packet_list.append(packet)
    return packet_list
