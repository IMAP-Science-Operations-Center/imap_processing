from space_packet_parser import parser, xtcedef
import bitstring

def decom_packets(packet_file, xtce_packet_definition):
    """Unpack data packet. In this function, we unpack and return data
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
    packet_parser = parser.PacketParser(packet_definition)

    packets = []

    with open(packet_file, "r") as file:
        for line in file:
            if not line.startswith("#"):
                # Split the line by semicolons
                # Discard the first value since it is only a counter
                hex_values = line.strip().split(";")[1::]

                binary_values = ""
                for h in hex_values:
                    # Convert hex to integer
                    # 16 is the base of hexadecimal
                    int_value = int(h, 16)

                    # Convert integer to binary and remove the '0b' prefix
                    bin_value = bin(int_value)[2:]

                    # Make sure each binary string is 8 bits long
                    bin_value_padded = bin_value.zfill(8)

                    # Append the padded binary string to the final string
                    binary_values += bin_value_padded

                packet_generator = packet_parser.generator(bitstring.ConstBitStream(bin=binary_values))

                for packet in packet_generator:
                    packets.append(packet)

    return packets
