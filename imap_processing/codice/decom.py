""" This is an example of how to use the space_packet_parser module to parse with a XTCE file.

"""

from pathlib import Path

from space_packet_parser import parser, xtcedef

# Define paths
packet_file = Path("RAW.bin")
xtce_document = Path("L0/p_cod_aut_test.xml")

packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
my_parser = parser.PacketParser(packet_definition, 0x460)

with packet_file.open("rb") as binary_data:
    packet_generator = my_parser.generator(binary_data)

    for packet in packet_generator:
        # Do something with the packet data
        print(packet.header["PKT_APID"])
        print(packet.data)
