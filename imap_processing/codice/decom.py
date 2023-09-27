"""Decommutate CODICE L0 data.

This is an example of how to use the 'space_packet_parser' module to parse with a
XTCE file specifically for the CODICE L0 data. This is a modified version of the
example found in the 'space_packet_parser' module documentation.
This is the start of CODICE L0 data processing.
"""

from pathlib import Path

from space_packet_parser import parser, xtcedef

# Define the APID. This is the APID for the CODICE L0 data that is in the
# 'RAW.bin' file. Data bins like 'RAW.bin' will encompass multiple APIDs.
# This is why we need to specify the APID.
# The APID should be in the packet definition file given by instrument team.
apid = 0x460

# Define paths
packet_file = Path("RAW.bin")
xtce_document = Path("L0/p_cod_aut_test.xml")

packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
my_parser = parser.PacketParser(packet_definition, apid)

with packet_file.open("rb") as binary_data:
    packet_generator = my_parser.generator(binary_data)

    for packet in packet_generator:
        # Do something with the packet data
        print(packet.header["PKT_APID"])
        print(packet.data)
