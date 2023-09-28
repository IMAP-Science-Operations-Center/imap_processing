from pathlib import Path

from bitstring import ReadError
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.glows.l0.glows_l0_data import GlowsDeL0, GlowsHistL0


def decom_packets(packet_file_path: str) -> tuple[list[GlowsHistL0], list[GlowsDeL0]]:
    """Decom GLOWS data packets using GLOWS packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data : tuple[list[GlowsHistL0], list[GlowsDeL0]]
        A tuple with two pieces: one list of the GLOWS histogram data, in GlowsHistL0
        instances, and one list of the GLOWS direct event data, in GlowsDeL0 instance
    """
    hist_apid = 1480
    de_apid = 1481

    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/GLX_COMBINED.xml"
    )

    hist_packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    histparser = parser.PacketParser(hist_packet_definition)

    histdata = []
    dedata = []

    with open(packet_file_path, "rb") as binary_data:
        try:
            hist_packets = histparser.generator(
                binary_data,
                buffer_read_size_bytes=5790778,
            )

            for packet in hist_packets:
                # Do something with the packet data
                if packet.header["PKT_APID"].derived_value == hist_apid:
                    hist_l0 = GlowsHistL0(packet)
                    histdata.append(hist_l0)

                if packet.header["PKT_APID"].derived_value == de_apid:
                    de_l0 = GlowsDeL0(packet)
                    dedata.append(de_l0)

        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        return histdata, dedata
