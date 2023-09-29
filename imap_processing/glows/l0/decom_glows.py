from enum import IntEnum
from pathlib import Path

from bitstring import ReadError
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0


class GlowsParams(IntEnum):
    """Enum class for Glows packet data.

    Attributes
    ----------
    HIST_APID : int
        Histogram packet APID
    DE_APID : int
        Direct event APID
    """

    HIST_APID = 1480
    DE_APID = 1481


def decom_packets(
    packet_file_path: str,
) -> tuple[list[HistogramL0], list[DirectEventL0]]:
    """Decom GLOWS data packets using GLOWS packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data : tuple[list[HistogramL0], list[DirectEventL0]]
        A tuple with two pieces: one list of the GLOWS histogram data, in GlowsHistL0
        instances, and one list of the GLOWS direct event data, in GlowsDeL0 instance
    """
    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/GLX_COMBINED.xml"
    )

    packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    glows_parser = parser.PacketParser(packet_definition)

    histdata = []
    dedata = []

    with open(packet_file_path, "rb") as binary_data:
        try:
            glows_packets = glows_parser.generator(
                binary_data,
                buffer_read_size_bytes=5790778,
            )

            for packet in glows_packets:
                # Do something with the packet data
                if packet.header["PKT_APID"].derived_value == GlowsParams.HIST_APID:
                    hist_l0 = HistogramL0(packet)
                    histdata.append(hist_l0)

                if packet.header["PKT_APID"].derived_value == GlowsParams.DE_APID:
                    de_l0 = DirectEventL0(packet)
                    dedata.append(de_l0)

        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        return histdata, dedata
