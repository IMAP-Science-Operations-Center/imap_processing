"""Decommutate GLOWS CCSDS packets using GLOWS packet definitions."""

from enum import Enum
from pathlib import Path

from space_packet_parser import definitions

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.glows import __version__
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0


class GlowsParams(Enum):
    """
    Enum class for Glows packet data.

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
    packet_file_path: Path,
) -> tuple[list[HistogramL0], list[DirectEventL0]]:
    """
    Decom GLOWS data packets using GLOWS packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data : tuple[list[HistogramL0], list[DirectEventL0]]
        A tuple with two pieces: one list of the GLOWS histogram data, in GlowsHistL0
        instances, and one list of the GLOWS direct event data, in GlowsDeL0 instance.
    """
    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/GLX_COMBINED.xml"
    )

    packet_definition = definitions.XtcePacketDefinition(xtce_document)

    histdata = []
    dedata = []

    filename = packet_file_path.name

    with open(packet_file_path, "rb") as binary_data:
        glows_packets = packet_definition.packet_generator(binary_data)

        for packet in glows_packets:
            apid = packet["PKT_APID"]
            # Do something with the packet data
            if apid == GlowsParams.HIST_APID.value:
                values = [item.raw_value for item in packet.user_data.values()]
                hist_l0 = HistogramL0(
                    __version__, filename, CcsdsData(packet.header), *values
                )
                histdata.append(hist_l0)

            if apid == GlowsParams.DE_APID.value:
                values = [item.raw_value for item in packet.user_data.values()]

                de_l0 = DirectEventL0(
                    __version__, filename, CcsdsData(packet.header), *values
                )
                dedata.append(de_l0)

        return histdata, dedata
