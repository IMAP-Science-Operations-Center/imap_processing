"""Decommutate GLOWS CCSDS packets using GLOWS packet definitions."""

from enum import Enum
from pathlib import Path

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.glows import __version__
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0
from imap_processing.utils import packet_generator, separate_ccsds_header_userdata


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

    histdata = []
    dedata = []

    filename = packet_file_path.name

    for packet in packet_generator(packet_file_path, xtce_document):
        apid = packet["PKT_APID"]
        # Do something with the packet data
        if apid == GlowsParams.HIST_APID.value:
            header, userdata = separate_ccsds_header_userdata(packet)
            hist_l0 = HistogramL0(
                __version__, filename, CcsdsData(header), *list(userdata.values())
            )
            histdata.append(hist_l0)

        if apid == GlowsParams.DE_APID.value:
            values = [item.raw_value for i, item in enumerate(packet.values()) if i > 6]
            header = {
                key: value for i, (key, value) in enumerate(packet.items()) if i <= 6
            }
            de_l0 = DirectEventL0(__version__, filename, CcsdsData(header), *values)
            dedata.append(de_l0)

    return histdata, dedata
