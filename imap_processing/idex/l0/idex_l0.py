"""Decommutate IDEX CCSDS packets."""

import logging

from imap_processing import decom, imap_module_directory

logger = logging.getLogger(__name__)


def decom_packets(packet_file: str) -> list:
    """
    Decom IDEX data packets using IDEX packet definition.

    Parameters
    ----------
    packet_file : str
        String to data packet path with filename.

    Returns
    -------
    list
        All the unpacked data.
    """
    xtce_filename = "idex_packet_definition.xml"
    xtce_file = f"{imap_module_directory}/idex/packet_definitions/{xtce_filename}"

    decom_packet_list = decom.decom_packets(packet_file, xtce_file)

    return list(decom_packet_list)
