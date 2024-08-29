"""Decommutate IDEX CCSDS packets."""

import logging
from pathlib import Path
from typing import Union

from imap_processing import decom, imap_module_directory

logger = logging.getLogger(__name__)


def decom_packets(packet_file: Union[str, Path]) -> list:
    """
    Decom IDEX data packets using IDEX packet definition.

    Parameters
    ----------
    packet_file : pathlib.Path | str
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
