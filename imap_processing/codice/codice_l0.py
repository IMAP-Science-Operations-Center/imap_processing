"""
Perform CoDICE L0 processing.

This module contains a function to decommutate CoDICE CCSDS packets using
XTCE packet definitions.

For more information on this process and the latest versions of the packet
definitions, see https://lasp.colorado.edu/galaxy/display/IMAP/CoDICE.

Notes
-----
    from imap_processing.codice.codice_l0 import decom_packets
    packet_file = '/path/to/raw_ccsds_20230822_122700Z_idle.bin'
    packet_list = decom_packets(packet_file)
"""

from pathlib import Path

from imap_processing import decom, imap_module_directory
from imap_processing.codice import constants


def decom_packets(packet_file: Path) -> list:
    """
    Decom CoDICE data packets using CoDICE packet definition.

    Parameters
    ----------
    packet_file : pathlib.Path
        Path to data packet path with filename.

    Returns
    -------
    list : list
        All the unpacked data.
    """
    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/{constants.PACKET_TO_XTCE_MAPPING[packet_file.name]}"
    )
    decom_packet_list: list = decom.decom_packets(packet_file, xtce_document)
    return decom_packet_list
