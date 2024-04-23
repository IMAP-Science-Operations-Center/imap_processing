"""Perform CoDICE L0 processing.

This module contains a function to decommutate CoDICE CCSDS packets using
XTCE packet definitions.

For more information on this process and the latest versions of the packet
definitions, see https://lasp.colorado.edu/galaxy/display/IMAP/CoDICE.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    packet_file = '/path/to/raw_ccsds_20230822_122700Z_idle.bin'
    packet_list = decom_packets(packet_file)
"""

from pathlib import Path

from imap_processing import decom, imap_module_directory

PACKET_TO_XTCE_MAPPING = {
    "raw_ccsds_20230822_122700Z_idle.bin": "P_COD_NHK.xml",
    "lo_fsw_view_5_ccsds.bin": "P_COD_LO_SW_SPECIES_COUNTS.xml",
    "lo_fsw_view_6_ccsds.bin": "P_COD_LO_NSW_SPECIES_COUNTS.xml",
}


def decom_packets(packet_file: Path) -> list:
    """Decom CoDICE data packets using CoDICE packet definition.

    Parameters
    ----------
    packet_file : pathlib.Path
        Path to data packet path with filename.

    Returns
    -------
    list : list
        all the unpacked data.
    """
    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/{PACKET_TO_XTCE_MAPPING[packet_file.name]}"
    )
    return decom.decom_packets(packet_file, xtce_document)
