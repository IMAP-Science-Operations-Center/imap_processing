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

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice import constants
from imap_processing.utils import packet_file_to_datasets


def decom_packets(packet_file: Path) -> dict[int, xr.Dataset]:
    """
    Decom CoDICE data packets using CoDICE packet definition.

    Parameters
    ----------
    packet_file : pathlib.Path
        Path to data packet path with filename.

    Returns
    -------
    packets : dict[int, xarray.Dataset]
        Mapping from apid to ``xarray`` dataset, one dataset per apid.
    """
    xtce_packet_definition = Path(
        f"{imap_module_directory}/codice/packet_definitions/{constants.PACKET_TO_XTCE_MAPPING[packet_file.name]}"
    )
    packets = packet_file_to_datasets(packet_file, xtce_packet_definition)

    return packets
