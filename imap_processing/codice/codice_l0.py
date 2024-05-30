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
    # Determine the mapping between L0 filename and corresponding XML filename
    # via the descriptor. For example:
    # packet_file = imap_codice_l0_lo-sw-species_YYYYMMDD_v001.pkts
    descriptor = (
        packet_file.stem.split("imap_codice_l0_")[
            -1
        ]  # lo-sw-species_YYYYMMDD_v001.pkts
        .split("_")[0]  # lo-sw-species
        .upper()  # LO-SW-SPECIES
        .replace("-", "_")  # LO_SW_SPECIES
    )
    filename = f"P_COD_{descriptor}.xml"  # P_COD_LO_SW_SPECIES.xml
    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/{filename}"
    )
    return decom.decom_packets(packet_file, xtce_document)
