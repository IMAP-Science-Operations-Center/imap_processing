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
    packet_to_xtce_mapping = {
        "imap_codice_l0_hi-counters-aggregated_20240429_v001.pkts": "P_COD_HI_INST_COUNTS_AGGREGATED.xml",  # noqa
        "imap_codice_l0_hi-counters-singles_20240429_v001.pkts": "P_COD_HI_INST_COUNTS_SINGLES.xml",  # noqa
        "imap_codice_l0_hi-omni_20240429_v001.pkts": "P_COD_HI_OMNI_SPECIES_COUNTS.xml",
        "imap_codice_l0_hi-sectored_20240429_v001.pkts": "P_COD_HI_SECT_SPECIES_COUNTS.xml",  # noqa
        "imap_codice_l0_hskp_20100101_v001.pkts": "P_COD_NHK.xml",
        "imap_codice_l0_lo-counters-aggregated_20240429_v001.pkts": "P_COD_LO_INST_COUNTS_AGGREGATED.xml",  # noqa
        "imap_codice_l0_lo-counters-singles_20240429_v001.pkts": "P_COD_LO_INST_COUNTS_SINGLES.xml",  # noqa
        "imap_codice_l0_lo-sw-angular_20240429_v001.pkts": "P_COD_LO_SW_ANGULAR_COUNTS.xml",  # noqa
        "imap_codice_l0_lo-nsw-angular_20240429_v001.pkts": "P_COD_LO_NSW_ANGULAR_COUNTS.xml",  # noqa
        "imap_codice_l0_lo-sw-priority_20240429_v001.pkts": "P_COD_LO_SW_PRIORITY_COUNTS.xml",  # noqa
        "imap_codice_l0_lo-nsw-priority_20240429_v001.pkts": "P_COD_LO_NSW_PRIORITY_COUNTS.xml",  # noqa
        "imap_codice_l0_lo-sw-species_20240429_v001.pkts": "P_COD_LO_SW_SPECIES_COUNTS.xml",  # noqa
        "imap_codice_l0_lo-nsw-species_20240429_v001.pkts": "P_COD_LO_NSW_SPECIES_COUNTS.xml",  # noqa
    }

    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/{packet_to_xtce_mapping[packet_file.name]}"
    )
    decom_packet_list: list = decom.decom_packets(packet_file, xtce_document)
    return decom_packet_list
