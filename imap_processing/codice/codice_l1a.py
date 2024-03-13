"""Perform CoDICE l1a processing.

This module contains functions to process decommutated CoDICE packets and create
L1a data products.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1a import codice_l1a
    packets = decom_packets(packet_file, xtce_document)
    cdf_filename = codice_l1a(packets)
"""

import logging

import imap_data_access
import space_packet_parser

from imap_processing.cdf.utils import write_cdf
from imap_processing.codice.utils import CODICEAPID, create_dataset
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)


def codice_l1a(packets: list[space_packet_parser.parser.Packet]) -> str:
    """Process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        Decom data list that contains all APIDs

    Returns
    -------
    cdf_filename : str
        The path to the CDF file that was created
    """
    # Group data by APID and sort by time
    grouped_data = group_by_apid(packets)

    for apid in grouped_data.keys():
        if apid == CODICEAPID.COD_NHK:
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")
            data = create_dataset(packets=sorted_packets)
        else:
            logger.debug(f"{apid} is currently not supported")

    file = imap_data_access.ScienceFilePath.generate_from_inputs(
        "codice", "l1a", "hk", "20210101", "20210102", "v01-01"
    )
    # Write data to CDF
    cdf_filename = write_cdf(data, file.construct_path())

    return cdf_filename
