"""IMAP-HI L1A processing module."""

import logging
from pathlib import Path

from imap_processing.hi.l0 import decom_hi
from imap_processing.hi.l1a.histogram import create_dataset as hist_create_dataset
from imap_processing.hi.l1a.housekeeping import process_housekeeping
from imap_processing.hi.l1a.science_direct_event import science_direct_event
from imap_processing.hi.utils import HIAPID
from imap_processing.utils import group_by_apid

logger = logging.getLogger(__name__)


def hi_l1a(packet_file_path: str | Path):
    """Process IMAP raw data to l1a.

    Parameters
    ----------
    packet_file_path : str
        Data packet file path

    Returns
    -------
    processed_data : list
        List of processed xarray dataset
    """
    unpacked_data = decom_hi.decom_packets(packet_file_path)

    # group data by apid
    grouped_data = group_by_apid(unpacked_data)

    # Process science to l1a.
    processed_data = []
    for apid in grouped_data.keys():
        if apid == HIAPID.H45_SCI_CNT:
            logger.info(
                "Processing histogram data for [%s] packets", HIAPID.H45_SCI_CNT.name
            )
            data = hist_create_dataset(grouped_data[apid])
        elif apid == HIAPID.H45_SCI_DE:
            logger.info(
                "Processing direct event data for [%s] packets", HIAPID.H45_SCI_DE.name
            )

            data = science_direct_event(grouped_data[apid])
        elif apid == HIAPID.H45_APP_NHK:
            logger.info(
                "Processing housekeeping data for [%s] packets", HIAPID.H45_APP_NHK.name
            )
            data = process_housekeeping(grouped_data[apid])
        else:
            raise RuntimeError(f"Encountered unexpected APID [{apid}]")
        processed_data.append(data)
    return processed_data
