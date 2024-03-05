"""Contains code to perform SWE L1a processing."""

import logging

from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
    create_dataset,
    filename_descriptors,
)
from imap_processing.utils import group_by_apid, sort_by_time


def swe_l1a(file_path):
    """Process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    file_path: pathlib.Path
        Path where data is downloaded

    Returns
    -------
    List
        List of xarray.Dataset
    """
    packets = decom_swe.decom_packets(file_path)

    processed_data = []
    # group data by appId
    grouped_data = group_by_apid(packets)

    for apid in grouped_data.keys():
        # If appId is science, then the file should contain all data of science appId
        if apid == SWEAPID.SWE_SCIENCE:
            # sort data by acquisition time
            sorted_packets = sort_by_time(grouped_data[apid], "ACQ_START_COARSE")
            logging.debug(
                "Processing science data for [%s] packets", len(sorted_packets)
            )
            data = swe_science(decom_data=sorted_packets)
        else:
            # If it's not science, we unpack, organize and save it as a dataset.
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")
            data = create_dataset(packets=sorted_packets)

        # write data to CDF
        mode = f"{data['APP_MODE'].data[0]}-" if apid == SWEAPID.SWE_APP_HK else ""
        descriptor = f"{mode}{filename_descriptors.get(apid)}"

        processed_data.append({"data": data, "descriptor": descriptor})
    return processed_data
