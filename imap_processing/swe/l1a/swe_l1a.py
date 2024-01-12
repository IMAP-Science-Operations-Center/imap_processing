"""Contains code to perform SWE L1a processing."""

import logging

from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
    create_dataset,
    filename_descriptors,
)
from imap_processing.utils import group_by_apid, sort_by_time


def swe_l1a(packets):
    """Process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    packets: list
        Decom data list that contains all appIds

    Returns
    -------
    pathlib.Path
        Path to where the CDF file was created.
        This is used to upload file from local to s3.
        TODO: test this later.
    """
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
        return write_cdf(
            data,
            descriptor=f"{mode}{filename_descriptors.get(apid)}",
        )
