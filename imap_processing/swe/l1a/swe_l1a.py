import logging
from pathlib import Path

from imap_processing.swe import __version__
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
    create_dataset,
    get_descriptor,
)
from imap_processing.write_to_cdf import write_to_cdf
from imap_processing.utils import group_by_apid, sort_by_met_time


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
    str
        Path name of where CDF file was created.
        This is used to upload file from local to s3.
        TODO: test this later.
    """
    # group data by appId
    grouped_data = group_by_apid(packets)

    for apid in grouped_data.keys():
        # If appId is science, then the file should contain all data of science appId
        if apid == SWEAPID.SWE_SCIENCE.value:
            # sort data by acquisition time
            sorted_packets = sort_by_met_time(grouped_data[apid], "ACQ_START_COARSE")
            logging.debug(
                "Processing science data for [%s] packets", len(sorted_packets)
            )
            data = swe_science(decom_data=sorted_packets)
        else:
            # If it's not science, we unpack, organize and save it as a dataset.
            sorted_packets = sort_by_met_time(grouped_data[apid], "SHCOARSE")
            data = create_dataset(packets=sorted_packets)

        current_dir = Path(__file__).parent
        # write data to CDF
        if apid == SWEAPID.SWE_APP_HK.value:
            return write_to_cdf(
                data,
                "swe",
                "l1a",
                version=__version__,
                mode=f"{data['APP_MODE'].data[0]}",
                description=get_descriptor(apid),
                directory=current_dir,
            )
        else:
            return write_to_cdf(
                data,
                "swe",
                "l1a",
                version=__version__,
                description=get_descriptor(apid),
                directory=current_dir,
            )
