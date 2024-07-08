"""Contains code to perform SWE L1a processing."""

import logging

import xarray as xr

from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)


def swe_l1a(file_path: str, data_version: str) -> xr.Dataset:
    """
    Will process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    file_path : str
        Path where data is downloaded.
    data_version : str
        Data version to write to CDF files and the Data_version CDF attribute.
        Should be in the format Vxxx.

    Returns
    -------
    List
        List of xarray.Dataset.
    """
    packets = decom_swe.decom_packets(file_path)

    # group data by appId
    grouped_data = group_by_apid(packets)

    # TODO: figure out how to handle non-science data error
    # Process science data packets
    # sort data by acquisition time
    sorted_packets = sort_by_time(grouped_data[SWEAPID.SWE_SCIENCE], "ACQ_START_COARSE")
    logger.debug("Processing science data for [%s] packets", len(sorted_packets))

    return swe_science(decom_data=sorted_packets, data_version=data_version)
