"""Contains code to perform SWE L1a processing."""

import logging

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def swe_l1a(packet_file: str, data_version: str) -> xr.Dataset:
    """
    Will process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    packet_file : str
        Path where the raw packet file is stored.
    data_version : str
        Data version to write to CDF files and the Data_version CDF attribute.
        Should be in the format Vxxx.

    Returns
    -------
    List
        List of xarray.Dataset.
    """
    xtce_document = (
        f"{imap_module_directory}/swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=False
    )

    # TODO: figure out how to handle non-science data
    return swe_science(
        l0_dataset=datasets_by_apid[SWEAPID.SWE_SCIENCE], data_version=data_version
    )
