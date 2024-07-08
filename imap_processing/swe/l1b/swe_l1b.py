"""Contains code to perform SWE L1b processing."""

import logging

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.swe.l1b.swe_l1b_science import swe_l1b_science
from imap_processing.swe.utils.swe_utils import SWEAPID
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)


def swe_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process data to L1B.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        The l1a data input.
    data_version : str
        Version of the data product being created.

    Returns
    -------
     data : xarray.Dataset
        Processed data to L1B.
    """
    apid = int(l1a_dataset.attrs["packet_apid"])

    # convert value from raw to engineering units as needed
    conversion_table_path = str(
        imap_module_directory / "swe/l1b/engineering_unit_convert_table.csv"
    )
    # Look up packet name from APID
    packet_name = next(packet for packet in SWEAPID if packet.value == apid)

    # Convert raw data to engineering units as needed
    eu_data = convert_raw_to_eu(
        l1a_dataset,
        conversion_table_path=conversion_table_path,
        packet_name=packet_name.name,
    )
    data = swe_l1b_science(eu_data, data_version)
    if data is None:
        logger.info("No data to write to CDF")
    return data
