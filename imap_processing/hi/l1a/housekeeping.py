"""Unpack IMAP-Hi housekeeping data."""

import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing.hi.hi_cdf_attrs import (
    hi_hk_l1a_attrs,
)
from imap_processing.utils import create_dataset, update_epoch_to_datetime


def process_housekeeping(packets: list[Packet]) -> xr.Dataset:
    """
    Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[space_packet_parser.ParsedPacket]
        Packet list.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    dataset = create_dataset(
        packets=packets, spacecraft_time_key="ccsds_met", skip_keys=["INSTR_SPECIFIC"]
    )
    # Update epoch to datetime
    dataset = update_epoch_to_datetime(dataset)

    # Add datalevel attrs
    dataset.attrs.update(hi_hk_l1a_attrs.output())
    return dataset
