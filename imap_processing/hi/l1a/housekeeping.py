"""Unpack IMAP-Hi housekeeping data."""

import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time
from imap_processing.hi.hi_cdf_attrs import (
    hi_hk_l1a_attrs,
)
from imap_processing.utils import create_dataset


def process_housekeeping(packets: list[Packet]) -> xr.Dataset:
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[Packet]
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    dataset = create_dataset(
        packets=packets, spacecraft_time_key="ccsds_met", skip_keys=["INSTR_SPECIFIC"]
    )
    epoch_converted_time = [
        calc_start_time(sc_time) for sc_time in dataset["ccsds_met"].data
    ]

    # update Epoch attrs
    epoch = xr.DataArray(
        epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )
    dataset = dataset.assign_coords(epoch=epoch)

    # Add datalevel attrs
    dataset.attrs.update(hi_hk_l1a_attrs.output())
    return dataset
