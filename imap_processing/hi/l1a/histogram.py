"""Unpack IMAP-Hi histogram data."""

import dataclasses

import numpy as np
import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time
from imap_processing.hi import hi_cdf_attrs

# TODO: Verify that these names are OK for counter variables in the CDF
# define the names of the 24 counter arrays
# contained in the histogram packet
QUALIFIED_COUNTERS = (
    "qual_ab",
    "qual_c1c2",
    "qual_ac1",
    "qual_bc1",
    "qual_abc1",
    "qual_ac1c2",
    "qual_bc1c2",
    "qual_abc1c2",
)
LONG_COUNTERS = (
    "long_a",
    "long_b",
    "long_c",
    "long_ab",
    "long_c1c2",
    "long_ac1",
    "long_bc1",
    "long_abc1",
    "long_ac1c2",
    "long_bc1c2",
    "long_abc1c2",
)
TOTAL_COUNTERS = ("total_a", "total_b", "total_c", "fee_de_sent", "fee_de_recd")


def create_dataset(packets: list[Packet]) -> xr.Dataset:
    """Create dataset for a number of Hi Histogram packets.

    Parameters
    ----------
    packets : list[Packet]
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    dataset = allocate_histogram_dataset(len(packets))

    # unpack the packets data into the Dataset
    for i_epoch, packet in enumerate(packets):
        dataset.epoch.data[i_epoch] = calc_start_time(
            packet.data["CCSDS_MET"].raw_value
        )
        dataset.ccsds_met[i_epoch] = packet.data["CCSDS_MET"].raw_value
        dataset.esa_step[i_epoch] = packet.data["ESA_STEP"].raw_value

        # unpack 24 arrays of 90 12-bit unsigned integers
        counters_binary_data = packet.data["COUNTERS"].raw_value
        counter_ints = [
            int(counters_binary_data[i * 12 : (i + 1) * 12], 2) for i in range(90 * 24)
        ]
        # populate the dataset with the unpacked integers
        for i_counter, counter in enumerate(
            (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS)
        ):
            dataset[counter][i_epoch] = counter_ints[
                i_counter * 90 : (i_counter + 1) * 90
            ]

    return dataset


def allocate_histogram_dataset(num_packets: int) -> xr.Dataset:
    """
    Allocate empty xr.Dataset for specified number of Hi Histogram packets.

    Parameters
    ----------
    num_packets : int
        The number of Hi Histogram packets to allocate space for
        in the xr.Dataset.

    Returns
    -------
    xr.Dataset
        Empty xr.Dataset ready to be filled with packet data
    """
    # preallocate the xr.DataArrays for all CDF attributes based on number of packets
    coords = dict()
    coords["epoch"] = xr.DataArray(
        np.empty(num_packets, dtype="datetime64[ns]"),
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )
    # Histogram data is binned in 90, 4-degree bins
    # TODO: Confirm whether to define bins by centers or edges. For now centers
    #    are assumed.
    coords["angle"] = xr.DataArray(
        np.arange(2, 360, 4),
        name="angle",
        dims=["angle"],
        attrs=hi_cdf_attrs.hi_hist_l1a_angle_attrs.output(),
    )
    data_vars = dict()
    data_vars["ccsds_met"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint32),
        dims=["epoch"],
        attrs=hi_cdf_attrs.ccsds_met_attrs.output(),
    )
    data_vars["esa_step"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint8),
        dims=["epoch"],
        attrs=hi_cdf_attrs.esa_step_attrs.output(),
    )

    for counter in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        data_vars[counter] = xr.DataArray(
            data=np.empty((num_packets, len(coords["angle"])), np.uint16),
            dims=["epoch", "angle"],
            attrs=dataclasses.replace(
                hi_cdf_attrs.hi_hist_l1a_counter_attrs,
                catdesc=f"Angular histogram of {counter} type events",
                fieldname=f"{counter} histogram",
                label_axis=counter,
            ).output(),
        )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=hi_cdf_attrs.hi_hist_l1a_global_attrs.output(),
    )
    return dataset
