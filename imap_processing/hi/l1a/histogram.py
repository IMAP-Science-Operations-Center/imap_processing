"""Unpack IMAP-Hi histogram data."""

import numpy as np
import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing import imap_module_directory
from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.cdf.utils import met_to_j2000ns

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
    """
    Create dataset for a number of Hi Histogram packets.

    Parameters
    ----------
    packets : list[space_packet_parser.ParsedPacket]
        Packet list.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    dataset = allocate_histogram_dataset(len(packets))

    # unpack the packets data into the Dataset
    for i_epoch, packet in enumerate(packets):
        dataset.epoch.data[i_epoch] = met_to_j2000ns(packet.data["CCSDS_MET"].raw_value)
        dataset.ccsds_met[i_epoch] = packet.data["CCSDS_MET"].raw_value
        dataset.esa_stepping_num[i_epoch] = packet.data["ESA_STEP"].raw_value

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
    Allocate empty xarray.Dataset for specified number of Hi Histogram packets.

    Parameters
    ----------
    num_packets : int
        The number of Hi Histogram packets to allocate space for
        in the xarray.Dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Empty xarray.Dataset ready to be filled with packet data.
    """
    cdf_manager = CdfAttributeManager(imap_module_directory / "cdf" / "config")
    cdf_manager.load_global_attributes("imap_hi_global_cdf_attrs.yaml")
    cdf_manager.load_variable_attributes("imap_hi_variable_attrs.yaml")
    # preallocate the xr.DataArrays for all CDF attributes based on number of packets
    coords = dict()
    coords["epoch"] = xr.DataArray(
        np.empty(num_packets, dtype="datetime64[ns]"),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("hi_hist_epoch"),
    )
    # Histogram data is binned in 90, 4-degree bins
    # TODO: Confirm whether to define bins by centers or edges. For now centers
    #    are assumed.
    coords["angle"] = xr.DataArray(
        np.arange(2, 360, 4),
        name="angle",
        dims=["angle"],
        attrs=cdf_manager.get_variable_attributes("hi_hist_angle"),
    )
    data_vars = dict()
    data_vars["ccsds_met"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint32),
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("hi_hist_ccsds_met"),
    )
    data_vars["esa_stepping_num"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint8),
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("hi_hist_esa_stepping_num"),
    )

    default_counter_attrs = cdf_manager.get_variable_attributes("hi_hist_counters")
    for counter_name in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        # Inject counter name into generic counter attributes
        counter_attrs = default_counter_attrs.copy()
        for key, val in counter_attrs.items():
            if isinstance(val, str) and "{counter_name}" in val:
                counter_attrs[key] = val.format(counter_name=counter_name)
        data_vars[counter_name] = xr.DataArray(
            data=np.empty((num_packets, len(coords["angle"])), np.uint16),
            dims=["epoch", "angle"],
            attrs=counter_attrs,
        )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_hi_l1a_hist_attrs"),
    )
    return dataset
