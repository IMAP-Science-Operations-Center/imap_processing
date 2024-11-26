"""Unpack IMAP-Hi histogram data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import convert_to_binary_string

# define the names of the 24 counter arrays
# contained in the histogram packet
QUALIFIED_COUNTERS = (
    "ab_qualified",
    "c1c2_qualified",
    "ac1_qualified",
    "bc1_qualified",
    "abc1_qualified",
    "ac1c2_qualified",
    "bc1c2_qualified",
    "abc1c2_qualified",
)
LONG_COUNTERS = (
    "a_first_only",
    "b_first_only",
    "c_first_only",
    "ab_long",
    "c1c2_long",
    "ac1_long",
    "bc1_long",
    "abc1_long",
    "ac1c2_long",
    "bc1c2_long",
    "abc1c2_long",
)
TOTAL_COUNTERS = ("a_total", "b_total", "c_total", "fee_de_recd", "fee_de_sent")


def create_dataset(input_ds: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for a number of Hi Histogram packets.

    Parameters
    ----------
    input_ds : xarray.Dataset
        Dataset of packets.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    dataset = allocate_histogram_dataset(len(input_ds.epoch))

    # TODO: Move into the allocate dataset function. Ticket: #700
    dataset["epoch"].data[:] = input_ds["epoch"].data
    dataset["ccsds_met"].data = input_ds["shcoarse"].data
    dataset["esa_stepping_num"].data = input_ds["esa_step"].data
    dataset["num_of_spins"].data = input_ds["num_of_spins"].data

    # unpack the counter binary blobs into the Dataset
    # TODO: Look into avoiding the for-loops below
    #       It seems like we could try to reshape the arrays and do some numpy
    #       broadcasting rather than for-loops directly here. Ticket: #700
    for i_epoch in range(input_ds["epoch"].size):
        for counter in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
            binary_str_val = convert_to_binary_string(input_ds[counter].data[i_epoch])
            # unpack array of 90 12-bit unsigned integers
            counter_ints = [
                int(binary_str_val[i * 12 : (i + 1) * 12], 2) for i in range(90)
            ]
            dataset[counter][i_epoch] = counter_ints

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
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
    # preallocate the xr.DataArrays for all CDF attributes based on number of packets
    coords = dict()
    coords["epoch"] = xr.DataArray(
        np.empty(num_packets, dtype="datetime64[ns]"),
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )
    # Histogram data is binned in 90, 4-degree bins
    coords["angle"] = xr.DataArray(
        np.arange(2, 360, 4),
        name="angle",
        dims=["angle"],
        attrs=attr_mgr.get_variable_attributes("hi_hist_angle"),
    )

    data_vars = dict()
    # Generate label variables
    data_vars["angle_label"] = xr.DataArray(
        coords["angle"].values.astype(str),
        name="angle_label",
        dims=["angle"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_hist_angle_label", check_schema=False
        ),
    )
    # Other data variables
    data_vars["ccsds_met"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint32),
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("hi_hist_ccsds_met"),
    )
    data_vars["esa_stepping_num"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint8),
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("hi_hist_esa_step"),
    )
    data_vars["num_of_spins"] = xr.DataArray(
        np.empty(num_packets, dtype=np.uint8),
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("hi_hist_esa_step"),
    )

    # Allocate xarray.DataArray objects for the 24 90-element histogram counters
    default_counter_attrs = attr_mgr.get_variable_attributes("hi_hist_counters")
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
        attrs=attr_mgr.get_global_attributes("imap_hi_l1a_hist_attrs"),
    )
    return dataset
