"""Unpack IMAP-Hi histogram data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

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
        Dataset of packets generated using the
        `imap_processing.utils.packet_file_to_datasets` function.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    dataset = update_packet_dataset(input_ds)

    # unpack the counter binary blobs into the Dataset
    for counter in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        # Interpret bytestrings for all epochs of current counter as uint8 array
        counter_uint8 = np.frombuffer(input_ds[counter].data.sum(), dtype=np.uint8)
        # Split into triplets of upper-byte, split-byte and lower-byte arrays
        upper_uint8, split_unit8, lower_uint8 = np.reshape(
            counter_uint8, (3, -1), order="F"
        ).astype(np.uint16)
        # Compute even indexed uint12 values from upper-byte and first 4-bits of
        # split-byte
        even_uint12 = (upper_uint8 << 4) + (split_unit8 >> 4)
        # Compute odd indexed uint12 values from lower 4-bits of split-byte and
        # lower-byte
        odd_uint12 = ((split_unit8 & (2**4 - 1)) << 8) + lower_uint8
        combined = np.reshape(np.column_stack((even_uint12, odd_uint12)), (-1, 90))
        # Assign dataset counter data
        dataset[counter].data = combined
        pass

    return dataset


def update_packet_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Update dataset generated from ccsds to match L1A CDF definition.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset read in from IMAP-Hi histogram CCSDS data.

    Returns
    -------
    dataset : xarray.Dataset
        Updated xarray.Dataset ready to be filled with histogram counter data.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    dataset.epoch.attrs.update(
        attr_mgr.get_variable_attributes("epoch"),
    )
    # Add the hist_angle coordinate
    # Histogram data is binned in 90, 4-degree bins
    attrs = attr_mgr.get_variable_attributes("hi_hist_angle")
    dataset.coords.update(
        {
            "angle": xr.DataArray(
                np.arange(2, 360, 4),
                name="angle",
                dims=["angle"],
                attrs=attrs,
            )
        }
    )
    # Rename shcoarse variable
    dataset = dataset.rename_vars({"shcoarse": "ccsds_met"})
    # Update existing variable attributes
    for var_name in [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "ccsds_met",
        "esa_step",
        "num_of_spins",
        "cksum",
    ]:
        attrs = attr_mgr.get_variable_attributes(f"hi_hist_{var_name}")
        dataset.data_vars[var_name].attrs.update(attrs)

    new_vars = dict()
    # Allocate xarray.DataArray objects for the 90-element histogram counters
    default_counter_attrs = attr_mgr.get_variable_attributes(
        "hi_hist_counters", check_schema=False
    )
    for counter_name in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        # Inject counter name into generic counter attributes
        counter_attrs = default_counter_attrs.copy()
        dtype = counter_attrs.pop("dtype")
        for key, val in counter_attrs.items():
            if isinstance(val, str) and "{counter_name}" in val:
                counter_attrs[key] = val.format(counter_name=counter_name)
        new_vars[counter_name] = xr.DataArray(
            data=np.empty((dataset.epoch.size, dataset.angle.size), dtype=dtype),
            dims=["epoch", "angle"],
            attrs=counter_attrs,
        )

    # Generate label variable for angle coordinate
    new_vars["angle_label"] = xr.DataArray(
        dataset.coords["angle"].values.astype(str),
        name="angle_label",
        dims=["angle"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_hist_angle_label", check_schema=False
        ),
    )

    dataset.update(new_vars)
    dataset.attrs.update(attr_mgr.get_global_attributes("imap_hi_l1a_hist_attrs"))
    return dataset
