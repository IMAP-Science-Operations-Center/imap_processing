"""Unpack IMAP-Hi histogram data."""

import numpy as np
import xarray as xr
from numpy._typing import NDArray

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
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # Rename shcoarse variable (do this first since it copies the input_ds)
    dataset = input_ds.rename_vars({"shcoarse": "ccsds_met"})

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
    # Populate 90-element histogram counters
    default_counter_attrs = attr_mgr.get_variable_attributes("hi_hist_counters")
    for counter_name in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        # Inject counter name into generic counter attributes
        counter_attrs = default_counter_attrs.copy()
        for key, val in counter_attrs.items():
            if isinstance(val, str) and "{counter_name}" in val:
                counter_attrs[key] = val.format(counter_name=counter_name)
        # Instantiate the counter DataArray
        new_vars[counter_name] = xr.DataArray(
            data=unpack_hist_counter(input_ds[counter_name].data.sum()),
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


def unpack_hist_counter(counter_bytes: bytes) -> NDArray[np.uint16]:
    """
    Unpack Hi SCI_CNT counter data for a single counter.

    Parameters
    ----------
    counter_bytes : bytes
        Sum individual bytes for all epochs of a Hi SCI_CNT counter.

    Returns
    -------
    output_array : numpy.ndarray[numpy.uint16]
        The unpacked 12-bit unsigned integers for the input bytes. The
        output array has a shape of (n, 90) where n is the number of SCI_CNT
        packets in the input dataset.
    """
    # Interpret bytes for all epochs of current counter as uint8 array
    counter_uint8 = np.frombuffer(counter_bytes, dtype=np.uint8)
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
    output_array = np.column_stack((even_uint12, odd_uint12)).reshape(-1, 90)
    return output_array
