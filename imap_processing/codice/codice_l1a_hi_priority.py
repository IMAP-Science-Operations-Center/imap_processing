"""CoDICE L1A Hi priority processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    ViewTabInfo,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_view_tab_info,
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_hi_priority(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
    """
    Process CoDICE Hi Priority L1A data.

    Parameters
    ----------
    unpacked_dataset : xarray.Dataset
        Unpacked dataset from L0 packet file.
    lut_file : Path
        Path to the LUT file for processing.

    Returns
    -------
    xarray.Dataset
        Processed L1A dataset for Hi Omni data.
    """
    # Get these values from unpacked data. These are used to
    # lookup in LUT table.
    table_id = unpacked_dataset["table_id"].values[0]
    view_id = unpacked_dataset["view_id"].values[0]
    apid = unpacked_dataset["pkt_apid"].values[0]
    plan_id = unpacked_dataset["plan_id"].values[0]
    plan_step = unpacked_dataset["plan_step"].values[0]

    logger.info(
        f"Processing species with - APID: {apid} / 0x{apid:X}, View ID: {view_id}, "
        f"Table ID: {table_id}, Plan ID: {plan_id}, Plan Step: {plan_step}"
    )

    # ========== Get LUT Data ===========
    # Read information from LUT
    sci_lut_data = read_sci_lut(lut_file, table_id)

    view_tab_info = get_view_tab_info(sci_lut_data, view_id, apid)

    view_tab_obj = ViewTabInfo(
        apid=apid,
        view_id=view_id,
        sensor=view_tab_info["sensor"],
        three_d_collapsed=view_tab_info["3d_collapse"],
        collapse_table=view_tab_info["collapse_table"],
    )

    if view_tab_obj.sensor != 1:
        raise ValueError("Unsupported sensor ID for Hi priority processing.")

    # ========= Get Epoch Time Data ===========
    # Epoch center time and delta
    epoch_center, deltas = get_codice_epoch_time(
        unpacked_dataset["acq_start_seconds"].values,
        unpacked_dataset["acq_start_subseconds"].values,
        unpacked_dataset["spin_period"].values,
        view_tab_obj,
    )

    # ========= Decompress and Calculate Reshape information ===========
    species_data = sci_lut_data["data_product_hi_tab"]["0"]["priority"]
    species_names = species_data.keys()
    logical_source_id = "imap_codice_l1a_hi-priority"
    compression_algorithm = constants.HI_COMPRESSION_ID_LOOKUP[view_tab_obj.view_id]

    # Decompress data using byte count information from decommed data
    binary_data_list = unpacked_dataset["data"].values
    byte_count_list = unpacked_dataset["byte_count"].values

    # The decompressed data in the shape of (epoch, n). Then reshape later.
    decompressed_data = [
        decompress(
            packet_data[:byte_count],
            compression_algorithm,
        )
        for (packet_data, byte_count) in zip(
            binary_data_list, byte_count_list, strict=False
        )
    ]

    num_packets = len(binary_data_list)

    # Reshape decompressed data to in below for loop:
    # (num_packets, collapse_shape[1](inst_az))
    collapse_shape = get_collapse_pattern_shape(
        sci_lut_data,
        view_tab_obj.sensor,
        view_tab_obj.collapse_table,
    )
    species_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets, collapse_shape[1]
    )
    species_data = species_data.astype(np.float64)
    # ========== Create CDF Dataset with Metadata ===========
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    l1a_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                met_to_ttj2000ns(epoch_center),
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
            ),
            "epoch_delta_minus": xr.DataArray(
                deltas,
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes(
                    "epoch_delta_minus", check_schema=False
                ),
            ),
            "epoch_delta_plus": xr.DataArray(
                deltas,
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes(
                    "epoch_delta_plus", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes(logical_source_id),
    )
    # Add first few unique variables
    l1a_dataset["spin_period"] = xr.DataArray(
        unpacked_dataset["spin_period"].values * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("spin_period"),
    )
    l1a_dataset["data_quality"] = xr.DataArray(
        unpacked_dataset["suspect"].values,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("data_quality"),
    )

    # Finally, add species data variables and their uncertainties
    for idx, species in enumerate(species_names):
        l1a_dataset[species] = xr.DataArray(
            species_data[
                :,
                idx,
            ],
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(species),
        )
        l1a_dataset[f"unc_{species}"] = xr.DataArray(
            np.sqrt(l1a_dataset[species].values),
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(species),
        )

    return l1a_dataset
