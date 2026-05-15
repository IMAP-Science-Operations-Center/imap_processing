"""CoDICE L1A Lo Counters aggregated processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.constants import HALF_SPIN_FILLVAL
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CoDICECompression,
    calculate_acq_time_per_step,
    get_codice_epoch_time,
    get_counters_aggregated_pattern,
    get_view_tab_obj,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_lo_counters_aggregated(
    group_ds: xr.Dataset,
    lut_file: Path,
    table_id: str,
    view_id: int,
    apid: int,
    plan_id: int,
    plan_step: int,
) -> xr.Dataset:
    """
    Process a single table-ID group of CoDICE Lo Counters Aggregated L1A data.

    Parameters
    ----------
    group_ds : xarray.Dataset
        Dataset filtered to a single table_id.
    lut_file : Path
        Path to the LUT file for processing.
    table_id : str
        The table ID for this group.
    view_id : int
        View ID (uniform across the product).
    apid : int
        APID (uniform across the product).
    plan_id : int
        Plan ID (uniform across the product).
    plan_step : int
        Plan step (uniform across the product).

    Returns
    -------
    xarray.Dataset
        Processed L1A dataset for input table-ID group.
    """
    logger.info(
        f"Processing aggregated with - APID: {apid} / 0x{apid:X}, View ID: {view_id}, "
        f"Table ID: {table_id}, Plan ID: {plan_id}, Plan Step: {plan_step}"
    )
    # ========== Get LUT Data ===========
    sci_lut_data, view_tab_obj = get_view_tab_obj(lut_file, table_id, view_id, apid)

    if view_tab_obj.sensor != 0:
        raise ValueError("Unsupported sensor ID for Lo processing.")

    # ========== Get Voltage Data from LUT ===========
    # Use plan id and plan step to get voltage data's table_number in ESA sweep table.
    # Voltage data is (128,)
    esa_table_number = sci_lut_data["plan_tab"][f"({plan_id}, {plan_step})"][
        "lo_stepping"
    ]
    voltage_data = sci_lut_data["esa_sweep_tab"][f"{esa_table_number}"]

    # ========= Decompress and Reshape Data ===========
    logical_source_id = "imap_codice_l1a_lo-counters-aggregated"
    # Counters is little bit different in how CDF variables are derived.
    # For singles, CDF variables are coming from 'product' tab. But for
    # counters aggregated, it's coming from 'collapsed' tab in JSON LUT.
    # In addition, for Lo, we include other inactive variables as zeros
    # in the CDF. For Hi, all variables except 'reserved{x}' are active,
    # so reserved variables are excluded unless it changes. This change
    # will affect CoDICE team and here.
    non_reserved_variables = get_counters_aggregated_pattern(
        sci_lut_data, view_tab_obj.sensor, view_tab_obj.collapse_table
    )
    non_reserved_keys = non_reserved_variables.keys()
    all_keys = sci_lut_data["collapse_lo"]["1"]["variables"].keys()
    # Dimensions
    num_variables = len(non_reserved_keys)
    spin_sector_pairs = non_reserved_variables["tcr"]
    esa_step = constants.NUM_ESA_STEPS
    # Decompress data using byte count information from decommed data
    binary_data_list = group_ds["data"].values
    byte_count_list = group_ds["byte_count"].values

    compression_algorithm = CoDICECompression(view_tab_obj.compression)

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

    counters_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        -1, esa_step, num_variables, spin_sector_pairs
    )

    # If data size is less than 128, pad with fillval to make it 128
    half_spin_per_esa_step = sci_lut_data["lo_stepping_tab"]["row_number"].get("data")
    if len(half_spin_per_esa_step) < constants.NUM_ESA_STEPS:
        pad_size = constants.NUM_ESA_STEPS - len(half_spin_per_esa_step)
        half_spin_per_esa_step = np.concatenate(
            (np.array(half_spin_per_esa_step), np.full(pad_size, HALF_SPIN_FILLVAL))
        )
    # Each group shares the same table_id, so all epochs use the same LUT values.
    half_spin_per_esa_step = np.tile(
        np.asarray(half_spin_per_esa_step).astype(np.uint8),
        (len(group_ds["acq_start_seconds"]), 1),
    )
    # Get acquisition time per esa step
    acquisition_time_per_step = calculate_acq_time_per_step(
        sci_lut_data["lo_stepping_tab"]
    )
    acquisition_time_per_step = np.tile(
        np.asarray(acquisition_time_per_step),
        (len(group_ds["acq_start_seconds"]), 1),
    )
    # For every energy after nso_half_spin, set data to fill values
    nso_half_spin = group_ds["nso_half_spin"].values
    nso_mask = (half_spin_per_esa_step >= nso_half_spin[:, np.newaxis]) | (
        half_spin_per_esa_step == HALF_SPIN_FILLVAL
    )
    counters_mask = nso_mask[:, :, np.newaxis, np.newaxis]
    counters_mask = np.broadcast_to(counters_mask, counters_data.shape)
    counters_data = counters_data.astype(np.float64)
    counters_data[counters_mask] = np.nan
    # Set half_spin_per_esa_step to (fillval) where nso_mask is True
    half_spin_per_esa_step[nso_mask] = HALF_SPIN_FILLVAL
    # Set acquisition_time_per_step to nan where nso_mask is True
    acquisition_time_per_step[nso_mask] = np.nan

    # ========= Get Epoch Time Data ===========
    # Epoch center time and delta
    epoch_center, deltas = get_codice_epoch_time(
        group_ds["acq_start_seconds"].values,
        group_ds["acq_start_subseconds"].values,
        group_ds["spin_period"].values,
        view_tab_obj,
    )

    # ========== Initialize CDF Dataset with Coordinates ===========
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
            "esa_step": xr.DataArray(
                np.arange(esa_step, dtype=np.uint8),
                dims=("esa_step",),
                attrs=cdf_attrs.get_variable_attributes("esa_step", check_schema=False),
            ),
            "half_spin_per_esa_step": xr.DataArray(
                half_spin_per_esa_step,
                dims=(
                    "epoch",
                    "esa_step",
                ),
                attrs=cdf_attrs.get_variable_attributes(
                    "half_spin_per_esa_step", check_schema=False
                ),
            ),
            "esa_step_label": xr.DataArray(
                np.arange(esa_step, dtype=np.uint8).astype(str),
                dims=("esa_step",),
                attrs=cdf_attrs.get_variable_attributes(
                    "esa_step_label", check_schema=False
                ),
            ),
            "spin_sector_pairs": xr.DataArray(
                np.arange(spin_sector_pairs, dtype=np.uint8),
                dims=("spin_sector_pairs",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector_pairs", check_schema=False
                ),
            ),
            "spin_sector_pairs_label": xr.DataArray(
                np.arange(spin_sector_pairs, dtype=np.uint8).astype(str),
                dims=("spin_sector_pairs",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector_pairs_label", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes(logical_source_id),
    )

    # Add first few unique variables
    l1a_dataset["spin_period"] = xr.DataArray(
        group_ds["spin_period"].values * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("spin_period"),
    )
    l1a_dataset["k_factor"] = xr.DataArray(
        np.array([constants.K_FACTOR]),
        dims=("k_factor",),
        attrs=cdf_attrs.get_variable_attributes("k_factor_attrs", check_schema=False),
    )
    l1a_dataset["voltage_table"] = xr.DataArray(
        np.array(voltage_data),
        dims=("esa_step",),
        attrs=cdf_attrs.get_variable_attributes("voltage_table", check_schema=False),
    )
    l1a_dataset["data_quality"] = xr.DataArray(
        group_ds["suspect"].values,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("data_quality"),
    )
    l1a_dataset["acquisition_time_per_esa_step"] = xr.DataArray(
        acquisition_time_per_step,
        dims=("epoch", "esa_step"),
        attrs=cdf_attrs.get_variable_attributes(
            "acquisition_time_per_esa_step", check_schema=False
        ),
    )
    # Rename vars
    group_ds = group_ds.rename(
        {
            k: v
            for k, v in [
                ("rgfo_energy_step", "rgfo_esa_step"),
                ("nso_energy_step", "nso_esa_step"),
            ]
            if k in group_ds
        }
    )
    # These variables were added to the packet definition after 20260129, so they only
    # exist in the dataset if packet_version > 1.
    # If they don't exist, initialize them with fill val arrays since they won't be
    # used in the NSO/RGFO masking logic but should still exist in l1a for SPDF
    # compliance/consistency.
    l1a_additional_vars = [
        "rgfo_spin_sector",
        "rgfo_esa_step",
        "nso_spin_sector",
        "nso_esa_step",
    ]
    for var in l1a_additional_vars:
        if var not in group_ds:
            group_ds[var] = np.full(group_ds.sizes["epoch"], fill_value=np.nan)

    # Carry over these variables from unpacked data to l1a_dataset
    l1a_carryover_vars = [
        "sw_bias_gain_mode",
        "st_bias_gain_mode",
        "rgfo_half_spin",
        "nso_half_spin",
        *l1a_additional_vars,
    ]
    # Loop through them since we need to set their attrs too
    for var in l1a_carryover_vars:
        l1a_dataset[var] = xr.DataArray(
            group_ds[var].values,
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(var),
        )
    # Finally, add data variables
    for idx, variable in enumerate(non_reserved_variables):
        # We don't store reserved variables in CDF
        attrs = cdf_attrs.get_variable_attributes(f"lo-{variable}")

        l1a_dataset[variable] = xr.DataArray(
            counters_data[:, :, idx, :],
            dims=("epoch", "esa_step", "spin_sector_pairs"),
            attrs=attrs,
        )

    for variable in all_keys:
        if variable in non_reserved_keys or variable.startswith("reserved"):
            continue
        attrs = cdf_attrs.get_variable_attributes(f"lo-{variable}")
        l1a_dataset[variable] = xr.DataArray(
            np.full(
                l1a_dataset["tcr"].shape,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            ),
            dims=("epoch", "esa_step", "spin_sector_pairs"),
            attrs=attrs,
        )
    # No uncertainty needed for Lo counters data
    return l1a_dataset
