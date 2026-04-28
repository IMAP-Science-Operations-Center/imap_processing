"""CoDICE L1A Lo priority processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.constants import HALF_SPIN_FILLVAL
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    CoDICECompression,
    ViewTabInfo,
    calculate_acq_time_per_step,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_view_tab_info,
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_lo_priority(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
    """
    Process CoDICE Lo Priority L1A data.

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
        compression=view_tab_info["compression"],
    )

    if view_tab_obj.sensor != 0:
        raise ValueError("Unsupported sensor ID for Lo priority processing.")

    # ========== Get Voltage Data from LUT ===========
    # Use plan id and plan step to get voltage data's table_number in ESA sweep table.
    # Voltage data is (128,)
    esa_table_number = sci_lut_data["plan_tab"][f"({plan_id}, {plan_step})"][
        "lo_stepping"
    ]
    voltage_data = sci_lut_data["esa_sweep_tab"][f"{esa_table_number}"]

    # ========= Get Epoch Time Data ===========
    # Epoch center time and delta
    epoch_center, deltas = get_codice_epoch_time(
        unpacked_dataset["acq_start_seconds"].values,
        unpacked_dataset["acq_start_subseconds"].values,
        unpacked_dataset["spin_period"].values,
        view_tab_obj,
    )

    # ========= Decompress and Calculate Reshape information ===========
    # Set needed metadata for Hi and Lo's different priority products
    if apid == CODICEAPID.COD_LO_SW_PRIORITY_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["priority"]["sw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-sw-priority"
        compression_algorithm = CoDICECompression(view_tab_obj.compression)
    elif apid == CODICEAPID.COD_LO_NSW_PRIORITY_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["priority"]["nsw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-nsw-priority"
        compression_algorithm = CoDICECompression(view_tab_obj.compression)
    else:
        raise ValueError("Unsupported APID for Lo priority processing.")

    # Decompress data using byte count information from decommed data
    binary_data_list = unpacked_dataset["data"].values
    byte_count_list = unpacked_dataset["byte_count"].values

    packet_version = unpacked_dataset["packet_version"].values[0]
    # The decompressed data in the shape of (epoch, n). Then reshape later.
    decompressed_data = [
        np.frombuffer(
            bytes(
                decompress(
                    packet_data[:byte_count],
                    compression_algorithm,
                )
            ),
            dtype=">u4",  # Big endian
        )
        # For newer packet versions, the decompressed data needs to be converted to
        # uint32
        if packet_version > 1
        else decompress(
            packet_data[:byte_count],
            compression_algorithm,
        )
        for (packet_data, byte_count) in zip(
            binary_data_list, byte_count_list, strict=False
        )
    ]

    num_packets = len(binary_data_list)

    # Reshape decompressed data to in below for loop:
    # (num_packets, num_species, esa_steps, collapse_shape[0](spin_sector))
    num_species = len(species_names)
    num_esa_steps = constants.NUM_ESA_STEPS
    collapse_shape = get_collapse_pattern_shape(
        sci_lut_data,
        view_tab_obj.sensor,
        view_tab_obj.collapse_table,
    )
    num_spin_sectors = collapse_shape[0]
    species_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets, num_species, num_esa_steps, num_spin_sectors
    )

    # If data size is less than 128, pad with fillval to make it 128
    half_spin_per_esa_step = sci_lut_data["lo_stepping_tab"]["row_number"].get("data")
    if len(half_spin_per_esa_step) < num_esa_steps:
        pad_size = num_esa_steps - len(half_spin_per_esa_step)
        half_spin_per_esa_step = np.concatenate(
            (np.array(half_spin_per_esa_step), np.full(pad_size, HALF_SPIN_FILLVAL))
        )

    # TODO: Handle epoch dependent acquisition time and half spin per esa step
    #   For now, just tile the same array for all epochs.
    #   Eventually we may have data from a day where the LUT changed. If this is the
    #  case, we need to split the data by epoch and assign different acquisition times
    half_spin_per_esa_step = np.tile(
        np.asarray(half_spin_per_esa_step).astype(np.uint8),
        (len(unpacked_dataset["acq_start_seconds"]), 1),
    )
    # Get acquisition time per esa step
    acquisition_time_per_step = calculate_acq_time_per_step(
        sci_lut_data["lo_stepping_tab"]
    )
    acquisition_time_per_step = np.tile(
        np.asarray(acquisition_time_per_step),
        (len(unpacked_dataset["acq_start_seconds"]), 1),
    )
    # ========== Apply NSO/RGFO Masking ===========
    # After FSW changes on 20260129, The Lo L1A product contains variables that
    # indicate the esa step and spin sector during which the RGFO or NSO limits are
    # triggered. The spin sector variable ranges from 0-11 and is the instrument
    # reported spin sector. The following algorithm defines when to assign NaN to the
    # priority data product due to NSO
    # operation:
    # 1. For half_spin > nso_half_spin a set all data to NaN
    # 2. For half_spin = nso_half_spin
    #   a. For spin_sector > nso_spin_sector a set all data to NaN
    #   b. For spin_sector = nso_spin_sector
    #       i. For esa_step > nso_esa_step a set all data to NaN
    # For every energy after nso_half_spin, set data to fill values
    # For data before 20260129 ( packet_version <=1 ) set all data to NaN where
    # half_spin > nso_half_spin
    packet_versions = unpacked_dataset["packet_version"].values
    nso_half_spin = unpacked_dataset["nso_half_spin"].values
    # TODO handle boundary days where the FSW changed halfway through the dataset. E.g
    # Some packet_version = 1 and some = 2
    if packet_versions[0] <= 1:
        # For half_spin >= NSO_half_spin, set to NaN
        half_spin_mask = (half_spin_per_esa_step >= nso_half_spin[:, np.newaxis]) | (
            half_spin_per_esa_step == HALF_SPIN_FILLVAL
        )
        species_mask = half_spin_mask[:, np.newaxis, :, np.newaxis]
        species_mask = np.broadcast_to(species_mask, species_data.shape)
    else:
        # nso_spin_sector and nso_esa_step for comparison. Shape (epoch, 1, 1)
        # to broadcast
        # Packet nso_spin_sector spans the full spin (0-23), but this product's
        # spin_sector dimension is half-spin indexed (0-11), so modulo 12 is
        # intentional to align packet NSO metadata with the data coordinates.
        nso_spin_sector = (
            unpacked_dataset["nso_spin_sector"].values[:, np.newaxis, np.newaxis] % 12
        )
        nso_esa_step = unpacked_dataset["nso_energy_step"].values[
            :, np.newaxis, np.newaxis
        ]
        # Create arrays for spin sectors and esa steps to compare with nso values.
        # Shape (1, 1, spin_sector) and (1, esa_step, 1)
        spin_sectors = np.arange(num_spin_sectors)[np.newaxis, np.newaxis, :]
        esa_steps = np.arange(num_esa_steps)[np.newaxis, :, np.newaxis]
        # Create a mask for half_spin > nso_half_spin. Shape (epoch, esa_step))
        # This will be used below to set half_spin_per_esa_step to fillval and
        # acquisition_time_per_step to NaN for those steps.
        half_spin_mask = (half_spin_per_esa_step > nso_half_spin[:, np.newaxis]) | (
            half_spin_per_esa_step == HALF_SPIN_FILLVAL
        )
        # Create a mask for the boundary condition where half_spin == nso_half_spin.
        at_boundary = (
            half_spin_per_esa_step[:, :, np.newaxis]
            == nso_half_spin[:, np.newaxis, np.newaxis]
        )
        boundary_half_spin_mask = (
            at_boundary
            &
            # For spin_sector > nso_spin_sector, set to NaN
            (
                (spin_sectors > nso_spin_sector)
                |
                # For spin_sector = nso_spin_sector and esa_step > nso_esa_step,
                # set to NaN
                ((spin_sectors == nso_spin_sector) & (esa_steps > nso_esa_step))
            )
        )
        # Combine masks. Shape (epoch, esa_step, spin_sector). This mask is True
        # where data should be set to NaN
        nso_mask = half_spin_mask[:, :, np.newaxis] | boundary_half_spin_mask
        # Expand nso_mask to (epoch, 1, esa_step, spin_sector) to apply to species_data.
        species_mask = np.broadcast_to(
            nso_mask[:, np.newaxis, :, :], species_data.shape
        )

    species_data = species_data.astype(np.float64)
    species_data[species_mask] = np.nan
    # Set half_spin_per_esa_step to (fillval) where nso_mask is True
    half_spin_per_esa_step[half_spin_mask] = HALF_SPIN_FILLVAL
    # Set acquisition_time_per_step to nan where nso_mask is True
    acquisition_time_per_step[half_spin_mask] = np.nan

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
            "esa_step": xr.DataArray(
                np.arange(128),
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
                np.arange(128).astype(str),
                dims=("esa_step",),
                attrs=cdf_attrs.get_variable_attributes(
                    "esa_step_label", check_schema=False
                ),
            ),
            "k_factor": xr.DataArray(
                np.array([constants.K_FACTOR]),
                dims=("k_factor",),
                attrs=cdf_attrs.get_variable_attributes(
                    "k_factor_attrs", check_schema=False
                ),
            ),
            "spin_sector": xr.DataArray(
                np.arange(collapse_shape[0], dtype=np.uint8),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector", check_schema=False
                ),
            ),
            "spin_sector_label": xr.DataArray(
                np.arange(collapse_shape[0]).astype(str),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector_label", check_schema=False
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
        unpacked_dataset["suspect"].values,
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
    unpacked_dataset = unpacked_dataset.rename(
        {
            k: v
            for k, v in [
                ("rgfo_energy_step", "rgfo_esa_step"),
                ("nso_energy_step", "nso_esa_step"),
            ]
            if k in unpacked_dataset
        }
    )
    # These variables were added to the packet definition after 20260129, so they only
    # exist in the unpacked dataset if packet_version > 1
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
        if var not in unpacked_dataset:
            unpacked_dataset[var] = np.full(
                unpacked_dataset.sizes["epoch"], fill_value=np.nan
            )

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
            unpacked_dataset[var].values,
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(var),
        )
    # Finally, add species data variables and their uncertainties
    for idx, species in enumerate(species_names):
        l1a_dataset[species] = xr.DataArray(
            species_data[:, idx, :, :],
            dims=("epoch", "esa_step", "spin_sector"),
            attrs=cdf_attrs.get_variable_attributes(species),
        )
        l1a_dataset[f"unc_{species}"] = xr.DataArray(
            np.sqrt(l1a_dataset[species].values),
            dims=("epoch", "esa_step", "spin_sector"),
            attrs=cdf_attrs.get_variable_attributes(species),
        )

    return l1a_dataset
