"""CoDICE Lo Angular L1A processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    ViewTabInfo,
    calculate_acq_time_per_step,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_view_tab_info,
    index_to_position,
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def _despin_species_data(
    species_data: np.ndarray, sci_lut_data: dict, view_tab_obj: ViewTabInfo
) -> np.ndarray:
    """
    Apply despinning mapping for angular products.

    Despinned data shape is (num_packets, num_species, 24, inst_az) where
    we expand spin_sector to 24 by filling with zeros in 12 to 24 or 0 to 11
    based on pixel orientation.

    Parameters
    ----------
    species_data : np.ndarray
        The species data array to be despun.
    sci_lut_data : dict
        The science LUT data used for despinning.
    view_tab_obj : ViewTabInfo
        The view table information object.

    Returns
    -------
    np.ndarray
        The despun species data array in
        (num_packets, num_species, esa_steps, 24, inst_az).
    """
    # species_data shape: (num_packets, num_species, esa_steps, *collapsed_dims)
    num_packets, num_species, esa_steps = species_data.shape[:3]
    collapsed_dims = species_data.shape[3:]
    inst_az_dim = collapsed_dims[-1]

    # Prepare despinning output: (num_packets, num_species, esa_steps, 24, inst_az_dim)
    # 24 is derived by multiplying spin sector dim from collapse table by 2
    spin_sector_len = constants.LO_DESPIN_SPIN_SECTORS
    despun_shape = (num_packets, num_species, esa_steps, spin_sector_len, inst_az_dim)
    despun_data = np.full(despun_shape, 0)

    # Pixel orientation array and mapping positions
    pixel_orientation = np.array(
        sci_lut_data["lo_stepping_tab"]["pixel_orientation"]["data"]
    )
    # index_to_position gets the position from collapse table. Eg.
    #   [1, 2, 3, 23, 24] for SW angular
    angular_position = index_to_position(sci_lut_data, 0, view_tab_obj.collapse_table)
    orientation_a = pixel_orientation == "A"
    orientation_b = pixel_orientation == "B"

    # Despin data based on orientation and angular position
    for pos_idx, position in enumerate(angular_position):
        if position <= 12:
            # Case 1: position 0-12, orientation A, append to first half
            despun_data[:, :, orientation_a, :12, pos_idx] = species_data[
                :, :, orientation_a, :, pos_idx
            ]
            # Case 2: position 13-24, orientation B, append to second half
            despun_data[:, :, orientation_b, 12:, pos_idx] = species_data[
                :, :, orientation_b, :, pos_idx
            ]
        else:
            # Case 3: position 13-24, orientation A, append to second half
            despun_data[:, :, orientation_a, 12:, pos_idx] = species_data[
                :, :, orientation_a, :, pos_idx
            ]
            # Case 4: position 0-12, orientation B, append to first half
            despun_data[:, :, orientation_b, :12, pos_idx] = species_data[
                :, :, orientation_b, :, pos_idx
            ]

    return despun_data


def l1a_lo_angular(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
    """
    L1A processing code.

    Parameters
    ----------
    unpacked_dataset : xarray.Dataset
        The decompressed and unpacked data from the packet file.
    lut_file : pathlib.Path
        Path to the LUT (Lookup Table) file used for processing.

    Returns
    -------
    xarray.Dataset
        The processed L1A dataset for the given species product.
    """
    # Get these values from unpacked data. These are used to
    # lookup in LUT table.
    table_id = unpacked_dataset["table_id"].values[0]
    view_id = unpacked_dataset["view_id"].values[0]
    apid = unpacked_dataset["pkt_apid"].values[0]
    plan_id = unpacked_dataset["plan_id"].values[0]
    plan_step = unpacked_dataset["plan_step"].values[0]

    logger.info(
        f"Processing angular with - APID: 0x{apid:X}, View ID: {view_id}, "
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

    if view_tab_obj.sensor != 0:
        raise ValueError("Unsupported sensor ID for Lo angular processing.")

    # ========= Decompress and Reshape Data ===========
    # Lookup SW or NSW species based on APID
    if view_tab_obj.apid == CODICEAPID.COD_LO_SW_ANGULAR_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["angular"]["sw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-sw-angular"
    elif view_tab_obj.apid == CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["angular"]["nsw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-nsw-angular"
    else:
        raise ValueError(f"Unknown apid {view_tab_obj.apid} in Lo species processing.")

    compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[view_tab_obj.view_id]
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

    # Look up collapse pattern using LUT table. This should return collapsed shape.
    collapsed_shape = get_collapse_pattern_shape(
        sci_lut_data, view_tab_obj.sensor, view_tab_obj.collapse_table
    )

    # Reshape decompressed data to:
    #   (num_packets, num_species, esa_steps, 12, 5)
    # 24 includes despinning spin sector. Then at later steps,
    # we handle despinning.
    num_packets = len(binary_data_list)
    esa_steps = constants.NUM_ESA_STEPS
    num_species = len(species_names)
    species_data = np.array(decompressed_data).reshape(
        num_packets, num_species, esa_steps, *collapsed_shape
    )

    # Despinning
    # ----------------
    species_data = _despin_species_data(species_data, sci_lut_data, view_tab_obj)

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
            "esa_step_label": xr.DataArray(
                np.arange(128).astype(str),
                dims=("esa_step",),
                attrs=cdf_attrs.get_variable_attributes(
                    "esa_step_label", check_schema=False
                ),
            ),
            "inst_az": xr.DataArray(
                index_to_position(sci_lut_data, 0, view_tab_obj.collapse_table),
                dims=("inst_az",),
                attrs=cdf_attrs.get_variable_attributes("inst_az", check_schema=False),
            ),
            "inst_az_label": xr.DataArray(
                index_to_position(sci_lut_data, 0, view_tab_obj.collapse_table).astype(
                    str
                ),
                dims=("inst_az",),
                attrs=cdf_attrs.get_variable_attributes(
                    "inst_az_label", check_schema=False
                ),
            ),
            "k_factor": xr.DataArray(
                np.array([constants.K_FACTOR]),
                dims=("k_factor",),
                attrs=cdf_attrs.get_variable_attributes("k_factor", check_schema=False),
            ),
            "spin_sector": xr.DataArray(
                np.arange(24, dtype=np.uint8),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector", check_schema=False
                ),
            ),
            "spin_sector_label": xr.DataArray(
                np.arange(24).astype(str),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector_label", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes(logical_source_id),
    )
    # Add first few unique variables
    l1a_dataset["k_factor"] = xr.DataArray(
        np.array([constants.K_FACTOR]),
        dims=("k_factor",),
        attrs=cdf_attrs.get_variable_attributes("k_factor_attrs", check_schema=False),
    )
    l1a_dataset["spin_period"] = xr.DataArray(
        unpacked_dataset["spin_period"].values * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("spin_period"),
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
    l1a_dataset["acquisition_time_per_step"] = xr.DataArray(
        calculate_acq_time_per_step(sci_lut_data["lo_stepping_tab"]),
        dims=("esa_step",),
        attrs=cdf_attrs.get_variable_attributes(
            "acquisition_time_per_step", check_schema=False
        ),
    )

    # Carry over these variables from unpacked data to l1a_dataset
    l1a_carryover_vars = [
        "sw_bias_gain_mode",
        "st_bias_gain_mode",
        "rgfo_half_spin",
        "nso_half_spin",
    ]
    # Loop through them since we need to set their attrs too
    for var in l1a_carryover_vars:
        l1a_dataset[var] = xr.DataArray(
            unpacked_dataset[var].values,
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(var),
        )

    # Finally, add species data variables and their uncertainties
    for species_data_idx, species in enumerate(species_names):
        species_attrs = cdf_attrs.get_variable_attributes("lo-angular-attrs")
        unc_attrs = cdf_attrs.get_variable_attributes("lo-angular-unc-attrs")

        direction = (
            "Sunward"
            if view_tab_obj.apid == CODICEAPID.COD_LO_SW_ANGULAR_COUNTS
            else "Non-Sunward"
        )
        # Replace {species} and {direction} in attrs
        species_attrs["CATDESC"] = species_attrs["CATDESC"].format(
            species=species, direction=direction
        )
        species_attrs["FIELDNAM"] = species_attrs["FIELDNAM"].format(
            species=species, direction=direction
        )
        l1a_dataset[species] = xr.DataArray(
            species_data[:, species_data_idx, :, :, :],
            dims=("epoch", "esa_step", "spin_sector", "inst_az"),
            attrs=species_attrs,
        )
        # Uncertainty data
        unc_attrs["CATDESC"] = unc_attrs["CATDESC"].format(
            species=species, direction=direction
        )
        unc_attrs["FIELDNAM"] = unc_attrs["FIELDNAM"].format(
            species=species, direction=direction
        )
        l1a_dataset[f"unc_{species}"] = xr.DataArray(
            np.sqrt(species_data[:, species_data_idx, :, :, :]),
            dims=("epoch", "esa_step", "spin_sector", "inst_az"),
            attrs=unc_attrs,
        )

    return l1a_dataset
