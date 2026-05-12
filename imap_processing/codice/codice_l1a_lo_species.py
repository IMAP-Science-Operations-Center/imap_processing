"""CoDICE Lo Species L1A processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.constants import (
    HALF_SPIN_FILLVAL,
    LO_IALIRT_VARIABLE_NAMES,
    LO_NSW_SPECIES_VARIABLE_NAMES,
    LO_SW_SPECIES_VARIABLE_NAMES,
)
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    CoDICECompression,
    calculate_acq_time_per_step,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_view_tab_obj,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_lo_species(  # noqa: PLR0912
    group_ds: xr.Dataset,
    lut_file: Path,
    table_id: str,
    view_id: int,
    apid: int,
    plan_id: int,
    plan_step: int,
) -> xr.Dataset:
    """
    Process a single table-ID group of CoDICE Lo Species L1A data.

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
        The processed L1A dataset for input table-ID group.
    """
    logger.info(
        f"Processing species with - APID: {apid} / 0x{apid:X}, View ID: {view_id}, "
        f"Table ID: {table_id}, Plan ID: {plan_id}, Plan Step: {plan_step}"
    )

    # ========== Get LUT Data ===========
    sci_lut_data, view_tab_obj = get_view_tab_obj(lut_file, table_id, view_id, apid)
    if view_tab_obj.sensor != 0:
        raise ValueError("Unsupported sensor ID for Lo species processing.")

    # ========= Decompress and Reshape Data ===========
    # Lookup SW or NSW species based on APID
    if view_tab_obj.apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
        actual_species_names = sci_lut_data["data_product_lo_tab"]["0"]["species"][
            "sw"
        ]["species_names"]
        desired_species_names = set(
            sci_lut_data["data_product_lo_tab"]["0"]["species"]["sw"][
                "desired_species_names"
            ]
            + LO_SW_SPECIES_VARIABLE_NAMES
        )
        logical_source_id = "imap_codice_l1a_lo-sw-species"
    elif view_tab_obj.apid == CODICEAPID.COD_LO_NSW_SPECIES_COUNTS:
        actual_species_names = sci_lut_data["data_product_lo_tab"]["0"]["species"][
            "nsw"
        ]["species_names"]
        desired_species_names = set(
            sci_lut_data["data_product_lo_tab"]["0"]["species"]["nsw"][
                "desired_species_names"
            ]
            + LO_NSW_SPECIES_VARIABLE_NAMES
        )
        logical_source_id = "imap_codice_l1a_lo-nsw-species"
        # Rename "cnoplus" to "junk" if we are processing NSW angular data. Although
        # cnoplus is in desired, and actual species name in the LUT, it is referencing
        # different "cnoplus" data his is to handle the bug in which the spacecraft was
        # sending data down "off by one" and getting mislabeled. The cnoplus data we
        # are referencing is actually data that we want to toss out and fill with
        # fill vals. This only affects data before the LUT was updated
        # (table_id 3978152295).
        if int(table_id) <= 3978152295:
            actual_species_names = [
                "junk" if name == "cnoplus" else name for name in actual_species_names
            ]
    elif view_tab_obj.apid == CODICEAPID.COD_LO_IAL:
        actual_species_names = sci_lut_data["data_product_lo_tab"]["0"]["ialirt"]["sw"][
            "species_names"
        ]
        desired_species_names = set(
            sci_lut_data["data_product_lo_tab"]["0"]["ialirt"]["sw"][
                "desired_species_names"
            ]
            + LO_IALIRT_VARIABLE_NAMES
        )
        # Note: ialirt does not produce a cdf for l1a so this is arbitrary.
        logical_source_id = "imap_codice_l1a_lo-sw-species"
    else:
        raise ValueError(f"Unknown apid {view_tab_obj.apid} in Lo species processing.")

    compression_algorithm = CoDICECompression(view_tab_obj.compression)
    # Decompress data using byte count information from decommed data
    binary_data_list = group_ds["data"].values
    byte_count_list = group_ds["byte_count"].values
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
    # For Lo species, it will be (1,)
    collapsed_shape = get_collapse_pattern_shape(
        sci_lut_data, view_tab_obj.sensor, view_tab_obj.collapse_table
    )

    # Reshape decompressed data to:
    #   (num_packets, num_species, esa_steps, *collapsed_shape)
    # where collapsed_shape is usually (1,) for Lo species.
    num_packets = len(binary_data_list)
    num_species = len(actual_species_names)
    esa_steps = constants.NUM_ESA_STEPS
    species_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets, num_species, esa_steps, *collapsed_shape
    )

    # If data size is less than 128, pad with fillval to make it 128
    half_spin_per_esa_step = sci_lut_data["lo_stepping_tab"]["row_number"].get("data")
    if len(half_spin_per_esa_step) < constants.NUM_ESA_STEPS:
        pad_size = constants.NUM_ESA_STEPS - len(half_spin_per_esa_step)
        half_spin_per_esa_step = np.concatenate(
            (np.array(half_spin_per_esa_step), np.full(pad_size, HALF_SPIN_FILLVAL))
        )

    acquisition_time_per_step = calculate_acq_time_per_step(
        sci_lut_data["lo_stepping_tab"]
    )
    # Get acquisition time per esa step. Each group shares the same table_id, so
    # all epochs within the group use the same LUT values.
    half_spin_per_esa_step = np.tile(
        np.asarray(
            half_spin_per_esa_step,
        ).astype(np.uint8),
        (len(group_ds["acq_start_seconds"]), 1),
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
    species_mask = nso_mask[:, np.newaxis, :, np.newaxis]
    species_mask = np.repeat(species_mask, num_species, 1)
    species_data = species_data.astype(np.float64)
    species_data[species_mask] = np.nan
    # Set half_spin_per_esa_step to (fillval) where nso_mask is True
    half_spin_per_esa_step[nso_mask] = HALF_SPIN_FILLVAL
    # Set acquisition_time_per_step to nan where nso_mask is True
    acquisition_time_per_step[nso_mask] = np.nan

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
        group_ds["acq_start_seconds"].values,
        group_ds["acq_start_subseconds"].values,
        group_ds["spin_period"].values,
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
                np.array([0], dtype=np.uint8),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector", check_schema=False
                ),
            ),
            "spin_sector_label": xr.DataArray(
                np.array(["0"]).astype(str),
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
        if var not in group_ds:
            group_ds[var] = np.full(group_ds.sizes["epoch"], fill_value=np.nan)

    # Carry over these variables from unpacked data to l1a_dataset
    l1a_carryover_vars = [
        "sw_bias_gain_mode",
        "st_bias_gain_mode",
        "rgfo_half_spin",
        "nso_half_spin",
        "packet_version",
        *l1a_additional_vars,
    ]
    # Loop through them since we need to set their attrs too
    for var in l1a_carryover_vars:
        l1a_dataset[var] = xr.DataArray(
            group_ds[var].values,
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes(var),
        )
    # Finally, add species data variables and their uncertainties
    # Loop through the species we want in the final dataset (desired_species_names) and
    # add them if they exist in the actual species names from the LUT.
    # This is to handle the bug in which the spacecraft was sending data down "off by
    # one" and getting mislabeled.
    for species in desired_species_names:
        if species not in actual_species_names:
            logger.warning(
                f"Desired species {species} not found in actual species names from "
                f"LUT. This species will be filled with fill values in the final "
                f"dataset. Actual species names: {actual_species_names}"
            )
            species_data_individual = np.full(species_data[:, 0, :, :].shape, np.nan)
        else:
            species_idx = actual_species_names.index(species)
            species_data_individual = species_data[:, species_idx, :, :]

        species_attrs = cdf_attrs.get_variable_attributes("lo-species-attrs")
        unc_attrs = cdf_attrs.get_variable_attributes("lo-species-unc-attrs")

        direction = (
            "Sunward"
            if view_tab_obj.apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS
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
            species_data_individual,
            dims=("epoch", "esa_step", "spin_sector"),
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
            np.sqrt(l1a_dataset[species].values),
            dims=("epoch", "esa_step", "spin_sector"),
            attrs=unc_attrs,
        )
    return l1a_dataset
