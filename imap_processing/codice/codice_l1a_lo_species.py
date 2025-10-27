"""CoDICE Lo Species L1A processing functions."""

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
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_lo_species(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
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
        f"Processing species with - APID: {apid}, View ID: {view_id}, "
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
        raise ValueError("Unsupported sensor ID for Lo species processing.")

    # ========= Decompress and Reshape Data ===========
    # Lookup SW or NSW species based on APID
    if view_tab_obj.apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["species"]["sw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-sw-species"
    elif view_tab_obj.apid == CODICEAPID.COD_LO_NSW_SPECIES_COUNTS:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["species"]["nsw"][
            "species_names"
        ]
        logical_source_id = "imap_codice_l1a_lo-nsw-species"
    elif view_tab_obj.apid == CODICEAPID.COD_LO_IAL:
        species_names = sci_lut_data["data_product_lo_tab"]["0"]["ialirt"]["sw"][
            "species_names"
        ]
        # Note: ialirt does not produce a cdf for l1a so this is arbitrary.
        logical_source_id = "imap_codice_l1a_lo-sw-species"
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
    # For Lo species, it will be (1,)
    collapsed_shape = get_collapse_pattern_shape(
        sci_lut_data, view_tab_obj.sensor, view_tab_obj.collapse_table
    )

    # Reshape decompressed data to:
    #   (num_packets, num_species, esa_steps, *collapsed_shape)
    # where collapsed_shape is usually (1,) for Lo species.
    num_packets = len(binary_data_list)
    num_species = len(species_names)
    esa_steps = constants.NUM_ESA_STEPS
    species_data = np.array(decompressed_data).reshape(
        num_packets, num_species, esa_steps, *collapsed_shape
    )

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
    for idx, species in enumerate(species_names):
        if view_tab_obj.apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS and species in [
            "heplus",
            "cnoplus",
        ]:
            species_attrs = cdf_attrs.get_variable_attributes("lo-pui-species-attrs")
            unc_attrs = cdf_attrs.get_variable_attributes("lo-pui-species-unc-attrs")
        else:
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
            species_data[:, idx, :, :],
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
            np.sqrt(species_data[:, idx, :, :]),
            dims=("epoch", "esa_step", "spin_sector"),
            attrs=unc_attrs,
        )

    return l1a_dataset
