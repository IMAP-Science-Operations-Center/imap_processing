"""CoDICE Hi Sectored L1A processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CoDICECompression,
    apply_replacements_to_attrs,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_energy_info,
    get_view_tab_obj,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_hi_sectored(
    group_ds: xr.Dataset,
    lut_file: Path,
    table_id: str,
    view_id: int,
    apid: int,
    plan_id: int,
    plan_step: int,
) -> xr.Dataset:
    """
    Process a single table-ID group of CoDICE Hi Sectored L1A data.

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
        f"Processing species with - APID: {apid} / 0x{apid:X}, View ID: {view_id}, "
        f"Table ID: {table_id}, Plan ID: {plan_id}, Plan Step: {plan_step}"
    )

    # ========== Get LUT Data ===========
    sci_lut_data, view_tab_obj = get_view_tab_obj(lut_file, table_id, view_id, apid)

    if view_tab_obj.sensor != 1:
        raise ValueError("Unsupported sensor ID for Hi Sectored processing.")

    # ========= Get Epoch Time Data ===========
    # Epoch center time and delta
    epoch_center, deltas = get_codice_epoch_time(
        group_ds["acq_start_seconds"].values,
        group_ds["acq_start_subseconds"].values,
        group_ds["spin_period"].values,
        view_tab_obj,
    )

    # ========= Decompress and Calculate Reshape information ===========
    species_data = sci_lut_data["data_product_hi_tab"]["0"]["sectored"]
    species_names = species_data.keys()
    logical_source_id = "imap_codice_l1a_hi-sectored"

    compression_algorithm = CoDICECompression(view_tab_obj.compression)
    # Decompress data using byte count information from decommed data
    binary_data_list = group_ds["data"].values
    byte_count_list = group_ds["byte_count"].values

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

    # Use chunks of (energy_x) to put data in its energy bins as done below.
    #   Eg. [15, 15, 15, 18, 18, 15, 18, 5, 1]
    # where each number is energy dimension for species 'x'.
    species_chunk_sizes = [
        len(species_data[species]["min_energy"]) for species in species_names
    ]

    # Reshape decompressed data to in below for loop:
    # (num_packets, num_species, energy_bins, spin_sector, inst_az)
    num_species = len(species_names)
    energy_bins = 8
    collapse_shape = get_collapse_pattern_shape(
        sci_lut_data,
        view_tab_obj.sensor,
        view_tab_obj.collapse_table,
    )
    if np.unique(species_chunk_sizes) != [energy_bins]:
        raise ValueError("Expected energy bins to be 8 for Hi Sectored data.")

    # Calculate collapsed size
    decompressed_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets, num_species, energy_bins, *collapse_shape
    )

    # ========== Create Dataset ===========
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
            "spin_sector": xr.DataArray(
                np.arange(collapse_shape[0]),
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
            "inst_az": xr.DataArray(
                np.arange(collapse_shape[1]),
                dims=("inst_az",),
                attrs=cdf_attrs.get_variable_attributes("inst_az", check_schema=False),
            ),
            "inst_az_label": xr.DataArray(
                np.arange(collapse_shape[1]).astype(str),
                dims=("inst_az",),
                attrs=cdf_attrs.get_variable_attributes(
                    "inst_az_label", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes(logical_source_id),
    )

    # Final data shape of each species is (epoch, energy_h, spin_sector, inst_az)
    for species_index, (species_name, data) in enumerate(species_data.items()):
        # Add coordinate for 'energy_{species_name}'
        energy_centers, energy_minus, energy_plus = get_energy_info(data)
        coord_attrs = cdf_attrs.get_variable_attributes(
            "hi-energy-attrs", check_schema=False
        )
        coord_attrs = apply_replacements_to_attrs(
            coord_attrs, {"species": species_name}
        )
        l1a_dataset = l1a_dataset.assign_coords(
            {
                f"energy_{species_name}": xr.DataArray(
                    np.array(energy_centers),
                    dims=(f"energy_{species_name}",),
                    attrs=coord_attrs,
                )
            }
        )
        species_label_attrs = cdf_attrs.get_variable_attributes(
            "energy_species_label", check_schema=False
        )
        species_label_attrs = apply_replacements_to_attrs(
            species_label_attrs, {"species": species_name}
        )
        l1a_dataset[f"energy_{species_name}_label"] = xr.DataArray(
            np.array(energy_centers).astype("str"),
            dims=(f"energy_{species_name}"),
            attrs=species_label_attrs,
        )
        # Add energy plus and minus variables
        minus_attrs = cdf_attrs.get_variable_attributes("hi-energy-delta-attrs")
        minus_attrs = apply_replacements_to_attrs(
            minus_attrs, {"species": species_name, "operation": "minus"}
        )
        l1a_dataset[f"energy_{species_name}_minus"] = xr.DataArray(
            energy_minus,
            dims=(f"energy_{species_name}",),
            attrs=minus_attrs,
        )
        plus_attrs = cdf_attrs.get_variable_attributes("hi-energy-delta-attrs")
        plus_attrs = apply_replacements_to_attrs(
            plus_attrs, {"species": species_name, "operation": "plus"}
        )
        l1a_dataset[f"energy_{species_name}_plus"] = xr.DataArray(
            energy_plus,
            dims=(f"energy_{species_name}",),
            attrs=plus_attrs,
        )

        # Extract species data from decompressed data:
        # - (num_packets, energy_bins, spin_sector, inst_az)
        # Unpacked data is in this order:
        #   (epoch, number_species, energy_{species}, inst_az, spin_sector)
        species_data = decompressed_data[:, species_index, :, :, :]
        # Now transpose to put data in the correct order for writing to CDF:
        #   (epoch, energy_{species}, spin_sector, inst_az)
        species_data = np.transpose(species_data, [0, 1, 3, 2])

        species_attrs = cdf_attrs.get_variable_attributes("hi-species-attrs")
        species_attrs = apply_replacements_to_attrs(
            species_attrs, {"species": species_name}
        )
        species_data = species_data.astype(np.float64)
        # Add DEPEND_2, DEPEND_3
        species_attrs["DEPEND_2"] = "spin_sector"
        species_attrs["LABL_PTR_2"] = "spin_sector_label"
        species_attrs["DEPEND_3"] = "inst_az"
        species_attrs["LABL_PTR_3"] = "inst_az_label"
        l1a_dataset[species_name] = xr.DataArray(
            species_data,
            dims=("epoch", f"energy_{species_name}", "spin_sector", "inst_az"),
            attrs=species_attrs,
        )
        # Uncertainty is sqrt of counts
        species_unc_attrs = cdf_attrs.get_variable_attributes("hi-species-unc-attrs")
        species_unc_attrs = apply_replacements_to_attrs(
            species_unc_attrs, {"species": species_name}
        )
        # Add DEPEND_2, DEPEND_3
        species_unc_attrs["DEPEND_2"] = "spin_sector"
        species_unc_attrs["LABL_PTR_2"] = "spin_sector_label"
        species_unc_attrs["DEPEND_3"] = "inst_az"
        species_unc_attrs["LABL_PTR_3"] = "inst_az_label"
        l1a_dataset[f"unc_{species_name}"] = xr.DataArray(
            np.sqrt(species_data),
            dims=("epoch", f"energy_{species_name}", "spin_sector", "inst_az"),
            attrs=species_unc_attrs,
        )

    # ========= Add Additional Variables ===========
    l1a_dataset["spin_period"] = xr.DataArray(
        group_ds["spin_period"].values * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("spin_period"),
    )
    l1a_dataset["data_quality"] = xr.DataArray(
        group_ds["suspect"].values,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("data_quality"),
    )

    return l1a_dataset
