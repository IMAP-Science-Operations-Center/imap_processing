"""CoDICE Hi Omni L1A processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    ViewTabInfo,
    apply_replacements_to_attrs,
    get_codice_epoch_time,
    get_energy_info,
    get_view_tab_info,
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_hi_omni(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
    """
    Process CoDICE Hi Omni L1A data.

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
        raise ValueError("Unsupported sensor ID for Hi processing.")

    # ========= Decompress and Reshape Data ===========
    species_data = sci_lut_data["data_product_hi_tab"]["0"]["omni"]
    species_names = species_data.keys()
    logical_source_id = "imap_codice_l1a_hi-omni"

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

    # ========= Get Epoch Time Data ===========
    # Epoch center time and delta
    epoch_center, deltas = get_codice_epoch_time(
        unpacked_dataset["acq_start_seconds"].values,
        unpacked_dataset["acq_start_subseconds"].values,
        unpacked_dataset["spin_period"].values,
        view_tab_obj,
    )

    three_d_collapsed = view_tab_obj.three_d_collapsed
    num_packets = len(binary_data_list)

    # Repeat deltas n_spins times to match new num_epochs
    n_spins = int(16 / three_d_collapsed)
    repeated_deltas = np.tile(deltas, n_spins)
    # Calculate center of new epoch times using this
    # formula:
    #   epoch_time = epoch_center + (i * delta)
    #   where i = 0 to n_spins.
    # We are repeating each center time 'n_spins' times to
    # get new epochs and then multiply by factor. Final and new epoch shape
    # is (num_packets * n_spins). It's in seconds.
    # TODO: why multiply by 2?
    epoch_times = (
        np.repeat(epoch_center, n_spins)
        + np.tile(np.arange(n_spins), num_packets)
        * np.repeat(deltas, n_spins)
        / 1e9
        * 2
    )

    # ========== Initialize CDF Dataset with Coordinates ===========
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    l1a_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                met_to_ttj2000ns(epoch_times),
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
            ),
            "epoch_delta_minus": xr.DataArray(
                repeated_deltas,
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes(
                    "epoch_delta_minus", check_schema=False
                ),
            ),
            "epoch_delta_plus": xr.DataArray(
                repeated_deltas,
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes(
                    "epoch_delta_plus", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes(logical_source_id),
    )

    # Reshape decompressed data to:
    #   decompressed_data -> (9, 480)
    # Then we will parse 480 data into species below for looping.
    decompressed_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets, n_spins * 120
    )

    # Use chunks of (energy_x) to put data in its energy bins as done below.
    #   Eg. [15, 15, 15, 18, 18, 15, 18, 5, 1]
    # where each number is energy dimension for species 'x'.
    species_chunk_sizes = [
        len(species_data[species]["min_energy"]) for species in species_names
    ]
    start_idx = 0
    for index, (species_name, data) in enumerate(species_data.items()):
        # Add coordinate for 'energy_{species_name}'
        centers, energy_minus, energy_plus = get_energy_info(data)
        energy_attrs = cdf_attrs.get_variable_attributes(
            "hi-energy-attrs", check_schema=False
        )
        energy_attrs = apply_replacements_to_attrs(
            energy_attrs, {"species": species_name}
        )
        l1a_dataset = l1a_dataset.assign_coords(
            {
                f"energy_{species_name}": xr.DataArray(
                    np.array(centers),
                    dims=(f"energy_{species_name}",),
                    attrs=energy_attrs,
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
            np.array(centers).astype("str"),
            dims=(f"energy_{species_name}"),
            attrs=species_label_attrs,
        )
        # Add energy minus variables
        delta_attrs = cdf_attrs.get_variable_attributes("hi-energy-delta-attrs")
        delta_attrs = apply_replacements_to_attrs(
            delta_attrs, {"species": species_name, "operation": "minus"}
        )
        l1a_dataset[f"energy_{species_name}_minus"] = xr.DataArray(
            energy_minus, dims=(f"energy_{species_name}",), attrs=delta_attrs
        )
        # Add energy plus variable
        delta_attrs = apply_replacements_to_attrs(
            delta_attrs, {"species": species_name, "operation": "plus"}
        )
        l1a_dataset[f"energy_{species_name}_plus"] = xr.DataArray(
            energy_plus, dims=(f"energy_{species_name}",), attrs=delta_attrs
        )

        # Now, we put species data into its energy bins using indices like this:
        # Eg. species h's 4 spins data are in these indices:
        #   All h energy data of first spin   = [0,4,8,12,… 56].
        #   All h energy data of second spin  = [1,5,9,…,57].
        #   All h energy data of third spin   = [2,6,10,…,58].
        #   All h energy data of fourth spin  = [3,7,11,…,59].
        # In other words, H - [0 - 59] contains 4 spins data of h's 15 energy bins
        # and repeated this pattern for other species in order.
        #   Eg. He3 - [60 - 119] and so on.

        chunk_size = species_chunk_sizes[index]
        # Now parse the decompressed data into species as mentioned in above comment
        # using start and end indices.
        # End indices is start + (chunk size * n_spins)
        end_idx = start_idx + chunk_size * n_spins
        # Get specie's data by (num_epochs, start_idx:end_idx)
        #   Eg. (9, 60) for H
        species_data = decompressed_data[:, start_idx:end_idx]
        # Reshape the data to (num_epochs, species_chunk_size, n_spins) to begin
        # getting data into it's final state.
        #   Eg. (9, 15, 4)
        species_data = species_data.reshape(-1, chunk_size, n_spins)
        # Then transpose into (num_epochs, n_spins, species_chunk_size) and reshape
        # into (num_epochs * n_spins, species_chunk_size) to get final state.
        #   Eg. (36, 15)
        species_data = species_data.transpose(0, 2, 1).reshape(-1, chunk_size)
        species_attrs = cdf_attrs.get_variable_attributes("hi-species-attrs")
        species_attrs = apply_replacements_to_attrs(
            species_attrs, {"species": species_name}
        )
        # Convert to float
        species_data = species_data.astype(np.float64)
        l1a_dataset[species_name] = xr.DataArray(
            species_data,
            dims=("epoch", f"energy_{species_name}"),
            attrs=species_attrs,
        )
        species_unc_attrs = cdf_attrs.get_variable_attributes("hi-species-unc-attrs")
        species_unc_attrs = apply_replacements_to_attrs(
            species_unc_attrs, {"species": species_name}
        )
        l1a_dataset[f"unc_{species_name}"] = xr.DataArray(
            np.sqrt(species_data),
            dims=("epoch", f"energy_{species_name}"),
            attrs=species_unc_attrs,
        )
        # Increment start index
        start_idx = end_idx

    # ========= Add Additional Variables ===========
    # Repeat spin_period and data_quality to match new epoch shape (num_epochs)
    l1a_dataset["spin_period"] = xr.DataArray(
        np.repeat(unpacked_dataset["spin_period"].values, n_spins)
        * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("spin_period"),
    )
    l1a_dataset["data_quality"] = xr.DataArray(
        np.repeat(unpacked_dataset["suspect"].values, n_spins),
        dims=("epoch",),
        attrs=cdf_attrs.get_variable_attributes("data_quality"),
    )

    return l1a_dataset
