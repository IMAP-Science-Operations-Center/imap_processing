"""CoDICE Hi I-ALiRT L1A processing functions."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    ViewTabInfo,
    get_codice_epoch_time,
    get_collapse_pattern_shape,
    get_energy_info,
    get_view_tab_info,
    read_sci_lut,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def l1a_ialirt_hi(unpacked_dataset: xr.Dataset, lut_file: Path) -> xr.Dataset:
    """
    Process CoDICE Hi I-ALiRT L1A data.

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
    table_id = unpacked_dataset["table_id"].values[0]
    view_id = unpacked_dataset["view_id"].values[0]
    apid = unpacked_dataset["pkt_apid"].values[0]
    plan_id = unpacked_dataset["plan_id"].values[0]
    plan_step = unpacked_dataset["plan_step"].values[0]

    logger.info(
        f"Processing species with - APID: {apid} / 0x{apid:X}, "
        f"View ID: {view_id}, Table ID: {table_id}, "
        f"Plan ID: {plan_id}, Plan Step: {plan_step}"
    )

    # ========== Get LUT Data ===========
    sci_lut_data = read_sci_lut(lut_file, table_id)
    view_tab_info = get_view_tab_info(sci_lut_data, view_id, apid)

    view_tab_obj = ViewTabInfo(
        apid=apid,
        view_id=view_id,
        sensor=view_tab_info["sensor"],
        three_d_collapsed=view_tab_info["3d_collapse"],
        collapse_table=view_tab_info["collapse_table"],
    )

    species_data = sci_lut_data["data_product_hi_tab"]["0"]["ialirt"]
    first_species = next(iter(species_data))
    centers, energy_minus, energy_plus = get_energy_info(species_data[first_species])

    compression_algorithm = constants.HI_COMPRESSION_ID_LOOKUP[view_tab_obj.view_id]

    binary_data_list = unpacked_dataset["data"].values
    byte_count_list = unpacked_dataset["byte_count"].values

    decompressed_data = [
        decompress(packet_data[:byte_count], compression_algorithm)
        for packet_data, byte_count in zip(
            binary_data_list, byte_count_list, strict=False
        )
    ]

    epoch_center, deltas = get_codice_epoch_time(
        unpacked_dataset["acq_start_seconds"].values,
        unpacked_dataset["acq_start_subseconds"].values,
        unpacked_dataset["spin_period"].values,
        view_tab_obj,
    )

    three_d_collapsed = view_tab_obj.three_d_collapsed
    num_packets = len(binary_data_list)
    n_spins = int(16 / three_d_collapsed)
    repeated_deltas = np.tile(deltas, n_spins)

    epoch_times = (
        np.repeat(epoch_center, n_spins)
        + np.tile(np.arange(n_spins), num_packets)
        * np.repeat(deltas, n_spins)
        / 1e9
        * 2
    )

    l1a_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                met_to_ttj2000ns(epoch_times),
                dims=("epoch",),
            ),
            "epoch_delta_minus": xr.DataArray(
                repeated_deltas,
                dims=("epoch",),
            ),
            "epoch_delta_plus": xr.DataArray(
                repeated_deltas,
                dims=("epoch",),
            ),
            f"energy_{first_species}_minus": xr.DataArray(
                energy_minus,
                dims=(f"energy_{first_species}",),
            ),
            f"energy_{first_species}_plus": xr.DataArray(
                energy_plus,
                dims=(f"energy_{first_species}",),
            ),
        },
    )

    energy_bins = 15
    collapse_shape = get_collapse_pattern_shape(
        sci_lut_data,
        view_tab_obj.sensor,
        view_tab_obj.collapse_table,
    )

    # Reshape data into (epoch, energy, n_spins, spin_sector, inst_az)
    decompressed_data = np.array(decompressed_data, dtype=np.uint32).reshape(
        num_packets,
        energy_bins,
        n_spins,
        *collapse_shape,
    )

    # Number of energy chunks.
    num_energy_chunk = [len(species_data[first_species]["min_energy"])]

    l1a_dataset = l1a_dataset.assign_coords(
        {f"energy_{first_species}": (f"energy_{first_species}", np.array(centers))}
    )

    chunk_size = num_energy_chunk[0]

    # This is rearranging data from (epoch, energy, n_spins, spin_sector, inst_az)
    # -> (epoch, n_spins, energy, spin_sector, inst_az) ->
    # finally (epoch * n_spins, energy,
    # spin_sector, inst_az)
    decompressed_data = decompressed_data.transpose(0, 2, 1, 3, 4).reshape(
        -1, chunk_size, *collapse_shape
    )

    l1a_dataset[first_species] = xr.DataArray(
        decompressed_data,
        dims=("epoch", f"energy_{first_species}", "spin_sector", "inst_az"),
    )

    l1a_dataset[f"unc_{first_species}"] = xr.DataArray(
        np.sqrt(decompressed_data),
        dims=("epoch", f"energy_{first_species}", "spin_sector", "inst_az"),
    )

    l1a_dataset["spin_period"] = xr.DataArray(
        np.repeat(unpacked_dataset["spin_period"].values, n_spins)
        * constants.SPIN_PERIOD_CONVERSION,
        dims=("epoch",),
    )

    l1a_dataset["data_quality"] = xr.DataArray(
        np.repeat(unpacked_dataset["suspect"].values, n_spins),
        dims=("epoch",),
    )

    return l1a_dataset
