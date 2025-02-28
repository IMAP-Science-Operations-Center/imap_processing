"""IMAP-HIT L2 data processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l2(dependencies: dict, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L2.

    Processes dependencies needed to create L2 data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of dependencies that are L1B xarray datasets
        for science data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of three L2 datasets.
    """
    # Create the attribute manager for this data level
    # attr_mgr = get_attribute_manager(data_version, "l2")

    # Create L2 datasets
    l2_datasets: list = []
    if "imap_hit_l1b_summed-rates" in dependencies:
        # Process science data to L2 datasets
        l1b_summed_rates_dataset = dependencies["imap_hit_l1a_count-rates"]
        l2_datasets.extend(process_summed_flux_data(l1b_summed_rates_dataset))
        logger.info("HIT L2 summed flux dataset created")

    return l2_datasets


def process_summed_flux_data(l1b_summed_rates_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L2 HIT summed flux data from L1B summed rates.

    Parameters
    ----------
    l1b_summed_rates_dataset : xarray.Dataset
        HIT L1B summed rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 summed flux dataset.
    """
    # TODO: determine where to use attr manager
    # TODO: determine where to pull ancillary data. Storing it locally for now

    # Create a new dataset to store the L1B summed flux data
    l1b_summed_flux_dataset = l1b_summed_rates_dataset.copy(deep=True)

    # Read in ancillary data which contains factors to convert L1B Summed count
    # rate data to L2 fluxes
    # (delta energy, geometry factor, efficiency, and b)
    # See equation 11 in the HIT algorithm document.
    # Load the validation data
    ancillary_file = (
        imap_module_directory
        / "hit/ancillary/imap_hit_l1b-to-l2-summed-factors-20250219_v002.csv"
    )
    ancillary_data = pd.read_csv(ancillary_file)
    ancillary_data.columns = ancillary_data.columns.str.lower().str.strip()

    particle_names = {
        "hydrogen": "H",
        "helium3": "He3",
        "helium4": "He4",
        "helium": "He",
        "carbon": "C",
        "nitrogen": "N",
        "oxygen": "O",
        "neon": "Ne",
        "sodium": "Na",
        "magnesium": "Mg",
        "aluminum": "Al",
        "silicon": "Si",
        "sulfur": "S",
        "argon": "Ar",
        "calcium": "Ca",
        "iron": "Fe",
        "nickel": "Ni",
    }

    # Calculate the summed flux using the ancillary table.
    for var in l1b_summed_flux_dataset.data_vars:
        if var != "dynamic_threshold_state" and "energy_" not in var:
            print(var)
            # Get the species name from the variable name
            if "_delta_" in var:
                # uncertainty variables (i.e. h_delta_plus, h_delta_minus)
                species = str(var).split("_")[0]
            else:
                species = var
            species_abbrev = particle_names[species]

            # Get the ancillary data for the species
            var_anc_data = ancillary_data[ancillary_data["species"] == species_abbrev]

            # Calculate the summed flux for each epoch and energy bin
            for epoch in range(l1b_summed_flux_dataset[var].shape[0]):
                for i, rate in enumerate(l1b_summed_flux_dataset[var][epoch].values):
                    energy_min = l1b_summed_flux_dataset[f"{species}_energy_min"][
                        i
                    ].values.item()
                    # TODO add check for max too after a new ancillary file is provided
                    #  fixing errors
                    # energy_max = l1b_summed_flux_dataset[f"{species}_energy_max"][
                    #     i
                    # ].values

                    # Get the ancillary data for this energy bin range
                    flux_factors = var_anc_data[
                        var_anc_data["lower energy (mev)"].astype(np.float32)
                        == energy_min
                    ]
                    delta_e_factor = flux_factors["delta e (mev)"].values[0]
                    geometry_factor = flux_factors["geometry factor (cm2 sr)"].values[0]
                    efficiency = flux_factors["efficiency"].values[0]
                    b = flux_factors["b"].values[0]
                    print(f"EPOCH: {epoch}")
                    print(f"ENERGY_INDEX: {i}")
                    print(f"RATE: {l1b_summed_flux_dataset[var][epoch][i].values}")

                    # Calculate the summed flux for this energy bin
                    l1b_summed_flux_dataset[var][epoch][i] = (
                        rate / (60 * delta_e_factor * geometry_factor * efficiency)
                    ) - b

                    print(f"FLUX: {l1b_summed_flux_dataset[var][epoch][i].values}")
                    print(energy_min)
                    print(f"delta_e_factor: {delta_e_factor}")
                    print(f"geometry_factor: {geometry_factor}")
                    print(f"efficiency: {efficiency}")
                    print(f"b: {b}\n")

            print(l1b_summed_flux_dataset[species].shape)

    return l1b_summed_flux_dataset


if __name__ == "__main__":
    from imap_processing import imap_module_directory
    from imap_processing.hit.l1a.hit_l1a import hit_l1a
    from imap_processing.hit.l1b.hit_l1b import process_summed_rates_data

    # L0 file path
    packet_file = imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"

    datasets = hit_l1a(packet_file, "001")
    counts = datasets[0]

    # Calculate livetime from the livetime counter
    livetime = counts["livetime_counter"] / 270

    summed_rates = process_summed_rates_data(counts, livetime)
    # print(summed_rates)
    # print(summed_rates["hydrogen"][0])

    l2_dataset = process_summed_flux_data(summed_rates)
