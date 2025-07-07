"""Decommutate HIT CCSDS data and create L1a data products."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    add_energy_variables,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l0.constants import (
    AZIMUTH_ANGLES,
    MOD_10_MAPPING,
    ZENITH_ANGLES,
)
from imap_processing.hit.l0.decom_hit import decom_hit

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)

# Fill value for missing data
fillval = -9223372036854775808


def hit_l1a(packet_file: str) -> list[xr.Dataset]:
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(packet_file)

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l1a")

    l1a_datasets = []

    # Process l1a data products
    if HitAPID.HIT_HSKP in datasets_by_apid:
        logger.info("Creating HIT L1A housekeeping dataset")
        l1a_datasets.append(
            process_housekeeping_data(
                datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1a_hk"
            )
        )
    if HitAPID.HIT_SCIENCE in datasets_by_apid:
        l1a_datasets.extend(
            process_science(datasets_by_apid[HitAPID.HIT_SCIENCE], attr_mgr)
        )
    return l1a_datasets


def subcom_sectorates(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Subcommutate sectorates data.

    Sectored rates data contains raw counts for 5 species and 10
    energy ranges. This function subcommutates the sectored
    rates data by organizing the counts by species. Which
    species and energy range the data belongs to is determined
    by taking the mod 10 value of the corresponding header
    minute count value in the dataset. A mapping of mod 10
    values to species and energy ranges is provided in constants.py.

    MOD_10_MAPPING = {
        0: {"species": "h", "energy_min": 1.8, "energy_max": 3.6},
        1: {"species": "h", "energy_min": 4, "energy_max": 6},
        2: {"species": "h", "energy_min": 6, "energy_max": 10},
        3: {"species": "he4", "energy_min": 4, "energy_max": 6},
        ...
        9: {"species": "fe", "energy_min": 4, "energy_max": 12}}

    The data is added to the dataset as new data fields named
    according to their species. They have 4 dimensions: epoch
    energy mean, azimuth, and zenith. The energy mean
    dimension is used to distinguish between the different energy
    ranges the data belongs to. The energy deltas for each species
    are also added to the dataset as new data fields.

    Parameters
    ----------
    sci_dataset : xarray.Dataset
        Xarray dataset containing parsed HIT science data.

    Returns
    -------
    sci_dataset : xarray.Dataset
        Xarray dataset with sectored rates data organized by species.
    """
    updated_dataset = sci_dataset.copy()

    # Calculate mod 10 values
    hdr_min_count_mod_10 = updated_dataset.hdr_minute_cnt.values % 10

    # Reference mod 10 mapping to initialize data structure for species and
    # energy ranges and add arrays with fill values for each science frame.
    num_frames = len(hdr_min_count_mod_10)
    data_by_species_and_energy_range = {
        key: {
            **value,
            "counts": np.full(
                (num_frames, len(AZIMUTH_ANGLES), len(ZENITH_ANGLES)),
                fill_value=fillval,
                dtype=np.int64,
            ),
        }
        for key, value in MOD_10_MAPPING.items()
    }

    # Update counts for science frames where data is available
    for i, mod_10 in enumerate(hdr_min_count_mod_10):
        data_by_species_and_energy_range[mod_10]["counts"][i] = updated_dataset[
            "sectorates"
        ].values[i]

    # H has 3 energy ranges, 4He, CNO, NeMgSi have 2, and Fe has 1.
    # Aggregate sectored rates and energy min/max values for each species.
    # First, initialize dictionaries to store rates and min/max energy values by species
    data_by_species: dict = {
        value["species"]: {"counts": [], "energy_min": [], "energy_max": []}
        for value in data_by_species_and_energy_range.values()
    }

    for value in data_by_species_and_energy_range.values():
        species = value["species"]
        data_by_species[species]["counts"].append(value["counts"])
        data_by_species[species]["energy_min"].append(value["energy_min"])
        data_by_species[species]["energy_max"].append(value["energy_max"])

    # Add sectored rates by species to the dataset
    for species, data in data_by_species.items():
        # Rates data has shape: energy_mean, epoch, azimuth, zenith
        # Convert rates to numpy array and transpose axes to get
        # shape: epoch, energy_mean, azimuth, zenith
        rates_data = np.transpose(np.array(data["counts"]), axes=(1, 0, 2, 3))

        updated_dataset[f"{species}_sectored_counts"] = xr.DataArray(
            data=rates_data,
            dims=["epoch", f"{species}_energy_mean", "azimuth", "zenith"],
            name=f"{species}_counts_sectored",
        )

        # Add energy mean and deltas for each species
        updated_dataset = add_energy_variables(
            updated_dataset,
            species,
            np.array(data["energy_min"]),
            np.array(data["energy_max"]),
        )

    return updated_dataset


def calculate_uncertainties(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate statistical uncertainties.

    Calculate the upper and lower uncertainties. The uncertainty for
    the raw Lev1A HIT data will be calculated as asymmetric Poisson
    uncertainty as prescribed in Gehrels 1986 (DOI: 10.1086/164079).
    See section 5.5 in the algorithm document for details.

    The upper uncertainty will be calculated as
        uncert_plus = sqrt(counts + 1) + 1

    The lower uncertainty will be calculated as
        uncert_minus = sqrt(counts)

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing counts data.

    Returns
    -------
    dataset : xarray.Dataset
        The dataset with added uncertainties for each counts data variable.
    """
    # Variables that aren't counts data and should be skipped in the calculation
    ignore_vars = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_dynamic_threshold_state",
        "hdr_leak_conv",
        "hdr_heater_duty_cycle",
        "hdr_code_ok",
        "hdr_minute_cnt",
        "livetime_counter",
        "h_energy_delta_minus",
        "h_energy_delta_plus",
        "he4_energy_delta_minus",
        "he4_energy_delta_plus",
        "cno_energy_delta_minus",
        "cno_energy_delta_plus",
        "nemgsi_energy_delta_minus",
        "nemgsi_energy_delta_plus",
        "fe_energy_delta_minus",
        "fe_energy_delta_plus",
    ]

    # Counts data that need uncertainties calculated
    count_vars = set(dataset.data_vars) - set(ignore_vars)

    # Calculate uncertainties for counts data variables.
    # Arrays with fill values (i.e. missing data) are skipped in this calculation
    # but are kept in the new data arrays to retain shape and dimensions.
    for var in count_vars:
        mask = dataset[var] != fillval  # Mask for valid values
        # Ensure that the values are positive before taking the square root
        safe_values_plus = np.maximum(dataset[var] + 1, 0).astype(np.float32)
        safe_values_minus = np.maximum(dataset[var], 0).astype(np.float32)

        dataset[f"{var}_stat_uncert_plus"] = xr.DataArray(
            np.where(
                mask, np.sqrt(safe_values_plus) + 1, dataset[var].astype(np.float32)
            ),
            dims=dataset[var].dims,
        )
        dataset[f"{var}_stat_uncert_minus"] = xr.DataArray(
            np.where(mask, np.sqrt(safe_values_minus), dataset[var].astype(np.float32)),
            dims=dataset[var].dims,
        )
    return dataset


def process_science(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes
) -> list[xr.Dataset]:
    """
    Will process science datasets for CDF products.

    Process binary science data for CDF creation. The data is
    grouped into science frames, decommutated and decompressed,
    and split into count rates and event datasets. Updates the
    dataset attributes and coordinates and data variable
    dimensions according to specifications in a cdf yaml file.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset containing HIT science data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    dataset : list
        A list of science datasets ready for CDF conversion.
    """
    logger.info("Creating HIT L1A science datasets")

    # Decommutate and decompress the science data
    sci_dataset = decom_hit(dataset)

    # Organize sectored rates by species type
    sci_dataset = subcom_sectorates(sci_dataset)

    # Split the science data into count rates and event datasets
    pha_raw_dataset = xr.Dataset(
        {"pha_raw": sci_dataset["pha_raw"]}, coords={"epoch": sci_dataset["epoch"]}
    )
    count_rates_dataset = sci_dataset.drop_vars("pha_raw")

    # Calculate uncertainties for count rates
    count_rates_dataset = calculate_uncertainties(count_rates_dataset)

    l1a_datasets: dict = {
        "imap_hit_l1a_counts": count_rates_dataset,
        "imap_hit_l1a_direct-events": pha_raw_dataset,
    }

    # Update attributes and dimensions
    for logical_source, ds in l1a_datasets.items():
        ds.attrs = attr_mgr.get_global_attributes(logical_source)

        # Assign attributes and dimensions to each data array in the Dataset
        for field in ds.data_vars.keys():
            try:
                ds[field].attrs = attr_mgr.get_variable_attributes(field)
            except KeyError:
                print(f"Field {field} not found in attribute manager.")
                logger.warning(f"Field {field} not found in attribute manager.")

        # check_schema=False to avoid attr_mgr adding stuff dimensions don't need
        for dim in ds.dims:
            ds[dim].attrs = attr_mgr.get_variable_attributes(dim, check_schema=False)
            # TODO: should labels be added as coordinates? Check with SPDF
            if dim != "epoch":
                label_array = xr.DataArray(
                    ds[dim].values.astype(str),
                    name=f"{dim}_label",
                    dims=[dim],
                    attrs=attr_mgr.get_variable_attributes(
                        f"{dim}_label", check_schema=False
                    ),
                )
                ds.coords[f"{dim}_label"] = label_array

        logger.info(f"HIT L1A dataset created for {logical_source}")

    return list(l1a_datasets.values())
