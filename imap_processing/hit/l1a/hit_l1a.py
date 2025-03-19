"""Decommutate HIT CCSDS data and create L1a data products."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l0.constants import MOD_10_MAPPING
from imap_processing.hit.l0.decom_hit import decom_hit

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)

# Fill value for missing data
fillval = -9223372036854775808


def hit_l1a(packet_file: str, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(packet_file)

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1a")

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


def subcom_sectorates(sci_dataset: xr.Dataset) -> None:
    """
    Subcommutate sectorates data.

    Sector rates data contains rates for 5 species and 10
    energy ranges. This function subcommutates the sector
    rates data by organizing the rates by species. Which
    species and energy range the data belongs to is determined
    by taking the mod 10 value of the corresponding header
    minute count value in the dataset. A mapping of mod 10
    values to species and energy ranges is provided in constants.py.

    MOD_10_MAPPING = {
        0: {"species": "H", "energy_min": 1.8, "energy_max": 3.6},
        1: {"species": "H", "energy_min": 4, "energy_max": 6},
        2: {"species": "H", "energy_min": 6, "energy_max": 10},
        3: {"species": "4He", "energy_min": 4, "energy_max": 6},
        ...
        9: {"species": "Fe", "energy_min": 4, "energy_max": 12}}

    The data is added to the dataset as new data fields named
    according to their species. They have 4 dimensions: epoch
    energy index, declination, and azimuth. The energy index
    dimension is used to distinguish between the different energy
    ranges the data belongs to. The energy min and max values for
    each species are also added to the dataset as new data fields.

    Parameters
    ----------
    sci_dataset : xarray.Dataset
        Xarray dataset containing parsed HIT science data.
    """
    # Calculate mod 10 values
    hdr_min_count_mod_10 = sci_dataset.hdr_minute_cnt.values % 10

    # Reference mod 10 mapping to initialize data structure for species and
    # energy ranges and add 8x15 arrays with fill values for each science frame.
    num_frames = len(hdr_min_count_mod_10)
    # TODO: add more specific dtype for rates (ex. int16) once this is defined by HIT
    data_by_species_and_energy_range = {
        key: {
            **value,
            "rates": np.full((num_frames, 8, 15), fill_value=fillval, dtype=int),
        }
        for key, value in MOD_10_MAPPING.items()
    }

    # Update rates for science frames where data is available
    for i, mod_10 in enumerate(hdr_min_count_mod_10):
        data_by_species_and_energy_range[mod_10]["rates"][i] = sci_dataset[
            "sectorates"
        ].values[i]

    # H has 3 energy ranges, 4He, CNO, NeMgSi have 2, and Fe has 1.
    # Aggregate sector rates and energy min/max values for each species.
    # First, initialize dictionaries to store rates and min/max energy values by species
    data_by_species: dict = {
        value["species"]: {"rates": [], "energy_min": [], "energy_max": []}
        for value in data_by_species_and_energy_range.values()
    }

    for value in data_by_species_and_energy_range.values():
        species = value["species"]
        data_by_species[species]["rates"].append(value["rates"])
        data_by_species[species]["energy_min"].append(value["energy_min"])
        data_by_species[species]["energy_max"].append(value["energy_max"])

    # Add sector rates by species to the dataset
    for species_type, data in data_by_species.items():
        # Rates data has shape: energy_index, epoch, declination, azimuth
        # Convert rates to numpy array and transpose axes to get
        # shape: epoch, energy_index, declination, azimuth
        rates_data = np.transpose(np.array(data["rates"]), axes=(1, 0, 2, 3))

        species = species_type.lower()
        sci_dataset[f"{species}_counts_sectored"] = xr.DataArray(
            data=rates_data,
            dims=["epoch", f"{species}_energy_index", "declination", "azimuth"],
            name=f"{species}_counts_sectored",
        )
        sci_dataset[f"{species}_energy_min"] = xr.DataArray(
            data=np.array(data["energy_min"], dtype=np.int8),
            dims=[f"{species}_energy_index"],
            name=f"{species}_energy_min",
        )
        sci_dataset[f"{species}_energy_max"] = xr.DataArray(
            data=np.array(data["energy_max"], dtype=np.int8),
            dims=[f"{species}_energy_index"],
            name=f"{species}_energy_max",
        )
        # add energy index coordinate to the dataset
        sci_dataset.coords[f"{species}_energy_index"] = xr.DataArray(
            np.arange(sci_dataset.sizes[f"{species}_energy_index"], dtype=np.int8),
            dims=[f"{species}_energy_index"],
            name=f"{species}_energy_index",
        )


def calculate_uncertainties(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate uncertainties for each counts data variable in the dataset.

    Calculate the upper and lower uncertainties. The uncertainty for
    the raw Lev1A HIT data will be calculated as asymmetric Poisson
    uncertainty as prescribed in Gehrels 1986 (DOI: 10.1086/164079).
    See section 5.5 in the algorithm document for details.

    The upper uncertainty will be calculated as
        DELTA_PLUS = sqrt(counts + 1) + 1

    The lower uncertainty will be calculated as
        DELTA_MINUS = sqrt(counts)

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
        "h_energy_min",
        "h_energy_max",
        "he4_energy_min",
        "he4_energy_max",
        "cno_energy_min",
        "cno_energy_max",
        "nemgsi_energy_min",
        "nemgsi_energy_max",
        "fe_energy_min",
        "fe_energy_max",
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

        dataset[f"{var}_delta_plus"] = xr.DataArray(
            np.where(
                mask, np.sqrt(safe_values_plus) + 1, dataset[var].astype(np.float32)
            ),
            dims=dataset[var].dims,
        )
        dataset[f"{var}_delta_minus"] = xr.DataArray(
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

    # Organize sector rates by species type
    subcom_sectorates(sci_dataset)

    # Split the science data into count rates and event datasets
    pha_raw_dataset = xr.Dataset(
        {"pha_raw": sci_dataset["pha_raw"]}, coords={"epoch": sci_dataset["epoch"]}
    )
    count_rates_dataset = sci_dataset.drop_vars("pha_raw")

    # Calculate uncertainties for count rates
    count_rates_dataset = calculate_uncertainties(count_rates_dataset)

    # Logical sources for the two products.
    logical_sources = ["imap_hit_l1a_counts", "imap_hit_l1a_direct-events"]

    datasets = []
    # Update attributes and dimensions
    for dataset, logical_source in zip(
        [count_rates_dataset, pha_raw_dataset], logical_sources
    ):
        dataset.attrs = attr_mgr.get_global_attributes(logical_source)

        # TODO: Add CDF attributes to yaml once they're defined for L1A science data
        # Assign attributes and dimensions to each data array in the Dataset
        for field in dataset.data_vars.keys():
            try:
                # Create a dict of dimensions using the DEPEND_I keys in the
                # attributes
                dims = {
                    key: value
                    for key, value in attr_mgr.get_variable_attributes(field).items()
                    if "DEPEND" in key
                }
                dataset[field].attrs = attr_mgr.get_variable_attributes(field)
                dataset[field].assign_coords(dims)
            except KeyError:
                print(f"Field {field} not found in attribute manager.")
                logger.warning(f"Field {field} not found in attribute manager.")

        # Skip schema check for epoch to prevent attr_mgr from adding the
        # DEPEND_0 attribute which isn't required for epoch
        dataset.epoch.attrs = attr_mgr.get_variable_attributes(
            "epoch", check_schema=False
        )

        datasets.append(dataset)

        logger.info(f"HIT L1A dataset created for {logical_source}")

    return datasets
