"""IMAP-HIT L1B data processing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    add_summed_particle_data_to_dataset,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l1b.constants import (
    FILLVAL_FLOAT32,
    FILLVAL_INT64,
    LIVESTIM_PULSES,
    SUMMED_PARTICLE_ENERGY_RANGE_MAPPING,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(dependencies: dict) -> list[xr.Dataset]:
    """
    Will process HIT data to L1B.

    Processes dependencies needed to create L1B data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of dependencies that are L1A xarray datasets
        for science data and a file path string to an L0 file
        for housekeeping data.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of four L1B datasets.
    """
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l1b")

    # Create L1B datasets
    l1b_datasets: list = []
    if "imap_hit_l0_raw" in dependencies:
        # Unpack ccsds file to xarray datasets
        packet_file = dependencies["imap_hit_l0_raw"]
        datasets_by_apid = get_datasets_by_apid(packet_file, derived=True)
        # TODO: update to raise error after all APIDs are included in the same
        #  raw files. currently science and housekeeping are in separate files.
        if HitAPID.HIT_HSKP in datasets_by_apid:
            # Process housekeeping to L1B.
            l1b_datasets.append(
                process_housekeeping_data(
                    datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
                )
            )
            logger.info("HIT L1B housekeeping dataset created")
    if "imap_hit_l1a_counts" in dependencies:
        # Process science data to L1B datasets
        l1a_counts_dataset = dependencies["imap_hit_l1a_counts"]
        l1b_datasets.extend(process_science_data(l1a_counts_dataset, attr_mgr))
        logger.info("HIT L1B science datasets created")

    return l1b_datasets


def process_science_data(
    l1a_counts_dataset: xr.Dataset, attr_mgr: ImapCdfAttributes
) -> list[xr.Dataset]:
    """
    Will create L1B science datasets for CDF products.

    Process L1A raw counts data to create L1B science data for
    CDF creation. This function will create three L1B science
    datasets: standard rates, summed rates, and sectored rates.
    It will also update dataset attributes, coordinates and
    data variable dimensions according to specifications in
    a CDF yaml file.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.
    attr_mgr : AttributeManager
        The attribute manager for the L1B data level.

    Returns
    -------
    dataset : list
        The processed L1B science datasets as xarray datasets.
    """
    logger.info("Creating HIT L1B science datasets")

    # TODO: Write functions to create the following datasets
    #  Process sectored rates dataset

    # Calculate fractional livetime from the livetime counter
    livetime = l1a_counts_dataset["livetime_counter"] / LIVESTIM_PULSES
    livetime = livetime.rename("livetime")

    # Process counts data to L1B datasets
    l1b_datasets: dict = {}
    l1b_datasets["imap_hit_l1b_standard-rates"] = process_standard_rates_data(
        l1a_counts_dataset, livetime
    )
    l1b_datasets["imap_hit_l1b_summed-rates"] = process_summed_rates_data(
        l1a_counts_dataset, livetime
    )
    l1b_datasets["imap_hit_l1b_sectored-rates"] = process_sectored_rates_data(
        l1a_counts_dataset, livetime
    )

    # Update attributes and dimensions
    for logical_source, dataset in l1b_datasets.items():
        dataset.attrs = attr_mgr.get_global_attributes(logical_source)

        # TODO: Add CDF attributes to yaml once they're defined for L1B science data
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

        logger.info(f"HIT L1B dataset created for {logical_source}")

    return list(l1b_datasets.values())


def initialize_l1b_dataset(l1a_counts_dataset: xr.Dataset, coords: list) -> xr.Dataset:
    """
    Initialize the L1B dataset.

    Create a dataset and add coordinates and the dynamic threshold state data array
    from the L1A counts dataset.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.
    coords : list
        A list of coordinates to assign to the L1B dataset.

    Returns
    -------
    l1b_dataset : xr.Dataset
        An L1B dataset with coordinates and dynamic threshold state.
    """
    l1b_dataset = xr.Dataset(
        coords={coord: l1a_counts_dataset.coords[coord] for coord in coords}
    )
    l1b_dataset["dynamic_threshold_state"] = l1a_counts_dataset[
        "hdr_dynamic_threshold_state"
    ]
    return l1b_dataset


def process_standard_rates_data(
    l1a_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1B standard rates data from L1A raw counts data.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B standard rates dataset.
    """
    # Initialize the L1B standard rates dataset with coordinates from the L1A dataset
    l1b_standard_rates_dataset = initialize_l1b_dataset(
        l1a_counts_dataset,
        coords=[
            "epoch",
            "gain",
            "sngrates_index",
            "coinrates_index",
            "pbufrates_index",
            "l2fgrates_index",
            "l2bgrates_index",
            "l3fgrates_index",
            "l3bgrates_index",
            "penfgrates_index",
            "penbgrates_index",
            "ialirtrates_index",
        ],
    )

    # Define fields from the L1A counts dataset to calculate standard rates from
    standard_rate_fields = [
        "sngrates",
        "coinrates",
        "pbufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "l4fgrates",
        "l4bgrates",
    ]

    for var in standard_rate_fields:
        # Add counts and uncertainty data to the dataset
        l1b_standard_rates_dataset[var] = l1a_counts_dataset[var]
        l1b_standard_rates_dataset[f"{var}_stat_uncert_minus"] = l1a_counts_dataset[
            f"{var}_stat_uncert_minus"
        ]
        l1b_standard_rates_dataset[f"{var}_stat_uncert_plus"] = l1a_counts_dataset[
            f"{var}_stat_uncert_plus"
        ]
        # Calculate rates using livetime
        l1b_standard_rates_dataset = calculate_rates(
            l1b_standard_rates_dataset, var, livetime
        )

    return l1b_standard_rates_dataset


def calculate_rates(
    dataset: xr.Dataset,
    var: str,
    livetime: xr.DataArray,
) -> xr.Dataset:
    """
    Calculate rates by dividing counts by livetime.

    Parameters
    ----------
    dataset : xr.Dataset
        The L1B dataset containing counts data.
    var : str
        The name of the variable to calculate rates for.
    livetime : xr.DataArray
        1D array of livetime values. Shape equals the
        number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The dataset with rates.
    """
    dataset[f"{var}"] = (dataset[f"{var}"] / livetime).astype(np.float32)
    dataset[f"{var}_stat_uncert_minus"] = (
        dataset[f"{var}_stat_uncert_minus"] / livetime
    ).astype(np.float32)
    dataset[f"{var}_stat_uncert_plus"] = (
        dataset[f"{var}_stat_uncert_plus"] / livetime
    ).astype(np.float32)

    return dataset


def sum_livetime_10min(livetime: xr.DataArray) -> xr.DataArray:
    """
    Sum livetime values in 10-minute intervals.

    Parameters
    ----------
    livetime : xr.DataArray
        1D array of livetime values. Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.DataArray
        Livetime summed over 10-minute intervals. Values repeated for each epoch in the
        10-minute intervals to match the original livetime array shape.
        [5,5,5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7,7,7].
    """
    livetime_10min_sum = [
        livetime[i : i + 10].sum().item() for i in range(0, len(livetime) - 9, 10)
    ]
    livetime_expanded = np.repeat(livetime_10min_sum, 10)
    return xr.DataArray(livetime_expanded, dims=livetime.dims, coords=livetime.coords)


def process_summed_rates_data(
    l1a_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1B summed rates data from L1A raw counts data.

    This function calculates summed rates for each particle type and energy range.
    The counts that are summed come from the l2fgrates, l3fgrates, and penfgrates
    data variables in the L1A counts data. These variables represent counts
    of different detector penetration ranges (Range 2, Range 3, and Range 4
    respectively). Only the energy ranges specified in the
    SUMMED_PARTICLE_ENERGY_RANGE_MAPPING dictionary are included in this product.

    The summed rates are calculated by summing the counts for each energy range and
    dividing by the livetime.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B summed rates dataset.
    """
    # Initialize the L1B summed rates dataset with coordinates from the L1A dataset
    l1b_summed_rates_dataset = initialize_l1b_dataset(
        l1a_counts_dataset, coords=["epoch"]
    )

    for particle, energy_ranges in SUMMED_PARTICLE_ENERGY_RANGE_MAPPING.items():
        # Sum counts for each energy range and add to dataset
        l1b_summed_rates_dataset = add_summed_particle_data_to_dataset(
            l1b_summed_rates_dataset,
            l1a_counts_dataset,
            particle,
            energy_ranges,
        )
        # Calculate rates using livetime
        l1b_summed_rates_dataset = calculate_rates(
            l1b_summed_rates_dataset, particle, livetime
        )

    return l1b_summed_rates_dataset


def subset_data_for_sectored_counts(
    l1a_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> tuple[xr.Dataset, xr.DataArray]:
    """
    Subset data for complete sets of sectored counts and corresponding livetime values.

    A set of sectored data starts with hydrogen and ends with iron and correspond to
    the mod 10 values 0-9. The livetime values from the previous 10 minutes are used
    to calculate the rates for each set since those counts are transmitted 10 minutes
    after they were collected. Therefore, only complete sets of sectored counts where
    livetime from the previous 10 minutes are available are included in the output.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.
    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.

    Returns
    -------
    tuple[xr.Dataset, xr.DataArray]
        Dataset of complete sectored counts and corresponding livetime values.
    """
    # Identify 10-minute intervals of complete sectored counts.
    bin_size = 10
    mod_10 = l1a_counts_dataset.hdr_minute_cnt.values % 10
    pattern = np.arange(bin_size)

    # Use sliding windows to find pattern matches
    matches = np.all(
        np.lib.stride_tricks.sliding_window_view(mod_10, bin_size) == pattern, axis=1
    )
    start_indices = np.where(matches)[0]

    # Filter out start indices that are less than or equal to the bin size
    # since the previous 10 minutes are needed for calculating rates
    if start_indices.size == 0:
        logger.error(
            "No data to process - valid start indices not found for "
            "complete sectored counts."
        )
        raise ValueError("No valid start indices found for complete sectored counts.")
    else:
        start_indices = start_indices[start_indices >= bin_size]

    # Subset data for complete sets of sectored counts.
    # Each set of sectored counts is 10 minutes long, so we take the indices
    # starting from the start indices and extend to the bin size of 10.
    # This creates a 1D array of indices that correspond to the complete
    # sets of sectored counts which is used to filter the L1A dataset and
    # create the L1B sectored rates dataset.
    data_indices = np.concatenate(
        [np.arange(idx, idx + bin_size) for idx in start_indices]
    )
    l1b_sectored_rates_dataset = l1a_counts_dataset.isel(epoch=data_indices)

    # Subset livetime values corresponding to the previous 10 minutes
    # for each start index. This ensures the livetime data aligns correctly
    # with the sectored counts for rate calculations.
    livetime_indices = np.concatenate(
        [np.arange(idx - bin_size, idx) for idx in start_indices]
    )
    livetime = livetime.isel(epoch=livetime_indices)

    return l1b_sectored_rates_dataset, livetime


def process_sectored_rates_data(
    l1a_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1B sectored rates data from L1A raw counts data.

    A complete set of sectored counts is taken over 10 science frames (10 minutes)
    where each science frame contains counts for one species and energy range.

    Species and energy ranges are as follows:

        H      1.8 - 3.6 MeV, 4.0 - 6.0 MeV, 6.0 - 10 MeV
        4He    4.0 - 6.0 MeV, 6.0 - 12.0 MeV
        CNO    4.0 - 6.0 MeV, 6.0 - 12.0 MeV
        NeMgSi 4.0 - 6.0 MeV, 6.0 - 12.0 MeV
        Fe     4.0 - 12.0 MeV

    Sectored counts data is transmitted 10 minutes after they are collected.
    To calculate rates, the sectored counts over 10 minutes need to be divided by
    the sum of livetime values from the previous 10 minutes.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B sectored rates dataset.
    """
    # TODO
    #  -filter by epoch values in day being processed.
    #   middle epoch (or mod 5 value for 6th frame)
    #  -consider refactoring calculate_rates function to handle sectored rates

    # Define particles and coordinates
    particles = ["h", "he4", "cno", "nemgsi", "fe"]

    # Extract relevant data variable names that start with a particle name
    data_vars = [
        str(var)
        for var in l1a_counts_dataset.data_vars
        if any(str(var).startswith(f"{p}_") for p in particles)
    ]

    # Subset data for complete sets of sectored counts and corresponding livetime values
    l1a_counts_dataset, livetime = subset_data_for_sectored_counts(
        l1a_counts_dataset, livetime
    )

    # Sum livetime over 10 minute intervals
    livetime_10min = sum_livetime_10min(livetime)

    # Initialize the L1B dataset with coordinates from the subset L1A dataset
    l1b_sectored_rates_dataset = initialize_l1b_dataset(
        l1a_counts_dataset,
        coords=[
            "epoch",
            "zenith",
            "azimuth",
            "h_energy_mean",
            "he4_energy_mean",
            "cno_energy_mean",
            "nemgsi_energy_mean",
            "fe_energy_mean",
        ],
    )

    # Dictionary to store variable rename mappings for L1B dataset
    rename_map = {}

    # # Compute rates, skipping fill values, and add to the L1B dataset
    for var in data_vars:
        if "sectored_counts" in var:
            # Determine the new variable name for the L1B dataset
            if "_sectored_counts" in var:
                new_var = var.replace("_sectored_counts", "")
            else:
                new_var = None
            if new_var:
                rename_map[var] = new_var

            # Since epoch times don't align, convert xarray data arrays to numpy arrays
            # to avoid rates being calculated along the epoch dimension.
            # Reshape livetime to match 4D shape of counts.
            counts = l1a_counts_dataset[var].values
            livetime_10min_reshaped = livetime_10min.values[:, None, None, None]
            rates = xr.DataArray(
                np.where(
                    counts != FILLVAL_INT64,
                    (counts / livetime_10min_reshaped).astype(np.float32),
                    FILLVAL_FLOAT32,
                ),
                dims=l1a_counts_dataset[var].dims,
            )
            l1b_sectored_rates_dataset[var] = rates
        else:
            # Add other data variables to the dataset
            l1b_sectored_rates_dataset[var] = l1a_counts_dataset[var]

    # Rename variables in L1B dataset
    if rename_map:
        l1b_sectored_rates_dataset = l1b_sectored_rates_dataset.rename(rename_map)

    return l1b_sectored_rates_dataset
