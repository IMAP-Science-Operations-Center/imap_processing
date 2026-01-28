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
    SECTORS,
    SUMMED_PARTICLE_ENERGY_RANGE_MAPPING,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(dependency: str | xr.Dataset, l1b_descriptor: str) -> xr.Dataset:
    """
    Will process HIT data to L1B.

    Processes dependencies needed to create L1B data products.

    Parameters
    ----------
    dependency : Union[str, xr.Dataset]
        Dependency is either an L1A xarray dataset to process
        science data or a file path string to an L0 file to
        process housekeeping data.
    l1b_descriptor : str
        The descriptor for the L1B dataset to create.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The processed L1B dataset.
    """
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l1b")

    l1b_dataset = None

    # Create L1B datasets
    if l1b_descriptor == "hk":
        # Unpack ccsds file to xarray datasets
        packet_file = dependency
        datasets_by_apid = get_datasets_by_apid(packet_file, derived=True)
        if HitAPID.HIT_HSKP in datasets_by_apid:
            # Process housekeeping to L1B.
            l1b_dataset = process_housekeeping_data(
                datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
            )
            logger.info("HIT L1B housekeeping dataset created")
    elif l1b_descriptor in ["standard-rates", "summed-rates", "sectored-rates"]:
        # Process science data to L1B datasets
        l1b_dataset = process_science_data(dependency, l1b_descriptor, attr_mgr)
        logger.info("HIT L1B science dataset created")
    else:
        logger.error(f"Unsupported descriptor for L1B processing: {l1b_descriptor}")
        raise ValueError(f"Unsupported descriptor: {l1b_descriptor}")

    return l1b_dataset


def process_science_data(
    l1a_counts_dataset: xr.Dataset, descriptor: str, attr_mgr: ImapCdfAttributes
) -> xr.Dataset:
    """
    Will create L1B science datasets for CDF products.

    This function processes L1A counts data to L1B science
    data for CDF creation. There are three L1B science
    datasets: standard rates, summed rates, and sectored rates.
    This function creates one dataset based on the descriptor
    provided. It will also update dataset attributes, coordinates and
    data variable dimensions according to specifications in
    a CDF yaml file.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset.
    descriptor : str
        The descriptor for the L1B dataset to create
        (e.g., "standard-rates", "summed-rates", "sectored-rates").
    attr_mgr : AttributeManager
        The attribute manager for the L1B data level.

    Returns
    -------
    dataset : xarray.Dataset
        A processed L1B science dataset.
    """
    logger.info("Creating HIT L1B science datasets")

    dataset = None
    logical_source = None

    # Calculate fractional livetime from the livetime counter
    livetime = livetime_fraction_calculation(l1a_counts_dataset["livetime_counter"])

    # Process counts data to an L1B dataset based on the descriptor
    if descriptor == "standard-rates":
        dataset = process_standard_rates_data(l1a_counts_dataset, livetime)
        logical_source = "imap_hit_l1b_standard-rates"
    elif descriptor == "summed-rates":
        dataset = process_summed_rates_data(l1a_counts_dataset, livetime)
        logical_source = "imap_hit_l1b_summed-rates"
    elif descriptor == "sectored-rates":
        dataset = process_sectored_rates_data(l1a_counts_dataset, livetime)
        logical_source = "imap_hit_l1b_sectored-rates"

    # Update attributes and dimensions
    if dataset and logical_source:
        dataset.attrs = attr_mgr.get_global_attributes(logical_source)
        # TODO: Add CDF attributes to yaml
        for field in dataset.data_vars.keys():
            try:
                # Create a dict of dimensions using the DEPEND_I keys in the attributes
                dims = {
                    key: value
                    for key, value in attr_mgr.get_variable_attributes(field).items()
                    if "DEPEND" in key
                }
                dataset[field].attrs = attr_mgr.get_variable_attributes(field)
                dataset[field].assign_coords(dims)
            except KeyError:
                logger.warning(f"Field {field} not found in attribute manager.")

        # Skip schema check for epoch to prevent attr_mgr from adding the
        # DEPEND_0 attribute which isn't required for epoch
        dataset.epoch.attrs = attr_mgr.get_variable_attributes(
            "epoch", check_schema=False
        )
        logger.info(f"HIT L1B dataset created for {logical_source}")

    return dataset


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


def process_sectored_rates_data(
    l1a_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1A raw counts data into L1B sectored rates.

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
    the sum of livetime values from the previous 10 minutes multiplied by a factor
    15 to account for the different inclination sectors (a single spacecraft
    rotation is split into 15 inclination ranges). See equation 11 in the algorithm
    document.

    NOTE: The L1A counts dataset has complete sets of sectored counts and livetime is
    already shifted to 10 minutes before the counts. This was handled in L1A processing.

    Parameters
    ----------
    l1a_counts_dataset : xr.Dataset
        The L1A counts dataset containing sectored counts.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B sectored rates dataset.
    """
    # TODO - consider refactoring calculate_rates function to handle sectored rates

    # Define particles and coordinates
    particles = ["h", "he4", "cno", "nemgsi", "fe"]

    # Extract relevant data variable names that start with a particle name
    data_vars = [
        str(var)
        for var in l1a_counts_dataset.data_vars
        if any(str(var).startswith(f"{p}_") for p in particles)
    ]

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
                    (counts / (SECTORS * livetime_10min_reshaped)).astype(np.float32),
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


def livetime_fraction_calculation(livetime_counter: xr.DataArray) -> xr.DataArray:
    """
    Calculate livetime fraction from the livetime counter.

    Parameters
    ----------
    livetime_counter : xr.DataArray
        1D array of livetime counter values.

    Returns
    -------
    xr.DataArray
        1D array of livetime fraction values.
    """
    livetime_fraction = xr.zeros_like(livetime_counter, dtype=np.float32)

    # Equation 8 in section 6.2 of the algorithm document
    livetime1 = livetime_counter <= 4101
    livetime2 = (livetime_counter > 4101) & (livetime_counter <= 16000)
    livetime3 = livetime_counter > 16000

    livetime_fraction[livetime1] = livetime_counter[livetime1] * 3.41e-5 + 0.14
    livetime_fraction[livetime2] = livetime_counter[livetime2] * 6.827e-5
    livetime_fraction[livetime3] = livetime_counter[livetime3] * 1.04e-9

    return livetime_fraction
