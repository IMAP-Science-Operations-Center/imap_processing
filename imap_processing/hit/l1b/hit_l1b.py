"""IMAP-HIT L1B data processing."""

import logging
from typing import NamedTuple

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l1b.constants import (
    PARTICLE_ENERGY_RANGE_MAPPING,
    livestim_pulses,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(dependencies: dict, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L1B.

    Processes dependencies needed to create L1B data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of dependencies that are L1A xarray datasets
        for science data and a file path string to an L0 file
        for housekeeping data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of four L1B datasets.
    """
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1b")

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
    raw_counts_dataset: xr.Dataset, attr_mgr: ImapCdfAttributes
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
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset.
    attr_mgr : AttributeManager
        The attribute manager for the L1B data level.

    Returns
    -------
    dataset : list
        The processed L1B science datasets as xarray datasets.
    """
    logger.info("Creating HIT L1B science datasets")

    # Logical sources for the three L1B science products.
    # TODO: add logical sources for other l1b products once processing functions
    #  are written. ""imap_hit_l1b_sectored-rates"
    logical_sources = ["imap_hit_l1b_standard-rates", "imap_hit_l1b_summed-rates"]

    # TODO: Write functions to create the following datasets
    #  Process sectored rates dataset

    # Calculate fractional livetime from the livetime counter
    livetime = raw_counts_dataset["livetime_counter"] / livestim_pulses

    # Create a standard rates dataset
    standard_rates_dataset = process_standard_rates_data(raw_counts_dataset, livetime)

    # Create a summed rates dataset
    summed_rates_dataset = process_summed_rates_data(raw_counts_dataset, livetime)

    l1b_science_datasets = []
    # Update attributes and dimensions
    for dataset, logical_source in zip(
        [standard_rates_dataset, summed_rates_dataset], logical_sources
    ):
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

        l1b_science_datasets.append(dataset)

        logger.info(f"HIT L1B dataset created for {logical_source}")

    return l1b_science_datasets


def process_standard_rates_data(
    raw_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1B standard rates data from raw L1A counts data.

    Parameters
    ----------
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B standard rates dataset.
    """
    # Create a new dataset to store the L1B standard rates
    l1b_standard_rates_dataset = xr.Dataset()

    # Add required coordinates from the raw_counts_dataset
    coords = [
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
    ]
    l1b_standard_rates_dataset = l1b_standard_rates_dataset.assign_coords(
        {coord: raw_counts_dataset.coords[coord] for coord in coords}
    )

    # Add dynamic threshold variable from the L1A raw counts dataset
    l1b_standard_rates_dataset["dynamic_threshold_state"] = raw_counts_dataset[
        "hdr_dynamic_threshold_state"
    ]
    l1b_standard_rates_dataset["dynamic_threshold_state"].attrs = raw_counts_dataset[
        "hdr_dynamic_threshold_state"
    ].attrs

    # Define fields from the raw_counts_dataset to calculate standard rates from
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

    # Calculate standard rates by dividing the raw counts by livetime for
    # data variables with names that contain a substring from a defined
    # list of field names.
    for var in raw_counts_dataset.data_vars:
        if var != "livetime_counter" and any(
            base_var in var for base_var in standard_rate_fields
        ):
            l1b_standard_rates_dataset[var] = raw_counts_dataset[var] / livetime

    return l1b_standard_rates_dataset


def create_particle_data_arrays(
    dataset: xr.Dataset,
    particle: str,
    num_energy_ranges: int,
    epoch_size: int,
) -> xr.Dataset:
    """
    Create empty data arrays for a given particle.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the data arrays to.

    particle : str
        The abbreviated particle name. Valid names are:
            h
            he3
            he4
            he
            c
            n
            o
            ne
            na
            mg
            al
            si
            s
            ar
            ca
            fe
            ni

    num_energy_ranges : int
        Number of energy ranges for the particle.
        Used to define the shape of the data arrays.

    epoch_size : int
        Used to define the shape of the data arrays.

    Returns
    -------
    dataset : xr.Dataset
        The dataset with the added data arrays.
    """
    dataset[f"{particle}"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
        name=f"{particle}",
    )
    dataset[f"{particle}_delta_minus"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
        name=f"{particle}_delta_minus",
    )
    dataset[f"{particle}_delta_plus"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
        name=f"{particle}_delta_plus",
    )
    dataset.coords[f"{particle}_energy_index"] = xr.DataArray(
        np.arange(num_energy_ranges, dtype=np.int8),
        dims=[f"{particle}_energy_index"],
        name=f"{particle}_energy_index",
    )
    return dataset


def calculate_summed_counts(
    raw_counts_dataset: xr.Dataset, count_indices: dict
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Calculate summed counts for a given energy range.

    Parameters
    ----------
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset that contains the l2fgrates, l3fgrates,
        and penfgrates data variables with the particle counts data
        needed for the calculation.

    count_indices : dict
        A dictionary containing the indices for particle counts to sum for a given
        energy range.
        R2=Indices for L2FGRATES, R3=Indices for L3FGRATES, R4=Indices for PENFGRATES.

    Returns
    -------
    summed_counts : xr.DataArray
        The summed counts.

    summed_counts_delta_minus : xr.DataArray
        The summed counts for delta minus uncertainty.

    summed_counts_delta_plus : xr.DataArray
        The summed counts for delta plus uncertainty.
    """
    summed_counts = (
        raw_counts_dataset["l2fgrates"][:, count_indices["R2"]].sum(axis=1)
        + raw_counts_dataset["l3fgrates"][:, count_indices["R3"]].sum(axis=1)
        + raw_counts_dataset["penfgrates"][:, count_indices["R4"]].sum(axis=1)
    )

    summed_counts_delta_minus = (
        raw_counts_dataset["l2fgrates_delta_minus"][:, count_indices["R2"]].sum(axis=1)
        + raw_counts_dataset["l3fgrates_delta_minus"][:, count_indices["R3"]].sum(
            axis=1
        )
        + raw_counts_dataset["penfgrates_delta_minus"][:, count_indices["R4"]].sum(
            axis=1
        )
    )

    summed_counts_delta_plus = (
        raw_counts_dataset["l2fgrates_delta_plus"][:, count_indices["R2"]].sum(axis=1)
        + raw_counts_dataset["l3fgrates_delta_plus"][:, count_indices["R3"]].sum(axis=1)
        + raw_counts_dataset["penfgrates_delta_plus"][:, count_indices["R4"]].sum(
            axis=1
        )
    )

    return summed_counts, summed_counts_delta_minus, summed_counts_delta_plus


class SummedCounts(NamedTuple):
    """A namedtuple to store summed counts and uncertainties."""

    summed_counts: xr.DataArray
    summed_counts_delta_minus: xr.DataArray
    summed_counts_delta_plus: xr.DataArray


def add_rates_to_dataset(
    dataset: xr.Dataset,
    particle: str,
    index: int,
    summed_counts: SummedCounts,
    livetime: xr.DataArray,
) -> xr.Dataset:
    """
    Add summed rates to the dataset.

    This function divides the summed counts by livetime to calculate
    the rates for a given particle then adds the rates to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the rates to.

    particle : str
        The abbreviated particle name. Valid names are:
            h
            he3
            he4
            he
            c
            n
            o
            ne
            na
            mg
            al
            si
            s
            ar
            ca
            fe
            ni

    index : int
        The index of the energy range.

    summed_counts : namedtuple
        A namedtuple containing the summed counts.
        SummedCounts(summed_counts, summed_counts_delta_minus,
                    summed_counts_delta_plus).

    livetime : xr.DataArray
        1D array of livetime values. Shape equals the number of epochs in the dataset.

    Returns
    -------
    dataset: xr.Dataset
        The dataset with the added rates.
    """
    dataset[f"{particle}"][:, index] = (summed_counts.summed_counts / livetime).astype(
        np.float32
    )
    dataset[f"{particle}_delta_minus"][:, index] = (
        summed_counts.summed_counts_delta_minus / livetime
    ).astype(np.float32)
    dataset[f"{particle}_delta_plus"][:, index] = (
        summed_counts.summed_counts_delta_plus / livetime
    ).astype(np.float32)
    return dataset


def add_energy_variables(
    dataset: xr.Dataset,
    particle: str,
    energy_min_values: np.ndarray,
    energy_max_values: np.ndarray,
) -> xr.Dataset:
    """
    Add energy min and max variables to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the energy variables to.
    particle : str
        The particle name.
    energy_min_values : np.ndarray
        The minimum energy values for each energy range.
    energy_max_values : np.ndarray
        The maximum energy values for each energy range.

    Returns
    -------
    xr.Dataset
        The dataset with the added energy variables.
    """
    dataset[f"{particle}_energy_min"] = xr.DataArray(
        data=np.array(energy_min_values, dtype=np.float32),
        dims=[f"{particle}_energy_index"],
        name=f"{particle}_energy_min",
    )
    dataset[f"{particle}_energy_max"] = xr.DataArray(
        data=np.array(energy_max_values, dtype=np.float32),
        dims=[f"{particle}_energy_index"],
        name=f"{particle}_energy_max",
    )
    return dataset


def process_summed_rates_data(
    raw_counts_dataset: xr.Dataset, livetime: xr.DataArray
) -> xr.Dataset:
    """
    Will process L1B summed rates data from raw L1A counts data.

    This function calculates summed rates for each particle type and energy range.
    The counts that are summed come from the l2fgrates, l3fgrates, and penfgrates
    data variables in the L1A counts data. These variables represent counts
    of different detector penetration ranges (Range 2, Range 3, and Range 4
    respectively). Only the energy ranges specified in the
    PARTICLE_ENERGY_RANGE_MAPPING dictionary are included in this product.

    The summed rates are calculated by summing the counts for each energy range and
    dividing by the livetime.

    Parameters
    ----------
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset.

    livetime : xr.DataArray
        1D array of livetime values calculated from the livetime counter.
        Shape equals the number of epochs in the dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B summed rates dataset.
    """
    # Create a new dataset to store the L1B standard rates
    l1b_summed_rates_dataset = xr.Dataset()

    # Assign the epoch coordinate from the l1a dataset
    l1b_summed_rates_dataset = l1b_summed_rates_dataset.assign_coords(
        {"epoch": raw_counts_dataset.coords["epoch"]}
    )

    # TODO: dynamic threshold might not be needed for this product.
    #  Need confirmation from HIT
    # Add dynamic threshold variable from L1A raw counts dataset
    l1b_summed_rates_dataset["dynamic_threshold_state"] = raw_counts_dataset[
        "hdr_dynamic_threshold_state"
    ]
    l1b_summed_rates_dataset["dynamic_threshold_state"].attrs = raw_counts_dataset[
        "hdr_dynamic_threshold_state"
    ].attrs

    # Calculate summed rates for each particle and add them to the dataset
    for particle, energy_ranges in PARTICLE_ENERGY_RANGE_MAPPING.items():
        l1b_summed_rates_dataset = create_particle_data_arrays(
            l1b_summed_rates_dataset,
            particle,
            len(energy_ranges),
            raw_counts_dataset.sizes["epoch"],
        )

        energy_min, energy_max = (
            np.zeros(len(energy_ranges), dtype=np.float32),
            np.zeros(len(energy_ranges), dtype=np.float32),
        )
        for i, energy_range in enumerate(energy_ranges):
            summed_counts, summed_counts_delta_minus, summed_counts_delta_plus = (
                calculate_summed_counts(raw_counts_dataset, energy_range)
            )

            # Create namedtuple to store summed counts and uncertainties
            summed_counts = SummedCounts(
                summed_counts, summed_counts_delta_minus, summed_counts_delta_plus
            )

            l1b_summed_rates_dataset = add_rates_to_dataset(
                l1b_summed_rates_dataset,
                particle,
                i,
                summed_counts,
                livetime,
            )
            energy_min[i], energy_max[i] = (
                energy_range["energy_min"],
                energy_range["energy_max"],
            )

        l1b_summed_rates_dataset = add_energy_variables(
            l1b_summed_rates_dataset, particle, energy_min, energy_max
        )

    return l1b_summed_rates_dataset
