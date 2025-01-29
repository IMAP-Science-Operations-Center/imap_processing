"""IMAP-HIT L1B data processing."""

import logging

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
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
        # Process housekeeping to L1B.
        l1b_datasets.append(
            process_housekeeping_data(
                datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
            )
        )
        logger.info("HIT L1B housekeeping dataset created")
    if "imap_hit_l1a_count-rates" in dependencies:
        # Process science data to L1B datasets
        l1a_counts_dataset = dependencies["imap_hit_l1a_count-rates"]
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
    #  are written. "imap_hit_l1b_summed-rates", "imap_hit_l1b_sectored-rates"
    logical_sources = ["imap_hit_l1b_standard-rates"]

    # TODO: Write functions to create the following datasets
    #  Process summed rates dataset
    #  Process sectored rates dataset

    # Create a standard rates dataset
    standard_rates_dataset = process_standard_rates_data(raw_counts_dataset)

    l1b_science_datasets = []
    # Update attributes and dimensions
    for dataset, logical_source in zip([standard_rates_dataset], logical_sources):
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


def process_standard_rates_data(raw_counts_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L1B standard rates data from raw L1A counts data.

    Parameters
    ----------
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset.

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

    # Add dynamic threshold field
    l1b_standard_rates_dataset["dynamic_threshold_state"] = raw_counts_dataset[
        "hdr_dynamic_threshold_state"
    ]

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

    # Calculate livetime from the livetime counter
    livetime = raw_counts_dataset["livetime"] / 270

    # Calculate standard rates by dividing the raw counts by livetime for
    # data variables with names that contain a substring from a defined
    # list of field names.
    for var in raw_counts_dataset.data_vars:
        if var != "livetime" and any(
            base_var in var for base_var in standard_rate_fields
        ):
            l1b_standard_rates_dataset[var] = raw_counts_dataset[var] / livetime

    return l1b_standard_rates_dataset
