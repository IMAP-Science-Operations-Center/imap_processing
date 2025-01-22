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
        for science data and a file path string to a CCSDS file
        for housekeeping data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of L1B datasets. Total of four datasets.
    """
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1b")

    # Create L1B datasets
    datasets: list = []
    if "imap_hit_l0_raw" in dependencies:
        # Unpack ccsds file to xarray datasets
        packet_file = dependencies["imap_hit_l0_raw"]
        datasets_by_apid = get_datasets_by_apid(packet_file, derived=True)
        # Process housekeeping to l1b.
        datasets.append(
            process_housekeeping_data(
                datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
            )
        )
        logger.info("HIT L1B housekeeping dataset created")
    if "imap_hit_l1a_count-rates" in dependencies:
        # Process science data to L1B
        l1a_counts_dataset = dependencies["imap_hit_l1a_count-rates"]
        # Process standard rates
        datasets.append(process_standard_rates_data(l1a_counts_dataset, attr_mgr))
        # TODO: Process summed rates
        # TODO: Process sectored rates
        pass

    return datasets


def process_standard_rates_data(
    raw_counts_dataset: xr.Dataset, attr_mgr: ImapCdfAttributes
) -> xr.Dataset:
    """
    Will process L1B standard rates data from raw L1A counts data.

    Parameters
    ----------
    raw_counts_dataset : xr.Dataset
        The L1A counts dataset.
    attr_mgr : AttributeManager
        The attribute manager for the data level.

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

    # Define list of fields from the raw_counts_dataset to calculate standard rates
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
    raw_counts_dataset.attrs.update(attr_mgr.get_attrs("imap_hit_l1b_standard-rates"))
    return l1b_standard_rates_dataset
