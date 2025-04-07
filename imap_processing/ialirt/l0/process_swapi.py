"""Functions to support I-ALiRT SWAPI processing."""

import logging

import numpy as np
import xarray as xr
from xarray import DataArray

from imap_processing.ialirt.utils.grouping import find_groups

# from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
# from imap_processing.swapi.l2.swapi_l2 import TIME_PER_BIN

logger = logging.getLogger(__name__)


def process_swapi_ialirt(unpacked_data: xr.Dataset) -> dict[str, DataArray]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.

    Returns
    -------
    swapi_data : dict
        Dictionary containing all data variables for SWAPI I-ALiRT product.
    """
    logger.info("Processing SWAPI.")

    sci_dataset = unpacked_data.sortby("epoch", ascending=True)

    grouped_dataset = find_groups(sci_dataset, (0, 11), "swapi_seq_number", "swapi_acq")

    for group in np.unique(grouped_dataset["group"]):
        # Sequence values for the group should be 0-11 with no duplicates.
        seq_values = grouped_dataset["swapi_seq_number"][
            (grouped_dataset["group"] == group)
        ]

        # Ensure no duplicates and all values from 0 to 11 are present
        if not np.array_equal(seq_values.astype(int), np.arange(12)):
            logger.info(
                f"SWAPI group {group} does not contain all sequence values from 0 to "
                f"11 without duplicates."
            )
            continue

    total_packets = len(grouped_dataset["swapi_seq_number"].data)

    # It takes 12 sequence data to make one full SWAPI sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence

    met_values = grouped_dataset["swapi_shcoarse"].data.reshape(total_full_sweeps, 12)[
        :, 0
    ]

    # raw_coin_count = process_sweep_data(grouped_dataset, "coin_cnt")
    # raw_coin_rate = raw_coin_count / TIME_PER_BIN

    swapi_data = {
        "met": met_values
        # more variables to go here
    }

    return swapi_data
