"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.swe.l1a.swe_science import decompressed_counts
from imap_processing.swe.l1b.swe_l1b_science import deadtime_correction

logger = logging.getLogger(__name__)


def decompress_counts(raw_counts) -> int:
    decompression_table = np.array([decompressed_counts(i) for i in range(256)])

    # Decompress using the precomputed table
    counts = decompression_table[raw_counts]

    return counts


def prepare_raw_counts(grouped_data, group):
    # Prepare raw counts array just for this group (60 epochs, 7 CEMs, 4 energy steps)
    raw_counts = np.zeros((60, 7, 4), dtype=np.uint8)

    # Extract counts for this group
    for cem in range(1, 8):
        for e in range(1, 5):
            key = f"swe_cem{cem}_e{e}"

            # Slice out just the data for this group
            raw_counts[:, cem - 1, e - 1] = grouped_data[key][
                (grouped_data["group"] == group).values
            ].values

    return raw_counts


def process_swe(accumulated_data: xr.Dataset) -> list[dict]:
    """
    Process SWE.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.

    Returns
    -------
    swe_data : list[dict]
        Dictionaries of the parsed data product.
    """
    logger.info("Processing SWE.")

    time_seconds = calculate_time(
        accumulated_data["swe_acq_sec"], accumulated_data["swe_acq_sub"], 1000000
    )
    accumulated_data["time_seconds"] = time_seconds

    grouped_data = find_groups(accumulated_data, (0, 59), "swe_seq", "time_seconds")
    unique_groups = np.unique(grouped_data["group"])
    swe_data = []

    for group in unique_groups:
        # Sequence values for the group should be 0-59 with no duplicates.
        seq_values = grouped_data["swe_seq"][(grouped_data["group"] == group).values]

        # Ensure no duplicates and all values from 0 to 59 are present
        if not np.array_equal(seq_values, np.arange(60)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"59 without duplicates."
            )
            continue

        # Prepare raw counts array just for this group (60 epochs, 7 CEMs, 4 energy steps)
        raw_counts = prepare_raw_counts(grouped_data, group)

        counts = decompress_counts(raw_counts)
        # acq_duration = 80 milliseconds (hardcode)
        corrected_counts = deadtime_correction(counts, 80 * 10 ^ 3)

        # TODO: start with normalizing the counts based on the
        #  geometric factors for each CEM detector

    return swe_data
