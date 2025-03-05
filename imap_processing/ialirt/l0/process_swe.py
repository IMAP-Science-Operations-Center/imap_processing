"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.swe.l1a.swe_science import decompressed_counts
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time


logger = logging.getLogger(__name__)


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
        accumulated_data["swe_acq_sec"],
        accumulated_data["swe_acq_sub"],
        1000000
    )
    accumulated_data["time_seconds"] = time_seconds

    grouped_data = find_groups(accumulated_data, (0, 59), "swe_seq",
                               "time_seconds")
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

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]

        # We know we can only have 8 bit numbers input, so iterate over all
        # possibilities once up front
        decompression_table = np.array([decompressed_counts(i) for i in range(256)])

        # Loop through each packet individually with a list comprehension and
        # perform the following steps:
        # 1. Turn the binary string  of 0s and 1s to an int
        # 2. Convert the int into a bytes object of length 1260 (10080 / 8)
        #    Eg. "0000000011110011" --> b'\x00\xf3'
        #    1260 = 15 seconds x 12 energy steps x 7 CEMs
        # 3. Read that bytes data to a numpy array of uint8 through the buffer protocol
        # 4. Reshape the data to 180 x 7
        raw_science_array = np.array(
            [
                np.frombuffer(binary_string, dtype=np.uint8).reshape(
                    180, swe_constants.N_CEMS
                )
                for binary_string in l0_dataset["science_data"].values
            ]
        )

        # Decompress the raw science data using numpy broadcasting logic
        # science_array will be the same shape as raw_science_array (npackets, 180, 7)
        science_array = decompression_table[raw_science_array]

    return swe_data
