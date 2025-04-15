"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.swe.l1a.swe_science import decompressed_counts
from imap_processing.swe.l1b.swe_l1b_science import (
    deadtime_correction,
    read_in_flight_cal_data,
)
from imap_processing.swe.utils.swe_constants import (
    ESA_VOLTAGE_ROW_INDEX_DICT,
    GEOMETRIC_FACTORS,
    N_CEMS,
)
from imap_processing.swe.utils.swe_utils import combine_acquisition_time

logger = logging.getLogger(__name__)


def decompress_counts(raw_counts: NDArray) -> NDArray:
    """
    Perform decompression of raw counts using a predefined decompression table.

    Parameters
    ----------
    raw_counts : np.ndarray
        Array of raw compressed counts with shape (n_energy, n_cem, n_phi).

    Returns
    -------
    counts : np.ndarray
        Array of decompressed counts with the same shape as raw_counts.
    """
    decompression_table = np.array([decompressed_counts(i) for i in range(256)])

    # Decompress using the precomputed table
    counts = decompression_table[raw_counts]

    return counts


def phi_to_bin(phi_values: NDArray) -> NDArray:
    """
    Convert phi values to corresponding bin indices.

    Parameters
    ----------
    phi_values : NDArray
        Array of phi values.

    Returns
    -------
    bin_indices : NDArray
        Array of bin indices.
    """
    # Ensure it wraps correctly within 0-29 bins
    return ((phi_values - 12) // 12) % 30


def prepare_raw_counts(grouped: xr.Dataset, cem_number: int = N_CEMS) -> NDArray:
    """
    Reformat raw counts into a 3D array binned by phi.

    Parameters
    ----------
    grouped : xr.Dataset
        Dataset containing grouped i-ALiRT packet data for 30 seconds.
    cem_number : int
        Number of CEMs (default 7).

    Returns
    -------
    raw_counts : NDArray
        Raw counts with shape (8, 7, 30).

    Notes
    -----
    Array of raw counts with shape (n_energy, n_cem, n_phi), where:
        - 8 corresponds to the 8 energy steps.
        - 7 corresponds to the 7 CEM detectors.
        - 30 corresponds to the 30 phi bins.
    """
    raw_counts = np.zeros((8, cem_number, 30), dtype=np.uint8)

    # Compute phi values and their corresponding bins
    # Example: energy steps 0-1 have the same phi;
    # energy steps 2-3 have the same phi, etc.
    # A depiction of this is shown in Figure 7 of the SWE Algorithm Document.
    phi_values = np.array(
        [
            (12 + 24 * grouped["swe_seq"].values) % 360,  # Energy steps 0 and 1
            (24 + 24 * grouped["swe_seq"].values) % 360,  # Energy steps 2 and 3
        ]
    )
    phi_bins = phi_to_bin(phi_values).astype(int)  # Get phi bin indices

    # Energy bin lookup table (indexed by quarter cycle)
    energy_bins = np.array(
        [
            [1, 5, 7, 3],  # 0-14 (first quarter cycle)
            [2, 6, 4, 0],  # 15-29 (second quarter cycle)
            [3, 7, 5, 1],  # 30-44 (third quarter cycle)
            [0, 4, 6, 2],  # 45-59 (fourth quarter cycle)
        ]
    )

    # The first 15 seconds is the first quarter cycle, etc.
    quarter_cycles = np.floor(grouped["swe_seq"] / 15).astype(int)
    e_bins = energy_bins[quarter_cycles]

    # Populate raw_counts
    for cem in range(1, cem_number + 1):  # 7 CEMs
        e1 = grouped[f"swe_cem{cem}_e1"].values
        e2 = grouped[f"swe_cem{cem}_e2"].values
        e3 = grouped[f"swe_cem{cem}_e3"].values
        e4 = grouped[f"swe_cem{cem}_e4"].values

        # Phi bins 0, 2...(12, 36, ...)
        raw_counts[e_bins[:, 0], cem - 1, phi_bins[0]] = e1
        raw_counts[e_bins[:, 1], cem - 1, phi_bins[0]] = e2
        # Phi bins 1, 3...(24, 48, ...)
        raw_counts[e_bins[:, 2], cem - 1, phi_bins[1]] = e3
        raw_counts[e_bins[:, 3], cem - 1, phi_bins[1]] = e4

    return raw_counts


def get_ialirt_energies() -> list:
    """
    Get the ESA voltages for I-ALiRT.

    Returns
    -------
    energy : list
        List of ESA voltage for I-ALiRT.

    Notes
    -----
    This is a subset of the ESA_VOLTAGE_ROW_INDEX_DICT.
    """
    energy = [k for k, v in ESA_VOLTAGE_ROW_INDEX_DICT.items() if 11 <= v <= 18]

    return energy


def normalize_counts(counts: NDArray, latest_cal: pd.Series) -> NDArray:
    """
    Normalize the counts using the latest calibration factor.

    Parameters
    ----------
    counts : np.ndarray
        Array of counts.
    latest_cal : pd.Series
        Array of latest calibration factors.

    Returns
    -------
    norm_counts : np.ndarray
        Array of normalized counts.
    """
    latest_cal = latest_cal.to_numpy()

    # Norm counts where counts are non-negative
    # TODO: confirm fv is counts with Ruth
    norm_counts = counts * (latest_cal / GEOMETRIC_FACTORS)[:, np.newaxis]
    norm_counts[norm_counts < 0] = 0

    return norm_counts


def process_swe(accumulated_data: xr.Dataset) -> list[dict]:
    """
    Create L1 data dictionary..

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

    # Calculate time in seconds
    time_seconds = combine_acquisition_time(
        accumulated_data["swe_acq_sec"], accumulated_data["swe_acq_sub"]
    )
    accumulated_data["time_seconds"] = time_seconds

    # Get total full cycle data available for processing.
    # There are 60 packets in a set so (0, 59) is the range.
    grouped_data = find_groups(accumulated_data, (0, 59), "swe_seq", "time_seconds")
    unique_groups = np.unique(grouped_data["group"])
    swe_data: list[dict] = []

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
        # Prepare raw counts array just for this group
        # (8 energy steps, 7 CEMs, 30 phi bins)
        group_mask = grouped_data["group"] == group
        grouped = grouped_data.sel(epoch=group_mask)

        # Split into Q1 & Q2 (swe_seq 0-29) and Q3 & Q4 (swe_seq 30-59)
        first_half = grouped.where(grouped["swe_seq"] < 30, drop=True)
        second_half = grouped.where(grouped["swe_seq"] >= 30, drop=True)

        # Prepare raw counts separately for both halves
        raw_counts_first_half = prepare_raw_counts(first_half)
        raw_counts_second_half = prepare_raw_counts(second_half)

        # Decompress the raw counts
        counts_first_half = decompress_counts(raw_counts_first_half)
        counts_second_half = decompress_counts(raw_counts_second_half)

        # Apply the deadtime correction
        # acq_duration = 80 milliseconds
        corrected_first_half = deadtime_correction(counts_first_half, 80 * 10**3)
        corrected_second_half = deadtime_correction(counts_second_half, 80 * 10**3)

        # Grab the latest calibration factor
        in_flight_cal_df = read_in_flight_cal_data()
        latest_cal = in_flight_cal_df.sort_values("met_time").iloc[-1][1::]

        normalized_first_half = normalize_counts(corrected_first_half, latest_cal)
        normalized_second_half = normalize_counts(corrected_second_half, latest_cal)

        # Sum over the 7 detectors
        summed_first_half = np.sum(normalized_first_half, axis=1)  # noqa: F841
        summed_second_half = np.sum(normalized_second_half, axis=1)  # noqa: F841

        # TODO: will continue here

    return swe_data
