"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.swe.l1a.swe_science import decompressed_counts
from imap_processing.swe.l1b.swe_l1b import (
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
    norm_counts = counts * (latest_cal / GEOMETRIC_FACTORS)[:, np.newaxis]
    norm_counts[norm_counts < 0] = 0

    return norm_counts


def find_bin_offsets(peak_bins, offsets):
    """Find the bins with offsets from the peak bins."""
    # Azimuth has 30 values.
    bin_0, bin_1 = (peak_bins + offsets[0]) % 30, (peak_bins + offsets[1]) % 30

    return bin_0, bin_1


def average_values_and_azimuth(peak_bins, summed_half_cycle, azimuth, offsets):
    """Find the counts and azimuth values from the offset bins."""
    # Find the bins with offsets from the peak bins.
    bin_0, bin_1 = find_bin_offsets(peak_bins, offsets)

    # Get the counts value for the offset bins at each energy level and average them.
    row_idx = np.arange(len(peak_bins))
    avg_counts = (
        summed_half_cycle[row_idx, bin_0] + summed_half_cycle[row_idx, bin_1]
    ) / 2
    avg_azimuth = (azimuth[bin_0] + azimuth[bin_1]) / 2

    return avg_counts, avg_azimuth


def find_min_counts(summed_half_cycle):
    # Find the maximum counts for each energy level
    cpeak = np.max(summed_half_cycle, axis=1)

    # Find the bin that corresponds to the maximum counts at each energy
    peak_az_bin = np.argmax(summed_half_cycle, axis=1)

    # Azimuth is always 12 degrees apart.
    azimuth = np.arange(12, 361, 12)
    azimuth_peak = azimuth[peak_az_bin]

    # Find the counts and azimuth values from the offset bins.
    # Bins +6 and +8 correspond to +90 degrees.
    counts_90, azimuth_90 = average_values_and_azimuth(
        peak_az_bin, summed_half_cycle, azimuth, (6, 8)
    )

    # Find the counts and azimuth values from the offset bins.
    # Bins +14 and +16 correspond to 180 degrees.
    counts_180, azimuth_180 = average_values_and_azimuth(
        peak_az_bin, summed_half_cycle, azimuth, (14, 16)
    )

    # Find the counts and azimuth values from the offset bins.
    # Bins +6 and +8 correspond to -90 degrees.
    counts_neg_90, azimuth_neg_90 = average_values_and_azimuth(
        peak_az_bin, summed_half_cycle, azimuth, (-6, -8)
    )

    counts_stacked = np.hstack(
        [
            counts_90[:, np.newaxis],
            counts_180[:, np.newaxis],
            counts_neg_90[:, np.newaxis],
        ]
    )
    azimuths_stacked = np.hstack(
        [
            azimuth_90[:, np.newaxis],
            azimuth_180[:, np.newaxis],
            azimuth_neg_90[:, np.newaxis],
        ]
    )

    # Find the minimum value for each energy level
    min_indices = np.argmin(counts_stacked, axis=1)
    azimuth_cmin = azimuths_stacked[np.arange(len(min_indices)), min_indices]
    cmin = np.min(counts_stacked, axis=1)

    return (
        cpeak,
        cmin,
        (counts_neg_90, counts_90, counts_180),
        (azimuth_neg_90, azimuth_90, azimuth_180),
        azimuth_peak,
        azimuth_cmin,
    )


def determine_streaming(numerator_1, numerator_2, denominator, threshold=1.75):
    """
    Returns True if any energy level satisfies the bidirectional streaming condition:
    (numerator_1 / denominator > threshold) and (numerator_2 / denominator > threshold)
    """
    ratio_1 = numerator_1 / denominator
    ratio_2 = numerator_2 / denominator

    return ((ratio_1 > threshold) & (ratio_2 > threshold)).astype(int)


def compute_bde(streaming_first_half, streaming_second_half, min_esa_steps=3):
    """
    Compute the Bidirectional Electron parameter (BDE).
    """
    count_first = np.sum(streaming_first_half)
    count_second = np.sum(streaming_second_half)

    return int((count_first >= min_esa_steps) or (count_second >= min_esa_steps))


def get_normalized_counts(normalized_first_half, normalized_second_half, time_quarter_cycle_1, time_quarter_cycle_2, time_quarter_cycle_3, time_quarter_cycle_4):

    # Normalized counts as a function of time
    # Sum over CEMs (axis=1) and azimuths (axis=2)
    summed_first = normalized_first_half.sum(axis=(1, 2))  # shape: (8,)
    summed_second = normalized_second_half.sum(axis=(1, 2))
    summed_counts = np.concatenate([summed_first, summed_second])

    times_first = np.full(8, np.nan)
    times_second = np.full(8, np.nan)

    # Assign values at specified indices
    times_first[[1, 5, 7, 3]] = time_quarter_cycle_1
    times_first[[2, 6, 4, 0]] = time_quarter_cycle_2
    times_second[[3, 7, 5, 1]] = time_quarter_cycle_3
    times_second[[0, 4, 6, 2]] = time_quarter_cycle_4
    summed_times = np.concatenate([times_first, times_second])

    return summed_counts, summed_times


def first_check_counterstreaming(normalized_first_half, normalized_second_half):
    # Sum over the 7 detectors
    summed_first_half = np.sum(normalized_first_half, axis=1)
    summed_second_half = np.sum(normalized_second_half, axis=1)

    # Find peaks, counts (-90, 90, 180), and azimuth (-90, 90, 180) values
    (
        cpeak_first_half,
        cmin_first_half,
        counts_first_half,
        azimuth_first_half,
        azimuth_peak,
        azimuth_cmin,
    ) = find_min_counts(summed_first_half)
    (
        cpeak_second_half,
        cmin_second_half,
        counts_second_half,
        azimuth_second_half,
        azimuth_peak,
        azimuth_cmin,
    ) = find_min_counts(summed_second_half)

    # First search for counter-streaming: determine_streaming_summed_cems
    streaming_summed_cems_first_half = determine_streaming(
        cpeak_first_half, counts_first_half[2], cmin_first_half
    )
    streaming_summed_cems_second_half = determine_streaming(
        cpeak_second_half, counts_second_half[2], cmin_second_half
    )

    # If either of the half cycles has bidirectional streaming for 3/5 energies then bde = 1
    bde_first_search = compute_bde(
        streaming_summed_cems_first_half, streaming_summed_cems_second_half
    )

    return bde_first_search


def second_check_counterstreaming(normalized_first_half, normalized_second_half):

    # Sum over azimuth.
    az_first_half = np.sum(normalized_first_half, axis=2)
    az_second_half = np.sum(normalized_second_half, axis=2)

    # Cmin is the average of the counts in CEMs 3, 4, and 5
    cmin_first_half = az_first_half[:, 2:5].mean(axis=1)
    cmin_second_half = az_second_half[:, 2:5].mean(axis=1)

    # determine_streaming_summed_azimuths
    streaming_summed_azimuths_first_half = determine_streaming(
        az_first_half[:, 0], az_second_half[:, 6], cmin_first_half
    )
    streaming_summed_azimuths_second_half = determine_streaming(
        az_second_half[:, 0], az_second_half[:, 6], cmin_second_half
    )

    bde_second_search = compute_bde(
        streaming_summed_azimuths_first_half, streaming_summed_azimuths_second_half
    )

    return bde_second_search


def process_swe(accumulated_data: xr.Dataset, in_flight_cal_files: list) -> list[dict]:
    """
    Create L1 data dictionary..

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.
    in_flight_cal_files : list
        List of path to the in-flight calibration files.

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
            logger.info(
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
        in_flight_cal_df = read_in_flight_cal_data(in_flight_cal_files)
        latest_cal = in_flight_cal_df.sort_values("met_time").iloc[-1][1::]

        normalized_first_half = normalize_counts(corrected_first_half, latest_cal)
        normalized_second_half = normalize_counts(corrected_second_half, latest_cal)

        bde_first_search = first_check_counterstreaming(
            normalized_first_half, normalized_second_half
        )
        bde_second_search = second_check_counterstreaming(
            normalized_first_half, normalized_second_half
        )

        # BDE value
        bde = max(bde_first_search, bde_second_search)

        summed_counts, summed_times = get_normalized_counts(
            normalized_first_half,
            normalized_second_half,
            first_half["time_seconds"].min(),
            first_half["time_seconds"].max(),
            second_half["time_seconds"].min(),
            second_half["time_seconds"].max(),
        )

        swe_data.append(
            {
                "met": summed_times,
                "normalized_counts": summed_counts,
                "bde": bde,
            }
        )

    return swe_data
