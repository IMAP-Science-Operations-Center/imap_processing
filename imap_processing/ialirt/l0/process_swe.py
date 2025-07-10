"""Functions to support I-ALiRT SWE processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc
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

# Energy bin lookup table (indexed by quarter cycle)
ENERGY_BINS = np.array(
    [
        [1, 5, 7, 3],  # 0 to 14 (Q1)
        [2, 6, 4, 0],  # 15 to 29 (Q2)
        [3, 7, 5, 1],  # 30 to 44 (Q3)
        [0, 4, 6, 2],  # 45 to 59 (Q4)
    ]
)


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

    Notes
    -----
    CEM is channel electron multiplier.
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

    # The first 15 seconds is the first quarter cycle, etc.
    quarter_cycles = np.floor(grouped["swe_seq"] / 15).astype(int)
    e_bins = ENERGY_BINS[quarter_cycles]

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


def find_bin_offsets(
    peak_bins: NDArray, offsets: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """
    Find the bins with offsets from the peak bins.

    Parameters
    ----------
    peak_bins : np.ndarray
        Bins that correspond to the maximum counts at each energy.
    offsets : tuple[int, int]
        Offset values for the bins.

    Returns
    -------
    bin_0 : np.ndarray
        First bin used for the average.
    bin_1 : np.ndarray
        Second bin used for the average.
    """
    # Azimuth has 30 values.
    # Therefore, anything greater than 30 should be wrapped around.
    bin_0, bin_1 = (peak_bins + offsets[0]) % 30, (peak_bins + offsets[1]) % 30

    return bin_0, bin_1


def average_counts(
    peak_bins: NDArray, summed_half_cycle: NDArray, offsets: tuple[int, int]
) -> NDArray:
    """
    Get the counts value for the offset bins at each energy level and average them.

    Parameters
    ----------
    peak_bins : np.ndarray
        Bins that corresponds to the maximum counts at each energy.
    summed_half_cycle : np.ndarray
        Counts summed over the 7 CEM detectors.
    offsets : tuple
        Offset values for the bins.
        Offsets +6 and +8 correspond to +90 degrees.
        Offsets +14 and +16 correspond to 180 degrees.
        Offsets -6 and -8 correspond to -90 degrees.

    Returns
    -------
    avg_counts : np.ndarray
        Average counts of offset bin.
    """
    # Find the bins with offsets from the peak bins.
    bin_0, bin_1 = find_bin_offsets(peak_bins, offsets)

    # Get the counts value for the offset bins at each energy level and average them.
    row_idx = np.arange(len(peak_bins))
    avg_counts = (
        summed_half_cycle[row_idx, bin_0] + summed_half_cycle[row_idx, bin_1]
    ) / 2

    return avg_counts


def find_min_counts(
    summed_half_cycle: NDArray,
) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray, NDArray]]:
    """
    Find min counts (cmin), defined as the minimum of counts_180 and counts_90.

    Parameters
    ----------
    summed_half_cycle : np.ndarray
        Counts summed over the 7 CEM detectors.

    Returns
    -------
    cpeak : np.ndarray
        Maximum counts for each energy level.
    cmin : np.ndarray
        Minimum of counts_neg_90, counts_90, counts_180.
    counts : tuple[np.ndarray, np.ndarray, np.ndarray]
        Counts at +/- 90 and 180 degrees from peak.
    """
    # Find the maximum counts for each energy level
    cpeak = np.max(summed_half_cycle, axis=1)

    # Find the bin that corresponds to the maximum counts at each energy
    peak_bin = np.argmax(summed_half_cycle, axis=1)

    # Find the counts in each offset bins.
    # Offsets +6 and +8 correspond to +90 degrees.
    counts_90 = average_counts(peak_bin, summed_half_cycle, (6, 8))

    # Find the counts in each offset bins.
    # Offsets +14 and +16 correspond to 180 degrees.
    counts_180 = average_counts(peak_bin, summed_half_cycle, (14, 16))

    # Find the counts in each offset bins.
    # Offsets -6 and -8 correspond to -90 degrees.
    counts_neg_90 = average_counts(peak_bin, summed_half_cycle, (-6, -8))

    counts_stacked = np.hstack(
        [
            counts_90[:, np.newaxis],
            counts_180[:, np.newaxis],
            counts_neg_90[:, np.newaxis],
        ]
    )

    # Find the minimum value for each energy level
    cmin = np.min(counts_stacked, axis=1)

    return cpeak, cmin, (counts_neg_90, counts_90, counts_180)


def determine_streaming(
    numerator_1: NDArray,
    numerator_2: NDArray,
    denominator: NDArray,
    threshold: float = 1.75,
) -> NDArray:
    """
    Determine if any energy level satisfies the bidirectional streaming condition.

    Parameters
    ----------
    numerator_1 : np.ndarray
        For the first streaming check : cpeak.
        For the second streaming check : ccem_1.
    numerator_2 : np.ndarray
        For the first streaming check : counts_180.
        For the second streaming check : ccem_7.
    denominator : np.ndarray
        For both streaming checks: cmin.
    threshold : float (optional)
        Threshold value for the streaming condition.

    Returns
    -------
    streaming_flag : np.ndarray
        Array of 1s and 0s indicating if the condition is satisfied.
    """
    ratio_1 = numerator_1 / denominator
    ratio_2 = numerator_2 / denominator

    return ((ratio_1 > threshold) & (ratio_2 > threshold)).astype(int)


def compute_bidirectional(
    streaming_first_half: NDArray,
    streaming_second_half: NDArray,
    min_esa_steps: int = 3,
) -> tuple[int, int]:
    """
    Compute the Bidirectional Electron parameter (BDE).

    Parameters
    ----------
    streaming_first_half : np.ndarray
        Array of 1s and 0s indicating bidirectional streaming for first half-cycle.
    streaming_second_half : np.ndarray
        Array of 1s and 0s indicating bidirectional streaming for second half-cycle.
    min_esa_steps : int (optional)
        Minimum number of ESA steps for bidirectional streaming.
        If either of the half cycles has bidirectional streaming for
        3/8 energies then bde = 1.

    Returns
    -------
    bde : tuple
        Indicator for counter-streaming.
    """
    count_first: int = int(np.sum(streaming_first_half))
    count_second: int = int(np.sum(streaming_second_half))

    return int(count_first >= min_esa_steps), int(count_second >= min_esa_steps)


def azimuthal_check_counterstreaming(
    summed_first_half: NDArray, summed_second_half: NDArray
) -> tuple[int, int]:
    """
    Check if counterstreaming is observed in azimuthal angle direction.

    Parameters
    ----------
    summed_first_half : np.ndarray
        Counts summed over the 7 CEM detectors for first half-cycle.
    summed_second_half : np.ndarray
        Counts summed over the 7 CEM detectors for second half-cycle.

    Returns
    -------
    bde : tuple
        Indicator for counter-streaming.
    """
    # Find peaks, cmin, counts (-90, 90, 180)
    cpeak_first_half, cmin_first_half, counts_first_half = find_min_counts(
        summed_first_half
    )
    cpeak_second_half, cmin_second_half, counts_second_half = find_min_counts(
        summed_second_half
    )

    # First search for counter-streaming
    streaming_first_half = determine_streaming(
        cpeak_first_half, counts_first_half[2], cmin_first_half
    )
    streaming_second_half = determine_streaming(
        cpeak_second_half, counts_second_half[2], cmin_second_half
    )

    # If either of the half cycles has bidirectional streaming
    # for 3/8 energies then bde = 1
    bde_first_search = compute_bidirectional(
        streaming_first_half, streaming_second_half
    )

    return bde_first_search


def polar_check_counterstreaming(
    summed_first_half: NDArray, summed_second_half: NDArray
) -> tuple[int, int]:
    """
    Check if counterstreaming is observed in the polar angle direction.

    Parameters
    ----------
    summed_first_half : np.ndarray
        Counts summed over the azimuth for first half-cycle.
    summed_second_half : np.ndarray
        Counts summed over the azimuth for second half-cycle.

    Returns
    -------
    bde : tuple
        Indicator for counter-streaming.
    """
    # Cmin is the average of the counts in CEMs 3, 4, and 5
    cmin_first_half = summed_first_half[:, 2:5].mean(axis=1)
    cmin_second_half = summed_second_half[:, 2:5].mean(axis=1)

    # Determine if streaming is observed.
    # Note: Bidirectional electron streaming at a given ESA step
    # is identified if both c_cem1/cmin and c_cem7/cmin are > 1.75.
    streaming_first_half = determine_streaming(
        summed_first_half[:, 0], summed_first_half[:, 6], cmin_first_half
    )
    streaming_second_half = determine_streaming(
        summed_second_half[:, 0], summed_second_half[:, 6], cmin_second_half
    )

    # If either of the half cycles has bidirectional streaming
    # for 3/8 energies then bde = 1
    bde_second_search = compute_bidirectional(
        streaming_first_half, streaming_second_half
    )

    return bde_second_search


def process_swe(accumulated_data: xr.Dataset, in_flight_cal_files: list) -> list[dict]:
    """
    Create L1 data dictionary.

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
    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(
        accumulated_data["sc_sclk_sec"], accumulated_data["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    accumulated_data["met"] = met

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

        # Sum over the 7 detectors
        summed_first_half_cem = np.sum(normalized_first_half, axis=1)
        summed_second_half_cem = np.sum(normalized_second_half, axis=1)
        bde_first_search = azimuthal_check_counterstreaming(
            summed_first_half_cem, summed_second_half_cem
        )
        # Sum over azimuth.
        summed_first_half_az = np.sum(normalized_first_half, axis=2)
        summed_second_half_az = np.sum(normalized_second_half, axis=2)
        bde_second_search = polar_check_counterstreaming(
            summed_first_half_az, summed_second_half_az
        )

        # BDE value
        bde_first_half = max(bde_first_search[0], bde_second_search[0])
        bde_second_half = max(bde_first_search[1], bde_second_search[1])

        # For normalized counts each ESA step is summed
        # over both azimuthal and polar angles
        # Sum over CEMs (axis=1) and azimuths (axis=2)
        summed_first = normalized_first_half.sum(axis=(1, 2))
        summed_second = normalized_second_half.sum(axis=(1, 2))

        swe_data.append(
            {
                "apid": 478,
                "met": int(grouped["met"].min()),
                "met_in_utc": met_to_utc(grouped["met"].min()).split(".")[0],
                "ttj2000ns": int(met_to_ttj2000ns(grouped["met"].min())),
                "swe_normalized_counts_half_1_esa": [int(val) for val in summed_first],
                "swe_normalized_counts_half_2_esa": [int(val) for val in summed_second],
                "swe_counterstreaming_electrons": max(bde_first_half, bde_second_half),
            }
        )

    return swe_data
