"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.swe.l1a.swe_science import decompressed_counts
from imap_processing.swe.l1b.swe_l1b_science import deadtime_correction, read_in_flight_cal_data
from imap_processing.swe.utils.swe_constants import GEOMETRIC_FACTORS, ESA_VOLTAGE_ROW_INDEX_DICT

logger = logging.getLogger(__name__)


def decompress_counts(raw_counts: NDArray) -> NDArray:
    """
    Decompress raw counts using a predefined decompression table.

    Parameters
    ----------
    raw_counts : np.ndarray
        Array of raw compressed counts with shape (n_time, n_cem, n_energy_step).

    Returns
    -------
    counts : np.ndarray
        Array of decompressed counts with the same shape as raw_counts.
    """
    decompression_table = np.array([decompressed_counts(i) for i in range(256)])

    # Decompress using the precomputed table
    counts = decompression_table[raw_counts]

    return counts


def prepare_raw_counts(grouped_data: xr.Dataset, group: int) -> np.ndarray:
    """
    Reformat raw counts into a 3D array binned by phi.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Dataset containing grouped i-ALiRT packet data for 1 minute.

    group : int
        Group number.

    Returns
    -------
    raw_counts : np.ndarray
        Array of raw counts with shape (30, 7, 4), where:
        - 30 corresponds to the 30 phi bins.
        - 7 corresponds to the 7 CEM detectors.
        - 4 corresponds to the 4 energy steps.
    """

    raw_counts = np.zeros((30, 7, 4), dtype=np.uint8)

    group_mask = (grouped_data["group"] == group)
    group_data = grouped_data.sel(epoch=group_mask)

    # Phi bins to index mapping (phis wrap at 360, so they all land in 0-29 bins)
    def phi_to_bin(phi):
        return ((phi - 12) // 24) % 30

    for i in range(len(group_data["epoch"])):
        phi_0 = (12 + 24 * i) % 360  # Energy steps 0 and 1
        phi_1 = (24 + 24 * i) % 360  # Energy steps 2 and 3

        phi_0_bin = phi_to_bin(phi_0)
        phi_1_bin = phi_to_bin(phi_1)

        for cem in range(1, 8):  # 7 CEMs
            # swe_cem#_e1 and swe_cem#_e2 -> phi_0
            raw_counts[phi_0_bin, cem - 1, 0] = group_data[f"swe_cem{cem}_e1"].values[i]
            raw_counts[phi_0_bin, cem - 1, 1] = group_data[f"swe_cem{cem}_e2"].values[i]

            # swe_cem#_e3 and swe_cem#_e4 -> phi_1
            raw_counts[phi_1_bin, cem - 1, 2] = group_data[f"swe_cem{cem}_e3"].values[i]
            raw_counts[phi_1_bin, cem - 1, 3] = group_data[f"swe_cem{cem}_e4"].values[i]

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

    # Calculate time in seconds
    # 1 second = 1,000,000 microseconds for swe_acq_sub
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

        # Prepare raw counts array just for this group
        # (60 epochs, 7 CEMs, 4 energy steps)
        raw_counts = prepare_raw_counts(grouped_data, group)

        counts = decompress_counts(raw_counts)
        # acq_duration = 80 milliseconds (hardcode)
        corrected_counts = deadtime_correction(counts, 80 * 10 ^ 3)

        # Grab the latest calibration factor
        in_flight_cal_df = read_in_flight_cal_data()
        latest_cal = in_flight_cal_df.sort_values("met_time").iloc[-1]

        # Same geometric factors as in the L1B processing
        geometric_factors = GEOMETRIC_FACTORS
        # These energies only for I-AliRT
        energy = ESA_VOLTAGE_ROW_INDEX_DICT[11:18]
        n_energy = len(energy)
        # 0.5 sec ~ 12 degree spin angle
        # 2 phi values / sec
        n_phi = 30
        # 7 sensors
        n_cems = 7

        # initialize phase space density and norm counts
        # Phase space density fv in units of s^3/cm^6
        fv = np.zeros((n_energy, n_cems, n_phi))
        norm_counts = np.zeros((n_energy, n_cems, n_phi))

        # 30 phi for each cycle

        for i in range(n_energy):
            for j in range(n_cems):
                for k in range(n_phi):
                    if counts[i][j][k] < 0:
                        fv[i][j][k] = 0.0
                    else:
                        norm_counts[i][j][k] = ccounts[i][j][k] * cal_factor[j] / gg[j]

        # Combine "spin_1" and "spin_2" to get the full cycle data
        # Combine "spin_3" and "spin_4" to get the full cycle data

    return swe_data
