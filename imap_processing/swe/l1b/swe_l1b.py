"""Contains code to perform SWE L1b science processing."""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from imap_data_access.processing_input import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swe.utils import swe_constants
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
    calculate_data_acquisition_time,
    combine_acquisition_time,
    read_lookup_table,
)
from imap_processing.utils import convert_raw_to_eu, packet_file_to_datasets

logger = logging.getLogger(__name__)


def get_esa_dataframe(esa_table_number: int) -> pd.DataFrame:
    """
    Read lookup table from file.

    Parameters
    ----------
    esa_table_number : int
        ESA table index number.

    Returns
    -------
    esa_steps : pandas.DataFrame
        ESA table_number and its associated values.
    """
    if esa_table_number not in [0, 1]:
        raise ValueError(f"Unknown ESA table number {esa_table_number}")

    # Get the lookup table DataFrame
    lookup_table = read_lookup_table()

    esa_steps = lookup_table.loc[lookup_table["table_index"] == esa_table_number]
    return esa_steps


def deadtime_correction(
    counts: np.ndarray, acq_duration: Union[int, npt.NDArray]
) -> npt.NDArray:
    """
    Calculate deadtime correction.

    Deadtime correction is a technique used in various fields, including
    nuclear physics, radiation detection, and particle counting, to compensate
    for the effects of the time period during which a detector is not able to
    record new events or measurements after detecting a previous event.
    This "deadtime" is essentially the time during which the detector is
    recovering from the previous detection and is unable to detect new events.

    In particle detectors, there is a finite time required for the detector to
    reset or recover after detecting a particle. During this deadtime, any
    subsequent particles that may have arrived go undetected. As a result,
    the recorded count rate appears to be lower than the actual count rate.

    Deadtime correction involves mathematically adjusting the measured count
    rates to compensate for this deadtime effect. This correction is crucial
    when dealing with high-intensity sources or particle fluxes, as the deadtime
    can significantly affect the accuracy of the measurements.

    Deadtime correction is important to ensure accurate measurements and data
    analysis in fields where event detection rates are high and where every
    detected event is critical for understanding physical processes.

    Parameters
    ----------
    counts : numpy.ndarray
        Counts data before deadtime corrections.
    acq_duration : int or numpy.ndarray
        This is ACQ_DURATION from science packet. acq_duration is in microseconds.

    Returns
    -------
    corrected_count : numpy.ndarray
        Corrected counts.
    """
    # deadtime is 360 ns
    deadtime = 360e-9
    if isinstance(acq_duration, int):
        # Convert acq_duration to a numpy array for consistency
        acq_duration = np.array([acq_duration])
    correct = 1.0 - (deadtime * (counts / (acq_duration[..., np.newaxis] * 1e-6)))
    # NOTE: 0.1 is defined in SWE algorithm document. It says
    # 'arbitrary x10 cutoff' in the document.
    correct = np.maximum(0.1, correct)
    corrected_count = counts.astype(np.float64) / correct
    return corrected_count


def convert_counts_to_rate(data: np.ndarray, acq_duration: np.ndarray) -> npt.NDArray:
    """
    Convert counts to rate using sampling time.

    acq_duration is ACQ_DURATION from science packet.

    Parameters
    ----------
    data : numpy.ndarray
        Counts data.
    acq_duration : numpy.ndarray
        Acquisition duration. acq_duration is in microseconds.

    Returns
    -------
    numpy.ndarray
        Count rates array in seconds.
    """
    # Convert microseconds to seconds without modifying the original acq_duration
    acq_duration_sec = acq_duration * 1e-6

    # Ensure acq_duration_sec is broadcastable to data
    if acq_duration_sec.ndim < data.ndim:
        acq_duration_sec = acq_duration_sec[
            ..., np.newaxis
        ]  # Add a new axis for broadcasting

    # Perform element-wise division
    count_rate = data.astype(np.float64) / acq_duration_sec
    return count_rate


def read_in_flight_cal_data(in_flight_cal_files: list) -> pd.DataFrame:
    """
    Read in-flight calibration data.

    In-flight calibration data file will contain rows where each line
    has 8 numbers, with the first being a time stamp in MET, and the next
    7 being the factors for the 7 detectors.

    This file will be updated weekly with new calibration data. In other
    words, one line of data will be added each week to the existing file.
    File will be in CSV format. Processing won't be kicked off until there
    is in-flight calibration data that covers science data.

    Parameters
    ----------
    in_flight_cal_files : list
        List of in-flight calibration files.

    Returns
    -------
    in_flight_cal_df : pandas.DataFrame
        DataFrame with in-flight calibration data.
    """
    column_names = [
        "met_time",
        "cem1",
        "cem2",
        "cem3",
        "cem4",
        "cem5",
        "cem6",
        "cem7",
    ]
    in_flight_cal_df = pd.concat(
        [
            pd.read_csv(file_path, header=0, names=column_names)
            for file_path in in_flight_cal_files
        ]
    )
    # Drop duplicates and keep only last occurrence
    in_flight_cal_df = in_flight_cal_df.drop_duplicates(
        subset=["met_time"], keep="last"
    )
    # Sort by 'met_time' column
    in_flight_cal_df = in_flight_cal_df.sort_values(by="met_time")
    return in_flight_cal_df


def calculate_calibration_factor(
    acquisition_times: np.ndarray, cal_times: np.ndarray, cal_data: np.ndarray
) -> npt.NDArray:
    """
    Calculate calibration factor using linear interpolation.

    Steps to calculate calibration factor:
        1. Convert input time to match time format in the calibration data file.
           Both times should be in S/C MET time.
        2. Find the nearest in time calibration data point.
        3. Linear interpolate between those two nearest time and get factor for
           input time.

    Parameters
    ----------
    acquisition_times : numpy.ndarray
        Data points to interpolate. Shape is (N_ESA_STEPS, N_ANGLE_SECTORS).
    cal_times : numpy.ndarray
        X-coordinates data points. Calibration times. Shape is (n,).
    cal_data : numpy.ndarray
        Y-coordinates data points. Calibration data of corresponding cal_times.
        Shape is (n, N_CEMS).

    Returns
    -------
    calibration_factor : numpy.ndarray
        Calibration factor for each CEM detector. Shape is
        (N_ESA_STEPS, N_ANGLE_SECTORS, N_CEMS) where last 7 dimension
        contains calibration factor for each CEM detector.
    """
    # Raise error if there is no pre or post time in cal_times. SWE does not
    # want to extrapolate calibration data.
    if (
        acquisition_times.min() < cal_times.min()
        or acquisition_times.max() > cal_times.max()
    ):
        raise ValueError(
            f"Acquisition min/max times: {acquisition_times.min()} to "
            f"{acquisition_times.max()}. "
            f"Calibration min/max times: {cal_times.min()} to {cal_times.max()}. "
            "Acquisition times should be within calibration time range."
        )

    # This line of code finds the indices of acquisition_times in cal_times where
    # acquisition_times should be inserted to maintain order. As a result, it finds
    # its nearest pre and post time from cal_times.
    input_time_indices = np.searchsorted(cal_times, acquisition_times)

    # Assign to a variable for better readability
    x = acquisition_times
    xp = cal_times
    fp = cal_data

    # Given this situation which will be the case for SWE data
    # where data will fall in between two calibration times and
    # not be exactly equal to any calibration time,
    #   >>> a = [1, 2, 3]
    #   >>> np.searchsorted(a, [2.5])
    #   array([2])
    # we need to use (j - 1) to get pre time indices. (j-1) is
    # pre time indices and j is post time indices.
    j = input_time_indices
    w = (x - xp[j - 1]) / (xp[j] - xp[j - 1])
    return fp[j - 1] + w[..., None] * (fp[j] - fp[j - 1])


def apply_in_flight_calibration(
    corrected_counts: np.ndarray,
    acquisition_time: np.ndarray,
    in_flight_cal_files: list,
) -> npt.NDArray:
    """
    Apply in flight calibration to full cycle data.

    These factors are used to account for changes in gain with time.

    They are derived from the weekly electron calibration data.

    Parameters
    ----------
    corrected_counts : numpy.ndarray
        Corrected count of full cycle data. Data shape is
        (N_ESA_STEPS, N_ANGLE_SECTORS, N_CEMS).
    acquisition_time : numpy.ndarray
        Acquisition time of full cycle data. Data shape is
        (N_ESA_STEPS, N_ANGLE_SECTORS).
    in_flight_cal_files : list
        List of in-flight calibration files.

    Returns
    -------
    corrected_counts : numpy.ndarray
        Corrected count of full cycle data after applying in-flight calibration.
        Array shape is (N_ESA_STEPS, N_ANGLE_SECTORS, N_CEMS).
    """
    # Read in in-flight calibration data
    in_flight_cal_df = read_in_flight_cal_data(in_flight_cal_files)
    # calculate calibration factor.
    # return shape of calculate_calibration_factor is
    # (N_ESA_STEPS, N_ANGLE_SECTORS, N_CEMS) where
    # last 7 dimension contains calibration factor for each CEM detector.
    cal_factor = calculate_calibration_factor(
        acquisition_time,
        in_flight_cal_df["met_time"].values,
        in_flight_cal_df.iloc[:, 1:].values,
    )
    # Apply to full cycle data
    return corrected_counts.astype(np.float64) * cal_factor


def find_cycle_starts(cycles: np.ndarray) -> npt.NDArray:
    """
    Find index of where new cycle started.

    Brandon Stone helped developed this algorithm.

    Parameters
    ----------
    cycles : numpy.ndarray
        Array that contains quarter cycle information.

    Returns
    -------
    first_quarter_indices : numpy.ndarray
        Array of indices of start cycle.
    """
    if cycles.size < swe_constants.N_QUARTER_CYCLES:
        return np.array([], np.int64)

    # calculate difference between consecutive cycles
    diff = cycles[1:] - cycles[:-1]

    # This uses sliding window to find index where cycle starts.
    # This is what this below code line is doing:
    # [1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0]      # Is cycle zero?
    # [1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1]      # Next diff is one?
    #   [1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0]    # Next diff is one?
    #     [0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0 0]  # Next diff is one?
    #
    # [0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0]      # And all?
    ione = diff == 1
    valid = (cycles == 0)[:-3] & ione[:-2] & ione[1:-1] & ione[2:]
    first_quarter_indices = np.where(valid)[0]
    return first_quarter_indices


def get_indices_of_full_cycles(quarter_cycle: np.ndarray) -> npt.NDArray:
    """
    Get indices of full cycles.

    Parameters
    ----------
    quarter_cycle : numpy.ndarray
        Array that contains quarter cycles information.

    Returns
    -------
    full_cycles_indices : numpy.ndarray
        1D array with indices of full cycle data.
    """
    indices_of_start = find_cycle_starts(quarter_cycle)
    # indices_of_start[..., None] creates array of shape(n, 1).
    #   Eg. [[3], [8]]
    # np.arange(4)[None, ...] creates array of shape(1, 4)
    #   Eg. [[0, 1, 2, 3]]
    # then we add both of them together to get an array of shape(n, 4)
    #   Eg. [[3, 4, 5, 6], [8, 9, 10, 11]]
    full_cycles_indices = (
        indices_of_start[..., None]
        + np.arange(swe_constants.N_QUARTER_CYCLES)[None, ...]
    )
    return full_cycles_indices.reshape(-1)


def get_esa_energy_pattern(esa_lut_file: Path, esa_table_num: int = 0) -> npt.NDArray:
    """
    Get energy in the checkerboard pattern of a full cycle of SWE data.

    This uses ESA Table index number to look up which pattern to use from
    ESA LUT. This is used in L2 to process data further.

    Parameters
    ----------
    esa_lut_file : pathlib.Path
        ESA LUT file.
    esa_table_num : int
        ESA table number. Default is 0.

    Returns
    -------
    energy_pattern : numpy.ndarray
        (esa_step, spin_sector) array with energies of cycle data.
    """
    esa_lut_df = pd.read_csv(esa_lut_file)
    # Get the pattern from the ESA LUT
    esa_table_df = esa_lut_df[esa_lut_df["table_idx"] == esa_table_num]

    # Now define variable to store pattern for the first two columns
    # because that pattern is repeated in the rest of the columns.
    first_two_columns = np.zeros((swe_constants.N_ESA_STEPS, 2), dtype=np.float64)
    # Get row indices of all four quarter cycles. Then minus 1 to get
    # the row indices in 0-23 instead of 1-24.
    cycle_row_indices = esa_table_df["v_index"].values - 1
    esa_v = esa_table_df["esa_v"].values
    # Reshaping the 'v_index' into 4 x 12 gets 12 repeated row_indices and
    # energy steps of each quarter cycle
    row_indices = cycle_row_indices.reshape(4, 12)
    esa_v = esa_v.reshape(4, 12)
    for i in range(4):
        # Split each quarter's 12 steps into 2 x 6 blocks for even
        # and odd columns
        even_odd_column_info = row_indices[i].reshape(2, 6)
        even_row_indices = even_odd_column_info[0]
        odd_row_indices = even_odd_column_info[1]

        # Get even and odd column's ESA voltage information
        esa_v_info = esa_v[i].reshape(2, 6)
        first_two_columns[even_row_indices, 0] = esa_v_info[0]
        first_two_columns[odd_row_indices, 1] = esa_v_info[1]

    # Repeat the first 2 column pattern 15 times across 30 columns
    # (2 columns x 15 = 30)
    energy_pattern = np.tile(first_two_columns, (1, 15))

    # Convert
    return energy_pattern


def get_checker_board_pattern(
    esa_lut_file: pd.DataFrame, esa_table_num: int = 0
) -> npt.NDArray:
    """
    Generate the checkerboard pattern index map for a full cycle of SWE data.

    Find indices of where full cycle data goes in the checkerboard pattern.
    This is used to populate full cycle data in the full cycle data array.
    This uses ESA Table index number to look up which pattern to use from
    ESA LUT.

    Parameters
    ----------
    esa_lut_file : pathlib.Path
        ESA LUT file.
    esa_table_num : int
        ESA table number. Default is 0.

    Returns
    -------
    checkerboard_pattern : numpy.ndarray
        (esa_step * spin_sector) array with indices of where each cycle data goes in.
    """
    esa_lut_df = pd.read_csv(esa_lut_file)
    # Get the pattern from the ESA LUT
    esa_table_df = esa_lut_df[esa_lut_df["table_idx"] == esa_table_num]

    # Now define variable to store pattern for the first two columns
    # because that pattern is repeated in the rest of the columns.
    first_two_columns = np.zeros((24, 2), dtype=np.int64)
    # Get row indices of all four quarter cycles. Then minus 1 to get
    # the row indices in 0-23 instead of 1-24.
    cycle_row_indices = esa_table_df["v_index"].values - 1
    esa_step = esa_table_df["esa_step"].values
    # Reshaping the 'v_index' into 4 x 12 gets 12 repeated row_indices and
    # energy steps of each quarter cycle
    row_indices = cycle_row_indices.reshape(4, 12)
    esa_step = esa_step.reshape(4, 12)
    for i in range(4):
        # Split each quarter's 12 steps into 2 x 6 blocks for even
        # and odd columns
        even_odd_column_info = row_indices[i].reshape(2, 6)
        even_row_indices = even_odd_column_info[0]
        odd_row_indices = even_odd_column_info[1]

        # Starting ESA step value for this quarter cycle. Eg.
        # 0, 180, 360, 540
        start_esa_step = esa_step[i][0]
        # Populate the first two columns of the checkerboard pattern
        # using the start_esa_step value. Eg.
        #  Even row indices: 0, 1, 2, 3, 4, 5
        #  Odd row indices: 6, 7, 8, 9, 10, 11
        first_two_columns[even_row_indices, 0] = np.arange(6) + start_esa_step
        first_two_columns[odd_row_indices, 1] = np.arange(6) + start_esa_step + 6

    # Repeat the first 2 column pattern 15 times across 30 columns
    # (2 columns x 15 = 30)
    base_pattern = np.tile(first_two_columns, (1, 15))

    # Generate increment offsets: [0, 0, 12, 12, ..., 168, 168] -
    # shape: (30,)
    column_offsets = np.repeat(np.arange(15) * 12, 2)
    increment_by = np.tile(column_offsets, (24, 1))

    # Final checkerboard pattern with index offsets applied
    checkerboard_pattern = base_pattern + increment_by
    return checkerboard_pattern


def populated_data_in_checkerboard_pattern(
    data_ds: xr.Dataset, checkerboard_pattern: npt.NDArray
) -> dict:
    """
    Put input data in the checkerboard pattern.

    Put these data variables from l1a data into the checkerboard pattern:
       a. science_data
       b. acq_start_coarse
       c. acq_start_fine
       d. acq_duration
       e. settle_duration
       f. esa_steps_number (This is created in the code and not from science packet)

       These last five variables are used to calculate acquisition time of each
       count data. Acquisition time and duration are carried in l1b for level 2
       and 3 processing.

    Parameters
    ----------
    data_ds : xarray.Dataset
        Input data to be populated in the checkerboard pattern.
    checkerboard_pattern : numpy.ndarray
        Array with indices of where each cycle data goes in the checkerboard
        pattern.

    Returns
    -------
    var_names : dict
        Dictionary with subset data populated in the checkerboard pattern.
    """
    # Flatten with top-down and left-right order. This will be used as
    # indices to take and put data in the checkerboard pattern.
    checkerboard_pattern = checkerboard_pattern.flatten(order="F")

    # Variables that need to be put in the checkerboard pattern
    var_names = {
        "science_data": np.empty(
            (
                0,
                swe_constants.N_ESA_STEPS,
                swe_constants.N_ANGLE_SECTORS,
                swe_constants.N_CEMS,
            )
        ),
        "acq_start_coarse": np.empty(
            (0, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        ),
        "acq_start_fine": np.empty(
            (0, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        ),
        "acq_duration": np.empty(
            (0, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        ),
        "settle_duration": np.empty(
            (0, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        ),
        "esa_step_number": np.empty(
            (0, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        ),
    }

    for var_name in var_names:
        # Reshape the data of input variable for easier processing.
        if var_name == "science_data":
            # Science data shape before reshaping is
            #   (number of packets, 180, 7)
            # Reshape it to
            #   (number of full cycle, 720, N_CEMS)
            data = data_ds[var_name].data.reshape(
                -1,
                swe_constants.N_QUARTER_CYCLES * swe_constants.N_QUARTER_CYCLE_STEPS,
                swe_constants.N_CEMS,
            )
            # Apply the checkerboard pattern directly
            populated_data = data[:, checkerboard_pattern, :]
            # Reshape back into (n, 24, 30, 7)
            populated_data = populated_data.reshape(
                -1,
                swe_constants.N_ESA_STEPS,
                swe_constants.N_ANGLE_SECTORS,
                swe_constants.N_CEMS,
                order="F",
            )
        elif var_name == "esa_step_number":
            # This needs to be created to capture information about esa step number
            # from 0 to 179 for each quarter cycle. This is used to calculate data
            # acquisition time.
            epoch_data = data_ds["epoch"].data
            total_cycles = epoch_data.reshape(-1, swe_constants.N_QUARTER_CYCLES).shape[
                0
            ]
            # Now repeat this pattern n number of cycles
            data = np.tile(
                np.tile(
                    np.arange(swe_constants.N_QUARTER_CYCLE_STEPS),
                    swe_constants.N_QUARTER_CYCLES,
                ),
                (total_cycles, 1),
            )
            # Apply the checkerboard pattern directly
            populated_data = data[:, checkerboard_pattern]
            # Reshape back into (n, 24, 30)
            populated_data = populated_data.reshape(
                -1, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS, order="F"
            )
        else:
            # Input shape is number of packets. Reshape it to
            #   (number of full cycle, 4)
            # This is because we have the same value for each quarter
            # cycle.
            data = data_ds[var_name].data.reshape(-1, swe_constants.N_QUARTER_CYCLES)
            # Repeat the data 180 times to match the checkerboard pattern
            data = np.repeat(data, swe_constants.N_QUARTER_CYCLE_STEPS).reshape(-1, 720)
            # Apply the checkerboard pattern directly
            populated_data = data[:, checkerboard_pattern]
            # Reshape back into (n, 24, 30)
            populated_data = populated_data.reshape(
                -1, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS, order="F"
            )

        # Save the populated data in the dictionary
        var_names[var_name] = populated_data

    return var_names


def filter_full_cycle_data(
    full_cycle_data_indices: np.ndarray, l1a_data: xr.Dataset
) -> xr.Dataset:
    """
    Filter metadata and science of packets that makes full cycles.

    Parameters
    ----------
    full_cycle_data_indices : numpy.ndarray
        Array with indices of full cycles.
    l1a_data : xarray.Dataset
        L1A dataset.

    Returns
    -------
    l1a_data : xarray.Dataset
        L1A dataset with filtered metadata.
    """
    for key, value in l1a_data.items():
        l1a_data[key] = value.data[full_cycle_data_indices]
    return l1a_data


def swe_l1b_science(dependencies: ProcessingInputCollection) -> xr.Dataset:
    """
    SWE l1b science processing.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        Object containing lists of dependencies that CLI dependency
        parameter received.

    Returns
    -------
    dataset : xarray.Dataset
        Processed l1b science data.
    """
    # Read science data
    science_files = dependencies.get_file_paths(descriptor="sci")
    l1a_data = load_cdf(science_files[0])

    total_packets = len(l1a_data["science_data"].data)

    l1a_data_copy = l1a_data.copy(deep=True)

    # First convert some science data to engineering units
    # ---------------------------------------------------------------
    apid = int(l1a_data_copy.attrs["packet_apid"])

    # convert value from raw to engineering units as needed
    conversion_table_path = dependencies.get_file_paths(descriptor="eu-conversion")[0]
    # Look up packet name from APID
    packet_name = next(packet for packet in SWEAPID if packet.value == apid)

    # Convert raw data to engineering units as needed
    l1a_data_copy = convert_raw_to_eu(
        l1a_data_copy,
        conversion_table_path=conversion_table_path,
        packet_name=packet_name.name,
    )

    # Filter out all in-flight calibration data
    # -----------------------------------------
    # If ESA lookup table number is in-flight calibration
    # mode, then skip all those data per SWE teams specification.
    # SWE team only wants in-flight calibration data to be processed
    # upto l1a. In-flight calibration data looks same as science data
    # but it only measures one energy or specific energy steps during
    # the whole duration. Right now, only index 0 in LUT collects
    # science data.
    science_data = l1a_data_copy["esa_table_num"].data == 0
    # Filter out all in-flight calibration data
    l1a_data_copy = l1a_data_copy.isel({"epoch": science_data})

    full_cycle_data_indices = get_indices_of_full_cycles(
        l1a_data_copy["quarter_cycle"].data
    )
    logger.debug(
        f"Quarter cycle data before filtering: {l1a_data_copy['quarter_cycle'].data}"
    )

    # Delete Raw Science Data from l1b and onwards
    del l1a_data_copy["raw_science_data"]

    if full_cycle_data_indices.size == 0:
        # Log that no data is found for science data
        logger.info("No full cycle data found. Skipping.")
        return None

    # In this case, we found incomplete cycle data. We need to filter
    # out all the data that does not make a full cycle.
    if len(full_cycle_data_indices) != total_packets:
        # Filter metadata and science data of packets that makes full cycles
        full_cycle_l1a_data = l1a_data_copy.isel({"epoch": full_cycle_data_indices})

        # Update total packets
        total_packets = len(full_cycle_data_indices)
        logger.debug(
            "Quarters cycle after filtering: "
            f"{full_cycle_l1a_data['quarter_cycle'].data}"
        )
        if len(full_cycle_data_indices) != len(
            full_cycle_l1a_data["quarter_cycle"].data
        ):
            raise ValueError(
                "Error: full cycle data indices and filtered quarter cycle data size "
                "mismatch"
            )

    # Main science processing steps
    # ---------------------------------------------------------------
    # 1. Populate data in the checkerboard pattern. This can return
    #    data in a dictionary.
    # 2. Apply deadtime correction to each count data
    # 3. Apply in-flight calibration to count data
    # 4. Convert counts to rate using acquisition duration

    # Read ESA lookup table
    esa_lut_files = dependencies.get_file_paths(descriptor="esa-lut")
    if len(esa_lut_files) > 1:
        logger.warning(
            f"More than one ESA lookup table file found: {esa_lut_files}. "
            "Using the first one."
        )

    # Get checkerboard pattern
    checkerboard_pattern = get_checker_board_pattern(esa_lut_files[0])

    # Put data in the checkerboard pattern
    populated_data = populated_data_in_checkerboard_pattern(
        full_cycle_l1a_data, checkerboard_pattern
    )
    acq_duration = populated_data["acq_duration"]
    acq_start_time = combine_acquisition_time(
        populated_data["acq_start_coarse"],
        populated_data["acq_start_fine"],
    )
    acq_time = calculate_data_acquisition_time(
        acq_start_time,
        populated_data["esa_step_number"],
        acq_duration,
        populated_data["settle_duration"],
    )
    corrected_count = deadtime_correction(populated_data["science_data"], acq_duration)

    # Read in-flight calibration data
    in_flight_cal_files = dependencies.get_file_paths(descriptor="l1b-in-flight-cal")

    inflight_applied_count = apply_in_flight_calibration(
        corrected_count, acq_time, in_flight_cal_files
    )

    count_rate = convert_counts_to_rate(inflight_applied_count, acq_duration)

    # Statistical uncertainty is sqrt(decompressed counts)
    # TODO: Update this if SWE like to include deadtime correciton.
    counts_stat_uncert = np.sqrt(populated_data["science_data"])

    # Store ESA energies of full cycle for L2 purposes.
    esa_energies = get_esa_energy_pattern(esa_lut_files[0])
    # Repeat energies to be in the same shape as the science data
    esa_energies = np.repeat(esa_energies, total_packets // 4).reshape(
        -1, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS
    )
    # Convert voltage to electron energy in eV by apply conversion factor
    esa_energies = esa_energies * swe_constants.ENERGY_CONVERSION_FACTOR
    # ------------------------------------------------------------------
    # Save data to dataset.
    # ------------------------------------------------------------------
    # Load CDF attrs
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("swe")
    cdf_attrs.add_instrument_variable_attrs("swe", "l1b")

    # One full cycle data combines four quarter cycles data.
    # Epoch will store center of each science meansurement using
    # third acquisition start time coarse and fine value
    # of four quarter cycle data packets. For example, we want to
    # get indices of 3rd quarter cycle data packet in each full cycle
    # and use that to calculate center time of data acquisition time.
    #   Quarter cycle indices: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...
    indices_of_center_time = np.arange(2, total_packets, swe_constants.N_QUARTER_CYCLES)

    center_time = combine_acquisition_time(
        full_cycle_l1a_data["acq_start_coarse"].data[indices_of_center_time],
        full_cycle_l1a_data["acq_start_fine"].data[indices_of_center_time],
    )

    epoch_time = xr.DataArray(
        met_to_ttj2000ns(center_time),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    esa_step = xr.DataArray(
        np.arange(swe_constants.N_ESA_STEPS),
        name="esa_step",
        dims=["esa_step"],
        attrs=cdf_attrs.get_variable_attributes("esa_step", check_schema=False),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    esa_step_label = xr.DataArray(
        esa_step.values.astype(str),
        name="esa_step_label",
        dims=["esa_step"],
        attrs=cdf_attrs.get_variable_attributes("esa_step_label", check_schema=False),
    )

    spin_sector = xr.DataArray(
        np.arange(swe_constants.N_ANGLE_SECTORS),
        name="spin_sector",
        dims=["spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("spin_sector", check_schema=False),
    )

    # NOTE: LABL_PTR_2 should be CDF_CHAR.
    spin_sector_label = xr.DataArray(
        spin_sector.values.astype(str),
        name="spin_sector_label",
        dims=["spin_sector"],
        attrs=cdf_attrs.get_variable_attributes(
            "spin_sector_label", check_schema=False
        ),
    )

    cycle = xr.DataArray(
        np.arange(swe_constants.N_QUARTER_CYCLES),
        name="cycle",
        dims=["cycle"],
        attrs=cdf_attrs.get_variable_attributes("cycle", check_schema=False),
    )

    cycle_label = xr.DataArray(
        cycle.values.astype(str),
        name="cycle_label",
        dims=["cycle"],
        attrs=cdf_attrs.get_variable_attributes("cycle_label", check_schema=False),
    )

    cem_id = xr.DataArray(
        np.arange(swe_constants.N_CEMS, dtype=np.int8),
        name="cem_id",
        dims=["cem_id"],
        attrs=cdf_attrs.get_variable_attributes("cem_id", check_schema=False),
    )

    # NOTE: LABL_PTR_3 should be CDF_CHAR.
    cem_id_label = xr.DataArray(
        cem_id.values.astype(str),
        name="cem_id_label",
        dims=["cem_id"],
        attrs=cdf_attrs.get_variable_attributes("cem_id_label", check_schema=False),
    )

    # Add science data and it's associated metadata into dataset.
    # SCIENCE_DATA has array of this shape:
    #   (n, 24, 30, 7)
    #   n = total number of full cycles
    #   24 rows --> 24 esa voltage measurements
    #   30 columns --> 30 spin angle measurements
    #   7 elements --> 7 CEMs counts
    #
    # The metadata array will need to have this shape:
    #   (n, 4)
    #   n = total number of full cycles
    #   4 rows --> metadata for each full cycle. Each element of 4 maps to
    #              metadata of one quarter cycle.

    # Create the science dataset
    science_dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "esa_step": esa_step,
            "spin_sector": spin_sector,
            "cem_id": cem_id,
            "cycle": cycle,
            "esa_step_label": esa_step_label,
            "spin_sector_label": spin_sector_label,
            "cem_id_label": cem_id_label,
            "cycle_label": cycle_label,
        },
        attrs=cdf_attrs.get_global_attributes("imap_swe_l1b_sci"),
    )

    science_dataset["science_data"] = xr.DataArray(
        count_rate,
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attrs.get_variable_attributes("science_data"),
    )
    science_dataset["counts_stat_uncert"] = xr.DataArray(
        counts_stat_uncert,
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attrs.get_variable_attributes("counts_stat_uncert"),
    )
    science_dataset["acquisition_time"] = xr.DataArray(
        acq_time,
        dims=["epoch", "esa_step", "spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("acquisition_time"),
    )
    science_dataset["acq_duration"] = xr.DataArray(
        acq_duration,
        dims=["epoch", "esa_step", "spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("acq_duration"),
    )

    science_dataset["esa_energy"] = xr.DataArray(
        esa_energies,
        dims=["epoch", "esa_step", "spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("esa_energy"),
    )

    # create xarray dataarray for each data field
    for key, value in full_cycle_l1a_data.items():
        if key in ["science_data", "acq_duration"]:
            continue
        varname = key.lower()
        science_dataset[varname] = xr.DataArray(
            value.data.reshape(-1, swe_constants.N_QUARTER_CYCLES),
            dims=["epoch", "cycle"],
            attrs=cdf_attrs.get_variable_attributes(varname),
        )

    logger.info("SWE L1b science processing completed")
    return science_dataset


def swe_l1b(dependencies: ProcessingInputCollection) -> list[xr.Dataset]:
    """
    SWE L1B processing.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        Object containing lists of dependencies that CLI dependency
        parameter received.

    Returns
    -------
    list[xarray.Dataset]
        List of processed datasets.
    """
    processed_datasets = []
    has_science_data = dependencies.get_file_paths(descriptor="sci")
    if has_science_data:
        # Process science data to L1B
        science_dataset = swe_l1b_science(dependencies)
        processed_datasets.append(science_dataset)

    # Process HK data using L0 file
    l0_files = dependencies.get_file_paths(descriptor="raw")
    if l0_files:
        xtce_document = (
            f"{imap_module_directory}/swe/packet_definitions/swe_packet_definition.xml"
        )
        datasets_by_apid = packet_file_to_datasets(
            l0_files[0], xtce_document, use_derived_value=True
        )
        if SWEAPID.SWE_APP_HK in datasets_by_apid:
            # Define minimal CDF attrs for the HK dataset
            imap_attrs = ImapCdfAttributes()
            imap_attrs.add_instrument_global_attrs("swe")
            imap_attrs.add_instrument_variable_attrs("swe", "l1b")
            hk_attrs = imap_attrs.get_variable_attributes("l1b_hk_attrs")
            hk_str_attrs = imap_attrs.get_variable_attributes("l1b_hk_string_attrs")
            epoch_attrs = imap_attrs.get_variable_attributes(
                "epoch", check_schema=False
            )

            l1b_hk_dataset = datasets_by_apid[SWEAPID.SWE_APP_HK]
            # Update CDF attrs
            l1b_hk_dataset["epoch"].attrs.update(epoch_attrs)
            l1b_hk_dataset.attrs.update(
                imap_attrs.get_global_attributes("imap_swe_l1b_hk")
            )
            for hk_var in l1b_hk_dataset.data_vars:
                if isinstance(l1b_hk_dataset[hk_var].data[0], str):
                    l1b_hk_dataset[hk_var].attrs.update(hk_str_attrs)
                else:
                    l1b_hk_dataset[hk_var].attrs.update(hk_attrs)
            processed_datasets.append(l1b_hk_dataset)

    return processed_datasets
