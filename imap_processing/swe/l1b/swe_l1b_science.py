"""Contains code to perform SWE L1b science processing."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swe.utils import swe_constants
from imap_processing.swe.utils.swe_utils import (
    calculate_data_acquisition_time,
    combine_acquisition_time,
    read_lookup_table,
)

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


def deadtime_correction(counts: np.ndarray, acq_duration: int) -> npt.NDArray:
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
    acq_duration : int
        This is ACQ_DURATION from science packet. acq_duration is in microseconds.

    Returns
    -------
    corrected_count : numpy.ndarray
        Corrected counts.
    """
    # deadtime is 360 ns
    deadtime = 360e-9
    correct = 1.0 - (deadtime * (counts / (acq_duration * 1e-6)))
    correct = np.maximum(0.1, correct)
    corrected_count = np.divide(counts, correct)
    return corrected_count.astype(np.float64)


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
    # convert microseconds to seconds
    acq_duration_sec = acq_duration * 1e-6
    count_rate = data / acq_duration_sec
    return count_rate.astype(np.float64)


def read_in_flight_cal_data() -> pd.DataFrame:
    """
    Read in-flight calibration data.

    In-flight calibration data file will contain rows where each line
    has 8 numbers, with the first being a time stamp in MET, and the next
    7 being the factors for the 7 detectors.

    This file will be updated weekly with new calibration data. In other
    words, one line of data will be added each week to the existing file.
    File will be in CSV format. Processing won't be kicked off until there
    is in-flight calibration data that covers science data.

    TODO: decide filename convention given this information. This function
    is a placeholder for reading in the calibration data until we decide on
    how to read calibration data through dependencies list.

    Returns
    -------
    in_flight_cal_df : pandas.DataFrame
        DataFrame with in-flight calibration data.
    """
    # TODO: Read in in-flight calibration file.

    # Define the column headers
    columns = ["met_time", "cem1", "cem2", "cem3", "cem4", "cem5", "cem6", "cem7"]

    # Create an empty DataFrame with the specified columns
    empty_df = pd.DataFrame(columns=columns)
    return empty_df


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
        error_msg = (
            f"Acquisition min/max times: {acquisition_times.min()} to "
            f"{acquisition_times.max()}. "
            f"Calibration min/max times: {cal_times.min()} to {cal_times.max()}. "
            "Acquisition times should be within calibration time range."
        )
        raise ValueError(error_msg)

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
    corrected_counts: np.ndarray, acquisition_time: np.ndarray
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

    Returns
    -------
    corrected_counts : numpy.ndarray
        Corrected count of full cycle data after applying in-flight calibration.
        Array shape is (N_ESA_STEPS, N_ANGLE_SECTORS, N_CEMS).
    """
    # Read in in-flight calibration data
    in_flight_cal_df = read_in_flight_cal_data()
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


def populate_full_cycle_data(
    l1a_data: xr.Dataset, packet_index: int, esa_table_num: int
) -> npt.NDArray:
    """
    Populate full cycle data array using esa lookup table and l1a_data.

    Parameters
    ----------
    l1a_data : xarray.Dataset
        L1a data with full cycle data only.
    packet_index : int
        Index of current packet in the whole packet list.
    esa_table_num : int
        ESA lookup table number.

    Returns
    -------
    full_cycle_ds : xarray.Dataset
        Full cycle data and its acquisition times.
    """
    esa_lookup_table = get_esa_dataframe(esa_table_num)

    # If esa lookup table number is 0, then populate using esa lookup table data
    # with information that esa step ramps up in even column and ramps down
    # in odd column every six steps.
    if esa_table_num == 0:
        # create new full cycle data array
        full_cycle_data = np.zeros(
            (
                swe_constants.N_ESA_STEPS,
                swe_constants.N_ANGLE_SECTORS,
                swe_constants.N_CEMS,
            )
        )
        # SWE needs to store acquisition time of each count data point
        # to use in level 2 processing to calculate
        # spin phase. This is done below by using information from
        # science packet.
        acquisition_times = np.zeros(
            (swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        )

        # Store acquisition duration for later calculation in this function
        acq_duration_arr = np.zeros(
            (swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS)
        )

        # Initialize esa_step_number and column_index.
        # esa_step_number goes from 0 to 719 range where
        # 720 came from 24 x 30. full_cycle_data array has
        # (N_ESA_STEPS, N_ANGLE_SECTORS) dimension.
        esa_step_number = 0
        # column_index goes from 0 to 29 range where
        # 30 came from 30 column in full_cycle_data array
        column_index = -1

        # Go through four quarter cycle data packets
        for index in range(swe_constants.N_QUARTER_CYCLES):
            decompressed_counts = l1a_data["science_data"].data[packet_index + index]
            # Do deadtime correction
            acq_duration = l1a_data["acq_duration"].data[packet_index + index]
            settle_duration = l1a_data["settle_duration"].data[packet_index + index]
            corrected_counts = deadtime_correction(decompressed_counts, acq_duration)

            # Each quarter cycle data should have same acquisition start time coarse
            # and fine value. We will use that as base time to calculate each
            # acquisition time for each count data.
            base_quarter_cycle_acq_time = combine_acquisition_time(
                l1a_data["acq_start_coarse"].data[packet_index + index],
                l1a_data["acq_start_fine"].data[packet_index + index],
            )

            # Go through each quarter cycle's 180 ESA measurements
            # and put counts rate in full cycle data array
            for step in range(180):
                # Get esa voltage value from esa lookup table and
                # use that to get row index in full data array
                esa_voltage_value = esa_lookup_table.loc[esa_step_number]["esa_v"]
                esa_voltage_row_index = swe_constants.ESA_VOLTAGE_ROW_INDEX_DICT[
                    esa_voltage_value
                ]

                # every six steps, increment column index
                if esa_step_number % 6 == 0:
                    column_index += 1
                # Put counts rate in full cycle data array
                full_cycle_data[esa_voltage_row_index][column_index] = corrected_counts[
                    step
                ]
                # Acquisition time (in seconds) of each count data point
                acquisition_times[esa_voltage_row_index][column_index] = (
                    calculate_data_acquisition_time(
                        base_quarter_cycle_acq_time,
                        esa_step_number,
                        acq_duration,
                        settle_duration,
                    )
                )
                # Store acquisition duration for later calculation
                acq_duration_arr[esa_voltage_row_index][column_index] = acq_duration
                esa_step_number += 1

            # reset column index for next quarter cycle
            column_index = -1
        # TODO: Apply in flight calibration to full cycle data

    # NOTE: We may get more lookup table with different setup when we get real
    # data. But for now, we are advice to continue with current setup and can
    # add/change it when we get real data.

    # Apply calibration based on in-flight calibration.
    calibrated_counts = apply_in_flight_calibration(full_cycle_data, acquisition_times)

    # Convert counts to rate
    counts_rate = convert_counts_to_rate(
        calibrated_counts, acq_duration_arr[:, :, np.newaxis]
    )

    # Store full cycle data in xr.Dataset for later use.
    full_cycle_ds = xr.Dataset(
        {
            "full_cycle_data": (["esa_step", "spin_sector", "cem_id"], counts_rate),
            "acquisition_time": (["esa_step", "spin_sector"], acquisition_times),
            "acq_duration": (["esa_step", "spin_sector"], acq_duration_arr),
        }
    )

    return full_cycle_ds


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


def swe_l1b_science(l1a_data: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    SWE l1b science processing.

    Parameters
    ----------
    l1a_data : xarray.Dataset
        Input data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Processed l1b data.
    """
    total_packets = len(l1a_data["science_data"].data)

    # Array to store list of table populated with data
    # of full cycles
    full_cycle_science_data = []
    # These two are carried in l1b for level 2 and 3 processing
    full_cycle_acq_times = []
    full_cycle_acq_duration = []
    packet_index = 0
    l1a_data_copy = l1a_data.copy(deep=True)

    full_cycle_data_indices = get_indices_of_full_cycles(l1a_data["quarter_cycle"].data)
    logger.debug(
        f"Quarter cycle data before filtering: {l1a_data_copy['quarter_cycle'].data}"
    )

    # Delete Raw Science Data from l1b and onwards
    del l1a_data_copy["raw_science_data"]

    if full_cycle_data_indices.size == 0:
        # Log that no data is found for science data
        return None

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

    # Go through each cycle and populate full cycle data
    for packet_index in range(0, total_packets, swe_constants.N_QUARTER_CYCLES):
        # get ESA lookup table information
        esa_table_num = l1a_data["esa_table_num"].data[packet_index]

        # If ESA lookup table number is in-flight calibration
        # data, then skip current cycle per SWE teams specification.
        # SWE team only wants in-flight calibration data to be processed
        # upto l1a. In-flight calibration data looks same as science data
        # but it only measures one energy steps during the whole duration.
        if esa_table_num == 1:
            continue

        full_cycle_ds = populate_full_cycle_data(
            full_cycle_l1a_data, packet_index, esa_table_num
        )

        # save full data array to file
        full_cycle_science_data.append(full_cycle_ds["full_cycle_data"].data)
        full_cycle_acq_times.append(full_cycle_ds["acquisition_time"].data)
        full_cycle_acq_duration.append(full_cycle_ds["acq_duration"].data)

    # ------------------------------------------------------------------
    # Save data to dataset.
    # ------------------------------------------------------------------
    # Load CDF attrs
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("swe")
    cdf_attrs.add_instrument_variable_attrs("swe", "l1b")
    cdf_attrs.add_global_attribute("Data_version", data_version)

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

    # Create the dataset
    dataset = xr.Dataset(
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

    dataset["science_data"] = xr.DataArray(
        full_cycle_science_data,
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attrs.get_variable_attributes("science_data"),
    )
    dataset["acquisition_time"] = xr.DataArray(
        full_cycle_acq_times,
        dims=["epoch", "esa_step", "spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("acquisition_time"),
    )
    dataset["acq_duration"] = xr.DataArray(
        full_cycle_acq_duration,
        dims=["epoch", "esa_step", "spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("acq_duration"),
    )

    # create xarray dataset for each metadata field
    for key, value in full_cycle_l1a_data.items():
        if key in ["science_data", "acq_duration"]:
            continue
        metadata_field = key.lower()
        dataset[metadata_field] = xr.DataArray(
            value.data.reshape(-1, swe_constants.N_QUARTER_CYCLES),
            dims=["epoch", "cycle"],
            attrs=cdf_attrs.get_variable_attributes(metadata_field),
        )

    logger.info("SWE L1b science processing completed")
    return dataset
