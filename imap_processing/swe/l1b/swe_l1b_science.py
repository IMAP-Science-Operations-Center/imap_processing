"""Contains code to perform SWE L1b science processing."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swe.utils.swe_utils import read_lookup_table

logger = logging.getLogger(__name__)

# ESA voltage and index in the final data table
esa_voltage_row_index_dict = {
    0.56: 0,
    0.78: 1,
    1.08: 2,
    1.51: 3,
    2.10: 4,
    2.92: 5,
    4.06: 6,
    5.64: 7,
    7.85: 8,
    10.92: 9,
    15.19: 10,
    21.13: 11,
    29.39: 12,
    40.88: 13,
    56.87: 14,
    79.10: 15,
    110.03: 16,
    153.05: 17,
    212.89: 18,
    296.14: 19,
    411.93: 20,
    572.99: 21,
    797.03: 22,
    1108.66: 23,
}


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


def convert_counts_to_rate(data: np.ndarray, acq_duration: int) -> npt.NDArray:
    """
    Convert counts to rate using sampling time.

    acq_duration is ACQ_DURATION from science packet.

    Parameters
    ----------
    data : numpy.ndarray
        Counts data.
    acq_duration : int
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


def calculate_calibration_factor(time: int) -> None:
    """
    Calculate calibration factor.

    Steps to calculate calibration factor:

    1. Convert input time to match time format in the calibration data file.
    2. Find the nearest in time calibration data point.
    3. Linear interpolate between those two nearest time and get factor for input time.

    What this function is doing:

    | 1. **Reading Calibration Data**: The function first reads a file containing
    |     calibration data for electron measurements over time. This data helps
    |     adjust or correct the measurements based on changes in the instrument's
    |     sensitivity.

    | 2. **Interpolating Calibration Factors**: Imagine you have several points on
    |     a graph, and you want to estimate values between those points. In our case,
    |     these points represent calibration measurements taken at different times.
    |     The function figures out which two calibration points are closest in time
    |     to the specific measurement time you're interested in.

    | 3. **Calculating Factors**: Once it finds these two nearby calibration points,
    |     the function calculates a correction factor by drawing a straight line
    |     between them (linear interpolation). This factor helps adjust the measurement
    |     to make it more accurate, considering how the instrument's sensitivity changed
    |     between those two calibration points.

    | 4. **Returning the Correction Factor**: Finally, the function returns this
    |     correction factor. You can then use this factor to adjust or calibrate your
    |     measurements at the specific time you're interested in. This ensures that
    |     your measurements are as accurate as possible, taking into account the
    |     instrument's changing sensitivity over time.

    Parameters
    ----------
    time : int
        Input time.
    """
    # NOTE: waiting on fake calibration data to write this.
    pass


def apply_in_flight_calibration(data: np.ndarray) -> None:
    """
    Apply in flight calibration to full cycle data.

    These factors are used to account for changes in gain with time.

    They are derived from the weekly electron calibration data.

    Parameters
    ----------
    data : numpy.ndarray
        Full cycle data array.
    """
    # calculate calibration factor
    # Apply to all data
    pass


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
        energy_steps = 24
        angle = 30
        cem_detectors = 7
        # create new full cycle data array
        full_cycle_data = np.zeros((energy_steps, angle, cem_detectors))
        # SWE needs to store acquisition time of each count data point
        # to use in level 2 processing to calculate
        # spin phase. This is done below by using information from
        # science packet.
        acquisition_times = np.zeros((energy_steps, angle, cem_detectors))

        # Initialize esa_step_number and column_index.
        # esa_step_number goes from 0 to 719 range where
        # 720 came from 24 x 30. full_cycle_data array has (24, 30)
        # dimension.
        esa_step_number = 0
        # column_index goes from 0 to 29 range where
        # 30 came from 30 column in full_cycle_data array
        column_index = -1

        # Go through four quarter cycle data packets
        for index in range(4):
            decompressed_counts = l1a_data["science_data"].data[packet_index + index]
            # Do deadtime correction
            acq_duration = l1a_data["acq_duration"].data[packet_index + index]
            settle_duration = l1a_data["settle_duration"].data[packet_index + index]
            corrected_counts = deadtime_correction(decompressed_counts, acq_duration)
            # Convert counts to rate
            counts_rate = convert_counts_to_rate(corrected_counts, acq_duration)

            # Each quarter cycle data should have same acquisition start time coarse
            # and fine value. We will use that as base time to calculate each
            # acquisition time for each count data. Acquisition time of each count
            # data point will be calculated using this formula:
            #   base_quarter_cycle_acq_time = acq_start_coarse +
            #                                 acq_start_fine / 1000000
            #   each_count_acq_time = base_quarter_cycle_acq_time +
            #                         (step * ( acq_duration + settle_duration) / 1000 )
            # where step goes from 0 to 179, acq_start_coarse is in seconds and
            # acq_start_fine is in microseconds and acq_duration is in milliseconds.
            base_quarter_cycle_acq_time = (
                l1a_data["acq_start_coarse"].data[packet_index + index]
                + l1a_data["acq_start_fine"].data[packet_index + index] / 1000000
            )

            # Go through each quarter cycle's 180 ESA measurements
            # and put counts rate in full cycle data array
            for step in range(180):
                # Get esa voltage value from esa lookup table and
                # use that to get row index in full data array
                esa_voltage_value = esa_lookup_table.loc[esa_step_number]["esa_v"]
                esa_voltage_row_index = esa_voltage_row_index_dict[esa_voltage_value]

                # every six steps, increment column index
                if esa_step_number % 6 == 0:
                    column_index += 1
                # Put counts rate in full cycle data array
                full_cycle_data[esa_voltage_row_index][column_index] = counts_rate[step]
                # Put acquisition time in acquisition_times array
                acquisition_times[esa_voltage_row_index][column_index] = (
                    base_quarter_cycle_acq_time
                    + (step * (acq_duration + settle_duration) / 1000)
                )
                esa_step_number += 1

            # reset column index for next quarter cycle
            column_index = -1
        # TODO: Apply in flight calibration to full cycle data

    # NOTE: We may get more lookup table with different setup when we get real
    # data. But for now, we are advice to continue with current setup and can
    # add/change it when we get real data.

    # Store count data and acquisition times of full cycle data in xr.Dataset
    full_cycle_ds = xr.Dataset(
        {
            "full_cycle_data": (["energy", "angle", "cem"], full_cycle_data),
            "sci_step_acq_time_sec": (["energy", "angle", "cem"], acquisition_times),
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
    if cycles.size < 4:
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
    full_cycles_indices = indices_of_start[..., None] + np.arange(4)[None, ...]
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
    full_cycle_acq_times = []
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
    for packet_index in range(0, total_packets, 4):
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
        full_cycle_acq_times.append(full_cycle_ds["sci_step_acq_time_sec"].data)

    # ------------------------------------------------------------------
    # Save data to dataset.
    # ------------------------------------------------------------------
    # Load CDF attrs
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("swe")
    cdf_attrs.add_instrument_variable_attrs("swe", "l1b")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    # Get epoch time of full cycle data and then reshape it to
    # (n, 4) where n = total number of full cycles and 4 = four
    # quarter cycle data metadata. For epoch's data, we take the first element
    # of each quarter cycle data metadata.
    epoch_time = xr.DataArray(
        l1a_data["epoch"].data[full_cycle_data_indices].reshape(-1, 4)[:, 0],
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    energy = xr.DataArray(
        np.arange(24),
        name="energy",
        dims=["energy"],
        attrs=cdf_attrs.get_variable_attributes("energy"),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    energy_label = xr.DataArray(
        energy.values.astype(str),
        name="energy_label",
        dims=["energy_label"],
        attrs=cdf_attrs.get_variable_attributes("energy_label"),
    )

    angle = xr.DataArray(
        np.arange(30),
        name="angle",
        dims=["angle"],
        attrs=cdf_attrs.get_variable_attributes("angle"),
    )

    # NOTE: LABL_PTR_2 should be CDF_CHAR.
    angle_label = xr.DataArray(
        angle.values.astype(str),
        name="angle_label",
        dims=["angle_label"],
        attrs=cdf_attrs.get_variable_attributes("angle_label"),
    )

    cycle = xr.DataArray(
        np.arange(4),
        name="cycle",
        dims=["cycle"],
        attrs=cdf_attrs.get_variable_attributes("cycle"),
    )

    cem = xr.DataArray(
        np.arange(7, dtype=np.float64),
        name="cem",
        dims=["cem"],
        attrs=cdf_attrs.get_variable_attributes("cem"),
    )

    # NOTE: LABL_PTR_3 should be CDF_CHAR.
    cem_label = xr.DataArray(
        cem.values.astype(str),
        name="cem_label",
        dims=["cem_label"],
        attrs=cdf_attrs.get_variable_attributes("cem_label"),
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
            "energy": energy,
            "angle": angle,
            "cem": cem,
            "cycle": cycle,
            "energy_label": energy_label,
            "angle_label": angle_label,
            "cem_label": cem_label,
        },
        attrs=cdf_attrs.get_global_attributes("imap_swe_l1b_sci"),
    )

    dataset["science_data"] = xr.DataArray(
        full_cycle_science_data,
        dims=["epoch", "energy", "angle", "cem"],
        attrs=cdf_attrs.get_variable_attributes("science_data"),
    )
    dataset["sci_step_acq_time_sec"] = xr.DataArray(
        full_cycle_acq_times,
        dims=["epoch", "energy", "angle", "cem"],
        attrs=cdf_attrs.get_variable_attributes("sci_step_acq_time_sec"),
    )

    # create xarray dataset for each metadata field
    for key, value in full_cycle_l1a_data.items():
        if key == "science_data":
            continue
        metadata_field = key.lower()
        dataset[metadata_field] = xr.DataArray(
            value.data.reshape(-1, 4),
            dims=["epoch", "cycle"],
            attrs=cdf_attrs.get_variable_attributes(metadata_field),
        )

    logger.info("SWE L1b science processing completed")
    return dataset
