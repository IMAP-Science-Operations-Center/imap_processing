import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import cdf_utils, imap_module_directory
from imap_processing.swe import swe_cdf_attrs

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


def read_lookup_table(table_index_value: int):
    """Read lookup table from file.

    Parameters
    ----------
    table_index_value : int
        ESA table index number
    """
    lookup_table_filepath = f"{imap_module_directory}/swe/l1b/swe_esa_lookup_table.csv"
    lookup_table = pd.read_csv(
        lookup_table_filepath,
        index_col="e_step",
    )

    if table_index_value == 0:
        return lookup_table.loc[lookup_table["table_index"] == 0]

    if table_index_value == 1:
        return lookup_table.loc[lookup_table["table_index"] == 1]

    raise ValueError("Error: Invalid table index value")


def deadtime_correction(counts: np.array, acq_duration: int):
    """Calculate deadtime correction.

    Deadtime correction is a technique used in various fields, including
    nuclear physics,radiation detection, and particle counting, to compensate
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
    counts : np.array
        counts data before deadtime corrections
    acq_duration : int
        This is ACQ_DURATION from science packet

    Returns
    -------
    np.array
        Corrected counts
    """
    # deadtime will be constant once it's defined.
    # This deadtime value is from previous mission. SWE
    # will give new one once they have it ready.
    # TODO: update deadtime when we get new number
    deadtime = 1.5e-6
    correct = 1.0 - (deadtime * counts / acq_duration)
    correct = np.maximum(0.1, correct)
    corrected_count = np.divide(counts, correct)
    return corrected_count


def convert_counts_to_rate(data: np.array, acq_duration: int):
    """Convert counts to rate using sampling time.

    acq_duration is ACQ_DURATION from science packet.


    Parameters
    ----------
    data : np.array
        counts data
    acq_duration : int
        Acquisition duration. acq_duration is in millieseconds

    Returns
    -------
    np.array
        Count rates array in seconds
    """
    return np.divide(data, acq_duration)


def calculate_calibration_factor(time):
    """Calculate calibration factor.

    Steps to calculate calibration factor:
    1. Convert input time to match time format in the calibration data file.
    2. Find the nearest in time calibration data point.
    3. Linear interpolate between those two nearest time and get factor for input time.

    What this function is doing:
    1. **Reading Calibration Data**: The function first reads a file containing
        calibration data for electron measurements over time. This data helps
        adjust or correct the measurements based on changes in the instrument's
        sensitivity.

    2. **Interpolating Calibration Factors**: Imagine you have several points on
        a graph, and you want to estimate values between those points. In our case,
        these points represent calibration measurements taken at different times.
        The function figures out which two calibration points are closest in time
        to the specific measurement time you're interested in.

    3. **Calculating Factors**: Once it finds these two nearby calibration points,
        the function calculates a correction factor by drawing a straight line
        between them (linear interpolation). This factor helps adjust the measurement
        to make it more accurate, considering how the instrument's sensitivity changed
        between those two calibration points.

    4. **Returning the Correction Factor**: Finally, the function returns this
        correction factor. You can then use this factor to adjust or calibrate your
        measurements at the specific time you're interested in. This ensures that
        your measurements are as accurate as possible, taking into account the
        instrument's changing sensitivity over time.
    """
    # NOTE: waiting on fake calibration data to write this.
    pass


def apply_in_flight_calibration(data):
    """Apply in flight calibration to full cycle data.

    These factors are used to account for changes in gain with time.

    They are derived from the weekly electron calibration data.

    Parameters
    ----------
    data : _type_
        _description_
    """
    # calculate calibration factor
    # Apply to all data
    pass


def populate_full_cycle_data(
    l1a_data: xr.Dataset, packet_index: int, esa_table_num: int
):
    """Populate full cycle data array using esa lookup table and l1a_data.

    Parameters
    ----------
    l1a_data : xr.Dataset
        L1a data with full cycle data only
    packet_index : int
        Index of current packet in the whole packet list.
    esa_table_num : int
        ESA lookup table number

    Returns
    -------
    np.array
        Array with full cycle data populated
    """
    esa_lookup_table = read_lookup_table(esa_table_num)

    # If esa lookup table number is 0, then populate using esa lookup table data
    # with information that esa step ramps up in even column and ramps down
    # in odd column every six steps.
    if esa_table_num == 0:
        # create new full cycle data array
        full_cycle_data = np.zeros((24, 30, 7))

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
            uncompressed_counts = l1a_data["SCIENCE_DATA"].data[packet_index + index]
            # Do deadtime correction
            acq_duration = l1a_data["ACQ_DURATION"].data[packet_index + index]
            corrected_counts = deadtime_correction(uncompressed_counts, acq_duration)
            # Convert counts to rate
            counts_rate = convert_counts_to_rate(corrected_counts, acq_duration)

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
                esa_step_number += 1

            # reset column index for next quarter cycle
            column_index = -1
        # TODO: Apply in flight calibration to full cycle data

    # NOTE: We may get more lookup table with different setup when we get real
    # data. But for now, we are advice to continue with current setup and can
    # add/change it when we get real data.

    return full_cycle_data


def find_cycle_starts(cycles: np.array):
    """Find index of where new cycle started.

    Brandon Stone helped developed this algorithm.

    Parameters
    ----------
    cycles : np.array
        Array that contains quarter cycle information.

    Returns
    -------
    np.array
        Array of indices of start cycle
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
    return np.where(valid)[0]


def get_indices_of_full_cycles(quarter_cycle: np.array):
    """Get indices of full cycles.

    Parameters
    ----------
    quarter_cycle : np.array
        Array that contains quarter cycles informations.

    Returns
    -------
    np.array
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


def filter_full_cycle_data(full_cycle_data_indices: np.array, l1a_data: xr.Dataset):
    """Filter metadata and science of packets that makes full cycles.

    Parameters
    ----------
    full_cycle_data_indices : np.array
        Array with indices of full cycles.
    l1a_data : xr.dataset
        L1A dataset

    Returns
    -------
    xr.dataset
        L1A dataset with filtered metadata.
    """
    for key, value in l1a_data.items():
        l1a_data[key] = value.data[full_cycle_data_indices]
    return l1a_data


def swe_l1b_science(l1a_data):
    """In this function, we cap.

    Parameters
    ----------
    l1a_data : xr.dataset
        L1A data
    """
    total_packets = len(l1a_data["SCIENCE_DATA"].data)

    all_data = []
    packet_index = 0
    l1a_data_copy = l1a_data.copy(deep=True)

    full_cycle_data_indices = get_indices_of_full_cycles(l1a_data["QUARTER_CYCLE"].data)

    # Delete Raw Science Data from l1b and onwards
    del l1a_data_copy["RAW_SCIENCE_DATA"]

    if full_cycle_data_indices.size == 0:
        # Log that no data is found for science data
        return None

    if len(full_cycle_data_indices) != total_packets:
        # Filter metadata and science data of packets that makes full cycles
        l1a_data_copy = filter_full_cycle_data(full_cycle_data_indices, l1a_data_copy)

        # Update total packets
        total_packets = len(full_cycle_data_indices)

    # Go through each cycle and populate full cycle data
    for packet_index in range(0, total_packets, 4):
        # get ESA lookup table information
        esa_table_num = l1a_data["ESA_TABLE_NUM"].data[packet_index]

        # If ESA lookup table number is in-flight calibration
        # data, then skip current cycle per SWE teams specification.
        # SWE team only wants in-flight calibration data to be processed
        # upto l1a. In-flight calibration data looks same as science data
        # but it only measures one energy steps during the whole duration.
        if esa_table_num == 1:
            continue

        full_cycle_data = populate_full_cycle_data(
            l1a_data_copy, packet_index, esa_table_num
        )

        # save full data array to file
        all_data.append(full_cycle_data)

    # ------------------------------------------------------------------
    # Save data to dataset.

    # Get Epoch time of full cycle data and then reshape it to
    # (n, 4) where n = total number of full cycles and 4 = four
    # quarter cycle data metadata. For Epoch's data, we take the first element
    # of each quarter cycle data metadata.
    epoch_time = xr.DataArray(
        l1a_data["Epoch"].data[full_cycle_data_indices].reshape(-1, 4)[:, 0],
        name="Epoch",
        dims=["Epoch"],
        attrs=cdf_utils.epoch_attrs,
    )

    int_attrs = swe_cdf_attrs.int_attrs
    int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs["LABLAXIS"] = "Energy"
    int_attrs["VALIDMAX"] = np.int64(24)
    energy = xr.DataArray(
        np.arange(24),
        name="Energy",
        dims=["Energy"],
        attrs=int_attrs,
    )

    int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs["LABLAXIS"] = "Angle"
    int_attrs["VALIDMAX"] = np.int64(30)
    angle = xr.DataArray(
        np.arange(30),
        name="Angle",
        dims=["Angle"],
        attrs=int_attrs,
    )

    int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs[
        "LABLAXIS"
    ] = "Quarter Cycle"
    int_attrs["VALIDMAX"] = np.int64(180)
    cycle = xr.DataArray(
        np.arange(4),
        name="Cycle",
        dims=["Cycle"],
        attrs=int_attrs,
    )

    float_attrs = swe_cdf_attrs.float_attrs
    float_attrs["CATDESC"] = float_attrs["FIELDNAM"] = float_attrs["LABLAXIS"] = "Rates"
    rates = xr.DataArray(
        np.arange(7, dtype=np.float64),
        name="Rates",
        dims=["Rates"],
        attrs=float_attrs,
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
            "Epoch": epoch_time,
            "Energy": energy,
            "Angle": angle,
            "Rates": rates,
            "Cycle": cycle,
        },
        attrs=swe_cdf_attrs.swe_l1b_global_attrs,
    )

    dataset["SCIENCE_DATA"] = xr.DataArray(
        all_data,
        dims=["Epoch", "Energy", "Angle", "Rates"],
        attrs=swe_cdf_attrs.l1b_science_attrs,
    )

    # create xarray dataset for each metadata field
    for key, value in l1a_data_copy.items():
        if key == "SCIENCE_DATA":
            continue
        # if key == "SPIN_PHASE":
        #     print(key, value)

        int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs["LABLAXIS"] = key
        # get int32's max since most of metadata is under 32-bits
        int_attrs["VALIDMAX"] = np.iinfo(np.int32).max
        int_attrs["DEPEND_O"] = "Epoch"
        int_attrs["DEPEND_2"] = "Cycle"
        dataset[key] = xr.DataArray(
            value.data.reshape(-1, 4),
            dims=["Epoch", "Cycle"],
            attrs=int_attrs,
        )
    return dataset
