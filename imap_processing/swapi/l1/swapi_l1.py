"""SWAPI level-1 processing code."""
import copy
import logging

import numpy as np
import xarray as xr

from imap_processing.swapi.swapi_utils import SWAPIAPID, SWAPIMODE, create_dataset
from imap_processing.utils import group_by_apid, sort_by_time


def check_for_bad_data(full_sweep_sci):
    """Check for bad data.

    Bad data indicator:

    |    1. SWP_HK.CHKSUM is wrong
    |    2. SWAPI mode (SWP_SCI.MODE) is not HVSCI
    |    3. PLAN_ID for current sweep should all be one value
    |    4. SWEEP_TABLE should all be one value.

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Science data that only contains full sweep data.

    Returns
    -------
    List
        List of sweep indices of bad data
    """
    # If PLAN_ID and SWEEP_TABLE is not same, the discard the
    # sweep data. PLAN_ID and SWEEP_TABLE should match. In other word,
    # PLAN_ID for current sweep should all be one value and
    # SWEEP_TABLE should all be one value.
    plan_id = full_sweep_sci["PLAN_ID_SCIENCE"].data.reshape(-1, 12)
    sweep_table = full_sweep_sci["SWEEP_TABLE"].data.reshape(-1, 12)
    # plan_id[1][0] = 1
    mode = full_sweep_sci["MODE"].data.reshape(-1, 12)

    total_sweeps = len(plan_id)

    bad_data_start_indices = []
    for index in range(total_sweeps):
        # print(np.all(sweep_table[index] != sweep_table[index][0]))
        if not np.all(sweep_table[index] == sweep_table[index][0]):
            logging.debug("SWEEP_TABLE is not same")
            bad_data_start_indices.append(index)

        if not np.all(plan_id[index] == plan_id[index][0]):
            logging.debug("PLAN_ID is not same")
            bad_data_start_indices.append(index)

        # TODO: change comparison to SWAPIMODE.HVSCI once we have
        # some HVSCI data
        if not np.all(mode[index] == SWAPIMODE.HVENG):
            logging.debug("MODE is not HVSCI")
            bad_data_start_indices.append(index)
    # TODO: add checks for checksum

    # Get bad data sweep start indices and create
    # sweep indices.
    # Eg.
    # From this: [0 1]
    # To this: [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [12 13 14 15 16 17 18 19 20 21 22 23]]
    cycle_start_indices = np.unique(bad_data_start_indices)
    bad_cycle_indices = np.array(
        [np.arange(n * 12, (n + 1) * 12) for n in cycle_start_indices]
    ).reshape(-1)
    return bad_cycle_indices


def decompress_count(count_data: np.ndarray, compression_flag: np.ndarray = None):
    """Decompress counts based on compression indicators.

    Decompression algorithm:
    There are 3 compression regions:

    |    1) 0 <= value <=65535
    |    2) 65536 <= value <= 1,048,575
    |    3) 1,048,576 <= value

    Pseudocode:

    | if XXX_RNG_ST0 == 0:          # Not compressed
    |    actual_value = XXX_CNT0
    | elif (XXX_RNG_ST0==1 && XXX_CNT0==0xFFFF):    # Overflow
    |    actual_value = <some constant that indicates overflow>
    | elif (XXX_RNG_ST0==1 && XXX_CNT0!=0xFFFF):
    |    actual_value = XXX_CNT0 * 16

    Parameters
    ----------
    count_data : numpy.ndarray
        Array with counts.
    compression_flag : numpy.ndarray
        Array with compression indicators.

    Returns
    -------
    numpy.ndarray
        Array with decompressed counts.
    """
    # Decompress counts based on compression indicators
    # If 0, value is already decompressed. If 1, value is compressed.
    # If 1 and count is 0xFFFF, value is overflow.
    new_count = copy.deepcopy(count_data)
    # This below line gives back row and column index for
    # compressed flags. Eg.
    # (array([0, 1, 2, 2]), array([ 4,  1, 20, 51]))
    compressed_count_indices = np.where(compression_flag == 1)

    for row, col in zip(*compressed_count_indices):
        if count_data[row, col] == 0xFFFF:  # Overflow
            new_count[row, col] = -1
        elif count_data[row, col] != 0xFFFF:
            new_count[row, col] = count_data[row, col] * 16
    return new_count


def filter_full_cycle_data(full_cycle_data_indices: np.ndarray, l1a_data: xr.Dataset):
    """Filter metadata and science of packets that makes full cycles.

    Parameters
    ----------
    full_cycle_data_indices : numpy.ndarray
        Array with indices of full cycles.
    l1a_data : xarray.Dataset
        L1A dataset

    Returns
    -------
    xarray.Dataset
        L1A dataset with filtered metadata.
    """
    # Had to create new xr.Dataset because Epoch shape and new data variables shapes was
    # different.
    full_sweep_dataset = xr.Dataset(
        coords={"Epoch": l1a_data["Epoch"].data[full_cycle_data_indices]}
    )
    for key, _value in l1a_data.items():
        full_sweep_dataset[key] = l1a_data[key][full_cycle_data_indices]
    return full_sweep_dataset


def find_sweep_starts(packets: xr.Dataset):
    """Find index of where new cycle started.

    Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0
    (Sequence number of set of steps in energy sweep);
    end of a sweep is marked by SWP_SCI.SEQ_NUMBER=11;
    In this function, we look for index of SEQ_NUMBER 0.

    Brandon Stone helped developed this algorithm.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset that contains SWP_SCI packets.

    Returns
    -------
    numpy.ndarray
        Array of indices of start cycle.
    """
    if packets["Epoch"].size < 12:
        return np.array([], np.int64)

    # calculate time difference between consecutive sweep
    diff = packets["Epoch"].data[1:] - packets["Epoch"].data[:-1]

    # This uses sliding window to find index where cycle starts.
    # This is what this below code line is doing:
    # [1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0]      # Is cycle zero?
    # [1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1]      # Next diff is one?
    #   [1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0]    # Next diff is one?
    #     [0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0 0]  # Next diff is one?
    #
    # [0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0]      # And all?

    ione = diff == 1

    valid = (
        (packets["SEQ_NUMBER"] == 0)[:-11]
        & ione[:-10]
        & ione[1:-9]
        & ione[2:-8]
        & ione[3:-7]
        & ione[4:-6]
        & ione[5:-5]
        & ione[6:-4]
        & ione[7:-3]
        & ione[8:-2]
        & ione[9:-1]
        & ione[10:]
    )
    return np.where(valid)[0]


def get_indices_of_full_sweep(packets: xr.Dataset):
    """Get indices of full cycles.

    Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0
    (Sequence number of set of steps in energy sweep);
    end of a sweep is marked by SWP_SCI.SEQ_NUMBER=11;
    all packets must be present to process a sweep.

    In this function, we get the indices of SEQ_NUMBER
    0 and then construct full sweep indices.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset that contains SEQ_NUMBER data informations.
        Eg. sci_dataset["SEQ_NUMBER"].data

    Returns
    -------
    numpy.ndarray
        1D array with indices of full cycle data.
    """
    indices_of_start = find_sweep_starts(packets)
    # find_sweep_starts[..., None] creates array of shape(n, 1).
    #   Eg. [[3], [8]]
    # np.arange(12)[None, ...] creates array of shape(1, 12)
    #   Eg. [[0, 1, 2, 3, ....., 11]]
    # then we add both of them together to get an array of shape(n, 4)
    #   Eg. [[3, 4, 5, 6,...14], [8, 9, 10, 11, ..., 19]]
    full_cycles_indices = indices_of_start[..., None] + np.arange(12)[None, ...]
    return full_cycles_indices.reshape(-1)


def process_sweep_data(full_sweep_sci, cem_prefix, total_full_sweeps):
    """Process full sweep data.

    Data from each packet comes like this:

    |    SEQ_NUMBER
    |    .
    |    PCEM_RNG_ST0
    |    SCEM_RNG_ST0
    |    COIN_RNG_ST0
    |    PCEM_RNG_ST1
    |    SCEM_RNG_ST1
    |    COIN_RNG_ST1
    |    PCEM_RNG_ST2
    |    SCEM_RNG_ST2
    |    COIN_RNG_ST2
    |    PCEM_RNG_ST3
    |    SCEM_RNG_ST3
    |    COIN_RNG_ST3
    |    PCEM_RNG_ST4
    |    SCEM_RNG_ST4
    |    COIN_RNG_ST4
    |    PCEM_RNG_ST5
    |    SCEM_RNG_ST5
    |    COIN_RNG_ST5
    |    PCEM_CNT0
    |    SCEM_CNT0
    |    COIN_CNT0
    |    PCEM_CNT1
    |    SCEM_CNT1
    |    COIN_CNT1
    |    PCEM_CNT2
    |    SCEM_CNT2
    |    COIN_CNT2
    |    PCEM_CNT3
    |    SCEM_CNT3
    |    COIN_CNT3
    |    PCEM_CNT4
    |    SCEM_CNT4
    |    COIN_CNT4
    |    PCEM_CNT5
    |    SCEM_CNT5
    |    COIN_CNT5

    When we read all packets and store data for above fields, it
    looks like this:

    |    SEQ_NUMBER   -> [0, 1, 2, 3, 4,..., 11, 1, 2, ......, 9, 10, 11]
    |    PCEM_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    SCEM_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    COIN_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    PCEM_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    SCEM_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    COIN_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    ....
    |    PCEM_RNG_ST5 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    SCEM_RNG_ST5 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    PCEM_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    SCEM_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    COIN_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    ....
    |    PCEM_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    SCEM_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
    |    COIN_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]

    This function reads each sweep data in this order:

    |    PCEM_CNT0 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    |    PCEM_CNT1 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    |    PCEM_CNT2 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    |    PCEM_CNT3 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    |    PCEM_CNT4 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    |    PCEM_CNT5 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

    This example show for PCEM_CNT but same logic applies
    to SCEM_CNT, COIN_CNT, PCEM_RNG, SCEM_RNG, and COIN_RNG.

    In the final L1A product of (total_number_of_sweep x 72) array where
    we store final PCEM, SCEM, COIN counts or compression indicator
    such as PCEM_RNG, SCEM_RNG, COIN_RNG,
    we want data in this order. Transpose of above layout

    |   0, 0, 0, 0, 0, 0,
    |   1, 1, 1, 1, 1, 1,
    |   2, 2, 2, 2, 2, 2,
    |   3, 3, 3, 3, 3, 3,
    |   ....,
    |   11, 11, 11, 11, 11, 11.

    Reordering in this order is reordering all data of
    sequence 0 first, then sequence 1, then sequence 2,
    and so on until sequence 11.

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Full dataset
    cem_prefix : str
        Indicate which CEM or its flag we are processing. Options are:

        |    PCEM_CNT
        |    SCEM_CNT
        |    COIN_CNT
        |    PCEM_RNG_ST
        |    SCEM_RNG_ST
        |    COIN_RNG_ST
    total_full_sweeps: int
        Total number of sweeps
    """
    all_cem_data = np.zeros((total_full_sweeps, 72))

    # First, concat all PCEM data
    current_cem_counts = np.concatenate(
        (
            full_sweep_sci[f"{cem_prefix}0"],
            full_sweep_sci[f"{cem_prefix}1"],
            full_sweep_sci[f"{cem_prefix}2"],
            full_sweep_sci[f"{cem_prefix}3"],
            full_sweep_sci[f"{cem_prefix}4"],
            full_sweep_sci[f"{cem_prefix}5"],
        ),
        axis=0,
    )

    # Next:
    # Reshape data by CEM, number of sweeps and sequence counts.
    # Therefore, the data shape is 6 x total_full_sweeps x 12
    # Output looks like this:
    # [
    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]

    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]

    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]

    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]

    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]

    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [ 1  2  3  4  5  6  7  8  9  10  11  12]
    # [ 2  3  4  5  6  7  8  9  10  11  12  13]]]
    # In other word, we grouped each cem's
    # data by full sweep.
    current_cem_counts = current_cem_counts.reshape(6, total_full_sweeps, 12)

    # Then, we go from above to
    # to this final output:
    # [
    # [[0  0  0  0  0  0]
    # [1  1  1  1  1  1]
    # [2  2  2  2  2  2]
    # [3  3  3  3  3  3]
    # [4  4  4  4  4  4]
    # [5  5  5  5  5  5]
    # [6  6  6  6  6  6]
    # [7  7  7  7  7  7]
    # [8  8  8  8  8  8]
    # [9  9  9  9  9  9]
    # [10 10 10 10 10 10]
    # [11 11 11 11 11 11]],
    # [[1  1  1  1  1  1]
    # [2  2  2  2  2  2]
    # [3  3  3  3  3  3]
    # ...
    # [12  12  12  12  12  12]],
    # [[2  2  2  2  2  2]
    # [3  3  3  3  3  3]
    # ...
    # [13  13  13  13  13  13]]
    # ]
    all_cem_data = np.stack(current_cem_counts, axis=-1)
    # This line just flatten the inner most array to
    # (total_full_sweeps x 72)
    all_cem_data = all_cem_data.reshape(total_full_sweeps, 72)
    return all_cem_data


def process_swapi_science(sci_dataset):
    """Process SWAPI science data.

    Parameters
    ----------
    dataset : xarray.Dataset
        L0 data
    """
    # ====================================================
    # Step 1: Filter full cycle data
    # ====================================================
    full_sweep_indices = get_indices_of_full_sweep(sci_dataset)
    full_sweep_sci = filter_full_cycle_data(full_sweep_indices, sci_dataset)
    # TODO: check for bad data
    # Find indices of bad sweep cycles
    bad_data_indices = check_for_bad_data(full_sweep_sci)
    if len(bad_data_indices) > 0:
        logging.info("Bad data detected")
        logging.info(bad_data_indices)
        # TODO: filter out bad data from full_sweep_sci
        # NOTE: may be use bad_data_indices to get good data indices
        # and pass to filter_full_cycle_data?

    # ====================================================
    # Step 2: Process full sweep data
    # ====================================================
    total_packets = len(full_sweep_sci["SEQ_NUMBER"].data)

    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence
    # These array will be of size (number of good sweep, 72)
    raw_pcem_count = process_sweep_data(full_sweep_sci, "PCEM_CNT", total_full_sweeps)
    raw_scem_count = process_sweep_data(full_sweep_sci, "SCEM_CNT", total_full_sweeps)
    raw_coin_count = process_sweep_data(full_sweep_sci, "COIN_CNT", total_full_sweeps)
    pcem_compression_flags = process_sweep_data(
        full_sweep_sci, "PCEM_RNG_ST", total_full_sweeps
    )
    scem_compression_flags = process_sweep_data(
        full_sweep_sci, "SCEM_RNG_ST", total_full_sweeps
    )
    coin_compression_flags = process_sweep_data(
        full_sweep_sci, "COIN_RNG_ST", total_full_sweeps
    )

    print(raw_pcem_count)
    swp_pcem_counts = decompress_count(raw_pcem_count, pcem_compression_flags)
    swp_scem_counts = decompress_count(raw_scem_count, scem_compression_flags)
    swp_coin_counts = decompress_count(raw_coin_count, coin_compression_flags)

    # ===================================================================
    # Step 3: Create xarray.Dataset
    # ===================================================================

    # Epoch time. Should be same dimension as number of good sweeps
    epoch_time = full_sweep_sci["Epoch"].data.reshape(total_full_sweeps, 12)[:, 0]
    epoch_time = xr.DataArray(
        epoch_time,
        name="Epoch",
        dims=["Epoch"],
    )

    # There are 72 energy steps
    energy = xr.DataArray(np.arange(72), name="Energy", dims=["Energy"])

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time, "Energy": energy},
    )

    dataset["SWP_PCEM_COUNTS"] = xr.DataArray(swp_pcem_counts, dims=["Epoch", "Energy"])
    dataset["SWP_SCEM_COUNTS"] = xr.DataArray(swp_scem_counts, dims=["Epoch", "Energy"])
    dataset["SWP_COIN_COUNTS"] = xr.DataArray(swp_coin_counts, dims=["Epoch", "Energy"])

    # L1 quality flags
    dataset["SWP_PCEM_FLAGS"] = xr.DataArray(
        pcem_compression_flags, dims=["Epoch", "Energy"]
    )
    dataset["SWP_SCEM_FLAGS"] = xr.DataArray(
        scem_compression_flags, dims=["Epoch", "Energy"]
    )
    dataset["SWP_COIN_FLAGS"] = xr.DataArray(
        coin_compression_flags, dims=["Epoch", "Energy"]
    )

    # ===================================================================
    # Step 4: Calculate uncertainty
    # ===================================================================
    # Uncertainty in counts formula:
    # Uncertainty is quantified for the PCEM, SCEM, and COIN counts.
    # The Poisson contribution is
    #   uncertainty = sqrt(count)
    dataset["SWP_PCEM_ERR"] = xr.DataArray(
        np.sqrt(swp_pcem_counts), dims=["Epoch", "Energy"]
    )
    dataset["SWP_SCEM_ERR"] = xr.DataArray(
        np.sqrt(swp_scem_counts), dims=["Epoch", "Energy"]
    )
    dataset["SWP_COIN_ERR"] = xr.DataArray(
        np.sqrt(swp_coin_counts), dims=["Epoch", "Energy"]
    )
    return dataset


def swapi_l1(packets):
    """Based on APID, process SWAPI L0 data to level 1.

    Parameters
    ----------
    packets : list
        List of decom packets
    """
    grouped_packets = group_by_apid(packets)
    for apid in grouped_packets.keys():
        # Right now, we only process SWP_HK and SWP_SCI
        # other packets are not process in this processing pipeline
        # If appId is science, then the file should contain all data of science appId
        sorted_packets = sort_by_time(grouped_packets[apid], "SHCOARSE")
        ds_data = create_dataset(sorted_packets)

        if apid == SWAPIAPID.SWP_SCI.value:
            process_swapi_science(ds_data)
            # TODO: save full sweep data to CDF
