"""SWAPI level-1 processing code."""
import collections
import copy

import numpy as np
import xarray as xr

from imap_processing.swapi.swapi_utils import SWAPIAPID, create_dataset
from imap_processing.utils import group_by_apid, sort_by_time


def group_full_sweep_data(packets):
    """Group data by PLAN_ID and SWEEP_TABLE.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    dict
        grouped data by PLAN_ID and SWEEP_TABLE.
    """
    grouped_data = collections.defaultdict(list)
    for packet in packets:
        plan_id = packet.data["PLAN_ID_SCIENCE"]
        sweep_table = packet.data["SWEEP_TABLE"]
        grouped_data.setdefault(f"{plan_id}_{sweep_table}", []).append(packet)
    return grouped_data


def find_sweep_starts(sweep: np.ndarray):
    """Find index of where new cycle started.

    Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0
    (Sequence number of set of steps in energy sweep);
    end of a sweep is marked by SWP_SCI.SEQ_NUMBER=11;
    In this function, we look for index of SEQ_NUMBER 0.

    Brandon Stone helped developed this algorithm.

    Parameters
    ----------
    sweep : numpy.ndarray
        Array that contains quarter cycle information.

    Returns
    -------
    numpy.ndarray
        Array of indices of start cycle.
    """
    if sweep.size < 12:
        return np.array([], np.int64)

    # calculate difference between consecutive sweep
    diff = sweep[1:] - sweep[:-1]

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
        (sweep == 0)[:-11]
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


def get_indices_of_full_sweep(seq_number: np.ndarray):
    """Get indices of full cycles.

    Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0
    (Sequence number of set of steps in energy sweep);
    end of a sweep is marked by SWP_SCI.SEQ_NUMBER=11;
    all packets must be present to process a sweep.

    In this function, we get the indices of SEQ_NUMBER
    0 and then construct full sweep indices.

    Parameters
    ----------
    seq_number : numpy.ndarray
        Array that contains SEQ_NUMBER data informations.
        Eg. sci_dataset["SEQ_NUMBER"].data

    Returns
    -------
    numpy.ndarray
        1D array with indices of full cycle data.
    """
    indices_of_start = find_sweep_starts(seq_number)
    # find_sweep_starts[..., None] creates array of shape(n, 1).
    #   Eg. [[3], [8]]
    # np.arange(12)[None, ...] creates array of shape(1, 12)
    #   Eg. [[0, 1, 2, 3, ....., 11]]
    # then we add both of them together to get an array of shape(n, 4)
    #   Eg. [[3, 4, 5, 6,...14], [8, 9, 10, 11, ..., 19]]
    full_cycles_indices = indices_of_start[..., None] + np.arange(12)[None, ...]
    return full_cycles_indices.reshape(-1)


def filter_full_cycle_data(full_cycle_data_indices: np.ndarray, l1a_data: xr.Dataset):
    """Filter metadata and science of packets that makes full cycles.

    Parameters
    ----------
    full_cycle_data_indices : numpy.ndarray
        Array with indices of full cycles.
    l1a_data : xarray.dataset
        L1A dataset

    Returns
    -------
    xarray.dataset
        L1A dataset with filtered metadata.
    """
    # Had to create new xr.Dataset because Epoch shape and new data variables shapes was
    # different.
    full_sweep_dataset = xr.Dataset(
        coords={"Epoch": l1a_data["Epoch"].data[full_cycle_data_indices]}
    )
    for key, value in l1a_data.items():
        full_sweep_dataset[key] = xr.DataArray(value.data[full_cycle_data_indices])
    return full_sweep_dataset


def decompress_count(count_data: np.ndarray, compression_flag: np.ndarray = None):
    """Decompress counts based on compression indicators.

    Decompression algorithm:
    There are 3 compression regions:
        1) 0 <= value <=65535
        2) 65536 <= value <= 1,048,575
        3) 1,048,576 <= value

        Pseudocode:
        if XXX_RNG_ST0 == 0:          # Not compressed
            actual_value = XXX_CNT0
        elif (XXX_RNG_ST0==1 && XXX_CNT0==0xFFFF):    # Overflow
            actual_value = <some constant that indicates overflow>
        elif (XXX_RNG_ST0==1 && XXX_CNT0!=0xFFFF):
            actual_value = XXX_CNT0 * 16

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
    compressed_count_indices = np.where(compression_flag == 1)[0]
    print("index", compressed_count_indices)
    for index in compressed_count_indices:
        if count_data[index] == 0xFFFF:  # Overflow
            new_count[index] = -1
        elif count_data[index] != 0xFFFF:
            new_count[index] = count_data[index] * 16
    return new_count


def process_cem_data(full_sweep_sci, cem_prefix, m, n):
    """Combine certain CEM's data or their flag.

    Here, it combines certain CEM's data or their flag
    data and then apply transformation to get
    data in the correct sequence order and in the final
    data shape required by SWAPI.

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Full dataset
    cem_prefix : str
        This will indicate which CEM or its flag we are processing.
        Options are:
            PCEM_CNT
            SCEM_CNT
            COIN_CNT
            PCEM_RNG_ST
            SCEM_RNG_ST
            COIN_RNG_ST
    m : int
        Start index of current sweep.
    n : int
        End index of current sweep.

    Returns
    -------
    numpy.ndarray
        Array with data in the correct sequence order.
    """
    # For each PCEM, SCEM, COIN, and their quality flags, the data
    # is in sequence order. But in final sweep data, we need to
    # flatten array in the correct order of sequence number.
    # Eg.
    # Here, we reorder data from this:
    # [
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    # ]
    #
    # to this:
    # [
    #   0  0  0  0  0  0
    #   1  1  1  1  1  1
    #   2  2  2  2  2  2
    #   3  3  3  3  3  3
    #   4  4  4  4  4  4
    #   5  5  5  5  5  5
    #   6  6  6  6  6  6
    #   7  7  7  7  7  7
    #   8  8  8  8  8  8
    #   9  9  9  9  9  9
    #   10 10 10 10 10 10
    #   11 11 11 11 11 11
    # ]
    return np.vstack(
        (
            full_sweep_sci[f"{cem_prefix}0"].data[m:n],
            full_sweep_sci[f"{cem_prefix}1"].data[m:n],
            full_sweep_sci[f"{cem_prefix}2"].data[m:n],
            full_sweep_sci[f"{cem_prefix}3"].data[m:n],
            full_sweep_sci[f"{cem_prefix}4"].data[m:n],
            full_sweep_sci[f"{cem_prefix}5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]


def process_full_sweep_data(full_sweep_sci, sweep_start_index, sweep_end_index):
    """Process full sweep data.

    Twelve consecutive seconds of science data correspond to a single sweep,
    which is the unit of L1a data and downstream products.

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Science data that only contains full sweep data.
    sweep_start_index : int
        Start index of current sweep.
    sweep_end_index: int
        End index of current sweep.

    Returns
    -------
    pcem_count : numpy.ndarray
        List of decompressed PCEM counts.
    scem_count : numpy.ndarray
        List of decompressed SCEM counts.
    coin_count : numpy.ndarray
        List of decompressedCOIN counts.
    pcem_rng_val : numpy.ndarray
        List of PCEM compression indicator.
    scem_rng_val : numpy.ndarray
        List of SCEM compression indicator.
    coin_rng_val : numpy.ndarray
        List of COIN compression indicator.
    """
    # current sweep start and end index
    m = sweep_start_index
    n = sweep_end_index

    # All of these count and compression flag is 1x72 array

    raw_pcem_count = process_cem_data(full_sweep_sci, "PCEM_CNT", m, n)
    raw_scem_count = process_cem_data(full_sweep_sci, "SCEM_CNT", m, n)
    raw_coin_count = process_cem_data(full_sweep_sci, "COIN_CNT", m, n)

    # Compression indicators
    # XXX_RNG_ST{step} --> 0: not compressed, 1: compressed
    pcem_compression_flag = process_cem_data(full_sweep_sci, "PCEM_RNG_ST", m, n)
    scem_compression_flag = process_cem_data(full_sweep_sci, "SCEM_RNG_ST", m, n)
    coin_compression_flag = process_cem_data(full_sweep_sci, "COIN_RNG_ST", m, n)

    # Decompress counts using compression flags
    pcem_count = decompress_count(raw_pcem_count, pcem_compression_flag)
    scem_count = decompress_count(raw_scem_count, scem_compression_flag)
    coin_count = decompress_count(raw_coin_count, coin_compression_flag)

    return (
        pcem_count,
        scem_count,
        coin_count,
        pcem_compression_flag,
        scem_compression_flag,
        coin_compression_flag,
    )


def check_for_bad_data(full_sweep_sci, sweep_start_index, sweep_end_index):
    """Check for bad data.

    Bad data indicator:
    SWP_HK.CHKSUM is wrong
    SWAPI mode (SWP_SCI.MODE) is not HVSCI
    Check for saturation. If count rates exceed 4.0 MHz using SWP_SCI.PCEM_CNT0
    through SWP_SCI.PCEM_CNT5 or SWP_SCI.SCEM_CNT0 through
    SWP_SCI.SCEM_CNT5 the sweep may later be discarded.
     SWP_HK.PCEM_RATE_ST = 1 at any point during the sweep. The count rate threshold
    for the PCEM counter has been exceeded once and has continued to be exceeded despite
    measures by FSW
     SWP_HK.SCEM_RATE_ST = 1 at any point during the sweep. The count rate threshold
    for the SCEM counter has been exceeded once and has continued to be exceeded despite
    measures by FSW
    Note: The FSW automatically transitions to SAFE if:
    PCEM current or voltage, SCEM current or voltage, or temperature, current, or
    voltage
    sensors experience two consecutive samples out-of-limit
    PCEM count rate or SCEM count rate experience six or more consecutive
    samples out-oflimit
    (note: count rate limit is 4.0 MHz for PCEM and SCEM).

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Science data that only contains full sweep data.
    sweep_start_index : int
        Start index of current sweep.
    sweep_end_index: int
        End index of current sweep.

    Returns
    -------
    bool
        True if bad data is found, False otherwise.
    """
    # current sweep start and end index
    m = sweep_start_index
    n = sweep_end_index

    # If PLAN_ID and SWEEP_TABLE is not same, the discard the
    # sweep data. PLAN_ID and SWEEP_TABLE should match
    if not np.all(full_sweep_sci["PLAN_ID_SCIENCE"].data[m:n]) and np.all(
        full_sweep_sci["SWEEP_TABLE"].data[m:n]
    ):
        # TODO: add log here
        return True
    # TODO: add other checks for bad data
    return False


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
    full_sweep_indices = get_indices_of_full_sweep(sci_dataset["SEQ_NUMBER"].data)
    full_sweep_sci = filter_full_cycle_data(full_sweep_indices, sci_dataset)

    # ====================================================
    # Step 2: Process full sweep data
    # ====================================================
    total_packets = len(full_sweep_sci["SEQ_NUMBER"].data)
    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence
    swp_pcem_counts = np.zeros((total_full_sweeps, 72))
    swp_scem_counts = np.zeros((total_full_sweeps, 72))
    swp_coin_counts = np.zeros((total_full_sweeps, 72))
    swp_pcem_comp = np.zeros((total_full_sweeps, 72))
    swp_scem_comp = np.zeros((total_full_sweeps, 72))
    swp_coin_comp = np.zeros((total_full_sweeps, 72))
    collections.defaultdict(list)

    # Step through each twelve packets that makes full sweep
    for sweep_index in range(0, total_packets, 12):
        # current sweep start index
        m = sweep_index
        # current sweep end index
        n = sweep_index + 12

        # Index in the final data array
        idx = sweep_index // 12
        (
            swp_pcem_counts[idx],
            swp_scem_counts[idx],
            swp_coin_counts[idx],
            swp_pcem_comp[idx],
            swp_scem_comp[idx],
            swp_coin_comp[idx],
        ) = process_full_sweep_data(
            full_sweep_sci=full_sweep_sci, sweep_start_index=m, sweep_end_index=n
        )

    # ===================================================================
    # Step 3: Create xarray.Dataset
    # ===================================================================

    # Get Epoch time of full sweep data and then reshape it to
    # (n, 12) where n = total number of full sweep data and 12 = 12
    # sequence data's metadata. For Epoch's data, we take the first element
    # of each 12 sequence data's metadata.
    epoch_time = xr.DataArray(
        sci_dataset["Epoch"].data[full_sweep_indices].reshape(-1, 12)[:, 0],
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
    dataset["SWP_PCEM_RNG_ST_COMP"] = xr.DataArray(
        swp_pcem_comp, dims=["Epoch", "Energy"]
    )
    dataset["SWP_SCEM_RNG_ST_COMP"] = xr.DataArray(
        swp_scem_comp, dims=["Epoch", "Energy"]
    )
    dataset["SWP_COIN_RNG_ST_COMP"] = xr.DataArray(
        swp_coin_comp, dims=["Epoch", "Energy"]
    )

    # Uncertainty in counts formula:
    # Uncertainty is quantified for the PCEM, SCEM, and COIN counts.
    # The Poisson contribution is
    # uncertainty = sqrt(count)
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
    """Process SWAPI L0 data to level 1.

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
            data = process_swapi_science(ds_data)
            print(data)
            # TODO: save full sweep data to CDF
