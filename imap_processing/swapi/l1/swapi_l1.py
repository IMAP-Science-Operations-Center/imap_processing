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
    """Uncompress counts based on compression indicators.

    Decompression algorithm:
    There are 3 compression regions:
        1) 0 <= value <=65535
        2) 65536 <= value <= 1,048,575
        3) 1,048,576 <= value

        Pseudocode:
        if XXX_RNG_ST0 == 0:          # Uncompressed
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
        Array with uncompressed counts.
    """
    # Uncompress counts based on compression indicators
    # If 0, value is already uncompressed. If 1, value is compressed.
    # If 1 and count is 0xFFFF, value is overflow.
    new_count = copy.deepcopy(count_data)
    compressed_count_indices = np.where(compression_flag == 1)[0]

    for index in compressed_count_indices:
        if count_data[index] == 0xFFFF:  # Overflow
            new_count[index] = -1
        elif count_data[index] != 0xFFFF:
            new_count[index] = count_data[index] * 16
    return new_count


def create_full_sweep_data(full_sweep_sci, sweep_index):
    """Process full sweep data.

    Twelve consecutive seconds of science data correspond to a single sweep,
    which is the unit of L1a data and downstream products.
        1. SCI data packets are produced at 1Hz, so 12 packets = 1 sweep.
        2. Find 12 packets spanning 12 seconds (look at SWP_SCI.SHCOARSE)
            with matching SWP_SCI.PLAN_ID (The PLAN ID used for the current
            science sweeping mode if any) and SWP_SCI.SWEEP_TABLE (Sweep table ID
            within the PLAN ID that is being used for the current science sweeping
            mode) to group packets by sweep
        3. Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0 (Sequence
            number of set of steps in energy sweep); end of a sweep is
            marked by SWP_SCI.SEQ_NUMBER=11; all packets must be present
            to process a sweep
        4. Only process complete sweeps (72 steps).

    Processing steps:
        * Define empty arrays for the column data (PCEM counts, SCEM counts,
            COIN counts, etc.
        * Save PCEM, SCEM, and COIN counts stored in
            SWP_SCI.PCEM_CNT0 through SWP_SCI.PCEM_CNT5,
            SWP_SCI.SCEM_CNT0 through SWP_SCI.SCEM_CNT5, and
            SWP_SCI.COIN_CNT0 through SWP_SCI.COIN_CNT5, respectively
            as 1x72 arrays to include the counts during the
            1st, 2nd, 3rd, 4th, 5th, and 6th 1/6-second time interval of
            each of the 12 consecutive packets.

            We got 1x72 array because we create one array for each
            PCEM, SCEM, COIN. We got 72 element because
            each packet contains xxxx_CNT0 to xxxx_CNT5 (6 counts).
            Since we have 12 packets for one sweep each containing
            6 counts, 6 x 12, gives us 72 elements.

            Decipher PCEM, SCEM, and COIN counts using
            SWP_SCI.PCEM_RNG_ST0 through SWP_SCI.PCEM_RNG_ST5,
            SWP_SCI.SCEM_RNG_ST0 through SWP_SCI.SCEM_RNG_ST5, and
            SWP_SCI.COIN_RNG_ST0 through SWP_SCI.COIN_RNG_ST5 as indicators
            of compressed count ranges that should be converted to raw format.


    In other word, data from each packet comes like this:
        SEQ_NUMBER
        .
        PCEM_RNG_ST0
        SCEM_RNG_ST0
        COIN_RNG_ST0
        PCEM_RNG_ST1
        SCEM_RNG_ST1
        COIN_RNG_ST1
        PCEM_RNG_ST2
        SCEM_RNG_ST2
        COIN_RNG_ST2
        PCEM_RNG_ST3
        SCEM_RNG_ST3
        COIN_RNG_ST3
        PCEM_RNG_ST4
        SCEM_RNG_ST4
        COIN_RNG_ST4
        PCEM_RNG_ST5
        SCEM_RNG_ST5
        COIN_RNG_ST5
        PCEM_CNT0
        SCEM_CNT0
        COIN_CNT0
        PCEM_CNT1
        SCEM_CNT1
        COIN_CNT1
        PCEM_CNT2
        SCEM_CNT2
        COIN_CNT2
        PCEM_CNT3
        SCEM_CNT3
        COIN_CNT3
        PCEM_CNT4
        SCEM_CNT4
        COIN_CNT4
        PCEM_CNT5
        SCEM_CNT5
        COIN_CNT5
    When we read all packets and filter all full sweep data, it
    looks like this:
        SEQ_NUMBER   -> [0, 1, 2, 3, 4,..., 11, 1, 2, ......, 9, 10, 11]
        PCEM_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        SCEM_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        COIN_RNG_ST0 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        PCEM_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        SCEM_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        COIN_RNG_ST1 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        ....
        PCEM_RNG_ST5 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        SCEM_RNG_ST5 -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        PCEM_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        SCEM_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        COIN_CNT_0   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        ....
        PCEM_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        SCEM_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]
        COIN_CNT_5   -> [x, x, x, x, x,..., x, x, x, ..., x, x, x]

    This function reads one full sweep data in this order:
        PCEM_CNT0 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        PCEM_CNT1 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        PCEM_CNT2 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        PCEM_CNT3 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        PCEM_CNT4 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        PCEM_CNT5 --> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

    This example show for PCEM_CNT but same logic applies
    to SCEM_CNT, COIN_CNT, PCEM_RNG, SCEM_RNG, and COIN_RNG.

    In the final L1A product of 1x72 array where we store
    final PCEM, SCEM, COIN counts or compression indicator
    such as PCEM_RNG, SCEM_RNG, COIN_RNG,
    we want data in this order. Transpose of above layout
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3,
    ....,
    11, 11, 11, 11, 11, 11.
    Reordering in this order is reordering all data of
    sequence 0 first, then sequence 1, then sequence 2,
    and so on until sequence 11.

    Parameters
    ----------
    full_sweep_sci : xarray.Dataset
        Science data that only contains full sweep data.
    sweep_index : int
        Start index of current sweep.

    Returns
    -------
    pcem_count : list
        List of PCEM counts.
    scem_count : list
        List of SCEM counts.
    coin_count : list
        List of COIN counts.
    pcem_rng_val : list
        List of PCEM compression indicator.
    scem_rng_val : list
        List of SCEM compression indicator.
    coin_rng_val : list
        List of COIN compression indicator.
    """
    # current sweep start and end index
    m = sweep_index
    n = sweep_index + 12

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
    raw_pcem_count = np.vstack(
        (
            full_sweep_sci["PCEM_CNT0"].data[m:n],
            full_sweep_sci["PCEM_CNT1"].data[m:n],
            full_sweep_sci["PCEM_CNT2"].data[m:n],
            full_sweep_sci["PCEM_CNT3"].data[m:n],
            full_sweep_sci["PCEM_CNT4"].data[m:n],
            full_sweep_sci["PCEM_CNT5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]

    raw_scem_count = np.vstack(
        (
            full_sweep_sci["SCEM_CNT0"].data[m:n],
            full_sweep_sci["SCEM_CNT1"].data[m:n],
            full_sweep_sci["SCEM_CNT2"].data[m:n],
            full_sweep_sci["SCEM_CNT3"].data[m:n],
            full_sweep_sci["SCEM_CNT4"].data[m:n],
            full_sweep_sci["SCEM_CNT5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]
    raw_coin_count = np.vstack(
        (
            full_sweep_sci["COIN_CNT0"].data[m:n],
            full_sweep_sci["COIN_CNT1"].data[m:n],
            full_sweep_sci["COIN_CNT2"].data[m:n],
            full_sweep_sci["COIN_CNT3"].data[m:n],
            full_sweep_sci["COIN_CNT4"].data[m:n],
            full_sweep_sci["COIN_CNT5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]

    # Compression indicators
    # XXX_RNG_ST{step} --> 0: uncompressed, 1: compressed
    pcem_compression_flag = np.vstack(
        (
            full_sweep_sci["PCEM_RNG_ST0"].data[m:n],
            full_sweep_sci["PCEM_RNG_ST1"].data[m:n],
            full_sweep_sci["PCEM_RNG_ST2"].data[m:n],
            full_sweep_sci["PCEM_RNG_ST3"].data[m:n],
            full_sweep_sci["PCEM_RNG_ST4"].data[m:n],
            full_sweep_sci["PCEM_RNG_ST5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]
    scem_compression_flag = np.vstack(
        (
            full_sweep_sci["SCEM_RNG_ST0"].data[m:n],
            full_sweep_sci["SCEM_RNG_ST1"].data[m:n],
            full_sweep_sci["SCEM_RNG_ST2"].data[m:n],
            full_sweep_sci["SCEM_RNG_ST3"].data[m:n],
            full_sweep_sci["SCEM_RNG_ST4"].data[m:n],
            full_sweep_sci["SCEM_RNG_ST5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]
    coin_compression_flag = np.vstack(
        (
            full_sweep_sci["COIN_RNG_ST0"].data[m:n],
            full_sweep_sci["COIN_RNG_ST1"].data[m:n],
            full_sweep_sci["COIN_RNG_ST2"].data[m:n],
            full_sweep_sci["COIN_RNG_ST3"].data[m:n],
            full_sweep_sci["COIN_RNG_ST4"].data[m:n],
            full_sweep_sci["COIN_RNG_ST5"].data[m:n],
        )
    ).T.reshape(1, 72)[0]

    # Uncompress counts using compression flags
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


def process_swapi_science(sci_dataset):
    """Process SWAPI science data.

    Parameters
    ----------
    dataset : xarray.Dataset
        L0 data
    """
    full_sweep_indices = get_indices_of_full_sweep(sci_dataset["SEQ_NUMBER"].data)
    full_sweep_sci = filter_full_cycle_data(full_sweep_indices, sci_dataset)
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
        m = sweep_index
        n = sweep_index + 12
        # If PLAN_ID and SWEEP_TABLE is not same, the discard the
        # sweep data. PLAN_ID and SWEEP_TABLE should match
        if np.all(full_sweep_sci["PLAN_ID_SCIENCE"].data[m:n]) and np.all(
            full_sweep_sci["SWEEP_TABLE"].data[m:n]
        ):
            # TODO: add log here
            continue
        # TODO: add other checks for bad data
        idx = sweep_index // 12

        (
            swp_pcem_counts[idx],
            swp_scem_counts[idx],
            swp_coin_counts[idx],
            swp_pcem_comp[idx],
            swp_scem_comp[idx],
            swp_coin_comp[idx],
        ) = create_full_sweep_data(full_sweep_sci, sweep_index)

    print(swp_pcem_counts)
    # Get Epoch time of full sweep data and then reshape it to
    # (n, 12) where n = total number of full sweep data and 12 = 12
    # sequence data's metadata. For Epoch's data, we take the first element
    # of each 12 sequence data's metadata.
    epoch_time = xr.DataArray(
        sci_dataset["Epoch"].data[full_sweep_indices].reshape(-1, 12)[:, 0],
        name="Epoch",
        dims=["Epoch"],
    )
    counts = xr.DataArray(np.arange(72), name="Counts", dims=["Counts"])

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time, "Counts": counts},
    )

    dataset["SWP_PCEM_COUNTS"] = xr.DataArray(swp_pcem_counts, dims=["Epoch", "Counts"])
    dataset["SWP_SCEM_COUNTS"] = xr.DataArray(swp_scem_counts, dims=["Epoch", "Counts"])
    dataset["SWP_COIN_COUNTS"] = xr.DataArray(swp_coin_counts, dims=["Epoch", "Counts"])

    # Uncertainty in counts formula:
    # Uncertainty is quantified for the PCEM, SCEM, and COIN counts.
    # The Poisson contribution is
    # uncertainty = sqrt(count)
    dataset["SWP_PCEM_ERR"] = xr.DataArray(
        np.sqrt(swp_pcem_counts), dims=["Epoch", "Counts"]
    )
    dataset["SWP_SCEM_ERR"] = xr.DataArray(
        np.sqrt(swp_scem_counts), dims=["Epoch", "Counts"]
    )
    dataset["SWP_COIN_ERR"] = xr.DataArray(
        np.sqrt(swp_coin_counts), dims=["Epoch", "Counts"]
    )

    # L1A quality flags
    dataset["SWP_PCEM_RNG_ST_COMP"] = xr.DataArray(
        swp_pcem_comp, dims=["Epoch", "Counts"]
    )
    dataset["SWP_SCEM_RNG_ST_COMP"] = xr.DataArray(
        swp_scem_comp, dims=["Epoch", "Counts"]
    )
    dataset["SWP_COIN_RNG_ST_COMP"] = xr.DataArray(
        swp_coin_comp, dims=["Epoch", "Counts"]
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
