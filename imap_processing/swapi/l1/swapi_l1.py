"""SWAPI level-1 processing code."""

import copy

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf import epoch_attrs
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swapi.swapi_utils import SWAPIAPID, SWAPIMODE
from imap_processing.utils import packet_file_to_datasets


def filter_good_data(full_sweep_sci: xr.Dataset) -> np.ndarray:
    """
    Filter out bad data sweep indices.

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
    numpy.ndarray
        Good data sweep indices.
    """
    # PLAN_ID for current sweep should all be one value and
    # SWEEP_TABLE should all be one value.
    plan_id = full_sweep_sci["plan_id_science"].data.reshape(-1, 12)
    sweep_table = full_sweep_sci["sweep_table"].data.reshape(-1, 12)

    mode = full_sweep_sci["mode"].data.reshape(-1, 12)

    sweep_indices = (sweep_table == sweep_table[:, 0, None]).all(axis=1)
    plan_id_indices = (plan_id == plan_id[:, 0, None]).all(axis=1)
    # TODO: change comparison to SWAPIMODE.HVSCI once we have
    # some HVSCI data
    # MODE should be HVSCI
    mode_indices = (mode == SWAPIMODE.HVENG).all(axis=1)
    bad_data_indices = sweep_indices & plan_id_indices & mode_indices

    # TODO: add checks for checksum

    # Get bad data sweep start indices and create
    # sweep indices.
    # Eg.
    # From this: [0 24]
    # To this: [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [24 25 26 27 28 29 30 31 32 33 34 35]]
    cycle_start_indices = np.where(bad_data_indices == 0)[0] * 12
    bad_cycle_indices = cycle_start_indices[..., None] + np.arange(12)[
        None, ...
    ].reshape(-1)

    # Use bad data cycle indices to find all good data indices.
    # Then that will used to filter good sweep data.
    all_indices = np.arange(len(full_sweep_sci["epoch"].data))
    good_data_indices = np.setdiff1d(all_indices, bad_cycle_indices)

    return good_data_indices


def decompress_count(
    count_data: np.ndarray, compression_flag: np.ndarray
) -> np.ndarray:
    """
    Will decompress counts based on compression indicators.

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

    # If data is compressed, decompress it
    compressed_indices = compression_flag == 1
    new_count[compressed_indices] *= 16
    # TODO: add data type check here.
    # If the data was compressed and the count was 0xFFFF, mark it as an overflow
    if np.any(count_data < 0):
        raise ValueError(
            "Count data type must be unsigned int and should not contain negative value"
        )
    new_count[compressed_indices & (count_data == 0xFFFF)] = -1
    return new_count


def find_sweep_starts(packets: xr.Dataset) -> np.ndarray:
    """
    Find index of where new cycle started.

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
    if packets["epoch"].size < 12:
        return np.array([], np.int64)

    # calculate time difference between consecutive sweep
    diff = packets["epoch"].data[1:] - packets["epoch"].data[:-1]

    # This uses sliding window to find index where cycle starts.
    # This is what this below code line is doing:
    # [1 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0]      # Is cycle zero?
    # [1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1]      # Next diff is one?
    #   [1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0]    # Next diff is one?
    #     [0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 0 0]  # Next diff is one?
    #
    # [0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0]      # And all?

    ione = diff == 1e9  # 1 second

    valid = (
        (packets["seq_number"] == 0)[:-11]
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


def get_indices_of_full_sweep(packets: xr.Dataset) -> np.ndarray:
    """
    Get indices of full cycles.

    Beginning of a sweep is marked by SWP_SCI.SEQ_NUMBER=0
    (Sequence number of set of steps in energy sweep);
    end of a sweep is marked by SWP_SCI.SEQ_NUMBER=11;
    all packets must be present to process a sweep.

    In this function, we get the indices of SEQ_NUMBER
    0 and then construct full sweep indices.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset that contains SEQ_NUMBER data information.
        Eg. sci_dataset["SEQ_NUMBER"].data.

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


def process_sweep_data(full_sweep_sci: xr.Dataset, cem_prefix: str) -> xr.Dataset:
    """
    Group full sweep data into correct sequence order.

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
        Full dataset.
    cem_prefix : str
        Indicate which CEM or its flag we are processing.
        Options are:

        |    PCEM_CNT
        |    SCEM_CNT
        |    COIN_CNT
        |    PCEM_RNG_ST
        |    SCEM_RNG_ST
        |    COIN_RNG_ST.

    Returns
    -------
    all_cem_data : xarray.Dataset
        Correctly order dataset.
    """
    # First, concat all CEM data
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
    current_cem_counts = current_cem_counts.reshape(6, -1, 12)

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
    #
    # [[1  1  1  1  1  1]
    # [2  2  2  2  2  2]
    # [3  3  3  3  3  3]
    # ...
    # [12  12  12  12  12  12]],
    #
    # [[2  2  2  2  2  2]
    # [3  3  3  3  3  3]
    # ...
    # [13  13  13  13  13  13]]
    # ]
    # In other word, we grouped by sequence. The shape
    # of this transformed array is total_full_sweeps x 12 x 6
    all_cem_data = np.stack(current_cem_counts, axis=-1)
    # This line just flatten the inner most array to
    # (total_full_sweeps x 72)
    all_cem_data = all_cem_data.reshape(-1, 72)
    return all_cem_data


def process_swapi_science(sci_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process SWAPI science data and create CDF file.

    Parameters
    ----------
    sci_dataset : xarray.Dataset
        L0 data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset.
    """
    # ====================================================
    # Step 1: Filter full cycle data
    # ====================================================
    full_sweep_indices = get_indices_of_full_sweep(sci_dataset)

    # Filter full sweep data using indices returned from above line
    full_sweep_sci = sci_dataset.isel({"epoch": full_sweep_indices})

    # Find indices of good sweep cycles
    good_data_indices = filter_good_data(full_sweep_sci)

    good_sweep_sci = full_sweep_sci.isel({"epoch": good_data_indices})

    # ====================================================
    # Step 2: Process good sweep data
    # ====================================================
    total_packets = len(good_sweep_sci["seq_number"].data)

    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence
    # These array will be of size (number of good sweep, 72)
    raw_pcem_count = process_sweep_data(good_sweep_sci, "pcem_cnt")
    raw_scem_count = process_sweep_data(good_sweep_sci, "scem_cnt")
    raw_coin_count = process_sweep_data(good_sweep_sci, "coin_cnt")
    pcem_compression_flags = process_sweep_data(good_sweep_sci, "pcem_rng_st")
    scem_compression_flags = process_sweep_data(good_sweep_sci, "scem_rng_st")
    coin_compression_flags = process_sweep_data(good_sweep_sci, "coin_rng_st")

    swp_pcem_counts = decompress_count(raw_pcem_count, pcem_compression_flags)
    swp_scem_counts = decompress_count(raw_scem_count, scem_compression_flags)
    swp_coin_counts = decompress_count(raw_coin_count, coin_compression_flags)

    # ===================================================================
    # Step 3: Create xarray.Dataset
    # ===================================================================

    # epoch time. Should be same dimension as number of good sweeps
    epoch_values = good_sweep_sci["epoch"].data.reshape(total_full_sweeps, 12)[:, 0]

    # Load the CDF attributes
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("swapi")
    cdf_manager.load_variable_attributes("imap_swapi_variable_attrs.yaml")

    epoch_time = xr.DataArray(
        epoch_values,
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
    )

    # There are 72 energy steps
    energy = xr.DataArray(
        np.arange(72),
        name="energy",
        dims=["energy"],
        attrs=cdf_manager.get_variable_attributes("energy"),
    )
    # LABL_PTR_1 should be CDF_CHAR.
    energy_label = xr.DataArray(
        energy.values.astype(str),
        name="energy_label",
        dims=["energy_label"],
        attrs=cdf_manager.get_variable_attributes("energy_label"),
    )

    # Add other global attributes
    cdf_manager.add_global_attribute("Data_version", data_version)
    cdf_manager.add_global_attribute(
        "sweep_table", f"{sci_dataset['sweep_table'].data[0]}"
    )
    cdf_manager.add_global_attribute(
        "plan_id", f"{sci_dataset['plan_id_science'].data[0]}"
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time, "energy": energy, "energy_label": energy_label},
        attrs=cdf_manager.get_global_attributes("imap_swapi_l1_sci"),
    )

    dataset["swp_pcem_counts"] = xr.DataArray(
        np.array(swp_pcem_counts, dtype=np.uint16),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("pcem_counts"),
    )
    dataset["swp_scem_counts"] = xr.DataArray(
        np.array(swp_scem_counts, dtype=np.uint16),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("scem_counts"),
    )
    dataset["swp_coin_counts"] = xr.DataArray(
        np.array(swp_coin_counts, dtype=np.uint16),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("coin_counts"),
    )

    # L1 quality flags
    # TODO: Should these be kept in raw format rather than derived into strings?
    dataset["swp_pcem_flags"] = xr.DataArray(
        np.array(pcem_compression_flags, dtype=np.uint8),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("pcem_flags"),
    )
    dataset["swp_scem_flags"] = xr.DataArray(
        np.array(scem_compression_flags, dtype=np.uint8),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("scem_flags"),
    )
    dataset["swp_coin_flags"] = xr.DataArray(
        np.array(coin_compression_flags, dtype=np.uint8),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("coin_flags"),
    )

    # ===================================================================
    # Step 4: Calculate uncertainty
    # ===================================================================
    # Uncertainty in counts formula:
    # Uncertainty is quantified for the PCEM, SCEM, and COIN counts.
    # The Poisson contribution is
    #   uncertainty = sqrt(count)
    dataset["swp_pcem_err"] = xr.DataArray(
        np.sqrt(swp_pcem_counts),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("pcem_uncertainty"),
    )
    dataset["swp_scem_err"] = xr.DataArray(
        np.sqrt(swp_scem_counts),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("scem_uncertainty"),
    )
    dataset["swp_coin_err"] = xr.DataArray(
        np.sqrt(swp_coin_counts),
        dims=["epoch", "energy"],
        attrs=cdf_manager.get_variable_attributes("coin_uncertainty"),
    )
    # TODO: when SWAPI gives formula to calculate this scenario:
    # Compression of counts also contributes to the uncertainty.
    # an empirical expression to estimate the error.

    return dataset


def swapi_l1(file_path: str, data_version: str) -> xr.Dataset:
    """
    Will process SWAPI level 0 data to level 1.

    Parameters
    ----------
    file_path : str
        Path to SWAPI L0 file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : xarray.Dataset
        Set of processed data.
    """
    xtce_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    datasets = packet_file_to_datasets(
        file_path, xtce_definition, use_derived_value=False
    )
    processed_data = []
    for apid, ds_data in datasets.items():
        # Right now, we only process SWP_HK and SWP_SCI
        # other packets are not process in this processing pipeline
        # If appId is science, then the file should contain all data of science appId

        if apid == SWAPIAPID.SWP_SCI.value:
            data = process_swapi_science(ds_data, data_version)
            processed_data.append(data)
        if apid == SWAPIAPID.SWP_HK.value:
            # Add HK datalevel attrs
            # TODO: ask SWAPI if we need to process HK if we can use WebPODA
            hk_attrs = ImapCdfAttributes()
            hk_attrs.add_instrument_global_attrs("swapi")
            hk_attrs.add_global_attribute("Data_version", data_version)
            ds_data.attrs.update(hk_attrs.get_global_attributes("imap_swapi_l1_hk"))
            processed_data.append(ds_data)

    return processed_data
