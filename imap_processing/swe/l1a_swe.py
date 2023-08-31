import bitstring
import numpy as np
import xarray as xr

from imap_processing import packet_definition_directory
from imap_processing.swe import decom_swe


def uncompress(cem_count):
    """This function uncompress a count.

    Parameters
    ----------
    cem_count : int
        CEM counts. Eg. 243

    Returns
    -------
    int
        uncompressed count. Eg. 40959
    """
    # index is the first four bits of input data
    # multi is the last four bits of input data
    index = cem_count // 16
    multi = cem_count % 16

    # uncompression formula
    # N = base[index] + multi * step_size[index] + (step_size[index] - 1) / 2
    # NOTE: for (step_size[index] - 1) / 2, we only keep the whole number part of the quotient
    base_value = calculate_base(index)
    step_size = 2 ** calculate_step_power(index)
    return base_value + (multi * step_size) + ((step_size - 1) // 2)


def calculate_base(index):
    if index == 0:
        return 0
    elif 1 <= index <= 6:
        return 2 ** (index + 3)
    elif index == 7:
        return 768
    elif index == 8:
        return 1024
    elif 9 <= index <= 15:
        return 2**index + 1024
    else:
        print("\nError! Switch for index failed in function N.")
        return 0


def calculate_step_power(index):
    if index == 0:
        return 0
    elif 1 <= index <= 4:
        return index - 1
    elif 5 <= index <= 7:
        return 4
    elif 8 <= index <= 9:
        return 5
    elif 10 <= index <= 15:
        return index - 4
    else:
        print("\nError! Switch for I failed in function R.")
        return 0


def get_swe_cem_counts():
    """SWE L1A algorithm steps:
    - Read data from SWE packet file
    - Uncompress data
    - Store metadata and data in attrs and DataArray of xarray respectively
    - Save complete data to cdf file
    """
    packet_file = f"{packet_definition_directory}/../swe/tests/science_block_20221116_163611Z_idle.bin"
    decom_data = decom_swe.decom_packets(packet_file)
    dataset = xr.Dataset()
    for each_raw_data in decom_data:
        metadata = {}
        for key, value in each_raw_data.header.items():
            metadata[key] = value.raw_value

        for key, value in each_raw_data.data.items():
            if key != "SCIENCE_DATA":
                metadata[key] = (
                    value.raw_value
                    if value.derived_value is None
                    else value.derived_value
                )

        # read raw data
        binary_data = each_raw_data.data["SCIENCE_DATA"].raw_value
        # read raw data as binary array using bitstring
        bit_array = bitstring.ConstBitStream(bin=binary_data)
        # chunk binary into 1260 unit 8-bits
        compressed_data = bit_array.readlist(["uint:8"] * 1260)
        # for each data, uncompress it
        uncompressed_data = [uncompress(i) for i in compressed_data]
        # reshape the data to 3D array. Array shapes correspond to this:
        # 7 (CEM_COUNTS) * 12 (STEPS_EACH_SECOND) * 15 (SECONDS)
        unpacked_data = np.array(uncompressed_data).reshape(15, 12, 7)
        xarray_data = xr.DataArray(
            unpacked_data, dims=["seconds", "steps_each_second", "cem_counts"]
        )
        xarray_data.attrs = metadata
        dataset.update({each_raw_data.data["ACQ_START_COARSE"].raw_value: xarray_data})


get_swe_cem_counts()
