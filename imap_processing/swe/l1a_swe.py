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


def swe_l1a():
    """SWE L1A algorithm steps:
        - Read data from SWE packet file
        - Uncompress data
        - Store metadata and data in attrs and DataArray of xarray respectively
        - Save complete data to cdf file
    Each L1A data will have this shape: 24 rows, 7 columns, and each cell in 24 x 7 table contains
    30 element array. These dimension maps to this:
        24 rows --> 24 energy steps
        7 column --> 7 CEMs value
        30 element --> 30 spin angles
    """
    packet_file = f"{packet_definition_directory}/../swe/tests/science_block_20221116_163611Z_idle.bin"
    decom_data = decom_swe.decom_packets(packet_file)
    dataset = xr.Dataset()
    # To get one full data, we need to get data from four spins where each spin
    # data is stored in one packet data. ESA_STEPS from metadata gives information about
    # which spin data is stored in which packet.
    # ESA_STEPS = 0 --> first spin data
    # ESA_STEPS = 1 --> second spin data
    # ESA_STEPS = 2 --> third spin data
    # ESA_STEPS = 3 --> fourth spin data

    # These indexes is where each spin's data goes in full data table.
    # It's like putting a puzzle together. One full data table in the
    # the algorithm document is of this shape ( 24, 30, 7). Once we
    # have populated data table with all four spins, we can reshape
    # it to what subsequent algorithm needs which is (24, 7, 30).
    spin_one_indexes = [1,  5, 9, 13, 17,  21, 23, 19, 15, 11, 7, 3]
    spin_two_indexes = [2,  6, 10, 14, 18,  22, 20, 16, 12, 8, 4, 0]
    spin_three_indexes = [3,  7, 11, 15, 19,  23, 21, 17, 13, 9, 5, 1]
    spin_four_indexes = [4,  8, 12, 16, 20,  24, 22, 18, 14, 10, 6, 2]

    item_index = 0
    while item_index < len(decom_data):
        # If ESA_STEPS is 0, we need to get data from first spin
        spin_number = decom_data[item_index].data["ESA_STEPS"].raw_value
        if spin_number == 0:
            # It should follow by 4 packets where each packet contains
            # one spin's data.
            one_full_data = np.zeros((24, 30, 7))

        # TODO: get metadata of each data and stores. Find out how to combine four
        # packet's metadata.

        # read raw data
        binary_data = decom_data[item_index].data["SCIENCE_DATA"].raw_value
        # read raw data as binary array using bitstring
        bit_array = bitstring.ConstBitStream(bin=binary_data)
        # chunk binary into 1260 units each with 8-bits
        byte_data = bit_array.readlist(["uint:8"] * 1260)
        # for each data, uncompress it. uncompressed data is a list of 1260
        # where 1260 = 15 seconds x 7 CEMs x 12 energy steps
        uncompressed_data = [uncompress(i) for i in byte_data]
        # TODO: find out how to populate full data array with four spins data.
