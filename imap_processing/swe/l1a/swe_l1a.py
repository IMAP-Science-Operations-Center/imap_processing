import bitstring
import numpy as np
import xarray as xr

from imap_processing import packet_definition_directory
from imap_processing.swe.l0 import decom_swe


def uncompress_counts(cem_count):
    """This function uncompress counts data.

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

    # This is look up table for the index to get
    # base and step_size to calculate the uncompressed count.
    uncompress_table = {
        0: {"base": 0, "step_size": 1},
        1: {"base": 16, "step_size": 2},
        2: {"base": 32, "step_size": 4},
        3: {"base": 64, "step_size": 8},
        4: {"base": 128, "step_size": 16},
        5: {"base": 256, "step_size": 16},
        6: {"base": 512, "step_size": 16},
        7: {"base": 768, "step_size": 16},
        8: {"base": 1024, "step_size": 32},
        9: {"base": 1536, "step_size": 32},
        10: {"base": 2048, "step_size": 64},
        11: {"base": 3072, "step_size": 128},
        12: {"base": 5120, "step_size": 256},
        13: {"base": 9216, "step_size": 512},
        14: {"base": 17408, "step_size": 1024},
        15: {"base": 33792, "step_size": 2048}
    }

    # uncompression formula from SWE algorithm document CN102D-D0001 and page 16.
    # N = base[index] + multi * step_size[index] + (step_size[index] - 1) / 2
    # NOTE: for (step_size[index] - 1) / 2, we only keep the whole number part of the quotient

    return uncompress_table[index]["base"] + (multi * uncompress_table[index]["step_size"]) + ((uncompress_table[index]["step_size"] - 1) // 2)


def read_metadata(data_packet):
    """This function read metadata from data packet.

    Parameters
    ----------
    data_packet : SWE data packet

    Returns
    -------
    dict
        metadata
    """
    metadata = {}
    for key, value in data_packet.header.items():
        metadata[key] = value.raw_value

    for key, value in data_packet.data.items():
        if key != "SCIENCE_DATA":
            metadata[key] = (
                value.raw_value
                if value.derived_value is None
                else value.derived_value
            )

    return metadata


def swe_l1a(packet_file: str):
    """SWE L1A algorithm steps:
        - Read data from each SWE packet file
        - Uncompress counts data
        - Store metadata and data in attrs and DataArray of xarray respectively
        - Save uncompress data to cdf file

    SWE collects data for 15 seconds. In each second, it collect data for 12 energy steps and at each energy step,
    it collects 7 data from each 7 CEMs.

    Each L1A data from each packet will have this shape: 15 rows, 12 columns, and each cell in 15 x 12 table contains
    7 element array. These dimension maps to this:
        15 rows --> 15 seconds
        12 column --> 12 energy steps in each second
        7 element --> 7 CEMs counts

    In L1A, we don't do anything besides read raw data, uncompress counts data and store data in 15 x 12 x 7 array.
    SWE want to keep all value as it is in L1A. Post L1A, we group data into full cycle and convert
    raw data to engineering data as needed.

    Parameters
    ----------
    packet_file : str
        packet file path

    Returns
    -------
    xarray.Dataset
        xarray dataset with data and metadata.
        data.shape = (number_of_packets, 15, 12, 7)
        attrs = metadata of each packet.
    """
    decom_data = decom_swe.decom_packets(packet_file)
    data_array = []
    metadata_dict = dict()

    for data_packet in decom_data:
        # read raw data
        binary_data = data_packet.data["SCIENCE_DATA"].raw_value
        # read raw data as binary array using bitstring
        # Eg. "0b0101010111"
        bit_array = bitstring.ConstBitStream(bin=binary_data)
        # chunk binary into 1260 units each with 8-bits
        byte_data = np.frombuffer(bit_array.bytes, dtype=np.uint8)
        # Uncompress counts. Uncompressed data is a list of 1260
        # where 1260 = 15 seconds x 7 CEMs x 12 energy steps
        uncompressed_data = [uncompress_counts(i) for i in byte_data]
        reshaped_data = np.array(uncompressed_data).reshape(15, 12, 7)
        # print(reshaped_data)
        data_array.append(reshaped_data)
        metadata_dict[data_packet.data["SHCOARSE"].raw_value] = read_metadata(data_packet)

    xarray_data = xr.DataArray(
        data_array,
        dims=["number_of_packets", "seconds", "energy_steps", "cem_counts"],
        attrs=metadata_dict,
    )
    # print(xarray_data.shape)
    # print(len(metadata_dict))
    # print(xarray_data.data)
    # print(xarray_data.attrs[406334173])
    return xr.Dataset({"data": xarray_data})


if __name__ == "__main__":
    packet_file = f"{packet_definition_directory}/../swe/tests/science_block_20221116_163611Z_idle.bin"
    swe_l1a = swe_l1a(packet_file=packet_file)
