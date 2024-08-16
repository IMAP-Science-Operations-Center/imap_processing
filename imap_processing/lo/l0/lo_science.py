import xarray as xr
import numpy as np
import numpy.typing as npt
from collections import namedtuple
from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
    decompress_int,
)

HistPacking = namedtuple(
    "HistPacking",
    [
        "bit_length",
        "section_length",
        "shape" # (azimuth, esa_step)
    ],
)

hist_data_meta = {
    "start_a": HistPacking(12, 504, (6, 7)),
    "start_c": HistPacking(12, 504, (6, 7)),
    "stop_b0": HistPacking(12, 504, (6, 7)),
    "stop_b3": HistPacking(12, 504, (6, 7)),
    "tof0": HistPacking(8, 336, (6, 7)),
    "tof1": HistPacking(8, 336, (6, 7)),
    "tof2": HistPacking(8, 336, (6, 7)),
    "tof3": HistPacking(8, 336, (6, 7)),
    "tof0_tof1": HistPacking(8, 3360, (60, 7)),
    "tof0_tof2": HistPacking(8, 3360, (60, 7)),
    "tof1_tof2": HistPacking(8, 3360, (60, 7)),
    "silver": HistPacking(8, 3360, (60, 7)),
    "disc_tof0": HistPacking(8, 336, (6, 7)),
    "disc_tof1": HistPacking(8, 336, (6, 7)),
    "disc_tof2": HistPacking(8, 336, (6, 7)),
    "disc_tof3": HistPacking(8, 336, (6, 7)),
    "pos0": HistPacking(12, 504, (6, 7)),
    "pos1": HistPacking(12, 504, (6, 7)),
    "pos2": HistPacking(12, 504, (6, 7)),
    "pos3": HistPacking(12, 504, (6, 7)),
    "hydrogen": HistPacking(8, 3360, (60, 7)),
    "oxygen": HistPacking(8, 3360, (60, 7)),
}


def parse_histogram(dataset: xr.Dataset, attr_mgr) -> xr.Dataset:
    hist_bin = dataset.sci_cnt

    # initialize the starting bit for the sections of data
    section_start = 0
    # for each field type in the histogram data
    for field in hist_data_meta:
        data_meta = hist_data_meta[field]
        # for each histgram binary string decompress
        # the data
        decompressed_data = [
            decompress(
                bin_str,
                data_meta.bit_length,
                section_start,
                data_meta.section_length)
            for bin_str in hist_bin.values
        ]

        # add on the epoch length (equal to number of packets) to the
        # field shape
        data_shape = (len(hist_bin), data_meta.shape[0], data_meta.shape[1])

        # get the dimension names from the CDF attr manager
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]

        # reshape the decompressed data
        shaped_data = np.array(decompressed_data).reshape(data_shape)
        # add the data to the dataset
        dataset[field] = xr.DataArray(shaped_data, dims=dims, attrs=attr_mgr.get_variable_attributes(field))

        # increment for the start of the next section
        section_start += data_meta.section_length

    return dataset


def decompress(bin_str: str, bits_per_index: int, section_start: int, section_length: int) -> list[int]:
    # select the decompression method based on the bit length
    # of the compressed data
    if bits_per_index == 8:
        decompress = Decompress.DECOMPRESS8TO16
    elif bits_per_index == 12:
        decompress = Decompress.DECOMPRESS12TO16
    else:
        raise ValueError(f"Invalid bits_per_index: {bits_per_index}")

    # parse the binary, convert to integers, and decompress
    decompressed_ints = [
        decompress_int(
            int(bin_str[i:i + bits_per_index], 2),
            decompress,
            DECOMPRESSION_TABLES,
        )
        for i in range(section_start, section_start + section_length, bits_per_index)
    ]

    return decompressed_ints