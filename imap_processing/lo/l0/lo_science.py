"""Processing function for Lo Science Data."""

from collections import namedtuple

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.decompression_tables.decompression_tables import (
    CASE_DECODER,
    DATA_BITS,
    DE_BIT_SHIFT,
)
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
        "shape",  # (azimuth, esa_step)
    ],
)

HIST_DATA_META = {
    # field: bit_length, section_length, shape
    "start_a": HistPacking(12, 504, (6, 7)),
    "start_c": HistPacking(12, 504, (6, 7)),
    "stop_b0": HistPacking(12, 504, (6, 7)),
    "stop_b3": HistPacking(12, 504, (6, 7)),
    "tof0_count": HistPacking(8, 336, (6, 7)),
    "tof1_count": HistPacking(8, 336, (6, 7)),
    "tof2_count": HistPacking(8, 336, (6, 7)),
    "tof3_count": HistPacking(8, 336, (6, 7)),
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


def parse_histogram(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Parse and decompress binary histogram data for Lo.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science counts from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Parsed and decompressed histogram data.
    """
    hist_bin = dataset.sci_cnt

    # initialize the starting bit for the sections of data
    section_start = 0
    # for each field type in the histogram data
    for field in HIST_DATA_META:
        data_meta = HIST_DATA_META[field]
        # for each histogram binary string decompress
        # the data
        decompressed_data = [
            decompress(
                bin_str, data_meta.bit_length, section_start, data_meta.section_length
            )
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
        shaped_data = np.array(decompressed_data, dtype=np.uint32).reshape(data_shape)
        # add the data to the dataset
        dataset[field] = xr.DataArray(
            shaped_data, dims=dims, attrs=attr_mgr.get_variable_attributes(field)
        )

        # increment for the start of the next section
        section_start += data_meta.section_length

    return dataset


def decompress(
    bin_str: str, bits_per_index: int, section_start: int, section_length: int
) -> list[int]:
    """
    Parse and decompress binary histogram data for Lo.

    Parameters
    ----------
    bin_str : str
        Binary string to decompress.
    bits_per_index : int
        Number of bits per index of the data section.
    section_start : int
        The start bit for the section of data.
    section_length : int
        The length of the section of data.

    Returns
    -------
    decompressed_ints : list[int]
        Decompressed integers for the data section.
    """
    # select the decompression method based on the bit length
    # of the compressed data
    if bits_per_index == 8:
        decompress = Decompress.DECOMPRESS8TO16
    elif bits_per_index == 12:
        decompress = Decompress.DECOMPRESS12TO16
    else:
        raise ValueError(f"Invalid bits_per_index: {bits_per_index}")

    # parse the binary and convert to integers
    raw_ints = [
        int(bin_str[i : i + bits_per_index], 2)
        for i in range(section_start, section_start + section_length, bits_per_index)
    ]

    # decompress raw integers
    decompressed_ints: list[int] = decompress_int(
        raw_ints,
        decompress,
        DECOMPRESSION_TABLES,
    )

    return decompressed_ints


def parse_events(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Parse and decompress binary direct event data for Lo.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Parsed and decompressed direct event data.
    """
    # get the binary data for the direct events
    de_time = []
    esa_step = []
    mode = []
    tof0 = []
    tof1 = []
    tof2 = []
    tof3 = []
    cksm = []
    pos = []
    for pkt_idx, de_count in enumerate(dataset["count"].values):
        bit_pos = 0
        for de in range(de_count):
            print("pkt_idx", pkt_idx)
            print("de_count", de_count)
            print("de", de)
            print("data length", len(dataset["data"].values[pkt_idx]))
            print("bit pos", bit_pos)
            print("shcoarse", dataset["shcoarse"][pkt_idx])
            case_number = int(dataset["data"].values[de][bit_pos : bit_pos + 4], 2)
            bit_pos += 4
            print("case number", case_number)

            de_time.append(
                int(
                    dataset["data"].values[pkt_idx][
                        bit_pos : bit_pos + DATA_BITS.DE_TIME
                    ],
                    2,
                )
            )
            bit_pos += DATA_BITS.DE_TIME
            print("de_time", de_time)
            esa_step.append(
                int(
                    dataset["data"].values[pkt_idx][
                        bit_pos : bit_pos + DATA_BITS.ESA_STEP
                    ],
                    2,
                )
            )
            bit_pos += DATA_BITS.ESA_STEP
            print("esa_step", esa_step)
            mode.append(
                int(
                    dataset["data"].values[pkt_idx][bit_pos : bit_pos + DATA_BITS.MODE],
                    2,
                )
            )
            bit_pos += DATA_BITS.MODE
            print("mode", mode)

            case_decoder = CASE_DECODER[(case_number, mode[-1])]

            if case_decoder.TOF0:
                tof0.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.TOF0
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.TOF0
            else:
                tof0.append(attr_mgr.get_variable_attributes("tof0")["FILLVAL"])
            if case_decoder.TOF1:
                tof1.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.TOF1
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.TOF1
            else:
                tof1.append(attr_mgr.get_variable_attributes("tof1")["FILLVAL"])
            if case_decoder.TOF2:
                tof2.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.TOF2
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.TOF2
            else:
                tof2.append(attr_mgr.get_variable_attributes("tof2")["FILLVAL"])
            if case_decoder.TOF3:
                tof3.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.TOF3
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.TOF3
            else:
                tof3.append(attr_mgr.get_variable_attributes("tof3")["FILLVAL"])
            if case_decoder.CKSM:
                cksm.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.CKSM
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.CKSM
            else:
                cksm.append(attr_mgr.get_variable_attributes("cksm")["FILLVAL"])
            if case_decoder.POS:
                pos.append(
                    int(
                        dataset["data"].values[pkt_idx][
                            bit_pos : bit_pos + DATA_BITS.POS
                        ],
                        2,
                    )
                    << DE_BIT_SHIFT
                )
                bit_pos += DATA_BITS.POS
            else:
                pos.append(attr_mgr.get_variable_attributes("pos")["FILLVAL"])

            print("tof0", tof0)
            print("tof1", tof1)
            print("tof2", tof2)
            print("tof3", tof3)
            print("cksm", cksm)
            print("pos", pos)

    dataset["de_time"] = de_time
    dataset["esa_step"] = esa_step
    dataset["mode"] = mode
    dataset["tof0"] = tof0
    dataset["tof1"] = tof1
    dataset["tof2"] = tof2
    dataset["tof3"] = tof3
    dataset["cksm"] = cksm
    dataset["pos"] = pos
    # print all these values:
    print("de_time", de_time)
    print("esa_step", esa_step)
    print("mode", mode)
    print("tof0", tof0)
    print("tof1", tof1)
    print("tof2", tof2)
    print("tof3", tof3)
    print("cksm", cksm)
    print("pos", pos)
