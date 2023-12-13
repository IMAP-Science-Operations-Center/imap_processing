"""Contains code to perform SWE L1a science processing."""

import collections
import dataclasses

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.swe import swe_cdf_attrs
from imap_processing.swe.utils.swe_utils import (
    add_metadata_to_array,
)


def decompressed_counts(cem_count):
    """Decompressed counts from the CEMs.

    Parameters
    ----------
    cem_count : int
        CEM counts. Eg. 243

    Returns
    -------
    int
        decompressed count. Eg. 40959
    """
    # index is the first four bits of input data
    # multi is the last four bits of input data
    index = cem_count // 16
    multi = cem_count % 16

    # This is look up table for the index to get
    # base and step_size to calculate the decompressed count.
    decompress_table = {
        0: {"base": 0, "step_size": 1},
        1: {"base": 16, "step_size": 1},
        2: {"base": 32, "step_size": 2},
        3: {"base": 64, "step_size": 4},
        4: {"base": 128, "step_size": 8},
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
        15: {"base": 33792, "step_size": 2048},
    }

    # decompression formula from SWE algorithm document CN102D-D0001 and page 16.
    # N = base[index] + multi * step_size[index] + (step_size[index] - 1) / 2
    # NOTE: for (step_size[index] - 1) / 2, we only keep the whole number part of
    # the quotient

    return (
        decompress_table[index]["base"]
        + (multi * decompress_table[index]["step_size"])
        + ((decompress_table[index]["step_size"] - 1) // 2)
    )


def swe_science(decom_data):
    """SWE L1a science processing.

    SWE L1A algorithm steps:
        - Read data from each SWE packet file
        - Uncompress counts data
        - Store metadata fields and data in DataArray of xarray
        - Save data to dataset.

    In each packet, SWE collects data for 15 seconds. In each second,
    it collect data for 12 energy steps and at each energy step,
    it collects 7 data from each 7 CEMs.

    Each L1A data from each packet will have this shape: 15 rows, 12 columns,
    and each cell in 15 x 12 table contains 7 element array.
    These dimension maps to this:

    |     15 rows --> 15 seconds
    |     12 column --> 12 energy steps in each second
    |     7 element --> 7 CEMs counts

    In L1A, we don't do anything besides read raw data, uncompress counts data and
    store data in 15 x 12 x 7 array.

    SWE want to keep all value as it is in L1A. Post L1A, we group data into full cycle
    and convert raw data to engineering data as needed.

    Parameters
    ----------
    packet_file : str
        packet file path

    Returns
    -------
    xarray.Dataset
        xarray dataset with data.
    """
    science_array = []
    raw_science_array = []

    metadata_arrays = collections.defaultdict(list)

    # We know we can only have 8 bit numbers input, so iterate over all
    # possibilities once up front
    decompression_table = np.array([decompressed_counts(i) for i in range(256)])

    for data_packet in decom_data:
        # read raw data
        binary_data = data_packet.data["SCIENCE_DATA"].raw_value
        # read binary string to an int and then convert it to
        # bytes. This is to convert the string to bytes.
        # Eg. "0000000011110011" --> b'\x00\xf3'
        # 1260 = 15 seconds x 12 energy steps x 7 CEMs
        byte_data = int(binary_data, 2).to_bytes(1260, byteorder="big")
        # convert bytes to numpy array of uint8
        raw_counts = np.frombuffer(byte_data, dtype=np.uint8)

        # Uncompress counts. Decompressed data is a list of 1260
        # where 1260 = 180 x 7 CEMs
        # Take the "raw_counts" indices/counts mapping from
        # decompression_table and then reshape the return
        uncompress_data = np.take(decompression_table, raw_counts).reshape(180, 7)
        # Save raw counts data as well
        raw_counts = raw_counts.reshape(180, 7)

        # Save data with its metadata field to attrs and DataArray of xarray.
        # Save data as np.int64 to be complaint with ISTP' FILLVAL
        science_array.append(uncompress_data.astype(np.int64))
        raw_science_array.append(raw_counts.astype(np.int64))
        metadata_arrays = add_metadata_to_array(data_packet, metadata_arrays)

    epoch_time = xr.DataArray(
        metadata_arrays["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    # TODO: add more descriptive description
    energy = xr.DataArray(
        np.arange(180),
        name="Energy",
        dims=["Energy"],
        attrs=dataclasses.replace(
            swe_cdf_attrs.int_base,
            catdesc="Energy's index value in the lookup table",
            fieldname="Energy Bins",
            label_axis="Energy Bins",
            units="",
        ).output(),
    )

    counts = xr.DataArray(
        np.arange(7),
        name="Counts",
        dims=["Counts"],
        attrs=dataclasses.replace(
            swe_cdf_attrs.int_base,
            catdesc="Counts",
            fieldname="Counts",
            label_axis="Counts",
            units="int",
        ).output(),
    )

    science_xarray = xr.DataArray(
        science_array,
        dims=["Epoch", "Energy", "Counts"],
        attrs=swe_cdf_attrs.l1a_science_attrs.output(),
    )

    raw_science_xarray = xr.DataArray(
        raw_science_array,
        dims=["Epoch", "Energy", "Counts"],
        attrs=swe_cdf_attrs.l1a_science_attrs.output(),
    )

    dataset = xr.Dataset(
        coords={
            "Epoch": epoch_time,
            "Energy": energy,
            "Counts": counts,
        },
        attrs=swe_cdf_attrs.swe_l1a_global_attrs.output(),
    )
    dataset["SCIENCE_DATA"] = science_xarray
    dataset["RAW_SCIENCE_DATA"] = raw_science_xarray

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        if key == "SHCOARSE":
            continue
        # TODO: figure out how to add more descriptive
        # description for each metadata field
        #
        # int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs["LABLAXIS"] = key
        # # get int32's max since most of metadata is under 32-bits
        # int_attrs["VALIDMAX"] = np.iinfo(np.int32).max
        # int_attrs["DEPEND_0"] = "Epoch"
        dataset[key] = xr.DataArray(
            value,
            dims=["Epoch"],
            attrs=dataclasses.replace(
                swe_cdf_attrs.swe_metadata_attrs,
                catdesc=key,
                fieldname=key,
                label_axis=key,
                depend_0="Epoch",
            ).output(),
        )

    return dataset
