"""Contains code to perform SWE L1a science processing."""

import collections
import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.swe.utils.swe_utils import (
    add_metadata_to_array,
)

logger = logging.getLogger(__name__)


def decompressed_counts(cem_count: int) -> int:
    """
    Decompressed counts from the CEMs.

    Parameters
    ----------
    cem_count : int
        CEM counts. Eg. 243.

    Returns
    -------
    int
        Decompressed count. Eg. 40959.
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


def swe_science(decom_data: list, data_version: str) -> xr.Dataset:
    """
    SWE L1a science processing.

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
    decom_data : list
        Decompressed packet data.

    data_version : str
        Data version for the 'Data_version' CDF attribute. This is the version of the
        output file.

    Returns
    -------
    dataset : xarray.Dataset
        The xarray dataset with data.
    """
    science_array = []
    raw_science_array = []

    metadata_arrays: np.array = collections.defaultdict(list)

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

    # Load CDF attrs
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("swe")
    cdf_attrs.add_instrument_variable_attrs("swe", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    epoch_converted_time = met_to_j2000ns(metadata_arrays["SHCOARSE"])
    epoch_time = xr.DataArray(
        epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    spin_angle = xr.DataArray(
        np.arange(180),
        name="spin_angle",
        dims=["spin_angle"],
        attrs=cdf_attrs.get_variable_attributes("spin_angle"),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    spin_angle_label = xr.DataArray(
        spin_angle.values.astype(str),
        name="spin_angle_label",
        dims=["spin_angle_label"],
        attrs=cdf_attrs.get_variable_attributes("spin_angle_label"),
    )

    polar_angle = xr.DataArray(
        np.arange(7),
        name="polar_angle",
        dims=["polar_angle"],
        attrs=cdf_attrs.get_variable_attributes("polar_angle"),
    )

    # NOTE: LABL_PTR_2 should be CDF_CHAR.
    polar_angle_label = xr.DataArray(
        polar_angle.values.astype(str),
        name="polar_angle_label",
        dims=["polar_angle_label"],
        attrs=cdf_attrs.get_variable_attributes("polar_angle_label"),
    )

    science_xarray = xr.DataArray(
        science_array,
        dims=["epoch", "spin_angle", "polar_angle"],
        attrs=cdf_attrs.get_variable_attributes("science_data"),
    )

    raw_science_xarray = xr.DataArray(
        raw_science_array,
        dims=["epoch", "spin_angle", "polar_angle"],
        attrs=cdf_attrs.get_variable_attributes("raw_counts"),
    )

    # Add APID to global attrs for following processing steps
    l1a_global_attrs = cdf_attrs.get_global_attributes("imap_swe_l1a_sci")
    # Formatting to string to be complaint with ISTP
    l1a_global_attrs["packet_apid"] = f"{decom_data[0].header['PKT_APID'].raw_value}"
    dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "spin_angle": spin_angle,
            "polar_angle": polar_angle,
            "spin_angle_label": spin_angle_label,
            "polar_angle_label": polar_angle_label,
        },
        attrs=l1a_global_attrs,
    )
    dataset["science_data"] = science_xarray
    dataset["raw_science_data"] = raw_science_xarray

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        # Lowercase the key to be complaint with ISTP's metadata field
        metadata_field = key.lower()
        dataset[metadata_field] = xr.DataArray(
            value,
            dims=["epoch"],
            attrs=cdf_attrs.get_variable_attributes(metadata_field),
        )

    logger.info("SWE L1A science data process completed")
    return dataset
