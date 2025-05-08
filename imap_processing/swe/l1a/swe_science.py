"""Contains code to perform SWE L1a science processing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swe.utils import swe_constants
from imap_processing.swe.utils.swe_utils import SWEAPID

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


def swe_science(l0_dataset: xr.Dataset) -> xr.Dataset:
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
    l0_dataset : xarray.Dataset
        Raw packet data from SWE stored as an xarray dataset.

    Returns
    -------
    dataset : xarray.Dataset
        The xarray dataset with data.
    """
    # We know we can only have 8 bit numbers input, so iterate over all
    # possibilities once up front
    decompression_table = np.array([decompressed_counts(i) for i in range(256)])

    # Loop through each packet individually with a list comprehension and
    # perform the following steps:
    # 1. Turn the binary string  of 0s and 1s to an int
    # 2. Convert the int into a bytes object of length 1260 (10080 / 8)
    #    Eg. "0000000011110011" --> b'\x00\xf3'
    #    1260 = 15 seconds x 12 energy steps x 7 CEMs
    # 3. Read that bytes data to a numpy array of uint8 through the buffer protocol
    # 4. Reshape the data to 180 x 7
    raw_science_array = np.array(
        [
            np.frombuffer(binary_string, dtype=np.uint8).reshape(
                180, swe_constants.N_CEMS
            )
            for binary_string in l0_dataset["science_data"].values
        ]
    )

    # Decompress the raw science data using numpy broadcasting logic
    # science_array will be the same shape as raw_science_array (npackets, 180, 7)
    science_array = decompression_table[raw_science_array]

    # Load CDF attrs
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("swe")
    cdf_attrs.add_instrument_variable_attrs("swe", "l1a")

    epoch_time = xr.DataArray(
        l0_dataset["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    spin_sector = xr.DataArray(
        np.arange(180),
        name="spin_sector",
        dims=["spin_sector"],
        attrs=cdf_attrs.get_variable_attributes("spin_sector", check_schema=False),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    spin_sector_label = xr.DataArray(
        spin_sector.values.astype(str),
        name="spin_sector_label",
        dims=["spin_sector"],
        attrs=cdf_attrs.get_variable_attributes(
            "spin_sector_label", check_schema=False
        ),
    )

    cem_id = xr.DataArray(
        np.arange(swe_constants.N_CEMS),
        name="cem_id",
        dims=["cem_id"],
        attrs=cdf_attrs.get_variable_attributes("cem_id", check_schema=False),
    )

    # NOTE: LABL_PTR_2 should be CDF_CHAR.
    cem_id_label = xr.DataArray(
        cem_id.values.astype(str),
        name="cem_id_label",
        dims=["cem_id"],
        attrs=cdf_attrs.get_variable_attributes("cem_id_label", check_schema=False),
    )

    science_xarray = xr.DataArray(
        science_array,
        dims=["epoch", "spin_sector", "cem_id"],
        attrs=cdf_attrs.get_variable_attributes("science_data"),
    )

    raw_science_xarray = xr.DataArray(
        raw_science_array,
        dims=["epoch", "spin_sector", "cem_id"],
        attrs=cdf_attrs.get_variable_attributes("raw_counts"),
    )

    # Add APID to global attrs for following processing steps
    l1a_global_attrs = cdf_attrs.get_global_attributes("imap_swe_l1a_sci")
    # Formatting to string to be complaint with ISTP
    l1a_global_attrs["packet_apid"] = SWEAPID.SWE_SCIENCE.value
    dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "spin_sector": spin_sector,
            "cem_id": cem_id,
            "spin_sector_label": spin_sector_label,
            "cem_id_label": cem_id_label,
        },
        attrs=l1a_global_attrs,
    )
    dataset["science_data"] = science_xarray
    dataset["raw_science_data"] = raw_science_xarray
    # TODO: Remove the header in packet_file_to_datasets
    #       The science_data variable is also in the l1 dataset with different values
    l0_dataset = l0_dataset.drop_vars(
        [
            "science_data",
            "version",
            "type",
            "sec_hdr_flg",
            "pkt_apid",
            "seq_flgs",
            "src_seq_ctr",
            "pkt_len",
        ]
    )
    for var_name, arr in l0_dataset.variables.items():
        arr.attrs = cdf_attrs.get_variable_attributes(var_name)
    dataset = dataset.merge(l0_dataset)

    logger.info("SWE L1A science data processing completed.")
    return dataset
