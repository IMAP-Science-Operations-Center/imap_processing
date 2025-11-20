"""Functions to support I-ALiRT CoDICE processing."""

import logging
import pathlib
from decimal import Decimal
from typing import Any

import numpy as np
import xarray as xr

from imap_processing.codice import constants
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.ialirt.utils.grouping import find_groups

logger = logging.getLogger(__name__)

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = Decimal(str(-1.0e31))
COD_LO_COUNTER = 232
COD_HI_COUNTER = 197
COD_LO_RANGE = range(0, 15)
COD_HI_RANGE = range(0, 5)


def process_ialirt_data_streams(
    grouped_data: list[bytearray],
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Process each I-ALiRT science data stream to extract individual data fields.

    Each data stream is converted to binary so that each metadata and science
    data field and their values can be separated out. These fields and values
    eventually will be stored in CDF data/support variables.

    Parameters
    ----------
    grouped_data : list[bytearray]
        A list of grouped I-ALiRT data.

    Returns
    -------
    science_values : list[str]
        The science values / data array portion of the I-ALiRT data in the form
        of a binary string.
    metadata_values : dict[str, list[int]]
        The extracted metadata fields and their values.
    """
    # Initialize placeholders for the processed data
    science_values = []
    metadata_values: dict[str, list[int]] = {}
    for field in constants.IAL_BIT_STRUCTURE:
        metadata_values[field] = []

    # Process each complete data stream
    for data_stream in grouped_data:
        try:
            # Convert the data to binary
            bit_string = "".join(f"{byte:08b}" for byte in data_stream)

            # Separate the data into its individual fields
            bit_position = 0
            for field in constants.IAL_BIT_STRUCTURE:
                # Convert from binary to integer
                value = int(
                    bit_string[
                        bit_position : bit_position + constants.IAL_BIT_STRUCTURE[field]
                    ],
                    2,
                )

                # If we encounter an SHCOARSE of 0, the packet is bad
                if field == "SHCOARSE" and value == 0:
                    raise ValueError("Bad packet encountered")

                metadata_values[field].append(value)
                bit_position += constants.IAL_BIT_STRUCTURE[field]
                if field == "BYTE_COUNT":
                    byte_count = value * 8  # Convert from bytes to number of bits

            # The rest is the data field, up to the byte count
            data_field = bit_string[bit_position : bit_position + byte_count]
            science_values.append(data_field)
        except ValueError:
            pass

    return science_values, metadata_values


def concatenate_bytes(grouped_data: xr.Dataset, group: int, sensor: str) -> bytearray:
    """
    Concatenate all data fields for a specific group into a single bytearray.

    Parameters
    ----------
    grouped_data : xr.Dataset
        The grouped CoDICE dataset containing cod_{sensor}_data_XX variables.
    group : int
        The group number to extract.
    sensor : str
        The sensor type, either 'lo' or 'hi'.

    Returns
    -------
    current_data_stream: bytearray
        The concatenated data stream for the selected group.
    """
    current_data_stream = bytearray()
    group_mask = (grouped_data["group"] == group).values

    cod_ranges = {
        "lo": COD_LO_RANGE,
        "hi": COD_HI_RANGE,
    }

    # Stack all cod_* fields into a 2D NumPy array [n_rows, n_fields]
    arrays = [
        grouped_data[f"cod_{sensor}_data_{field:02}"].values[group_mask]
        for field in cod_ranges[sensor]
    ]

    # Shape → (n_fields, n_rows)
    stacked = np.vstack(arrays)

    # Transpose to get (n_rows, n_fields), then flatten row-wise
    flattened = stacked.T.flatten()

    # Convert to bytes and extend the stream
    current_data_stream.extend(np.uint8(flattened).tobytes())

    return current_data_stream


def create_xarray_dataset(
    science_values: list,
    metadata_values: dict,
    sensor: str,
    lut_file: pathlib.Path,
) -> xr.Dataset:
    """
    Create a xarray Dataset from science and metadata values.

    Parameters
    ----------
    science_values : list
        List of binary strings (bit representations) for each species.
    metadata_values : dict
        Dictionary of metadata values.
    sensor : str
        The sensor type, either 'lo' or 'hi'.
    lut_file : pathlib.Path
        Path to the LUT file.

    Returns
    -------
    xr.Dataset
        The constructed xarray Dataset compatible with l1a_lo_species().
    """
    apid = {"lo": 1152, "hi": 1168}

    packet_bytes = [
        int(bits, 2).to_bytes(len(bits) // 8, byteorder="big")
        for bits in science_values
    ]

    # Fake epoch time.
    num_epochs = len(np.array(metadata_values["ACQ_START_SECONDS"]))
    epoch = np.arange(num_epochs)

    epoch_time = xr.DataArray(epoch, name="epoch", dims=["epoch"])
    dataset = xr.Dataset(coords={"epoch": epoch_time})

    # Metadata value for each field
    for key, value in metadata_values.items():
        data = np.array(value)
        dataset[key.lower()] = xr.DataArray(data, dims=["epoch"])

    dataset["data"] = xr.DataArray(np.array(packet_bytes, dtype=object), dims=["epoch"])
    dataset["pkt_apid"] = xr.DataArray(
        np.full(len(epoch), apid[sensor]), dims=["epoch"]
    )

    return dataset


def process_codice(
    dataset: xr.Dataset,
    lut_path: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create final data products.

    Parameters
    ----------
    dataset : xr.Dataset
        Decommed L0 data.
    lut_path : pathlib.Path
        L1A LUT path.

    Returns
    -------
    codice_data : tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        Dictionary of final data product.

    Notes
    -----
    This function is incomplete and will need to be updated to include the
    necessary calculations and data products.
    - Calculate rates (assume 4 minutes per group)
    - Calculate L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
    - Calculate the public data products
    """
    grouped_cod_lo_data = find_groups(
        dataset, (0, COD_LO_COUNTER), "cod_lo_counter", "cod_lo_acq"
    )
    grouped_cod_hi_data = find_groups(
        dataset, (0, COD_HI_COUNTER), "cod_hi_counter", "cod_hi_acq"
    )
    unique_cod_lo_groups = np.unique(grouped_cod_lo_data["group"])
    unique_cod_hi_groups = np.unique(grouped_cod_hi_data["group"])

    cod_lo_grouped = []
    cod_hi_grouped = []

    # Processing for l1a.
    if unique_cod_lo_groups.size > 0:
        for group in unique_cod_lo_groups:
            cod_lo_data_stream = concatenate_bytes(grouped_cod_lo_data, group, "lo")

            # Decompress binary stream
            cod_lo_grouped.append(cod_lo_data_stream)

        cod_lo_science_values, cod_lo_metadata_values = process_ialirt_data_streams(
            cod_lo_grouped
        )
        cod_lo_dataset = create_xarray_dataset(
            cod_lo_science_values, cod_lo_metadata_values, "lo", lut_path
        )
        result = l1a_lo_species(cod_lo_dataset, lut_path)  # noqa

    if unique_cod_hi_groups.size > 0:
        for group in unique_cod_hi_groups:
            cod_hi_data_stream = concatenate_bytes(grouped_cod_hi_data, group, "hi")

            # Decompress binary stream
            cod_hi_grouped.append(cod_hi_data_stream)

        cod_hi_science_values, cod_hi_metadata_values = process_ialirt_data_streams(
            cod_hi_grouped
        )
        cod_hi_dataset = create_xarray_dataset(  # noqa
            cod_hi_science_values, cod_hi_metadata_values, "hi", lut_path
        )

    # TODO: calculate rates
    #       This will be done in codice.codice_l1b

    # TODO: calculate L2 CoDICE pseudodensities
    #       This will be done in codice.codice_l2

    # TODO: calculate the public data products
    #       This will be done in this module

    # Create mock dataset for I-ALiRT SIT
    # TODO: Once I-ALiRT test data is acquired that actually has data in it,
    #       we should be able to properly populate the I-ALiRT data, but for
    #       now, just create lists of dicts.
    cod_lo_data: list[dict[str, Any]] = []
    cod_hi_data: list[dict[str, Any]] = []

    return cod_lo_data, cod_hi_data
