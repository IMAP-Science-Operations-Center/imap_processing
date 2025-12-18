"""Functions to support I-ALiRT CoDICE processing."""

import logging
import pathlib
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.codice import constants
from imap_processing.codice.codice_l1a_ialirt_hi import l1a_ialirt_hi
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.codice_l1b import convert_to_rates
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc

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


def convert_to_intensities(
    cod_hi_l1b_data: xr.Dataset, l2_lut_path: pathlib.Path, species: str
) -> NDArray:
    """
    Calculate intensities.

    Parameters
    ----------
    cod_hi_l1b_data : xr.Dataset
        L1b data.
    l2_lut_path : pathlib.Path
        L2 LUT path.
    species : str
        CoDICE Hi species.

    Returns
    -------
    intensity : np.array
        L2 CoDICE-Hi intensities.

    Notes
    -----
    Equation from section 13.1 in the CoDICE Algorithm Document.
    """
    # Average of the hydrogen efficiencies.
    efficiencies_df = pd.read_csv(l2_lut_path)
    species_efficiency = efficiencies_df.sort_values(by="energy_bin")
    eps_ig = species_efficiency[["group_0", "group_1", "group_2", "group_3"]].to_numpy(
        float
    )

    # For omni over 3 SSDs:
    g_g = constants.L2_GEOMETRIC_FACTOR * constants.IALIRT_HI_NUMBER_OF_SSD_PER_GROUP

    # Calculate energy passband from L1B data
    energy_passbands = (
        cod_hi_l1b_data[f"energy_{species}_plus"]
        + cod_hi_l1b_data[f"energy_{species}_minus"]
    ).values[:, np.newaxis]

    denom = g_g * eps_ig * energy_passbands  # (15, 4)
    # reshape to broadcast along h's first and third dimensions
    denom = denom[None, :, None, :]

    # Rates in shape (n_spins, energy, spin_sector, inst_az - this is group)
    h = cod_hi_l1b_data[species].values

    # Final intensities with same shape as h
    intensity = h / denom  # shape (4, 15, 4, 4); units #/(cm^2 sr s MeV/nuc)

    return intensity


def process_codice(
    dataset: xr.Dataset,
    l1a_lut_path: pathlib.Path,
    l2_lut_path: pathlib.Path,
    sensor: str,
    l2_geometric_factor_path: Path | None = None,
) -> tuple:
    """
    Create final data products.

    Parameters
    ----------
    dataset : xr.Dataset
        Decommed L0 data.
    l1a_lut_path : pathlib.Path
        L1A LUT path.
    l2_lut_path : pathlib.Path
        L2 LUT path.
    sensor : str
        Sensor (codice_hi or codice_lo).
    l2_geometric_factor_path : pathlib.Path
        Optional geometric factor path based on the sensor (required by Lo).

    Returns
    -------
    cod_lo_data : dict
        Dictionary of final data product.
    codice_hi_data : dict
        Dictionary of final data product.

    Notes
    -----
    This function is incomplete and will need to be updated to include the
    necessary calculations and data products.
    - Calculate rates (assume 4 minutes per group)
    - Calculate L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
    - Calculate the public data products
    """
    logger.info("Processing CoDICE.")

    codice_lo_data: list[dict[str, Any]] = []
    codice_hi_data: list[dict[str, Any]] = []

    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(dataset["sc_sclk_sec"], dataset["sc_sclk_sub_sec"], 256)
    # Add required parameters.
    dataset["met"] = met

    if sensor == "codice_lo":
        logger.info("Processing CoDICE-Lo.")
        grouped_cod_lo_data = find_groups(
            dataset, (0, COD_LO_COUNTER), "cod_lo_counter", "cod_lo_acq"
        )
        unique_cod_lo_groups = np.unique(grouped_cod_lo_data["group"])

    if sensor == "codice_hi":
        logger.info("Processing CoDICE-Hi.")
        grouped_cod_hi_data = find_groups(
            dataset, (0, COD_HI_COUNTER), "cod_hi_counter", "cod_hi_acq"
        )
        unique_cod_hi_groups = np.unique(grouped_cod_hi_data["group"])

    if sensor == "codice_lo" and unique_cod_lo_groups.size > 0:
        for group in unique_cod_lo_groups:
            cod_lo_data_stream = concatenate_bytes(grouped_cod_lo_data, group, "lo")

            # Decompress binary stream
            met = grouped_cod_lo_data["met"][
                (grouped_cod_lo_data["group"] == group).values
            ]

            cod_lo_science_values, cod_lo_metadata_values = process_ialirt_data_streams(
                [cod_lo_data_stream]
            )
            cod_lo_dataset = create_xarray_dataset(
                cod_lo_science_values, cod_lo_metadata_values, "lo", l1a_lut_path
            )
            result = l1a_lo_species(cod_lo_dataset, l1a_lut_path)  # noqa

    if sensor == "codice_hi" and unique_cod_hi_groups.size > 0:
        for group in unique_cod_hi_groups:
            cod_hi_data_stream = concatenate_bytes(grouped_cod_hi_data, group, "hi")

            # Decompress binary stream
            met = grouped_cod_hi_data["met"][
                (grouped_cod_hi_data["group"] == group).values
            ]

            cod_hi_science_values, cod_hi_metadata_values = process_ialirt_data_streams(
                [cod_hi_data_stream]
            )
            cod_hi_dataset = create_xarray_dataset(
                cod_hi_science_values, cod_hi_metadata_values, "hi", l1a_lut_path
            )
            l1a_hi = l1a_ialirt_hi(cod_hi_dataset, l1a_lut_path)
            l1b_hi = convert_to_rates(
                l1a_hi,
                "hi-ialirt",
            )
            l2_hi = convert_to_intensities(l1b_hi, l2_lut_path, "h")
            # Put in Decimal format so DynamoDB can read it.
            dec_l2_hi = np.vectorize(lambda x: Decimal(f"{float(x):.3f}"))(
                l2_hi
            ).tolist()

            codice_hi_data.append(
                {
                    "apid": 478,
                    "met": int(met[0]),
                    "met_in_utc": met_to_utc(met[0]).split(".")[0],
                    "ttj2000ns": int(met_to_ttj2000ns(met[0])),
                    "instrument": f"{sensor}",
                    f"{sensor}_epoch": [int(epoch) for epoch in l1b_hi["epoch"]],
                    f"{sensor}_l2_hi": dec_l2_hi,
                }
            )

    return codice_lo_data, codice_hi_data
