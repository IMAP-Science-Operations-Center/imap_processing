"""Functions to support I-ALiRT CoDICE processing."""

import datetime
import logging
from collections import namedtuple
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.codice import constants
from imap_processing.codice.codice_l1a_ialirt_hi import l1a_ialirt_hi
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.codice_l1b import convert_to_rates
from imap_processing.codice.codice_l2 import (
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    process_lo_species_intensity,
)
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import (
    et_to_utc,
    ttj2000ns_to_et,
)

logger = logging.getLogger(__name__)

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = Decimal(str(-1.0e31))
COD_LO_COUNTER = 232
COD_HI_COUNTER = 197
COD_LO_RANGE = range(0, 15)
COD_HI_RANGE = range(0, 5)

COD_LO_L2 = namedtuple(
    "COD_LO_L2",
    [
        "c_over_o_abundance",
        "mg_over_o_abundance",
        "fe_over_o_abundance",
        "c_plus_6_over_c_plus_5",
        "o_plus_7_over_o_plus_6",
        "fe_low_over_fe_high",
    ],
)


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
    cod_hi_l1b_data: xr.Dataset, l2_lut_path: Path, species: str
) -> NDArray:
    """
    Calculate intensities.

    Parameters
    ----------
    cod_hi_l1b_data : xr.Dataset
        L1b data.
    l2_lut_path : Path
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
    species_efficiency = efficiencies_df.sort_values(by="energy_bin")[
        efficiencies_df["species"] != "GF"
    ]
    eps_ig = species_efficiency[["group_0", "group_1", "group_2", "group_3"]].to_numpy(
        float
    )

    # For omni over 3 SSDs:
    g_g = efficiencies_df[efficiencies_df["species"] == "GF"][
        ["group_0", "group_1", "group_2", "group_3"]
    ].to_numpy(float)

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


def calculate_ratios(
    cod_lo_l1b_data: xr.Dataset,
    l2_lut_path: Path,
    l2_geometric_factor_path: Path | None,
) -> COD_LO_L2:
    """
    Calculate CoDICE-Lo L2 data products.

    Parameters
    ----------
    cod_lo_l1b_data : xarray.Dataset
        Data in xarray format.
    l2_lut_path : Path
        Efficiency lookup table.
    l2_geometric_factor_path : Path
        Geometric factor lookup table.

    Returns
    -------
    c_over_o_abundance : float
        Ratio of C over O.
    mg_over_o_abundance : float
        Ratio of Mg over O.
    fe_over_o_abundance : float
        Ratio of Fe over O.
    c_plus_6_over_c_plus_5 : np.array
        Ratio of C+6 over C+5.
    o_plus_7_over_o_plus_6 : np.array
        Ratio of O+7 over O+6.
    fe_low_over_fe_high : np.array
        Ratio of Fe low over Fe high.
    """
    geometric_factor_lookup = get_geometric_factor_lut(None, l2_geometric_factor_path)
    geometric_factors = compute_geometric_factors(
        cod_lo_l1b_data, geometric_factor_lookup
    )

    efficiency_lookup = get_efficiency_lut(None, l2_lut_path)
    efficiencies = efficiency_lookup[efficiency_lookup["product"] == "sw"]
    intensity = process_lo_species_intensity(
        cod_lo_l1b_data,
        constants.LO_IALIRT_VARIABLE_NAMES,
        geometric_factors,
        efficiencies,
        constants.SOLAR_WIND_POSITIONS,
    )
    pseudo_density_dict = {}

    for species in constants.LO_IALIRT_VARIABLE_NAMES:
        pseudo_density = (
            intensity[species]
            * np.sqrt(cod_lo_l1b_data["energy_table"])
            * np.sqrt(constants.LO_IALIRT_M_OVER_Q[species])
        )  # (epoch, esa_step, spin_sector)

        summed_pseudo_density = pseudo_density.sum(dim="esa_step").squeeze(
            "spin_sector"
        )  # (epoch,)
        pseudo_density_dict[species] = summed_pseudo_density.values

    # Denominator.
    # Note that outside of this test a zero value denominator
    # will lead to a null value.
    # The use of zeros here is only to match the test data as
    # confirmed by the instrument team.
    o_abundance_denom = (
        pseudo_density_dict["oplus6"]
        + pseudo_density_dict["oplus7"]
        + pseudo_density_dict["oplus8"]
    )

    c_over_o_abundance_num = (
        pseudo_density_dict["cplus5"] + pseudo_density_dict["cplus6"]
    )
    mg_over_o_abundance_num = pseudo_density_dict["mg"]
    fe_over_o_abundance_num = (
        pseudo_density_dict["fe_loq"] + pseudo_density_dict["fe_hiq"]
    )

    if float(o_abundance_denom) != 0:
        c_over_o_abundance = c_over_o_abundance_num / o_abundance_denom
        mg_over_o_abundance = mg_over_o_abundance_num / o_abundance_denom
        fe_over_o_abundance = fe_over_o_abundance_num / o_abundance_denom

        c_over_o_abundance = Decimal(f"{float(c_over_o_abundance):.3f}")
        mg_over_o_abundance = Decimal(f"{float(mg_over_o_abundance):.3f}")
        fe_over_o_abundance = Decimal(f"{float(fe_over_o_abundance):.3f}")
    else:
        c_over_o_abundance, mg_over_o_abundance, fe_over_o_abundance = (
            FILLVAL_FLOAT32,
            FILLVAL_FLOAT32,
            FILLVAL_FLOAT32,
        )

    if float(pseudo_density_dict["cplus5"]) != 0:
        c_plus_6_over_c_plus_5 = (
            pseudo_density_dict["cplus6"] / pseudo_density_dict["cplus5"]
        )

        c_plus_6_over_c_plus_5 = Decimal(f"{float(c_plus_6_over_c_plus_5):.3f}")
    else:
        c_plus_6_over_c_plus_5 = FILLVAL_FLOAT32

    if float(pseudo_density_dict["oplus6"]) != 0:
        o_plus_7_over_o_plus_6 = (
            pseudo_density_dict["oplus7"] / pseudo_density_dict["oplus6"]
        )
        o_plus_7_over_o_plus_6 = Decimal(f"{float(o_plus_7_over_o_plus_6):.3f}")
    else:
        o_plus_7_over_o_plus_6 = FILLVAL_FLOAT32

    if float(pseudo_density_dict["fe_hiq"]) != 0:
        fe_low_over_fe_high = (
            pseudo_density_dict["fe_loq"] / pseudo_density_dict["fe_hiq"]
        )
        fe_low_over_fe_high = Decimal(f"{float(fe_low_over_fe_high):.3f}")
    else:
        fe_low_over_fe_high = FILLVAL_FLOAT32

    return COD_LO_L2(
        c_over_o_abundance=c_over_o_abundance,
        mg_over_o_abundance=mg_over_o_abundance,
        fe_over_o_abundance=fe_over_o_abundance,
        c_plus_6_over_c_plus_5=c_plus_6_over_c_plus_5,
        o_plus_7_over_o_plus_6=o_plus_7_over_o_plus_6,
        fe_low_over_fe_high=fe_low_over_fe_high,
    )


def process_codice(
    dataset: xr.Dataset,
    l1a_lut_path: Path,
    l2_lut_path: Path,
    sensor: str,
    l2_geometric_factor_path: Path | None = None,
) -> tuple:
    """
    Create final data products.

    Parameters
    ----------
    dataset : xr.Dataset
        Decommed L0 data.
    l1a_lut_path : Path
        L1A LUT path.
    l2_lut_path : Path
        L2 LUT path.
    sensor : str
        Sensor (codice_hi or codice_lo).
    l2_geometric_factor_path : Path
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
                cod_lo_science_values, cod_lo_metadata_values, "lo"
            )
            l1a_lo = l1a_lo_species(cod_lo_dataset, l1a_lut_path)
            l1b_lo = cast(
                xr.Dataset,
                convert_to_rates(
                    l1a_lo,
                    "lo-ialirt",
                ),
            )
            mid_measurement = int((l1b_lo["epoch"][0] + l1b_lo["epoch"][-1]) // 2)
            yyyymmdd = datetime.datetime.strptime(
                et_to_utc(ttj2000ns_to_et(mid_measurement)), "%Y-%m-%dT%H:%M:%S.%f"
            ).strftime("%Y%m%d")
            l1b_lo.attrs["Logical_file_id"] = (
                f"imap_ialirt_l1_realtime_{yyyymmdd}_v000.cdf"
            )

            l2_lo = calculate_ratios(l1b_lo, l2_lut_path, l2_geometric_factor_path)

            codice_lo_data.append(
                _populate_instrument_header_items(met)
                | {
                    "instrument": f"{sensor}",
                    "codice_lo_epoch": int(l1a_lo["epoch"]),
                    f"{sensor}_c_over_o_abundance": l2_lo.c_over_o_abundance,
                    f"{sensor}_mg_over_o_abundance": l2_lo.mg_over_o_abundance,
                    f"{sensor}_fe_over_o_abundance": l2_lo.fe_over_o_abundance,
                    f"{sensor}_c_plus_6_over_c_plus_5": l2_lo.c_plus_6_over_c_plus_5,
                    f"{sensor}_o_plus_7_over_o_plus_6": l2_lo.o_plus_7_over_o_plus_6,
                    f"{sensor}_fe_low_over_fe_high": l2_lo.fe_low_over_fe_high,
                }
            )

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
                cod_hi_science_values, cod_hi_metadata_values, "hi"
            )
            l1a_hi = l1a_ialirt_hi(cod_hi_dataset, l1a_lut_path)
            l1b_hi = convert_to_rates(
                l1a_hi,
                "hi-ialirt",
            )
            l2_hi = convert_to_intensities(l1b_hi, l2_lut_path, "h")
            # Put in Decimal format so DynamoDB can read it.
            dec_l2_hi = np.vectorize(lambda x: Decimal(f"{float(x):.4f}"))(
                l2_hi
            ).tolist()

            codice_hi_data.append(
                _populate_instrument_header_items(met)
                | {
                    "instrument": f"{sensor}",
                    f"{sensor}_epoch": [int(epoch) for epoch in l1b_hi["epoch"]],
                    f"{sensor}_h": dec_l2_hi,
                }
            )

    return codice_lo_data, codice_hi_data
