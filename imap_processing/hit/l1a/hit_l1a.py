"""Decommutate HIT CCSDS data and create L1a data products."""

import logging
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


class HitAPID(IntEnum):
    """
    HIT APID Mappings.

    Attributes
    ----------
    HIT_HSKP: int
        Housekeeping
    HIT_SCIENCE : int
        Science
    HIT_IALRT : int
        I-ALiRT
    """

    HIT_HSKP = 1251
    HIT_SCIENCE = 1252
    HIT_IALRT = 1253


def hit_l1a(packet_file: str, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    # TODO add logging

    # Unpack ccsds file
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/" "hit_packet_definitions.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
    )

    # Create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # Process science to l1a.
    processed_data = []
    for apid in datasets_by_apid:
        if apid == HitAPID.HIT_HSKP:
            housekeeping_dataset = process_housekeeping(
                datasets_by_apid[apid], attr_mgr
            )
            processed_data.append(housekeeping_dataset)
        elif apid == HitAPID.HIT_SCIENCE:
            # TODO complete science data processing
            print("Skipping science data for now")
            # science_dataset = process_science(datasets_by_apid[apid], attr_mgr)
        else:
            raise Exception(f"Unknown APID [{apid}]")

    return processed_data


def concatenate_leak_variables(dataset: xr.Dataset) -> xr.Dataset:
    """
    Concatenate leak variables in the dataset.

    Updates the housekeeping dataset to replace the individual
    leak_i_00, leak_i_01, ..., leak_i_63 variables with a single
    leak_i variable as a 2D array. This variable represents
    leakage current [Voltage] data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing 64 leak variables.

    Returns
    -------
    dataset : xarray.Dataset
        Updated dataset with concatenated leak variables.
    """
    # Stack 64 leak variables (leak_00, leak_01, ..., leak_63)
    leak_vars = [dataset[f"leak_i_{i:02d}"] for i in range(64)]

    # Concatenate along 'leak_index' and reorder dimensions
    stacked_leaks = xr.concat(leak_vars, dim="leak_index").transpose(
        "epoch", "leak_index"
    )
    dataset["leak_i"] = stacked_leaks

    # Drop the individual leak variables
    updated_dataset = dataset.drop_vars([f"leak_i_{i:02d}" for i in range(64)])

    return updated_dataset


def process_science(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Will process science dataset for CDF product.

    Process binary science data for CDF creation. The data is
    grouped into science frames, decommutated and decompressed,
    and split into count rates and event datasets. Updates the
    dataset attributes and coordinates and data variable
    dimensions according to specifications in a cdf yaml file.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing HIT science data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
    logger.info("Creating HIT L1A science datasets")

    # Logical sources for the two products.
    # logical_sources = ["imap_hit_l1a_sci-counts", "imap_hit_l1a_pulse-height-event"]

    # TODO: Complete this function
    #  - call decom_hit.py to decommutate the science data
    #  - split the science data into count rates and event datasets
    #  - update dimensions and add attributes to the dataset and data arrays
    #  - return list of two datasets (count rates and events)?

    # logger.info("HIT L1A event dataset created")
    # logger.info("HIT L1A count rates dataset created")

    return dataset


def process_housekeeping(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes
) -> xr.Dataset:
    """
    Will process housekeeping dataset for CDF product.

    Updates the housekeeping dataset to replace with a single
    leak_i variable as a 2D array. Also updates the dataset
    attributes and coordinates and data variable dimensions
    according to specifications in a cdf yaml file.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing HIT housekeeping data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
    logger.info("Creating HIT L1A housekeeping dataset")

    logical_source = "imap_hit_l1a_hk"

    # Drop keys that are not CDF data variables
    drop_keys = [
        "pkt_apid",
        "sc_tick",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    ]

    # Drop variables not needed for CDF
    dataset = dataset.drop_vars(drop_keys)

    # Create data arrays for dependencies
    adc_channels = xr.DataArray(
        np.arange(64, dtype=np.uint16),
        name="adc_channels",
        dims=["adc_channels"],
        attrs=attr_mgr.get_variable_attributes("adc_channels"),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    adc_channels_label = xr.DataArray(
        adc_channels.values.astype(str),
        name="adc_channels_label",
        dims=["adc_channels_label"],
        attrs=attr_mgr.get_variable_attributes("adc_channels_label"),
    )

    # Update dataset coordinates and attributes
    dataset = dataset.assign_coords(
        {
            "adc_channels": adc_channels,
            "adc_channels_label": adc_channels_label,
        }
    )
    dataset.attrs = attr_mgr.get_global_attributes(logical_source)

    # Stack 64 leak variables (leak_00, leak_01, ..., leak_63)
    dataset = concatenate_leak_variables(dataset)

    # Assign attributes and dimensions to each data array in the Dataset
    for field, data in dataset.data_vars.items():
        # Create a list of dimensions using the DEPEND_I keys in the
        # attributes
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]
        dataset[field] = xr.DataArray(
            data,
            dims=dims,
            attrs=attr_mgr.get_variable_attributes(field),
        )

    logger.info("HIT L1A housekeeping dataset created")
    return dataset
