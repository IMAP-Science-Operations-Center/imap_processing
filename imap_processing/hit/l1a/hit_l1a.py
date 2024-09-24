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
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
        use_derived_value=False,
    )

    # Create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # Process science to l1a.
    if HitAPID.HIT_HSKP in datasets_by_apid:
        datasets_by_apid[HitAPID.HIT_HSKP] = process_housekeeping(
            datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr
        )
    if HitAPID.HIT_SCIENCE in datasets_by_apid:
        # TODO complete science data processing
        print("Skipping science data for now")
        datasets_by_apid[HitAPID.HIT_SCIENCE] = process_science(
            datasets_by_apid[HitAPID.HIT_SCIENCE], attr_mgr
        )

    return list(datasets_by_apid.values())


def concatenate_leak_variables(
    dataset: xr.Dataset, adc_channels: xr.DataArray
) -> xr.Dataset:
    """
    Concatenate leak variables in the dataset.

    Updates the housekeeping dataset to replace the individual
    leak_i_00, leak_i_01, ..., leak_i_63 variables with a single
    leak_i variable as a 2D array. "i" here represents current
    in the leakage current [Voltage] data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing 64 leak variables.
    adc_channels : xarray.DataArray
        DataArray to be used as a dimension for the concatenated leak variables.

    Returns
    -------
    dataset : xarray.Dataset
        Updated dataset with concatenated leak variables.
    """
    # Stack 64 leak variables (leak_00, leak_01, ..., leak_63)
    leak_vars = [dataset[f"leak_i_{i:02d}"] for i in range(64)]

    # Concatenate along 'adc_channels' and reorder dimensions
    stacked_leaks = xr.concat(leak_vars, dim=adc_channels).transpose(
        "epoch", "adc_channels"
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
        np.arange(64, dtype=np.uint8),
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
    dataset = concatenate_leak_variables(dataset, adc_channels)

    # Assign attributes and dimensions to each data array in the Dataset
    for field in dataset.data_vars.keys():
        # Create a dict of dimensions using the DEPEND_I keys in the
        # attributes
        dims = {
            key: value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        }
        dataset[field].attrs = attr_mgr.get_variable_attributes(field)
        dataset[field].assign_coords(dims)

    dataset.epoch.attrs = attr_mgr.get_variable_attributes("epoch")

    return dataset
