"""
Classes and functions used in HIT processing.

This module contains utility classes and functions that are used by
HIT processing modules.
"""

from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import packet_file_to_datasets


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


def get_datasets_by_apid(
    packet_file: str, derived: bool = False
) -> dict[int, xr.Dataset]:
    """
    Get datasets by APID from a CCSDS packet file.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    derived : bool, optional
        Flag to use derived values, by default False.
        Only set to True to get engineering units for L1B
        housekeeping data product.

    Returns
    -------
    datasets_by_apid : dict[int, xr.Dataset]
        Dictionary of xarray datasets by APID.
    """
    # Unpack ccsds file
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )
    datasets_by_apid: dict[int, xr.Dataset] = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
        use_derived_value=derived,
    )
    return datasets_by_apid


def get_attribute_manager(data_version: str, level: str) -> ImapCdfAttributes:
    """
    Create an attribute manager for the HIT data products.

    Parameters
    ----------
    data_version : str
        Version of the data product being created.
    level : str
        Data level of the product being created.

    Returns
    -------
    attr_mgr : ImapCdfAttributes
        Attribute manager to set CDF attributes.
    """
    # Create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level=level)
    attr_mgr.add_global_attribute("Data_version", data_version)
    return attr_mgr


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


def process_housekeeping_data(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Will process housekeeping dataset for CDF product.

    Updates the housekeeping dataset with a single 2D leak_i
    variable. Also updates the dataset attributes, coordinates
    and data variable dimensions according to specifications in
    a cdf yaml file. This function is used for both L1A and L1B
    housekeeping data products.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing HIT housekeeping data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    logical_source : str
        Logical source of the data -> imap_hit_l1a_hk or imap_hit_l1b_hk.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
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
