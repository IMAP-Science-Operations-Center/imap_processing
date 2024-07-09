"""IMAP-HIT L1B data processing."""

import logging
from dataclasses import fields

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(l1a_dataset: xr.Dataset, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L1B.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        L1A data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    cdf_filepaths : xarray.Dataset
        L1B processed data.
    """
    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level="l1b")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # TODO: Check for type of L1A dataset and determine what L1B products to make
    #   Need more info from instrument teams. Work with housekeeping data for now
    logical_source = "imap_hit_l1b_hk"

    # Create datasets
    datasets = []
    if "_hk" in logical_source:
        dataset = create_hk_dataset(attr_mgr)
        datasets.append(dataset)
    elif "_sci" in logical_source:
        # process science data. placeholder for future code
        pass

    return datasets


# TODO: This is going to work differently when we have sample data
def create_hk_dataset(attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Create a housekeeping dataset.

    Parameters
    ----------
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    hk_dataset : xarray.dataset
        Dataset with all data product fields in xarray.DataArray.
    """
    logger.info("Creating datasets for HIT L1B data")

    # TODO: TEMPORARY. Need to update to use the L1B data class once that exists.
    #  Using l1a housekeeping data class for now since l1b housekeeping has the
    #  same data fields
    data_fields = fields(Housekeeping)

    # TODO define keys to skip. This will change later.
    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]

    logical_source = "imap_hit_l1b_hk"

    # Create fake data for now

    # Convert integers into datetime64[s]
    epoch_converted_time = met_to_j2000ns([0, 1, 2])

    # Shape for dims
    n_epoch = 3
    n_channels = 64

    # Create xarray data arrays for dependencies
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    adc_channels = xr.DataArray(
        np.arange(n_channels, dtype=np.uint16),
        name="adc_channels",
        dims=["adc_channels"],
        attrs=attr_mgr.get_variable_attributes("adc_channels"),
    )

    # Create xarray dataset
    hk_dataset = xr.Dataset(
        coords={"epoch": epoch_time, "adc_channels": adc_channels},
        attrs=attr_mgr.get_global_attributes(logical_source),
    )

    # Create xarray data array for each data field
    for data_field in data_fields:
        field = data_field.name.lower()
        if field not in skip_keys:
            # Create a list of all the dimensions using the DEPEND_I keys in the
            # attributes
            dims = [
                value
                for key, value in attr_mgr.get_variable_attributes(field).items()
                if "DEPEND" in key
            ]

            # TODO: This is temporary.
            #  The data will be set in the data class when that's created
            if field == "leak_i":
                # 2D array - needs two dims
                hk_dataset[field] = xr.DataArray(
                    np.ones((n_epoch, n_channels), dtype=np.uint16),
                    dims=dims,
                    attrs=attr_mgr.get_variable_attributes(field),
                )
            elif field in [
                "preamp_l234a",
                "preamp_l1a",
                "preamp_l1b",
                "preamp_l234b",
                "temp0",
                "temp1",
                "temp2",
                "temp3",
                "analog_temp",
                "hvps_temp",
                "idpu_temp",
                "lvps_temp",
                "ebox_3d4vd",
                "ebox_5d1vd",
                "ebox_p12va",
                "ebox_m12va",
                "ebox_p5d7va",
                "ebox_m5d7va",
                "ref_p5v",
                "l1ab_bias",
                "l2ab_bias",
                "l34a_bias",
                "l34b_bias",
                "ebox_p2d0vd",
            ]:
                hk_dataset[field] = xr.DataArray(
                    np.ones(3, dtype=np.float16),
                    dims=dims,
                    attrs=attr_mgr.get_variable_attributes(field),
                )
            else:
                hk_dataset[field] = xr.DataArray(
                    [1, 1, 1],
                    dims=dims,
                    attrs=attr_mgr.get_variable_attributes(field),
                )

    logger.info("HIT L1B datasets created")
    return hk_dataset
