"""IMAP-HIT L1B data processing."""

import logging
from dataclasses import fields

import numpy as np
import xarray as xr

from imap_processing import utils
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.hit import hit_cdf_attrs
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(l1a_dataset: xr.Dataset):
    """Process HIT data to L1B.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        l1A data

    Returns
    -------
    xarray.Dataset
        L1B processed data
    """
    # TODO: Check for type of L1A dataset and determine what L1B products to make
    #   Need more info from instrument teams. Work with housekeeping data for now
    logical_source = "imap_hit_l1b_hk"

    # Create datasets
    datasets = []
    if "_hk" in logical_source:
        dataset = create_hk_dataset()
        datasets.append(dataset)

    # Create CDF files
    logger.info("Creating CDF files for HIT L1B data")
    cdf_filepaths = []
    for dataset in datasets:
        cdf_file = write_cdf(dataset)
        cdf_filepaths.append(cdf_file)
    logger.info(f"L1B CDF files created: {cdf_filepaths}")
    return cdf_filepaths


# TODO: This is going to work differently when we have sample data
def create_hk_dataset():
    """Create a housekeeping dataset.

    Returns
    -------
    xr.dataset
        Dataset with all data product fields in xr.DataArray
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

    # Create fake data for now

    # Convert integers into datetime64[s]
    epoch_converted_time = [utils.calc_start_time(time) for time in [0, 1, 2]]

    # Create xarray data arrays for dependencies
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    adc_channels = xr.DataArray(
        np.arange(3, dtype=np.uint16),
        name="adc_channels",
        dims=["adc_channels"],
        attrs=hit_cdf_attrs.l1b_hk_attrs["adc_channels"].output(),
    )

    # Create xarray dataset
    hk_dataset = xr.Dataset(
        coords={"epoch": epoch_time, "adc_channels": adc_channels},
        attrs=hit_cdf_attrs.hit_hk_l1b_attrs.output(),
    )

    # Create xarray data array for each data field
    for data_field in data_fields:
        field = data_field.name.lower()
        if field not in skip_keys:
            # Create a list of all the dimensions using the DEPEND_I keys in the
            # attributes
            dims = [
                value
                for key, value in hit_cdf_attrs.l1b_hk_attrs[field].output().items()
                if "DEPEND" in key
            ]

            # TODO: This is temporary.
            #  The data will be set in the data class when that's created
            if field == "leak_i":
                # 2D array - needs two dims
                hk_dataset[field] = xr.DataArray(
                    [[0, 0, 1], [0, 1, 0], [0, 0, 1]],
                    # np.random.randint(2, size=(65, 3)),
                    dims=dims,
                    attrs=hit_cdf_attrs.l1b_hk_attrs[field].output(),
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
                    attrs=hit_cdf_attrs.l1b_hk_attrs[field].output(),
                )
            else:
                hk_dataset[field] = xr.DataArray(
                    [1, 1, 1],
                    dims=dims,
                    attrs=hit_cdf_attrs.l1b_hk_attrs[field].output(),
                )

    logger.info("HIT L1B datasets created")
    return hk_dataset
