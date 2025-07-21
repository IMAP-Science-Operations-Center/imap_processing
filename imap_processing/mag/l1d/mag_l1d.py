"""Module for generating Level 1d magnetic field data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1d.mag_l1d_data import MagL1d, MagL1dConfiguration
from imap_processing.mag.l2.mag_l2_data import ValidFrames


def mag_l1d(
    calibration_dataset: xr.Dataset,
    input_mago_norm: xr.Dataset,
    input_magi_norm: xr.Dataset,
    day_to_process: np.datetime64,
    input_mago_burst: xr.Dataset = None,
    input_magi_burst: xr.Dataset = None,
) -> list[xr.Dataset]:
    """
    Generate Level 1d magnetic field data from Level 1b/1c data.

    Both norm and burst mode are calculated at the same time. Normal mode L1C data is
    required, burst mode L1B data is optional.

    Parameters
    ----------
    calibration_dataset : xr.Dataset
        The calibration dataset to use for processing. Generated from multiple L1D
        ancillary files using MagAncillaryCombiner class.
    input_mago_norm : xr.Dataset
        The MAGo normal mode input dataset (MAG L1C).
    input_magi_norm : xr.Dataset
        The MAGi normal mode input dataset (MAG L1C).
    day_to_process : np.datetime64
        The day to process, in np.datetime64[D] format. This is used to select the
        correct ancillary parameters and to remove excessive data from the output.
    input_mago_burst : xr.Dataset, optional
        The MAGo burst mode input dataset (MAG L1B). If not provided, burst mode will
        not be calculated.
    input_magi_burst : xr.Dataset, optional
        The MAGi burst mode input dataset (MAG L1B). If not provided, burst mode will
        not be calculated.

    Returns
    -------
    list[xr.Dataset]
        A list containing the generated Level 1d dataset(s).
    """
    day: np.datetime64 = day_to_process.astype("datetime64[D]")

    # Read configuration out of file
    config = MagL1dConfiguration(calibration_dataset, day)

    # Only the first 3 components are used for L1d
    mago_vectors = input_mago_norm["vectors"].data[:, :3]
    magi_vectors = input_magi_norm["vectors"].data[:, :3]

    # TODO: verify that MAGO is primary sensor for all vectors before applying
    #  gradiometry

    l1d_norm = MagL1d(
        vectors=mago_vectors,
        epoch=input_mago_norm["epoch"].data,
        range=input_mago_norm["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(input_mago_norm["epoch"].data)),
        quality_bitmask=np.zeros(len(input_mago_norm["epoch"].data)),
        data_mode=DataMode.NORM,
        magi_vectors=magi_vectors,
        magi_range=input_magi_norm["vectors"].data[:, 3],
        magi_epoch=input_magi_norm["epoch"].data,
        config=config,
    )

    # TODO: L1D attributes
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    attributes.add_instrument_variable_attrs("mag", "l2")
    l1d_norm.rotate_frame(ValidFrames.SRF)

    output_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    # TODO: Output ancillary files
    return [output_dataset]
