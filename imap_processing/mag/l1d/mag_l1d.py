"""Module for generating Level 1d magnetic field data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1d.mag_l1d_data import MagL1d, MagL1dConfiguration
from imap_processing.mag.l2.mag_l2_data import ValidFrames


def mag_l1d(
    science_data: list[xr.Dataset],
    calibration_dataset: xr.Dataset,
    day_to_process: np.datetime64,
) -> list[xr.Dataset]:
    """
    Generate Level 1d magnetic field data from Level 1b/1c data.

    Both norm and burst mode are calculated at the same time. Normal mode MAGO and MAGI
    L1C data is required, burst mode MAGO and MAGI L1B data is optional.

    Parameters
    ----------
    science_data : list[xr.Dataset]
        The list of input datasets containing the MAG L1C and L1B data. This is required
        to have at least one normal mode dataset for MAGo and MAGi, and optionally
        burst mode datasets for MAGo and MAGi. There cannot be duplicates, so two
        norm-mago files is invalid.
    calibration_dataset : xr.Dataset
        The calibration dataset to use for processing. Generated from multiple L1D
        ancillary files using MagAncillaryCombiner class.
    day_to_process : np.datetime64
        The day to process, in np.datetime64[D] format. This is used to select the
        correct ancillary parameters and to remove excessive data from the output.

    Returns
    -------
    list[xr.Dataset]
        A list containing the generated Level 1d dataset(s).
    """
    input_magi_norm = None
    input_mago_norm = None
    input_magi_burst = None
    input_mago_burst = None
    for dataset in science_data:
        source = dataset.attrs.get("Logical_source", "")
        if "norm-magi" in source:
            input_magi_norm = dataset
        elif "norm-mago" in source:
            input_mago_norm = dataset
        elif "burst-magi" in source:
            input_magi_burst = dataset
        elif "burst-mago" in source:
            input_mago_burst = dataset
        else:
            raise ValueError(f"Input data has invalid logical source {source}")

    if input_magi_norm is None or input_mago_norm is None:
        raise ValueError(
            "Both MAGo and MAGi normal mode datasets are required for L1d processing."
        )

    day: np.datetime64 = day_to_process.astype("datetime64[D]")

    output_datasets = []

    # Read configuration out of file
    config = MagL1dConfiguration(calibration_dataset, day)

    # Only the first 3 components are used for L1d
    mago_vectors = input_mago_norm["vectors"].data[:, :3]
    magi_vectors = input_magi_norm["vectors"].data[:, :3]

    # TODO: verify that MAGO is primary sensor for all vectors before applying
    #  gradiometry

    # TODO: L1D attributes
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    attributes.add_instrument_variable_attrs("mag", "l2")

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
        day=day,
    )

    l1d_norm.rotate_frame(ValidFrames.SRF)
    norm_srf_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    l1d_norm.rotate_frame(ValidFrames.DSRF)
    norm_dsrf_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    output_datasets.append(norm_srf_dataset)
    output_datasets.append(norm_dsrf_dataset)

    if input_mago_burst is not None and input_magi_burst is not None:
        # If burst data is provided, use it to create the burst L1d dataset
        mago_burst_vectors = input_mago_burst["vectors"].data[:, :3]
        magi_burst_vectors = input_magi_burst["vectors"].data[:, :3]

        l1d_burst = MagL1d(
            vectors=mago_burst_vectors,
            epoch=input_mago_burst["epoch"].data,
            range=input_mago_burst["vectors"].data[:, 3],
            global_attributes={},
            quality_flags=np.zeros(len(input_mago_burst["epoch"].data)),
            quality_bitmask=np.zeros(len(input_mago_burst["epoch"].data)),
            data_mode=DataMode.BURST,
            magi_vectors=magi_burst_vectors,
            magi_range=input_magi_burst["vectors"].data[:, 3],
            magi_epoch=input_magi_burst["epoch"].data,
            config=config,
            spin_offsets=l1d_norm.spin_offsets,
            day=day,
        )
        l1d_burst.rotate_frame(ValidFrames.SRF)
        burst_srf_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        l1d_burst.rotate_frame(ValidFrames.DSRF)
        burst_dsrf_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        output_datasets.append(burst_srf_dataset)
        output_datasets.append(burst_dsrf_dataset)

    # TODO: Output ancillary files
    return output_datasets
