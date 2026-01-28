"""Module for generating Level 1d magnetic field data."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1d.mag_l1d_data import MagL1d, MagL1dConfiguration
from imap_processing.mag.l2.mag_l2_data import ValidFrames


def mag_l1d(  # noqa: PLR0912
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
        instrument_mode = source.split("_")[-1]
        match instrument_mode:
            case "norm-magi":
                input_magi_norm = dataset
            case "norm-mago":
                input_mago_norm = dataset
            case "burst-magi":
                input_magi_burst = dataset
            case "burst-mago":
                input_mago_burst = dataset
            case _:
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

    # Verify that MAGO is primary sensor for all vectors before applying gradiometry
    if not input_mago_norm.attrs.get("all_vectors_primary", 1):
        config.apply_gradiometry = False

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

    # Nominally, this is expected to create MAGO data. However, if the configuration
    # setting for always_output_mago is set to False, it will create MAGI data.

    l1d_norm.rotate_frame(ValidFrames.SRF)
    norm_srf_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    l1d_norm.rotate_frame(ValidFrames.DSRF)
    norm_dsrf_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    l1d_norm.rotate_frame(ValidFrames.GSE)
    norm_gse_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    l1d_norm.rotate_frame(ValidFrames.RTN)
    norm_rtn_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
    output_datasets.append(norm_srf_dataset)
    output_datasets.append(norm_dsrf_dataset)
    output_datasets.append(norm_gse_dataset)
    output_datasets.append(norm_rtn_dataset)

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

        # TODO: frame specific attributes may be required
        l1d_burst.rotate_frame(ValidFrames.SRF)
        burst_srf_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        l1d_burst.rotate_frame(ValidFrames.DSRF)
        burst_dsrf_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        l1d_burst.rotate_frame(ValidFrames.GSE)
        burst_gse_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        l1d_burst.rotate_frame(ValidFrames.RTN)
        burst_rtn_dataset = l1d_burst.generate_dataset(attributes, day_to_process)
        output_datasets.append(burst_srf_dataset)
        output_datasets.append(burst_dsrf_dataset)
        output_datasets.append(burst_gse_dataset)
        output_datasets.append(burst_rtn_dataset)

    # Output ancillary files
    # Add spin offsets dataset from normal mode processing
    if l1d_norm.spin_offsets is not None:
        spin_offset_dataset = l1d_norm.generate_spin_offset_dataset()
        spin_offset_dataset.attrs["Logical_source"] = "imap_mag_l1d_spin-offsets"
        output_datasets.append(spin_offset_dataset)

    # Add gradiometry offsets dataset if gradiometry was applied
    if l1d_norm.config.apply_gradiometry and hasattr(l1d_norm, "gradiometry_offsets"):
        gradiometry_dataset = l1d_norm.gradiometry_offsets.copy()
        gradiometry_dataset.attrs["Logical_source"] = (
            "imap_mag_l1d_gradiometry-offsets-norm"
        )
        output_datasets.append(gradiometry_dataset)

        # Also add burst gradiometry offsets if burst data was processed
        if input_mago_burst is not None and input_magi_burst is not None:
            if hasattr(l1d_burst, "gradiometry_offsets"):
                burst_gradiometry_dataset = l1d_burst.gradiometry_offsets.copy()
                burst_gradiometry_dataset.attrs["Logical_source"] = (
                    "imap_mag_l1d_gradiometry-offsets-burst"
                )
                output_datasets.append(burst_gradiometry_dataset)

    return output_datasets
