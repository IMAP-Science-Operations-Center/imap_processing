"""Module to run MAG L2 processing."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1b.mag_l1b import calibrate_vector
from imap_processing.mag.l2.mag_l2_data import MagL2


def mag_l2(
    calibration_dataset: xr.Dataset,
    offset_dataset: xr.Dataset,
    input_data: xr.Dataset,
    data_version: str,
) -> list[xr.Dataset]:
    """
    Complete MAG L2 processing.

    Input data can be burst or normal mode, but MUST match the file in offset_dataset.
    TODO: retrieve the file from offset_dataset in cli.py.
    Offset dataset will be split by norm/burst but will include all other data.
    Calibration dataset is the same for all runs.

    MAGi data is not used unless we indicate it.

    Parameters
    ----------
    calibration_dataset : xr.Dataset
        Calibration ancillary file input.
    offset_dataset : xr.Dataset
        Offset ancillary file input.
    input_data : xr.Dataset
        Input data from MAG L1C or L1B.
    data_version : str
        Version of output file.

    Returns
    -------
    list[xr.Dataset]
        List of xarray datasets ready to write to CDF file. Expected to be four outputs
        for different frames.
    """
    # TODO we may need to combine multiple calibration datasets into one timeline.

    # TODO set from offsets file
    always_output_mago = True

    vectors = apply_calibration_matrix(
        input_data["vectors"].data, calibration_dataset, always_output_mago
    )

    basic_test_data = MagL2(
        vectors[:, :3],  # level 2 vectors don't include range
        input_data["epoch"].data,
        input_data["vectors"].data[:, 3],
        {"Data_version": data_version},
        np.zeros(len(input_data["epoch"].data)),
        np.zeros(len(input_data["epoch"].data)),
        DataMode.NORM,
    )
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    # temporarily point to l1c
    attributes.add_instrument_variable_attrs("mag", "l1c")
    return [basic_test_data.generate_dataset(attributes)]


def apply_calibration_matrix(
    vectors: np.ndarray, calibration_dataset: xr.Dataset, use_mago: bool = True
) -> np.ndarray:
    """
    Apply the calibration file to the vectors to rotate them in space.

    Parameters
    ----------
    vectors : np.ndarray
        (n, 4) array of vectors to rotate and timeshift.
    calibration_dataset : xr.Dataset
        Ancillary file input for calibration.
    use_mago : bool
        Use the MAGo calibration matrix. Default is True.

    Returns
    -------
    np.ndarray
        Rotated and timeshifted vectors.
    """
    if use_mago:
        # TODO these variable names will change but the structure is the same
        calibration_data = calibration_dataset["MFOTOURFO"]
    else:
        calibration_data = calibration_dataset["MFITOURFI"]

    # TODO will need to combine multiple files here
    # TODO: Check validity of the calibration file?
    output_vectors: np.ndarray = np.apply_along_axis(
        calibrate_vector, 1, vectors, calibration_data
    )

    return output_vectors
