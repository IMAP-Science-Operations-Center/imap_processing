"""Module to run MAG L2 processing."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l2.mag_l2_data import MagL2


def mag_l2(
    calibration_dataset: xr.Dataset,
    offset_dataset: xr.Dataset,
    input_data: xr.Dataset,
) -> list[xr.Dataset]:
    """
    Complete MAG L2 processing.

    Input data can be burst or normal mode, but MUST match the file in offset_dataset.
    TODO: retrieve the file from offset_dataset in cli.py.

    Parameters
    ----------
    calibration_dataset : xr.Dataset
        Calibration ancillary file input.
    offset_dataset : xr.Dataset
        Offset ancillary file input.
    input_data : xr.Dataset
        Input data from MAG L1C or L1B.

    Returns
    -------
    list[xr.Dataset]
        List of xarray datasets ready to write to CDF file. Expected to be four outputs
        for different frames.
    """
    # TODO we may need to combine multiple calibration datasets into one timeline.

    basic_test_data = MagL2(
        input_data["vectors"].data[:, :3],  # level 2 vectors don't include range
        input_data["epoch"].data,
        input_data["vectors"].data[:, 3],
        {},
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
    calibration_dataset: xr.Dataset, vectors: np.ndarray
) -> np.ndarray:
    """
    Apply the calibration file to the vectors to rotate them in space.

    Parameters
    ----------
    calibration_dataset : xr.Dataset
        Ancillary file input for calibration.
    vectors : np.ndarray
        (n, 4) array of vectors to rotate and timeshift.

    Returns
    -------
    np.ndarray
        Rotated and timeshifted vectors.
    """
    raise NotImplementedError
