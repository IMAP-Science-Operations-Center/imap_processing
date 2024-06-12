"""MAG L1B Processing."""

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import load_cdf


def mag_l1b(input_dataset: xr.Dataset, version: str) -> xr.Dataset:
    """
    Process MAG L1B data from L1A data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process
    version : str
        The version of the output data

    Returns
    -------
    output_dataset : xr.Dataset
        The processed dataset
    """
    # TODO:
    # Read in calibration file
    # multiply all vectors by calibration file

    dims = [["direction"]]
    new_dims = [["direction"]]
    # TODO: This should definitely be loaded from AWS
    calibration_dataset = load_cdf(
        Path(__file__).parent / "imap_calibration_mag_20240229_v01.cdf"
    )
    # TODO: Check validity of time range for calibration
    # TODO: pick MFOTOURFO for mago and MFITOURI for magi

    calibration_matrix = calibration_dataset["MFOTOURFO"]

    l1b_fields = xr.apply_ufunc(
        calibrate,
        input_dataset["vectors"],
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
        keep_attrs=True,
        kwargs={"calibration_matrix": calibration_matrix},
    )

    output_dataset = input_dataset.copy()
    output_dataset["vectors"] = l1b_fields

    # TODO add/update attributes
    return output_dataset


def calibrate(input_vector: np.ndarray, calibration_matrix: xr.DataArray = None):
    """Apply calibration matrix to input vector."""
    # First, get the index of calibration_matrix from the range (input_vector[3])
    # Then matrix multiply (input_vector[:3] * calibration_matrix[range])
    # print(calibration_matrix)
    range = input_vector[3]
    return np.matmul(input_vector[:3], calibration_matrix.values[range - 1])
