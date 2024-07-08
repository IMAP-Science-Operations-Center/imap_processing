"""MAG L1B Processing."""

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf


def mag_l1b(input_dataset: xr.Dataset, version: str) -> xr.Dataset:
    """
    Will process MAG L1B data from L1A data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    version : str
        The version of the output data.

    Returns
    -------
    output_dataset : xr.Dataset
        The processed dataset.
    """
    # TODO:
    # Read in calibration file
    # multiply all vectors by calibration file

    output_dataset = mag_l1b_processing(input_dataset)
    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_global_attribute("Data_version", version)

    # Variable attributes can remain the same as L1A
    input_logical_source = input_dataset.attrs["Logical_source"]
    if isinstance(input_dataset.attrs["Logical_source"], list):
        input_logical_source = input_dataset.attrs["Logical_source"][0]

    logical_source = input_logical_source.replace("l1a", "l1b")
    output_dataset.attrs = attribute_manager.get_global_attributes(logical_source)

    return output_dataset


def mag_l1b_processing(input_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process MAG L1B data from L1A data.

    MAG L1B is almost identical to L1A, with only the vectors and attributes getting
    updated. All non-vector variables are the same.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.

    Returns
    -------
    output_dataset : xr.Dataset
        L1b dataset.
    """
    # TODO: There is a time alignment step that will add a lot of complexity.
    # This needs to be done once we have some SPICE time data.

    dims = [["direction"]]
    new_dims = [["direction"]]
    # TODO: This should definitely be loaded from AWS
    calibration_dataset = load_cdf(
        Path(__file__).parent / "imap_calibration_mag_20240229_v01.cdf"
    )
    # TODO: Check validity of time range for calibration
    if "mago" in input_dataset.attrs["Logical_source"][0]:
        calibration_matrix = calibration_dataset["MFOTOURFO"]
    else:
        calibration_matrix = calibration_dataset["MFITOURFI"]

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


def calibrate(
    input_vector: np.ndarray, calibration_matrix: xr.DataArray = None
) -> np.ndarray:
    """
    Apply calibration matrix to input vector.

    The calibration matrix contains 3x3 matrices for each range. To calibrate the input
    vector, we take the range (which is the fourth value of the vector) to determine
    the correct calibration matrix. We then multiply the input vector by that matrix.

    Parameters
    ----------
    input_vector : numpy.ndarray
        The input vector to calibrate [x, y, z, range].
    calibration_matrix : xr.DataArray
        The full set of calibration matrices, for each range. Size is ((3, 3, 4)).

    Returns
    -------
    updated_vector : numpy.ndarray
        Calibrated vector.
    """
    updated_vector = input_vector.copy()

    updated_vector[:3] = np.matmul(
        input_vector[:3], calibration_matrix.values[:, :, int(input_vector[3])]
    )

    return updated_vector
