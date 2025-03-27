import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.L2.mag_l2_data import MagL2


def mag_l2(
    calibration_dataset: xr.Dataset, offset_dataset: xr.Dataset, input_data: xr.Dataset
) -> xr.Dataset:
    """
    Complete MAG L2 processing.

    Input data can be burst or normal mode, but MUST match the file in offset_dataset.
    TODO: retrieve the file from offset_dataset in cli.py.

    Parameters
    ----------
    calibration_dataset
    offset_dataset
    input_data

    Returns
    -------

    """
    basic_test_data = MagL2(
        input_data["vectors"].data[:, :3],  # level 2 vectors don't include range
        input_data["epoch"].data,
        input_data["vectors"].data[:, 3],
        {},
        np.zeros(len(input_data["epoch"].data)),
        np.zeros(len(input_data["epoch"].data)),
    )
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    # temporarily point to l1c
    attributes.add_instrument_variable_attrs("mag", "l1c")

    return basic_test_data.generate_dataset(attributes)


def apply_calibration_matrix(
    calibration_dataset: xr.Dataset, vectors: np.ndarray
) -> np.ndarray:
    pass


def apply_offsets(
    offset_dataset: xr.Dataset, input_timestamps: np.ndarray, vectors: np.ndarray
) -> MagL2:
    pass
