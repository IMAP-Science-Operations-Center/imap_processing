import numpy as np
import xarray as xr

from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration


def mag_l1d(
    calibration_dataset: xr.Dataset,
    input_mago: xr.Dataset,
    input_magi: xr.Dataset,
    day_to_process: np.datetime64,
) -> list[xr.Dataset]:
    day: np.datetime64 = day_to_process.astype("datetime64[D]")

    calibration_matrix_mago = retrieve_matrix_from_l2_calibration(
        calibration_dataset, day, use_mago=True
    )

    calibration_matrix_magi = retrieve_matrix_from_l2_calibration(
        calibration_dataset, day, use_mago=False
    )

    l1d = MagL1d()

    return []
