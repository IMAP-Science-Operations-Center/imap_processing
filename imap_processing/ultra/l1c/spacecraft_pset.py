"""Calculate Pointing Set Grids."""

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_background_rates,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

# TODO: This is a placeholder for the API lookup table directory.
TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


def calculate_spacecraft_pset(
    de_dataset: xr.Dataset,
    extendedspin_dataset: xr.Dataset,
    cullingmask_dataset: xr.Dataset,
    name: str,
) -> xr.Dataset:
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing de data.
    extendedspin_dataset : xarray.Dataset
        Dataset containing extendedspin data.
    cullingmask_dataset : xarray.Dataset
        Dataset containing cullingmask data.
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict: dict[str, np.ndarray] = {}

    v_mag_dps_spacecraft = np.linalg.norm(de_dataset["velocity_dps_sc"].values, axis=1)
    vhat_dps_spacecraft = (
        de_dataset["velocity_dps_sc"].values / v_mag_dps_spacecraft[:, np.newaxis]
    )

    intervals, _, energy_bin_geometric_means = build_energy_bins()
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_spacecraft,
        de_dataset["energy_spacecraft"].values,
        intervals,
        nside=128,
    )
    healpix = np.arange(n_pix)

    # calculate background rates
    background_rates = get_background_rates()

    # TODO: calculate sensitivity and interpolate based on energy.

    # Calculate exposure
    constant_exposure = TEST_PATH / "ultra_90_dps_exposure.csv"
    df_exposure = pd.read_csv(constant_exposure)
    exposure_pointing = get_spacecraft_exposure_times(df_exposure)

    # For ISTP, epoch should be the center of the time bin.
    pset_dict["epoch"] = de_dataset.epoch.data[0].astype(np.int64)
    pset_dict["counts"] = counts
    pset_dict["latitude"] = latitude
    pset_dict["longitude"] = longitude
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates
    pset_dict["exposure_factor"] = exposure_pointing
    pset_dict["healpix"] = healpix
    pset_dict["energy_bin_delta"] = np.diff(intervals, axis=1).squeeze()

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
