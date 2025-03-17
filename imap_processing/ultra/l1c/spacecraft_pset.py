"""Calculate Pointing Set Grids."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins, get_spacecraft_histogram, get_background_rates
from imap_processing.spice.time import ttj2000ns_to_et


def calculate_spacecraft_pset(
    de_dataset: xr.Dataset, name: str, data_version: str
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
    data_version : str
        Version of the data.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    v_mag_dps_spacecraft = np.linalg.norm(de_dataset["velocity_dps_sc"].values, axis=1)
    vhat_dps_spacecraft = de_dataset["velocity_dps_sc"].values / v_mag_dps_spacecraft[:, np.newaxis]

    intervals, _, energy_bin_geometric_means = build_energy_bins()
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_spacecraft,
        de_dataset["energy_spacecraft"].values,
        intervals,
        nside=128,
    )
    healpix = np.arange(n_pix)

    background_rates = get_background_rates()
    # TODO: exposure and sensitivity go here.

    # For ISTP, epoch should be the center of the time bin.
    pset_dict["epoch"] = ttj2000ns_to_et(np.mean(de_dataset.epoch.data[[0, -1]]).astype(
        np.int64
    ))
    pset_dict["counts"] = counts
    pset_dict["latitude_bin_center"] = latitude
    pset_dict["longitude_bin_center"] = longitude
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates
    pset_dict["healpix"] = healpix

    dataset = create_dataset(pset_dict, name, "l1c", data_version)

    return dataset
