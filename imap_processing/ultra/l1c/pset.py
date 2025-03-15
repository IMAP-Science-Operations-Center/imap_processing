"""Calculate Pointing Set Grids."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins, get_spacecraft_histogram, get_background_rates


def calculate_pset(
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

    # TODO: What to do here?
    epoch = de_dataset.coords["epoch"].values

    v_mag_dps_spacecraft = np.linalg.norm(de_dataset["velocity_dps_sc"].values, axis=1)
    vhat_dps_spacecraft = de_dataset["velocity_dps_sc"].values / v_mag_dps_spacecraft[:, np.newaxis]

    intervals, energy_midpoints = build_energy_bins()
    counts, latitude, longitude, healpix_number = get_spacecraft_histogram(
        vhat_dps_spacecraft,
        de_dataset["tof_energy"].values,
        intervals,
        nside=128,
    )

    background_rates = get_background_rates()

    dataset = create_dataset(pset_dict, name, "l1c", data_version)

    return dataset
