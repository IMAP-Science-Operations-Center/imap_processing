"""Calculate Pointing Set Grids."""

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr

from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_helio_pset(
    de_dataset: xr.Dataset,
    extendedspin_dataset: xr.Dataset,
    cullingmask_dataset: xr.Dataset,
    name: str,
    ancillary_files: dict,
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
    ancillary_files : dict
        Ancillary files.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    # TODO: Fill in the rest of this later.
    pset_dict: dict[str, np.ndarray] = {}
    healpix = np.arange(hp.nside2npix(128))
    _, _, energy_bin_geometric_means = build_energy_bins()

    pset_dict["epoch"] = de_dataset.epoch.data[:1].astype(np.int64)
    pset_dict["pixel_index"] = healpix
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["exposure_factor"] = np.zeros(len(healpix), dtype=np.uint8)[
        np.newaxis, ...
    ]

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
