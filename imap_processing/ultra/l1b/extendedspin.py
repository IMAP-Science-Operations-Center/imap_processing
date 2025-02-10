"""Calculate Extended Spin."""

import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b_culling import (
    flag_attitude,
    flag_spin,
    get_energy_histogram,
    get_spin,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_extendedspin(
    hk_dataset: xr.Dataset,
    rates_dataset: xr.Dataset,
    de_dataset: xr.Dataset,
    name: str,
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    hk_dataset : xarray.Dataset
        Dataset containing l1a hk data.
    rates_dataset : xarray.Dataset
        Dataset containing l1a rates data.
    de_dataset : xarray.Dataset
        Dataset containing l1b de data.
    name : str
        Name of the dataset.
    data_version : str
        Version of the data.

    Returns
    -------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    """
    extendedspin_dict = {}
    rates_qf, spin, energy_midpoints, n_sigma_per_energy = flag_spin(
        de_dataset["event_times"].values,
        de_dataset["energy"].values,
    )
    spin_number = get_spin(de_dataset["event_times"].values)
    count_rates, _, counts, _ = get_energy_histogram(
        spin_number, de_dataset["energy"].values
    )
    attitude_qf, spin_rates, spin_period, spin_starttime = flag_attitude(
        de_dataset["event_times"].values
    )

    # These will be the coordinates.
    extendedspin_dict["spin_number"] = spin
    extendedspin_dict["energy_bin_geometric_mean"] = energy_midpoints

    extendedspin_dict["ena_rates"] = count_rates
    extendedspin_dict["ena_rates_threshold"] = n_sigma_per_energy
    extendedspin_dict["spin_start_time"] = spin_starttime
    extendedspin_dict["spin_period"] = spin_period
    extendedspin_dict["spin_rate"] = spin_rates
    extendedspin_dict["quality_attitude"] = attitude_qf
    extendedspin_dict["quality_ena_rates"] = rates_qf

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b", data_version)

    return extendedspin_dataset
