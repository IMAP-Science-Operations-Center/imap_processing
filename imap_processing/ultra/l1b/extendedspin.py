"""Calculate Extended Spin."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_extendedspin(rates_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    rates_dataset : xarray.Dataset
        Dataset containing rates data.
    name : str
        Name of the dataset.

    Returns
    -------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    """
    extendedspin_dict = {}

    epoch = rates_dataset.coords["epoch"].values

    # Placeholder for calculations
    extendedspin_dict["epoch"] = epoch
    extendedspin_dict["spin_number"] = np.zeros(len(epoch), dtype=np.uint64)
    extendedspin_dict["spin_start_time"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["avg_spin_period"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_start_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_stop_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_coin_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_processed_events"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_rejected_events"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["quality_hk"] = np.zeros(len(epoch), dtype=np.uint16)
    extendedspin_dict["quality_attitude"] = np.zeros(len(epoch), dtype=np.uint16)
    extendedspin_dict["quality_instruments"] = np.zeros(len(epoch), dtype=np.uint16)

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b")

    return extendedspin_dataset
