"""Calculate Extended Spin."""

import numpy as np
import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b_culling import (
    count_rejected_events_per_spin,
    flag_attitude,
    flag_hk,
    flag_imap_instruments,
    flag_rates,
    get_energy_histogram,
    get_pulses_per_spin,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

FILLVAL_UINT16 = 65535


def calculate_extendedspin(
    dict_datasets: dict[str, xr.Dataset],
    name: str,
    instrument_id: int,
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    dict_datasets : dict
        Dictionary containing all the datasets.
    name : str
        Name of the dataset.
    instrument_id : int
        Instrument ID.

    Returns
    -------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    """
    aux_dataset = dict_datasets[f"imap_ultra_l1a_{instrument_id}sensor-aux"]
    rates_dataset = dict_datasets[f"imap_ultra_l1a_{instrument_id}sensor-rates"]
    de_dataset = dict_datasets[f"imap_ultra_l1b_{instrument_id}sensor-de"]

    extendedspin_dict = {}
    rates_qf, spin, energy_midpoints, n_sigma_per_energy = flag_rates(
        de_dataset["spin"].values,
        de_dataset["energy"].values,
    )
    count_rates, _, counts, _ = get_energy_histogram(
        de_dataset["spin"].values, de_dataset["energy"].values
    )
    attitude_qf, spin_rates, spin_period, spin_starttime = flag_attitude(
        de_dataset["spin"].values, aux_dataset
    )
    # TODO: We will add to this later
    hk_qf = flag_hk(de_dataset["spin"].values)
    inst_qf = flag_imap_instruments(de_dataset["spin"].values)

    # Get the first epoch for each spin.
    mask = xr.DataArray(np.isin(de_dataset["spin"], spin), dims="epoch")
    filtered_dataset = de_dataset.where(mask, drop=True)
    _, first_indices = np.unique(filtered_dataset["spin"].values, return_index=True)
    first_epochs = filtered_dataset["epoch"].values[first_indices]

    # Get the number of pulses per spin.
    pulses = get_pulses_per_spin(rates_dataset)

    # Track rejected events in each spin based on
    # quality flags in de l1b data.
    rejected_counts = count_rejected_events_per_spin(
        de_dataset["spin"].values,
        de_dataset["quality_scattering"].values,
        de_dataset["quality_outliers"].values,
    )
    # These will be the coordinates.
    extendedspin_dict["epoch"] = first_epochs
    extendedspin_dict["spin_number"] = spin
    extendedspin_dict["energy_bin_geometric_mean"] = energy_midpoints

    extendedspin_dict["ena_rates"] = count_rates
    extendedspin_dict["ena_rates_threshold"] = n_sigma_per_energy
    extendedspin_dict["spin_start_time"] = spin_starttime
    extendedspin_dict["spin_period"] = spin_period
    extendedspin_dict["spin_rate"] = spin_rates
    extendedspin_dict["start_pulses_per_spin"] = pulses.start_per_spin
    extendedspin_dict["stop_pulses_per_spin"] = pulses.stop_per_spin
    extendedspin_dict["coin_pulses_per_spin"] = pulses.coin_per_spin
    extendedspin_dict["rejected_events_per_spin"] = rejected_counts
    extendedspin_dict["quality_attitude"] = attitude_qf
    extendedspin_dict["quality_ena_rates"] = rates_qf
    extendedspin_dict["quality_hk"] = hk_qf
    extendedspin_dict["quality_instruments"] = inst_qf

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b")

    return extendedspin_dataset
