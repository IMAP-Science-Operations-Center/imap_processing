"""Calculate Extended Spin."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapRatesUltraFlags
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    count_rejected_events_per_spin,
    expand_bin_flags_to_spins,
    flag_attitude,
    flag_high_energy,
    flag_hk,
    flag_imap_instruments,
    flag_low_voltage,
    flag_rates,
    get_binned_energy_ranges,
    get_binned_spins_edges,
    get_energy_histogram,
    get_energy_range_flags,
    get_pulses_per_spin,
)
from imap_processing.ultra.l1c.l1c_lookup_utils import build_energy_bins
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

FILLVAL_UINT16 = 65535
FILLVAL_FLOAT32 = -1.0e31


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
    status_dataset = dict_datasets[f"imap_ultra_l1b_{instrument_id}sensor-status"]

    extendedspin_dict = {}
    rates_qf, spin, energy_bin_geometric_mean, n_sigma_per_energy = flag_rates(
        de_dataset["spin"].values,
        de_dataset["energy"].values,
    )
    count_rates, _, _counts, _ = get_energy_histogram(
        de_dataset["spin"].values, de_dataset["energy"].values
    )
    attitude_qf, spin_rates, spin_period, spin_starttime = flag_attitude(
        de_dataset["spin"].values, aux_dataset
    )
    # TODO: We will add to this later
    hk_qf = flag_hk(de_dataset["spin"].values)
    inst_qf = flag_imap_instruments(de_dataset["spin"].values)

    spin_bin_size = UltraConstants.SPIN_BIN_SIZE
    spin_tbin_edges = get_binned_spins_edges(
        spin, spin_period, spin_starttime, spin_bin_size
    )
    voltage_qf = flag_low_voltage(spin_tbin_edges, status_dataset)
    voltage_qf = expand_bin_flags_to_spins(len(spin), voltage_qf, spin_bin_size)
    # Get energy bins used at l1c
    intervals, _, _ = build_energy_bins()
    # Get the energy ranges
    energy_ranges = get_binned_energy_ranges(intervals)
    energy_bin_flags = get_energy_range_flags(energy_ranges)
    # Calculate the high energy quality flags using the de dataset with low voltage
    # events removed. Use the same spin and energy bins that
    # were used for low voltage flags to maintain consistency in the flags.
    valid_voltage_spins = spin[np.where(voltage_qf == 0)]
    valid_de_spins = np.isin(de_dataset["spin"].values, valid_voltage_spins)
    de_dataset_filtered = de_dataset.isel(epoch=valid_de_spins)
    energy_thresholds = UltraConstants.HIGH_ENERGY_CULL_THRESHOLDS
    high_energy_qf = flag_high_energy(
        de_dataset_filtered,
        spin_tbin_edges,
        energy_ranges,
        energy_bin_flags,
        energy_thresholds,
        instrument_id,
    )
    # Get the number of pulses per spin.
    pulses = get_pulses_per_spin(aux_dataset, rates_dataset)

    # Track rejected events in each spin based on
    # quality flags in de l1b data.
    rejected_counts = count_rejected_events_per_spin(
        de_dataset["spin"].values,
        de_dataset["quality_scattering"].values,
        de_dataset["quality_outliers"].values,
    )

    # These will be the coordinates.
    extendedspin_dict["spin_number"] = spin
    extendedspin_dict["energy_bin_geometric_mean"] = energy_bin_geometric_mean

    extendedspin_dict["ena_rates"] = count_rates
    extendedspin_dict["ena_rates_threshold"] = n_sigma_per_energy
    extendedspin_dict["spin_start_time"] = spin_starttime
    extendedspin_dict["spin_period"] = spin_period
    extendedspin_dict["spin_rate"] = spin_rates

    # Get index of pulses.unique_spins corresponding to each spin.
    idx: NDArray[np.intp] = np.searchsorted(pulses.unique_spins, spin)

    # Validate that the spin values match
    valid = (idx < pulses.unique_spins.size) & (pulses.unique_spins[idx] == spin)

    start_per_spin = np.full(len(spin), FILLVAL_FLOAT32, dtype=np.float32)
    stop_per_spin = np.full(len(spin), FILLVAL_FLOAT32, dtype=np.float32)
    coin_per_spin = np.full(len(spin), FILLVAL_FLOAT32, dtype=np.float32)

    # Fill only the valid ones
    start_per_spin[valid] = pulses.start_per_spin[idx[valid]]
    stop_per_spin[valid] = pulses.stop_per_spin[idx[valid]]
    coin_per_spin[valid] = pulses.coin_per_spin[idx[valid]]

    # Expand binned quality flags to individual spins.
    high_energy_qf = expand_bin_flags_to_spins(len(spin), high_energy_qf, spin_bin_size)
    # account for rates spins which are not in the direct event spins
    extendedspin_dict["start_pulses_per_spin"] = start_per_spin
    extendedspin_dict["stop_pulses_per_spin"] = stop_per_spin
    extendedspin_dict["coin_pulses_per_spin"] = coin_per_spin
    extendedspin_dict["rejected_events_per_spin"] = rejected_counts
    extendedspin_dict["quality_attitude"] = attitude_qf
    extendedspin_dict["quality_ena_rates"] = rates_qf
    extendedspin_dict["quality_hk"] = hk_qf
    extendedspin_dict["quality_instruments"] = inst_qf
    extendedspin_dict["quality_low_voltage"] = voltage_qf  # shape (nspin,)
    # TODO calculate flags for high energy (SEPS) and statistics culling
    # Initialize these flags to NONE for now.
    extendedspin_dict["quality_statistics"] = np.full_like(
        voltage_qf, ImapRatesUltraFlags.NONE.value, np.uint16
    )  # shape (nspin,)
    extendedspin_dict["quality_high_energy"] = high_energy_qf  # shape (nspin,)
    # Add an array of flags for each energy bin. Shape: (n_energy_bins)
    extendedspin_dict["energy_range_flags"] = energy_bin_flags
    # Add energy ranges  Shape: (n_energy_bins + 1)
    extendedspin_dict["energy_range_edges"] = np.array(energy_ranges)

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b")

    return extendedspin_dataset
