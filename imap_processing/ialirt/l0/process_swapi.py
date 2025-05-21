"""Functions to support I-ALiRT SWAPI processing."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import erf
from xarray import DataArray

from imap_processing import imap_module_directory
from imap_processing.ialirt.constants import IalirtSwapiConstants as Consts
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import TIME_PER_BIN

logger = logging.getLogger(__name__)


def count_rate(
    energy_pass: float, speed: float, density: float, temp: float
) -> float | np.ndarray:
    """
    Compute SWAPI count rate for provided energy passband, speed, density and temp.

    This model for coincidence count rate was developed by the SWAPI instrument
    science team, detailed on page 52 of the IMAP SWAPI Instrument Algorithms Document.

    Parameters
    ----------
    energy_pass : float
        Energy passband [eV].
    speed : float
        Bulk solar wind speed [km/s].
    density : float
        Proton density [cm^-3].
    temp : float
        Temperature [K].

    Returns
    -------
    count_rate : float | np.ndarray
        Particle coincidence count rate.
    """
    # thermal velocity of solar wind ions
    thermal_velocity = np.sqrt(2 * Consts.boltz * temp / Consts.prot_mass)
    beta = 1 / (thermal_velocity**2)
    # convert energy to Joules
    center_speed = np.sqrt(2 * energy_pass * 1.60218e-19 / Consts.prot_mass)
    speed = speed * 1000  # convert km/s to m/s
    density = density * 1e6  # convert 1/cm**3 -to 1/m**3

    return (
        (density * Consts.eff_area * (beta / np.pi) ** (3 / 2))
        * (np.exp(-beta * (center_speed**2 + speed**2 - 2 * center_speed * speed)))
        * np.sqrt(np.pi / (beta * speed * center_speed))
        * erf(np.sqrt(beta * speed * center_speed) * (Consts.az_fov / 2))
        * (
            center_speed**4
            * Consts.speed_ew
            * np.arcsin(thermal_velocity / center_speed)
        )
    )


def optimize_pseudo_parameters(
    count_rates: np.ndarray, energy_passbands: Optional[np.ndarray] = None
) -> (dict)[str, list[float]]:
    """
    Find the pseudo speed (u), density (n) and temperature (T) of solar wind.

    Fit a curve to calculated count rate values as a function of energy passbands.

    Parameters
    ----------
    count_rates : np.ndarray
        Particle coincidence count rates.
    energy_passbands : np.ndarray, default None
        Energy passbands, passed in only for testing purposes.

    Returns
    -------
    solution_dict : dict
        Dictionary containing the optimized speed, density, and temperature values for
        each sweep included in the input count_rates array.
    """
    if not energy_passbands.any():
        # Read in energy passbands
        energy_data = pd.read_csv(
            f"{imap_module_directory}/tests/ialirt/test_data/ialirt_test_data.csv"
        )
        energy_passbands = energy_data["Energy [eV/q]"].to_numpy()

    # Initial guess pulled from page 52 of the IMAP SWAPI Instrument Algorithms Document
    initial_param_guess = np.array([550, 5.27, 1e5])
    solution_dict = {  # type: ignore
        "pseudo_speed": [],
        "pseudo_density": [],
        "pseudo_temperature": [],
    }

    for sweep in np.arange(count_rates.shape[0]):
        current_sweep_count_rates = count_rates[sweep, :]
        sol = curve_fit(
            count_rate,
            energy_passbands,
            current_sweep_count_rates,
            initial_param_guess,
        )
        solution_dict["pseudo_speed"].append(sol[0][0])
        solution_dict["pseudo_density"].append(sol[0][1])
        solution_dict["pseudo_temperature"].append(sol[0][2])

    return solution_dict


def process_swapi_ialirt(unpacked_data: xr.Dataset) -> dict[str, DataArray]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.

    Returns
    -------
    swapi_data : dict
        Dictionary containing all data variables for SWAPI I-ALiRT product.
    """
    logger.info("Processing SWAPI.")

    sci_dataset = unpacked_data.sortby("epoch", ascending=True)

    grouped_dataset = find_groups(sci_dataset, (0, 11), "swapi_seq_number", "swapi_acq")

    for group in np.unique(grouped_dataset["group"]):
        # Sequence values for the group should be 0-11 with no duplicates.
        seq_values = grouped_dataset["swapi_seq_number"][
            (grouped_dataset["group"] == group)
        ]

        # Ensure no duplicates and all values from 0 to 11 are present
        if not np.array_equal(seq_values.astype(int), np.arange(12)):
            logger.info(
                f"SWAPI group {group} does not contain all sequence values from 0 to "
                f"11 without duplicates."
            )
            continue

    total_packets = len(grouped_dataset["swapi_seq_number"].data)

    # It takes 12 sequence data to make one full SWAPI sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence

    met_values = grouped_dataset["swapi_shcoarse"].data.reshape(total_full_sweeps, 12)[
        :, 0
    ]

    raw_coin_count = process_sweep_data(grouped_dataset, "swapi_coin_cnt")
    raw_coin_rate = raw_coin_count / TIME_PER_BIN

    solution = optimize_pseudo_parameters(raw_coin_rate)

    swapi_data = {
        "met": met_values,
        "pseudo_speed": solution["pseudo_speed"],
        "pseudo_density": solution["pseudo_density"],
        "pseudo_temperature": solution["pseudo_temperature"],
    }

    return swapi_data
