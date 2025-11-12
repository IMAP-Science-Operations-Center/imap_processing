"""Functions to support I-ALiRT SWAPI processing."""

import logging
from decimal import Decimal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import erf

from imap_processing.ialirt.constants import IalirtSwapiConstants as Consts
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

NUM_IALIRT_ENERGY_STEPS = 63


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
    density = density * 1e6  # convert 1/cm**3 to 1/m**3

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
    count_rates: np.ndarray,
    count_rate_error: np.ndarray,
    energy_passbands: np.ndarray,
) -> (dict)[str, list[float]]:
    """
    Find the pseudo speed (u), density (n) and temperature (T) of solar wind particles.

    Fit a curve to calculated count rate values as a function of energy passband.

    Parameters
    ----------
    count_rates : np.ndarray
        Particle coincidence count rates.
    count_rate_error : np.ndarray
        Standard deviation of the coincidence count rates parameter.
    energy_passbands : np.ndarray, default None
        Energy values, taken from the SWAPI lookup table.

    Returns
    -------
    solution_dict : dict
        Dictionary containing the optimized speed, density, and temperature values for
        each sweep included in the input count_rates array.
    """
    solution_dict = {  # type: ignore
        "pseudo_speed": [],
        "pseudo_density": [],
        "pseudo_temperature": [],
    }

    for sweep in np.arange(count_rates.shape[0]):
        current_sweep_count_rates = count_rates[sweep, :]
        current_sweep_count_rate_errors = count_rate_error[sweep, :]
        # Find the max count rate, and use the 5 points surrounding it
        max_index = np.argmax(current_sweep_count_rates)
        initial_speed_guess = np.sqrt(energy_passbands[max_index]) * Consts.speed_coeff
        initial_param_guess = np.array(
            [
                initial_speed_guess,
                5 * (400 / initial_speed_guess) ** 2,
                60000 * (initial_speed_guess / 400) ** 2,
            ]
        )
        sol = curve_fit(
            f=count_rate,
            xdata=energy_passbands.take(
                range(max_index - 3, max_index + 3), mode="clip"
            ),
            ydata=current_sweep_count_rates.take(
                range(max_index - 3, max_index + 3), mode="clip"
            ),
            sigma=current_sweep_count_rate_errors.take(
                range(max_index - 3, max_index + 3), mode="clip"
            ),
            p0=initial_param_guess,
        )
        solution_dict["pseudo_speed"].append(sol[0][0])
        solution_dict["pseudo_density"].append(sol[0][1])
        solution_dict["pseudo_temperature"].append(sol[0][2])

    return solution_dict


def process_swapi_ialirt(
    unpacked_data: xr.Dataset, calibration_lut_table: pd.DataFrame
) -> list[dict]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.
    calibration_lut_table : pd.DataFrame
        DataFrame containing the contents of the SWAPI esa-unit-conversion lookup table.

    Returns
    -------
    swapi_data : dict
        Dictionary containing all data variables for SWAPI I-ALiRT product.
    """
    logger.info("Processing SWAPI.")

    sci_dataset = unpacked_data.sortby("epoch", ascending=True)

    met = calculate_time(
        sci_dataset["sc_sclk_sec"], sci_dataset["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    sci_dataset["met"] = met
    met_values = []
    incomplete_groups = []

    grouped_dataset = find_groups(sci_dataset, (0, 11), "swapi_seq_number", "met")

    if grouped_dataset.group.size == 0:
        logger.warning(
            "There was an issue with the SWAPI grouping process, returning empty data."
        )
        return []

    for group in np.unique(grouped_dataset["group"]):
        # Sequence values for the group should be 0-11 with no duplicates.
        seq_values = grouped_dataset["swapi_seq_number"][
            (grouped_dataset["group"] == group)
        ]

        met_values.append(
            int(grouped_dataset["met"][(grouped_dataset["group"] == group).values][0])
        )

        # Ensure no duplicates and all values from 0 to 11 are present
        if not np.array_equal(seq_values.astype(int), np.arange(12)):
            incomplete_groups.append(group)
            continue

    if incomplete_groups:
        logger.info(
            f"The following swapi groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )

    raw_coin_count = process_sweep_data(grouped_dataset, "swapi_coin_cnt")
    # Subset to only the relevant I-ALiRT energy steps
    raw_coin_count = raw_coin_count[:, :NUM_IALIRT_ENERGY_STEPS]
    raw_coin_rate = raw_coin_count / SWAPI_LIVETIME
    count_rate_error = np.sqrt(raw_coin_count) / SWAPI_LIVETIME

    # Extract energy values from the calibration lookup table file
    calibration_lut_table["timestamp"] = pd.to_datetime(
        calibration_lut_table["timestamp"], format="%m/%d/%Y %H:%M"
    )
    calibration_lut_table["timestamp"] = calibration_lut_table["timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    # Find the sweep's energy data for the latest time, where sweep_id == 2
    subset = calibration_lut_table[
        (calibration_lut_table["timestamp"] == calibration_lut_table["timestamp"].max())
        & (calibration_lut_table["Sweep #"] == 2)
    ]
    if subset.empty:
        energy_passbands = np.full(NUM_IALIRT_ENERGY_STEPS, np.nan, dtype=np.float64)
    else:
        subset = subset.sort_values(["timestamp", "ESA Step #"])
        energy_passbands = (
            subset["Energy"][:NUM_IALIRT_ENERGY_STEPS].to_numpy().astype(float)
        )

    solution = optimize_pseudo_parameters(
        raw_coin_rate, count_rate_error, energy_passbands
    )

    swapi_data = []

    for entry in np.arange(0, len(solution["pseudo_speed"])):
        swapi_data.append(
            {
                "apid": 478,
                "met": int(met_values[entry]),
                "met_in_utc": met_to_utc(met_values[entry]).split(".")[0],
                "ttj2000ns": int(met_to_ttj2000ns(met_values[entry])),
                "instrument": "swapi",
                "swapi_pseudo_proton_speed": Decimal(
                    f"{solution['pseudo_speed'][entry]:.3f}"
                ),
                "swapi_pseudo_proton_density": Decimal(
                    f"{solution['pseudo_density'][entry]:.3f}"
                ),
                "swapi_pseudo_proton_temperature": Decimal(
                    f"{solution['pseudo_temperature'][entry]:.3f}"
                ),
            }
        )

    return swapi_data
