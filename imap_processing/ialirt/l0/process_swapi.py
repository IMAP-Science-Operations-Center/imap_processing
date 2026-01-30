"""Functions to support I-ALiRT SWAPI processing."""

import logging
from decimal import Decimal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import erf

from imap_processing.ialirt.constants import IalirtSwapiConstants as Consts
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

NUM_IALIRT_ENERGY_STEPS = 63
FILLVAL_FLOAT32 = -1.0e31


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

    # see comment on Consts.temporary_density_factor
    density = density * Consts.temporary_density_factor

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
) -> np.ndarray:
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
    pseudo_params : np.ndarray
        Pseudo speed, pseudo density, pseudo temperature.
    """
    # Find the max count rate, and use the 5 points surrounding it
    max_index = np.argmax(count_rates)
    initial_speed_guess = np.sqrt(energy_passbands[max_index]) * Consts.speed_coeff
    initial_param_guess = np.array(
        [
            initial_speed_guess,
            5 * (400 / initial_speed_guess) ** 2,
            60000 * (initial_speed_guess / 400) ** 2,
        ]
    )

    sol = None

    try:
        five_point_range = range(max_index - 2, max_index + 2 + 1)
        xdata = energy_passbands.take(five_point_range, mode="clip")
        ydata = count_rates.take(five_point_range, mode="clip")
        sigma = count_rate_error.take(five_point_range, mode="clip")
        curve_fit_output = curve_fit(
            f=count_rate,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            p0=initial_param_guess,
        )

        # If covariance matrix is not finite, scipy failed to converge to a
        # solution and could just be reporting the initial guess
        covariance_matrix_is_finite = np.all(np.isfinite(curve_fit_output[1]))

        # fit has failed if R^2 < 0.7
        yfit = count_rate(xdata, *curve_fit_output[0])
        r2 = 1 - np.sum((ydata - yfit) ** 2) / np.sum((ydata - ydata.mean()) ** 2)
        r2_is_acceptable = r2 >= 0.7

        if covariance_matrix_is_finite and r2_is_acceptable:
            sol = curve_fit_output[0]
    except RuntimeError:
        logger.error("curve_fit failed")
        sol = None

    # report speed only if fit fails
    if sol is None:
        sol = initial_param_guess.copy()
        sol[1:] = FILLVAL_FLOAT32

    return sol


def geometric_mean(
    swapi_met_list: list,
    pseudo_speed_list: list,
    pseudo_proton_density_list: list,
    pseudo_proton_temperature_list: list,
) -> tuple:
    """
    Find moving geometric mean of SWAPI data.

    Parameters
    ----------
    swapi_met_list : list
        Mission elapsed time for first time measurement of SWAPI sweep.
    pseudo_speed_list : list
        Pseudo speed for SWAPI sweep.
    pseudo_proton_density_list : list
        Pseudo proton density for SWAPI sweep.
    pseudo_proton_temperature_list : list
        Pseudo proton temperature for SWAPI sweep.

    Returns
    -------
    avg_swapi_met : float
        Average swapi met value.
    avg_proton_density : float
        Average proton density value.
    avg_pseudo_speed : float
        Average pseudo speed value.
    avg_proton_temperature : float
        Average proton temperature value.
    """
    met_arr = np.asarray(swapi_met_list)

    # If any of the values are equal to nan then do not include that index.
    valid = (
        ~np.isnan(pseudo_speed_list)
        & ~np.isnan(pseudo_proton_density_list)
        & ~np.isnan(pseudo_proton_temperature_list)
    )

    pseudo_speed_arr = np.asarray(pseudo_speed_list)[valid]
    avg_pseudo_speed = np.exp(np.mean(np.log(pseudo_speed_arr)))

    density_arr = np.asarray(pseudo_proton_density_list)[valid]
    avg_proton_density = np.exp(np.mean(np.log(density_arr)))

    temperature_arr = np.asarray(pseudo_proton_temperature_list)[valid]
    avg_proton_temperature = np.exp(np.mean(np.log(temperature_arr)))

    avg_swapi_met = np.mean(met_arr[valid])

    return avg_swapi_met, avg_proton_density, avg_pseudo_speed, avg_proton_temperature


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
    incomplete_groups = []
    swapi_data = []
    pseudo_proton_speed_list = []
    pseudo_proton_density_list = []
    pseudo_proton_temperature_list = []
    swapi_met_list = []

    # Extract energy values from the calibration lookup table file
    calibration_lut_table["timestamp"] = pd.to_datetime(
        calibration_lut_table["timestamp"]
    )

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
        met = grouped_dataset["met"][(grouped_dataset["group"] == group).values]

        swapi_met = grouped_dataset["swapi_acq"][
            (grouped_dataset["group"] == group).values
        ]
        mid_measurement = int((swapi_met[0] + swapi_met[-1]) // 2)

        # Ensure no duplicates and all values from 0 to 11 are present
        if not np.array_equal(seq_values.values.astype(int), np.arange(12)):
            incomplete_groups.append(group)
            continue

        grouped_subset = grouped_dataset.sel(epoch=grouped_dataset.group == group)

        raw_coin_count = process_sweep_data(grouped_subset, "swapi_coin_cnt")
        # I-ALiRT packets have counts compressed by a factor of 16.
        # Add 8 to avoid having counts truncated to 0 and to avoid
        # counts being systematically too low
        raw_coin_count = raw_coin_count * 16 + 8
        # Subset to only the relevant I-ALiRT energy steps
        raw_coin_count = raw_coin_count[:, :NUM_IALIRT_ENERGY_STEPS]
        raw_coin_rate = raw_coin_count / SWAPI_LIVETIME
        count_rate_error = np.sqrt(raw_coin_count) / SWAPI_LIVETIME

        sweep_id = int(grouped_subset["swapi_version"].values[0])
        subset_sweep = calibration_lut_table[
            calibration_lut_table["Sweep #"] == sweep_id
        ]

        # Find the sweep's energy data for the latest time
        subset = subset_sweep[
            (subset_sweep["timestamp"] == subset_sweep["timestamp"].max())
        ]
        if subset.empty:
            raise ValueError(
                f"No esa unit conversion available for sweep {sweep_id}. "
                f"Check lookup table?"
            )
        else:
            subset = subset.sort_values(["timestamp", "ESA Step #"])
            energy_passbands = (
                subset["Energy"][:NUM_IALIRT_ENERGY_STEPS].to_numpy().astype(float)
            )

        pseudo_speed, pseudo_density, pseudo_temperature = optimize_pseudo_parameters(
            raw_coin_rate.squeeze(), count_rate_error.squeeze(), energy_passbands
        )

        pseudo_proton_speed_list.append(pseudo_speed)
        pseudo_proton_density_list.append(pseudo_density)
        pseudo_proton_temperature_list.append(pseudo_temperature)
        swapi_met_list.append(mid_measurement)

        # Begin averaging after 1 minute has passed (5 sweeps) and make certain that
        # the data is sequential (~12 s cadence).
        if len(swapi_met_list) >= 5 and np.all(
            np.isclose(np.diff(swapi_met_list[-5:]), 12.0, atol=0.05)
        ):
            (
                avg_swapi_met,
                avg_pseudo_proton_density,
                avg_pseudo_proton_speed,
                avg_pseudo_proton_temperature,
            ) = geometric_mean(
                swapi_met_list[-5:],
                pseudo_proton_speed_list[-5:],
                pseudo_proton_density_list[-5:],
                pseudo_proton_temperature_list[-5:],
            )

            # replace nans (resulting from geometric means that
            # include fill values) with fill values
            (
                avg_pseudo_proton_speed,
                avg_pseudo_proton_density,
                avg_pseudo_proton_temperature,
            ) = np.nan_to_num(
                (
                    avg_pseudo_proton_speed,
                    avg_pseudo_proton_density,
                    avg_pseudo_proton_temperature,
                ),
                nan=FILLVAL_FLOAT32,
            )

            swapi_data.append(
                _populate_instrument_header_items(met)
                | {
                    "instrument": "swapi",
                    "swapi_epoch": int(met_to_ttj2000ns(avg_swapi_met)),
                    "swapi_pseudo_proton_speed": Decimal(
                        f"{avg_pseudo_proton_speed:.3f}"
                    ),
                    "swapi_pseudo_proton_density": Decimal(
                        f"{avg_pseudo_proton_density:.3f}"
                    ),
                    "swapi_pseudo_proton_temperature": Decimal(
                        f"{avg_pseudo_proton_temperature:.3f}"
                    ),
                }
            )
    if incomplete_groups:
        logger.info(
            f"The following swapi groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )

    return swapi_data
