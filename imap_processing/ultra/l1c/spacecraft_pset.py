"""Calculate Pointing Set Grids."""

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.ultra.l1b.lookup_utils import (
    get_nominal_for_by_spin_phase,
    get_scattering_coefficients,
    mask_below_fwhm_scattering_threshold,
)
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_efficiencies_and_geometric_function,
    get_spacecraft_background_rates,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_pixels_within_scattering_threshold(
    for_indices_by_spin_phase: np.ndarray,
    theta_and_phi: np.ndarray,
    ancillary_files: dict,
    instrument_id: int,
) -> list:
    """
    Calculate pixels within the FWHM scattering threshold for each spin phase step.

    Parameters
    ----------
    for_indices_by_spin_phase : np.ndarray
        A 2D boolean array where cols are spin phase steps are rows are HEALPix pixels.
        True indicates pixels that are within the Field of Regard (FOR) at that
        spin phase.
    theta_and_phi : np.ndarray
        A 2D array where the first column is theta values and the second column is
        phi values for each HEALPix pixel.
    ancillary_files : dict
        Dictionary containing ancillary files.
    instrument_id : int,
        Instrument ID, either 45 or 90.

    Returns
    -------
    exposure_pointing_adjusted : list
        A Nested list of arrays indicating pixels within the scattering threshold.
        The outer list indicates spin phase steps, the middle list indicates energy
        bins, and the inner arrays contain indices indicating pixels that are below
        the FWHM scattering threshold.
    """
    pixels_below_scattering = []
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    steps = for_indices_by_spin_phase.shape[1]
    # Using the lookup table, get the indices of the pixels inside the FOR at the
    # current spin phase step.
    theta = theta_and_phi[:, 0]
    phi = theta_and_phi[:, 1]
    theta_coeffs, phi_coeffs = get_scattering_coefficients(
        ancillary_files, instrument_id, theta, phi
    )
    # The "for_indices_by_spin_phase" lookup table contains the boolean values of each
    # pixel at each spin phase step, indicating whether the pixel is inside the FOR.
    # It starts at Spin-phase = 0, and increments in fine steps (1 ms), spinning the
    # spacecraft in the despun frame. At each iteration, query for the pixels in the
    # FOR, and calculate whether the FWHM value is below the threshold at the energy.
    for i in range(steps):
        # Calculate spin phase for the current iteration
        for_inds = for_indices_by_spin_phase[:, i]
        pixels_below_scattering_for_energy = []
        for energy_idx in range(len(energy_bin_geometric_means)):
            # Get a mask for pixels below the FWHM scattering threshold
            energy = int(energy_bin_geometric_means[energy_idx])
            scattering_mask = mask_below_fwhm_scattering_threshold(
                theta_coeffs[for_inds], phi_coeffs[for_inds], energy
            )
            pixels_below_scattering_for_energy.append(
                np.where(for_inds)[0][scattering_mask]
            )
        pixels_below_scattering.append(pixels_below_scattering_for_energy)

    return pixels_below_scattering


def calculate_spacecraft_pset(
    de_dataset: xr.Dataset,
    extendedspin_dataset: xr.Dataset,
    cullingmask_dataset: xr.Dataset,
    rates_dataset: xr.Dataset,
    params_dataset: xr.Dataset,
    name: str,
    ancillary_files: dict,
    instrument_id: int,
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
    rates_dataset : xarray.Dataset
        Dataset containing image rates data.
    params_dataset : xarray.Dataset
        Dataset containing image parameters data.
    name : str
        Name of the dataset.
    ancillary_files : dict
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict: dict[str, np.ndarray] = {}
    sensor = parse_filename_like(name)["sensor"][0:2]

    v_mag_dps_spacecraft = np.linalg.norm(de_dataset["velocity_dps_sc"].values, axis=1)
    vhat_dps_spacecraft = (
        de_dataset["velocity_dps_sc"].values / v_mag_dps_spacecraft[:, np.newaxis]
    )

    intervals, _, energy_bin_geometric_means = build_energy_bins()
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_spacecraft,
        de_dataset["energy_spacecraft"].values,
        intervals,
        nside=128,
    )
    healpix = np.arange(n_pix)

    # Get lookup table for FOR indices by spin phase step
    for_indices_by_spin_phase, theta_and_phi, ra_and_dec = (
        get_nominal_for_by_spin_phase(ancillary_files, instrument_id)
    )
    pixels_below_scattering = calculate_pixels_within_scattering_threshold(
        for_indices_by_spin_phase, theta_and_phi, ancillary_files, instrument_id
    )
    # calculate efficiency and geometric function as a function of energy
    efficiencies, geometric_function = get_efficiencies_and_geometric_function(
        pixels_below_scattering, theta_and_phi, ancillary_files
    )
    # TODO handle sensitivity
    # sensitivity = interpolate_sensitivity(efficiencies, geometric_function)

    # Calculate exposure
    constant_exposure = ancillary_files["l1c-90sensor-dps-exposure"]
    df_exposure = pd.read_csv(constant_exposure)

    exposure_pointing = get_spacecraft_exposure_times(
        df_exposure, rates_dataset, params_dataset, pixels_below_scattering
    )

    # Calculate background rates
    background_rates = get_spacecraft_background_rates(
        rates_dataset,
        sensor,
        ancillary_files,
        intervals,
        cullingmask_dataset["spin_number"].values,
    )

    # For ISTP, epoch should be the center of the time bin.
    pset_dict["epoch"] = de_dataset.epoch.data[:1].astype(np.int64)
    pset_dict["counts"] = counts[np.newaxis, ...]
    pset_dict["latitude"] = latitude[np.newaxis, ...]
    pset_dict["longitude"] = longitude[np.newaxis, ...]
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates[np.newaxis, ...]
    pset_dict["exposure_factor"] = exposure_pointing[np.newaxis, ...]
    pset_dict["pixel_index"] = healpix
    pset_dict["energy_bin_delta"] = np.diff(intervals, axis=1).squeeze()[
        np.newaxis, ...
    ]
    # pset_dict["sensitivity"] = sensitivity[np.newaxis, ...]

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
