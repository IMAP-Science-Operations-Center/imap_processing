"""Contains tools for lookup tables for l1c."""

import logging

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_scattering_coefficients,
    load_scattering_lookup_tables,
)
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins

logger = logging.getLogger(__name__)


def mask_below_fwhm_scattering_threshold(
    theta_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    energy: int,
) -> np.ndarray:
    """
    Determine indices of theta and phi values below the FWHM scattering threshold.

    For each phi and theta, calculate the FWHM using the formula:
    FWHM = A*E^g
    If Phi FWHM or Theta FWHM > the scattering requirements from the table above,
    mask the instrument frame pixel.

    Parameters
    ----------
    theta_coeffs : NDArray
        Coefficients for theta FWHM calculation (a and g) for each pixel.
    phi_coeffs : NDArray
        Coefficients for phi FWHM calculation (a and g) for each pixel.
    energy : int
        Energy in keV.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating indices below the scattering threshold.
    """
    scattering_thresholds = UltraConstants.ULTRA_FWHM_SCATTERING_CULLING_THRESHOLDS
    # Calculate FWHM for theta and phi
    fwhm_theta = theta_coeffs[..., 0] * energy ** theta_coeffs[..., 1]
    fwhm_phi = phi_coeffs[..., 0] * energy ** phi_coeffs[..., 1]

    try:
        # Get the scattering threshold based on the energy
        threshold = next(
            threshold
            for energy_range, threshold in scattering_thresholds.items()
            if energy_range[0] <= energy < energy_range[1]
        )
    except StopIteration:
        logger.warning(
            f"Energy {energy} keV is out of bounds for scattering thresholds. Using "
            f"zero for as threshold."
        )
        threshold = 0
    # Combine conditions for both theta and phi
    return np.logical_and(fwhm_theta <= threshold, fwhm_phi <= threshold)


def calculate_pixels_within_scattering_threshold(
    for_indices_by_spin_phase: np.ndarray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
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
    theta_vals : np.ndarray
        A 2D array of theta values for each HEALPix pixel at each spin phase step.
    phi_vals : np.ndarray
         A 2D array of phi values for each HEALPix pixel at each spin phase step.
    ancillary_files : dict
        Dictionary containing ancillary files.
    instrument_id : int,
        Instrument ID, either 45 or 90.

    Returns
    -------
    pixels_below_scattering : list
        A Nested list of arrays indicating pixels within the scattering threshold.
        The outer list indicates spin phase steps, the middle list indicates energy
        bins, and the inner arrays contain indices indicating pixels that are below
        the FWHM scattering threshold.
    """
    # Load scattering coefficient lookup table
    scattering_luts = load_scattering_lookup_tables(ancillary_files, instrument_id)
    pixels_below_scattering = []
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    steps = for_indices_by_spin_phase.shape[1]
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
            # Using the lookup table, get the indices of the pixels inside the FOR at
            # the current spin phase step.
            theta = theta_vals[for_inds, i]
            phi = phi_vals[for_inds, i]
            theta_coeffs, phi_coeffs = get_scattering_coefficients(
                theta, phi, lookup_tables=scattering_luts
            )
            scattering_mask = mask_below_fwhm_scattering_threshold(
                theta_coeffs, phi_coeffs, energy
            )
            pixels_below_scattering_for_energy.append(
                np.where(for_inds)[0][scattering_mask]
            )
        pixels_below_scattering.append(pixels_below_scattering_for_energy)

    return pixels_below_scattering


def get_spacecraft_pointing_lookup_tables(
    ancillary_files: dict, instrument_id: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Get indices of pixels in the nominal FOR as a function of spin phase.

    This function also returns the theta / phi values in the instrument frame per spin
    phase, right ascension / declination values in the SC frame, and boundary scale
    factors for each pixel at each spin phase.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.

    Returns
    -------
    for_indices_by_spin_phase : NDArray
        A 2D boolean array of shape (npix, n_spin_phase_steps).
        True indicates pixels that are within the Field of Regard (FOR) at that
        spin phase.
    theta_vals : NDArray
        A 2D array of theta values for each HEALPix pixel at each spin phase step.
    phi_vals : NDArray
         A 2D array of phi values for each HEALPix pixel at each spin phase step.
    ra_and_dec : NDArray
        A 2D array of right ascension and declination values for each HEALPix pixel.
    boundary_scale_factors : NDArray
        A 2D array of boundary scale factors for each HEALPix pixel at each spin phase
        step.
    """
    theta_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-theta-n32"
    phi_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-phi-n32"
    index_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-index-n32"
    bsf_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-bsf-n32"

    theta_vals = pd.read_csv(
        ancillary_files[theta_descriptor], header=None, skiprows=1
    ).to_numpy(dtype=float)[:, 2:]
    phi_vals = pd.read_csv(
        ancillary_files[phi_descriptor], header=None, skiprows=1
    ).to_numpy(dtype=float)[:, 2:]
    index_grid = pd.read_csv(
        ancillary_files[index_descriptor], header=None, skiprows=1
    ).to_numpy(dtype=float)
    boundary_scale_factors = pd.read_csv(
        ancillary_files[bsf_descriptor], header=None, skiprows=1
    ).to_numpy(dtype=float)[:, 2:]

    ra_and_dec = index_grid[:, :2]  # Shape (npix, 2)
    # This array indicates whether each pixel is in the nominal FOR at each spin phase
    # step (15000 steps for a full rotation with 1 ms resolution).
    for_indices_by_spin_phase = np.nan_to_num(index_grid[:, 2:], nan=0).astype(
        bool
    )  # Shape (npix, 15000)
    return (
        for_indices_by_spin_phase,
        theta_vals,
        phi_vals,
        ra_and_dec,
        boundary_scale_factors,
    )
