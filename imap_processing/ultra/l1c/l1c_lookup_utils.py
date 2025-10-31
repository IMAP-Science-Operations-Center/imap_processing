"""Contains tools for lookup tables for l1c."""

import logging

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_scattering_coefficients,
    get_scattering_thresholds,
    load_scattering_lookup_tables,
)

logger = logging.getLogger(__name__)


def mask_below_fwhm_scattering_threshold(
    theta_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    energy: np.ndarray,
    scattering_thresholds: np.ndarray,
) -> tuple[NDArray, NDArray, NDArray]:
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
    energy : NDArray
        Energy corresponding to each theta and phi val in keV.
    scattering_thresholds : dict
        Scattering thresholds corresponding to each energy.

    Returns
    -------
    scattering_mask : numpy.ndarray
        Boolean array indicating indices below the scattering threshold.
    fwhm_theta : numpy.ndarray
        Calculated FWHM values for theta.
    fwhm_phi : numpy.ndarray
        Calculated FWHM values for phi.
    """
    # Calculate FWHM for all pixels and all energies
    fwhm_theta = theta_coeffs[..., 0:1] * (
        energy ** theta_coeffs[..., 1:2]
    )  # (npix, energy.shape[1])
    fwhm_phi = phi_coeffs[..., 0:1] * (
        energy ** phi_coeffs[..., 1:2]
    )  # (npix, energy.shape[1])

    thresholds = scattering_thresholds[np.newaxis, :]  # (1, energy.shape[1])

    # Combine conditions for both theta and phi.
    # shape = (npix, energy.shape[1])
    scattering_mask = np.logical_and(fwhm_theta <= thresholds, fwhm_phi <= thresholds)
    return scattering_mask, fwhm_theta, fwhm_phi


def calculate_fwhm_spun_scattering(
    for_indices_by_spin_phase: np.ndarray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    ancillary_files: dict,
    instrument_id: int,
) -> tuple[list, NDArray, NDArray, NDArray]:
    """
    Calculate FWHM scattering values for each pixel, energy bin, and spin phase step.

    This function also calculates a mask for pixels that are below the FWHM threshold.

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
    scattering_fwhm_theta : NDArray
        Calculated FWHM scatting values for theta at each energy bin and averaged
        over spin phase.
    scattering_fwhm_phi : NDArray
        Calculated FWHM scatting values for theta at each energy bin and averaged
        over spin phase.
    scattering_thresholds_for_energy_mean : NDArray
        Scattering thresholds corresponding to each energy bin.
    """
    # Load scattering coefficient lookup table
    scattering_luts = load_scattering_lookup_tables(ancillary_files, instrument_id)
    pixels_below_scattering = []
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    # Load scattering thresholds for the energy bin geometric means
    scattering_thresholds_for_energy_mean = get_scattering_thresholds_for_energy(
        energy_bin_geometric_means, ancillary_files
    )
    # Initialize arrays to accumulate FWHM values for averaging
    fwhm_theta_sum = np.zeros(
        (len(energy_bin_geometric_means), for_indices_by_spin_phase.shape[0])
    )
    fwhm_phi_sum = np.zeros_like(fwhm_theta_sum)
    sample_count = np.zeros_like(fwhm_theta_sum)

    steps = for_indices_by_spin_phase.shape[1]
    energies = energy_bin_geometric_means[np.newaxis, :]
    # The "for_indices_by_spin_phase" lookup table contains the boolean values of each
    # pixel at each spin phase step, indicating whether the pixel is inside the FOR.
    # It starts at Spin-phase = 0, and increments in fine steps (1 ms), spinning the
    # spacecraft in the despun frame. At each iteration, query for the pixels in the
    # FOR, and calculate whether the FWHM value is below the threshold at the energy.
    for i in range(steps):
        # Calculate spin phase for the current iteration
        for_inds = for_indices_by_spin_phase[:, i]

        # Skip if no pixels in FOR
        if not np.any(for_inds):
            logger.info(f"No pixels found in FOR at spin phase step {i}")
            pixels_below_scattering.append(
                [
                    np.array([], dtype=int)
                    for _ in range(len(energy_bin_geometric_means))
                ]
            )
            continue
        # Using the lookup table, get the indices of the pixels inside the FOR at
        # the current spin phase step.
        theta = theta_vals[for_inds, i]
        phi = phi_vals[for_inds, i]
        theta_coeffs, phi_coeffs = get_scattering_coefficients(
            theta, phi, lookup_tables=scattering_luts
        )
        # Get a mask for pixels below the FWHM scattering threshold
        scattering_mask, fwhm_theta, fwhm_phi = mask_below_fwhm_scattering_threshold(
            theta_coeffs,
            phi_coeffs,
            energies,
            scattering_thresholds=scattering_thresholds_for_energy_mean,
        )
        # Extract pixel indices for each energy
        for_pixel_indices = np.where(for_inds)[0]
        pixels_below_scattering_for_energy = []

        for energy_idx in range(len(energy_bin_geometric_means)):
            valid_pixels = scattering_mask[:, energy_idx]
            pixels_below_scattering_for_energy.append(for_pixel_indices[valid_pixels])

        pixels_below_scattering.append(pixels_below_scattering_for_energy)
        # Accumulate FWHM values for averaging
        fwhm_theta_sum[:, for_inds] += fwhm_theta.T
        fwhm_phi_sum[:, for_inds] += fwhm_phi.T
        sample_count[:, for_inds] += 1

    fwhm_phi_avg = np.zeros_like(fwhm_phi_sum)
    fwhm_theta_avg = np.zeros_like(fwhm_theta_sum)
    np.divide(fwhm_phi_sum, sample_count, out=fwhm_phi_avg, where=sample_count != 0)
    np.divide(fwhm_theta_sum, sample_count, out=fwhm_theta_avg, where=sample_count != 0)
    return (
        pixels_below_scattering,
        fwhm_theta_avg,
        fwhm_phi_avg,
        scattering_thresholds_for_energy_mean,
    )


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
    theta_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-theta"
    phi_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-phi"
    index_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-index"
    bsf_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-bsf"

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


def get_scattering_thresholds_for_energy(
    energy: np.ndarray, ancillary_files: dict
) -> np.ndarray:
    """
    Find the scattering thresholds for each energy bin.

    Parameters
    ----------
    energy : np.ndarray
        Array of energy values in keV.
    ancillary_files : dict
        Dictionary containing ancillary files.

    Returns
    -------
    np.ndarray
        Array of scattering thresholds for each energy bin.
    """
    scattering_thresholds = get_scattering_thresholds(ancillary_files)
    # Get thresholds for all energies
    thresholds = []
    for e in energy:
        try:
            threshold = next(
                threshold
                for energy_range, threshold in scattering_thresholds.items()
                if energy_range[0] <= e < energy_range[1]
            )
        except StopIteration:
            logger.warning(
                f"Energy {e} keV is out of bounds for scattering thresholds. Using"
                f" zero for as threshold."
            )

            threshold = 0
        thresholds.append(threshold)
    return np.array(thresholds)


def get_static_deadtime_ratios(
    sensor_id: int, ancillary_files: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get static deadtime ratios.

    These should only be used when the instrument is operating in a mode that does not
    produce sectored rates data.

    Parameters
    ----------
    sensor_id : int
        Sensor ID, either 45 or 90.
    ancillary_files : dict
        Dictionary containing ancillary files.

    Returns
    -------
    np.ndarray
        Array of static deadtime ratios for each energy bin.
    """
    descriptor = f"l1c-{sensor_id}sensor-static-dead-times"
    df = pd.read_csv(ancillary_files[descriptor])
    # Drop any rows that are duplicates. We only want unique spin phase and dead time
    # ratio pairs.
    df = df.drop_duplicates()
    return df["Spin Phase (deg)"].to_numpy(dtype=float), df["Dead Time Ratio"].to_numpy(
        dtype=float
    )


def build_energy_bins() -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """
    Build energy bin boundaries.

    Returns
    -------
    intervals : list[tuple[float, float]]
        Energy bins.
    energy_midpoints : np.ndarray
        Array of energy bin midpoints.
    energy_bin_geometric_means : np.ndarray
        Array of geometric means of energy bins.
    """
    # Create energy bins.
    energy_bin_edges = np.array(UltraConstants.PSET_ENERGY_BIN_EDGES)
    energy_midpoints = (energy_bin_edges[:-1] + energy_bin_edges[1:]) / 2

    intervals = [
        (float(energy_bin_edges[i]), float(energy_bin_edges[i + 1]))
        for i in range(len(energy_bin_edges) - 1)
    ]
    energy_bin_geometric_means = np.sqrt(energy_bin_edges[:-1] * energy_bin_edges[1:])

    return intervals, energy_midpoints, energy_bin_geometric_means
