"""Contains tools for lookup tables for l1c."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
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
    for_indices_by_spin_phase: xr.DataArray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    ancillary_files: dict,
    instrument_id: int,
    reject_scattering: bool = False,
) -> tuple[xr.DataArray, NDArray, NDArray, NDArray]:
    """
    Calculate FWHM scattering values for each pixel, energy bin, and spin phase step.

    This function also calculates a mask for pixels that are below the FWHM threshold.

    Parameters
    ----------
    for_indices_by_spin_phase : xarray.DataArray
        A 2D boolean array where cols are spin phase steps are rows are HEALPix pixels.
        True indicates pixels that are within the Field of Regard (FOR) at that
        spin phase.
    theta_vals : np.ndarray
        2D or 3D array of theta values. Shape is either (spin_phase_step, npix)
        or (spin_phase_step, energy_bins, npix) when energy-dependent scattering
        rejection is used.
    phi_vals : np.ndarray
        Array of phi values with the same shape as `theta_vals`, giving the
        corresponding phi for each pixel (and energy, if present).
    ancillary_files : dict
        Dictionary containing ancillary files.
    instrument_id : int,
        Instrument ID, either 45 or 90.
    reject_scattering : bool
        Whether to reject pixels based on scattering thresholds.

    Returns
    -------
    valid_spun_pixels : xarray.DataArray
       Boolean array indicating, for each spin phase step, energy_bin, pixel,
       the pixel is inside the Field Of Regard (FOR) and whether the pixel is inside the
       FOR at that spin phase and its computed FWHM at that energy is below the
       scattering threshold. If reject_scattering is False, this will just reflect
       the FOR mask (for_indices_by_spin_phase).
    scattering_fwhm_theta : NDArray
        Calculated FWHM scatting values for theta at each energy bin and averaged
        over spin phase.
    scattering_fwhm_phi : NDArray
        Calculated FWHM scatting values for theta at each energy bin and averaged
        over spin phase.
    scattering_thresholds_for_energy_mean : NDArray
        Scattering thresholds corresponding to each energy bin.
    """
    # Check shapes of theta phi, and index arrays
    index_shape = for_indices_by_spin_phase.shape
    if theta_vals.shape != index_shape or phi_vals.shape != index_shape:
        raise ValueError(
            "Shape mismatch between FOR indices and theta/phi values. "
            f"FOR indices shape: {index_shape}, "
            f"theta shape: {theta_vals.shape}, "
            f"phi shape: {phi_vals.shape}."
        )
    # Load scattering coefficient lookup table
    scattering_luts = load_scattering_lookup_tables(ancillary_files, instrument_id)
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    # Load scattering thresholds for the energy bin geometric means
    scattering_thresholds_for_energy_mean = get_scattering_thresholds_for_energy(
        energy_bin_geometric_means, ancillary_files
    )
    n_pix = for_indices_by_spin_phase.sizes["pixel"]
    # Initialize arrays to accumulate FWHM values for averaging
    fwhm_theta_sum = np.zeros((len(energy_bin_geometric_means), n_pix))
    fwhm_phi_sum = np.zeros_like(fwhm_theta_sum)
    sample_count = np.zeros_like(fwhm_theta_sum)

    steps = for_indices_by_spin_phase.sizes["spin_phase_step"]
    energies = energy_bin_geometric_means[np.newaxis, :]
    # Initialize DataArray to hold boolean of valid pixels at each spin phase step
    # If reject_scattering if false, this will just be the FOR mask.
    spun_dims = ("spin_phase_step", "energy", "pixel")
    if reject_scattering:
        valid_pixels = xr.DataArray(
            np.zeros((steps, len(energy_bin_geometric_means), n_pix), dtype=bool),
            dims=spun_dims,
        )
    elif "energy" not in for_indices_by_spin_phase.sizes:
        valid_pixels = for_indices_by_spin_phase.expand_dims(
            {"energy": len(energy_bin_geometric_means)}
        ).transpose(*spun_dims)
    else:
        valid_pixels = for_indices_by_spin_phase
    # The "for_indices_by_spin_phase" lookup table contains the boolean values of each
    # pixel at each spin phase step, indicating whether the pixel is inside the FOR.
    # It starts at Spin-phase = 0, and increments in fine steps (1 ms), spinning the
    # spacecraft in the despun frame. At each iteration, query for the pixels in the
    # FOR, and calculate whether the FWHM value is below the threshold at the energy.
    for i in range(steps):
        for_inds = for_indices_by_spin_phase.isel(spin_phase_step=i).values

        if for_inds.ndim > 1:
            # Energy dependent FOR indices
            for e_ind in range(len(energy_bin_geometric_means)):
                for_inds_energy = for_inds[e_ind, :]

                # Skip if no pixels in FOR
                if not np.any(for_inds_energy):
                    continue

                theta = theta_vals[i, e_ind, for_inds_energy]
                phi = phi_vals[i, e_ind, for_inds_energy]
                theta_coeffs, phi_coeffs = get_scattering_coefficients(
                    theta.data, phi.data, lookup_tables=scattering_luts
                )
                # Calculate scattering mask for specified energy
                energy = energy_bin_geometric_means[e_ind : e_ind + 1][np.newaxis, :]
                scattering_mask, fwhm_theta, fwhm_phi = (
                    mask_below_fwhm_scattering_threshold(
                        theta_coeffs,
                        phi_coeffs,
                        energy,
                        scattering_thresholds=scattering_thresholds_for_energy_mean[
                            e_ind : e_ind + 1
                        ],
                    )
                )
                # If rejecting scattering, store the mask
                if reject_scattering:
                    valid_pixels[i, e_ind, for_inds_energy] = scattering_mask.flatten()

                # Accumulate FWHM values
                fwhm_theta_sum[e_ind, for_inds_energy] += fwhm_theta.flatten()
                fwhm_phi_sum[e_ind, for_inds_energy] += fwhm_phi.flatten()
                sample_count[e_ind, for_inds_energy] += 1
        else:
            # Energy independent FOR indices
            if not np.any(for_inds):
                continue

            theta = theta_vals[i, for_inds]
            phi = phi_vals[i, for_inds]
            theta_coeffs, phi_coeffs = get_scattering_coefficients(
                theta, phi, lookup_tables=scattering_luts
            )
            scattering_mask, fwhm_theta, fwhm_phi = (
                mask_below_fwhm_scattering_threshold(
                    theta_coeffs,
                    phi_coeffs,
                    energies,
                    scattering_thresholds=scattering_thresholds_for_energy_mean,
                )
            )

            if reject_scattering:
                valid_pixels[i, :, for_inds] = scattering_mask.T

            # Accumulate FWHM values
            fwhm_theta_sum[:, for_inds] += fwhm_theta.T
            fwhm_phi_sum[:, for_inds] += fwhm_phi.T
            sample_count[:, for_inds] += 1

    fwhm_phi_avg = np.zeros_like(fwhm_phi_sum)
    fwhm_theta_avg = np.zeros_like(fwhm_theta_sum)
    np.divide(fwhm_phi_sum, sample_count, out=fwhm_phi_avg, where=sample_count != 0)
    np.divide(fwhm_theta_sum, sample_count, out=fwhm_theta_avg, where=sample_count != 0)
    return (
        valid_pixels,
        fwhm_theta_avg,
        fwhm_phi_avg,
        scattering_thresholds_for_energy_mean,
    )


def get_spacecraft_pointing_lookup_tables(
    ancillary_files: dict, instrument_id: int
) -> tuple[xr.DataArray, NDArray, NDArray, NDArray, xr.DataArray]:
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
    for_indices_by_spin_phase : xarray.DataArray
        A 2D boolean array of shape (n_spin_phase_steps,npix).
        True indicates pixels that are within the Field of Regard (FOR) at that
        spin phase.
    theta_vals : NDArray
        A 2D array of theta values for each HEALPix pixel at each spin phase step.
    phi_vals : NDArray
         A 2D array of phi values for each HEALPix pixel at each spin phase step.
    ra_and_dec : NDArray
        A 2D array of right ascension and declination values for each HEALPix pixel.
    boundary_scale_factors : xarray.DataArray
        A 2D array of boundary scale factors for each HEALPix pixel at each spin phase
        step.
    """
    theta_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-theta"
    phi_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-phi"
    index_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-index"
    bsf_descriptor = f"l1c-{instrument_id}sensor-sc-pointing-bsf"

    theta_vals = (
        pd.read_csv(ancillary_files[theta_descriptor], header=None, skiprows=1)
        .to_numpy(dtype=float)[:, 2:]
        .T
    )
    phi_vals = (
        pd.read_csv(ancillary_files[phi_descriptor], header=None, skiprows=1)
        .to_numpy(dtype=float)[:, 2:]
        .T
    )
    index_grid = (
        pd.read_csv(ancillary_files[index_descriptor], header=None, skiprows=1)
        .to_numpy(dtype=float)
        .T
    )
    boundary_scale_factors = (
        pd.read_csv(ancillary_files[bsf_descriptor], header=None, skiprows=1)
        .to_numpy(dtype=float)[:, 2:]
        .T
    )

    ra_and_dec = index_grid[:2, :]  # Shape (npix, 2)
    # This array indicates whether each pixel is in the nominal FOR at each spin phase
    # step (15000 steps for a full rotation with 1 ms resolution).
    for_indices_by_spin_phase = xr.DataArray(
        np.nan_to_num(index_grid[2:, :], nan=0).astype(bool),
        dims=("spin_phase_step", "pixel"),
    )
    boundary_scale_factors = xr.DataArray(
        boundary_scale_factors, dims=("spin_phase_step", "pixel")
    )
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


def build_energy_bins(
    energy_bin_edges: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """
    Build energy bin boundaries.

    Parameters
    ----------
    energy_bin_edges : numpy.ndarray, optional
        List of energy bin edges. If None, uses default UltraConstants.PSET_ENERGY_BIN.

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
    if energy_bin_edges is None:
        energy_bin_edges = np.array(UltraConstants.PSET_ENERGY_BIN_EDGES)
        logger.info(
            f"No energy bin file found, using default pointing bin energy"
            f" edges {energy_bin_edges}"
        )

    energy_midpoints = (energy_bin_edges[:-1] + energy_bin_edges[1:]) / 2

    intervals = [
        (float(energy_bin_edges[i]), float(energy_bin_edges[i + 1]))
        for i in range(len(energy_bin_edges) - 1)
    ]
    energy_bin_geometric_means = np.sqrt(energy_bin_edges[:-1] * energy_bin_edges[1:])

    return intervals, energy_midpoints, energy_bin_geometric_means
