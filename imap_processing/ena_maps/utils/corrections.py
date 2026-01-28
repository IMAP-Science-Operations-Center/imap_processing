"""L2 corrections common to multiple IMAP ENA instruments."""

import logging
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial
from scipy.constants import electron_volt, erg, proton_mass

from imap_processing.ena_maps.ena_maps import (
    LoHiBasePointingSet,
)
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.time import ttj2000ns_to_et

logger = logging.getLogger(__name__)

# Tell ruff to ignore ambiguous Greek letters in formulas in this file
# ruff: noqa: RUF003

# Create a TypeVar to represent the specific class being passed in
# Bound to LoHiBasePointingSet, meaning it must be LoHiBasePointingSet
# or a subclass of it
LoHiBasePsetSubclass = TypeVar("LoHiBasePsetSubclass", bound=LoHiBasePointingSet)

# Physical constants for Compton-Getting correction
# Units: electron_volt = [J / eV]
#        erg = [J / erg]
# To get [erg / eV], => electron_volt [J / eV] / erg [J / erg] = erg_per_ev [erg / eV]
ERG_PER_EV = electron_volt / erg  # erg per eV - unit conversion factor
# Units: proton_mass = [kg]
# Here, we convert proton_mass to grams
PROTON_MASS_GRAMS = proton_mass * 1e3  # proton mass in grams


class PowerLawFluxCorrector:
    """
    IMAP-Lo flux correction algorithm implementation.

    Based on Section 5 of the Mapping Algorithm Document. Applies corrections for
    ESA transmission integration over energy bandpass using iterative
    predictor-corrector scheme to estimate source fluxes from observed fluxes.

    Parameters
    ----------
    coeffs_file : str or Path
        Location of CSV file containing ESA transmission coefficients.
    """

    def __init__(self, coeffs_file: str | Path):
        """Initialize PowerLawFluxCorrector."""
        # Load the csv file
        eta_coeffs_df = pd.read_csv(coeffs_file, index_col="esa_step")
        # Create a lookup dictionary to get the correct np.polynomial.Polynomial
        # for a given esa_step
        coeff_columns = ["M0", "M1", "M2", "M3", "M4", "M5"]
        self.polynomial_lookup = {
            row.name: Polynomial(row[coeff_columns].values)
            for _, row in eta_coeffs_df.iterrows()
        }

    def eta_esa(self, k: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        Calculate ESA transmission scale factor η_esa,k(γ) for each energy level.

        Parameters
        ----------
        k : np.ndarray
            Energy levels (1D array of ESA steps).
        gamma : np.ndarray
            Power-law slopes. Can be 1D (n_energy,) or multi-dimensional
            (n_energy, ...spatial_dims...).

        Returns
        -------
        np.ndarray
            ESA transmission scale factors. Shape matches gamma.
        """
        k = np.atleast_1d(k)
        gamma = np.atleast_1d(gamma)
        eta = np.empty_like(gamma)

        # Loop over energy levels only (first axis)
        for i, esa_step in enumerate(k):
            # Evaluate polynomial for all spatial pixels at this energy level
            eta[i] = self.polynomial_lookup[esa_step](gamma[i])
            # Negative transmissions get set to 1
            eta[i] = np.where(eta[i] < 0, 1.0, eta[i])

        return eta

    @staticmethod
    def estimate_power_law_slope(
        fluxes: np.ndarray,
        energies: np.ndarray,
        uncertainties: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Estimate power-law slopes γ_k for each energy level using vectorized operations.

        Implements equations (36)-(41) from the Mapping Algorithm Document v7
        with proper boundary handling. Uses extended arrays with repeated
        endpoints for unified calculation, and handles zero fluxes by falling
        back to linear differencing or returning NaN where both central and
        linear differencing fail.

        Parameters
        ----------
        fluxes : np.ndarray
            Array of differential fluxes with shape (n_energy, n_pixels).
        energies : np.ndarray
            Array of energy levels [E_1, E_2, ..., E_7]. Must be 1D.
        uncertainties : np.ndarray, optional
            Array of flux uncertainties. Shape must match fluxes.

        Returns
        -------
        gamma : np.ndarray
            Array of power-law slopes. Shape (n_energy, n_pixels).
        delta_gamma : np.ndarray or None
            Array of uncertainty slopes (if uncertainties provided). Shape
            (n_energy, n_pixels).
        """
        # Compute logs, setting non-positive fluxes to NaN
        log_fluxes = np.log(np.where(fluxes > 0, fluxes, np.nan))
        log_energies = np.log(energies)

        # Pad with NaN so central differencing naturally falls back to one-sided
        # Interior points use central differencing equation:
        #     gamma_k = ln(J_{k+1}/J_{k-1}) / ln(E_{k+1}/E_{k-1})
        # Left boundary uses linear forward differencing:
        #     gamma_k = ln(J_{k+1}/J_{k}) / ln(E_{k+1}/E_{k})
        # Right boundary uses linear backward differencing:
        #     gamma_k = ln(J_{k}/J_{k-1}) / ln(E_{k}/E_{k-1})

        # Pad along energy axis (first axis) with NaN
        # fluxes has shape (n_energy, n_pixels)
        log_extended_fluxes = np.pad(
            log_fluxes, ((1, 1), (0, 0)), constant_values=np.nan
        )
        log_extended_energies = np.pad(log_energies, (1, 1), constant_values=np.nan)

        # Broadcast energies to match flux shape:
        # (n_energy + 2,) -> (n_energy + 2, n_pixels)
        log_extended_energies_broadcast = np.broadcast_to(
            log_extended_energies[:, np.newaxis],
            log_extended_fluxes.shape,
        )

        # Create index arrays with same shape as fluxes
        # Start with central differencing indices: left=k-1, right=k+1
        # In the extended array, original index k corresponds to extended index k+1
        n_energies = energies.shape[0]
        left_indices = np.broadcast_to(
            np.arange(n_energies)[:, np.newaxis], fluxes.shape
        ).copy()
        right_indices = np.broadcast_to(
            (np.arange(n_energies) + 2)[:, np.newaxis], fluxes.shape
        ).copy()

        # Check if central differencing is valid
        central_invalid = ~(
            np.isfinite(np.take_along_axis(log_extended_fluxes, left_indices, axis=0))
            & np.isfinite(
                np.take_along_axis(log_extended_fluxes, right_indices, axis=0)
            )
        )

        # For invalid central differencing, try forward differencing: left=k, right=k+1
        left_indices[central_invalid] += 1

        # Check if forward differencing is valid
        forward_invalid = ~(
            np.isfinite(np.take_along_axis(log_extended_fluxes, left_indices, axis=0))
            & np.isfinite(
                np.take_along_axis(log_extended_fluxes, right_indices, axis=0)
            )
        )

        # For invalid forward differencing, try backward: left=k-1, right=k
        need_backward = central_invalid & forward_invalid
        left_indices[need_backward] -= 1  # Back to k-1
        right_indices[need_backward] -= 1  # Change from k+1 to k

        # Extract final flux and energy values using the computed indices
        left_log_fluxes = np.take_along_axis(log_extended_fluxes, left_indices, axis=0)
        right_log_fluxes = np.take_along_axis(
            log_extended_fluxes, right_indices, axis=0
        )
        left_log_energies = np.take_along_axis(
            log_extended_energies_broadcast, left_indices, axis=0
        )
        right_log_energies = np.take_along_axis(
            log_extended_energies_broadcast, right_indices, axis=0
        )

        # Compute power-law slopes
        valid = np.isfinite(left_log_fluxes) & np.isfinite(right_log_fluxes)
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = np.where(
                valid,
                (right_log_fluxes - left_log_fluxes)
                / (right_log_energies - left_log_energies),
                0.0,
            )

        # Compute uncertainty slopes
        delta_gamma = np.zeros_like(fluxes, dtype=float)
        if uncertainties is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_unc_sq = (uncertainties / fluxes) ** 2
            extended_rel_unc_sq = np.pad(
                rel_unc_sq, ((1, 1), (0, 0)), constant_values=np.nan
            )

            left_rel_unc_sq = np.take_along_axis(
                extended_rel_unc_sq, left_indices, axis=0
            )
            right_rel_unc_sq = np.take_along_axis(
                extended_rel_unc_sq, right_indices, axis=0
            )

            delta_gamma = np.where(
                valid,
                np.sqrt(left_rel_unc_sq + right_rel_unc_sq)
                / (right_log_energies - left_log_energies),
                0.0,
            )

        return gamma, delta_gamma

    def predictor_corrector_iteration(
        self,
        observed_fluxes: np.ndarray,
        observed_uncertainties: np.ndarray,
        energies: np.ndarray,
        max_iterations: int = 20,
        convergence_threshold: float = 0.005,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate source fluxes using iterative predictor-corrector scheme.

        Implements the algorithm from Appendix A of the Mapping Algorithm Document.
        Fully vectorized to process all spatial pixels simultaneously, with
        per-pixel convergence tracking.

        Parameters
        ----------
        observed_fluxes : np.ndarray
            Array of observed fluxes. Shape (n_energy,) or
            (n_energy, ...spatial_dims...).
        observed_uncertainties : numpy.ndarray
            Array of observed uncertainties. Shape must match observed_fluxes.
        energies : np.ndarray
            Array of energy levels (1D).
        max_iterations : int, optional
            Maximum number of iterations, by default 20.
        convergence_threshold : float, optional
            RMS convergence criterion, by default 0.005 (0.5%).

        Returns
        -------
        source_fluxes : np.ndarray
            Final estimate of source fluxes. Shape matches observed_fluxes.
        source_uncertainties : np.ndarray
            Final estimate of source uncertainties. Shape matches observed_fluxes.
        n_iterations : np.ndarray
            Number of iterations run for each pixel. Shape matches spatial dims
            of input.
        """
        n_levels = observed_fluxes.shape[0]
        energy_levels = np.arange(n_levels) + 1

        # Initial power-law estimate from observed fluxes
        gamma_initial, _ = self.estimate_power_law_slope(observed_fluxes, energies)

        # Initial source flux estimate
        eta_initial = self.eta_esa(energy_levels, gamma_initial)
        source_fluxes_n = observed_fluxes / eta_initial
        source_uncertainties = observed_uncertainties / eta_initial

        # Track which pixels have converged and iteration count per pixel
        converged = np.zeros(observed_fluxes.shape[1:], dtype=bool)
        n_iterations = np.zeros(observed_fluxes.shape[1:], dtype=int)

        for iteration in range(max_iterations):
            # Get mask for unconverged pixels
            not_converged = ~converged

            # Only process unconverged pixels
            source_fluxes_active = source_fluxes_n[:, not_converged]
            observed_fluxes_active = observed_fluxes[:, not_converged]
            observed_uncertainties_active = observed_uncertainties[:, not_converged]
            gamma_initial_active = gamma_initial[:, not_converged]

            # Store previous iteration for unconverged pixels
            source_fluxes_prev = source_fluxes_active.copy()

            # Predictor step - only for unconverged pixels
            gamma_pred, _ = self.estimate_power_law_slope(
                source_fluxes_active, energies
            )
            gamma_half = 0.5 * (gamma_initial_active + gamma_pred)

            # Predictor source flux estimate
            eta_half = self.eta_esa(energy_levels, gamma_half)
            source_fluxes_half = observed_fluxes_active / eta_half

            # Corrector step
            gamma_corr, _ = self.estimate_power_law_slope(source_fluxes_half, energies)
            gamma_n = 0.5 * (gamma_pred + gamma_corr)

            # Final source flux estimate for this iteration
            eta_final = self.eta_esa(energy_levels, gamma_n)
            source_fluxes_new = observed_fluxes_active / eta_final
            source_uncertainties_new = observed_uncertainties_active / eta_final

            # Check convergence for unconverged pixels
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios_sq = (source_fluxes_new / source_fluxes_prev) ** 2
            # Compute chi per pixel (mean over energy axis)
            chi_n = np.sqrt(np.mean(ratios_sq, axis=0)) - 1

            # Determine which pixels converged this iteration
            # Start with all False, then set True for newly converged pixels
            newly_converged = np.zeros_like(converged)
            newly_converged[not_converged] = chi_n < convergence_threshold
            n_iterations[newly_converged] = iteration + 1

            # Update source fluxes and uncertainties for unconverged pixels
            source_fluxes_n[:, not_converged] = source_fluxes_new
            source_uncertainties[:, not_converged] = source_uncertainties_new

            # Update converged mask
            converged |= newly_converged

            # If all pixels have converged, exit early
            if np.all(converged):
                break

        # Set iteration count for pixels that didn't converge
        n_iterations[~converged] = max_iterations

        return source_fluxes_n, source_uncertainties, n_iterations

    def apply_flux_correction(
        self,
        flux: xr.DataArray,
        flux_stat_unc: xr.DataArray,
        energies: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Apply flux correction to observed fluxes.

        Iterative predictor-corrector scheme is applied to all spatial pixels
        simultaneously using vectorized operations to correct fluxes and
        statistical uncertainties.

        Parameters
        ----------
        flux : xarray.DataArray
            Input flux. Must have "energy" dimension corresponding to energies.
            Can have arbitrary additional spatial dimensions.
        flux_stat_unc : xarray.DataArray
            Statistical uncertainty for input fluxes. Shape and dimensions must
            match flux.
        energies : xarray.DataArray
            Array of energy levels in units of eV or keV. Must be 1D with
            "energy" dimension.

        Returns
        -------
        tuple[xarray.DataArray, xarray.DataArray]
            Corrected fluxes and flux uncertainties with same shape and dimensions
            as input.
        """
        # Stack all non-energy dimensions into a single "pixel" dimension
        # This converts to shape (energy, pixel) for processing
        spatial_dims = [d for d in flux.dims if "energy" not in d]

        if spatial_dims:
            flux_stacked = flux.stack(flux_pixel=spatial_dims)
            flux_stat_unc_stacked = flux_stat_unc.stack(flux_pixel=spatial_dims)
        else:
            # If only energy dimension exists, add a dummy pixel dimension
            flux_stacked = flux.expand_dims("flux_pixel")
            flux_stat_unc_stacked = flux_stat_unc.expand_dims("flux_pixel")

        # Call vectorized predictor-corrector iteration on 2D arrays
        corrected_flux_stacked, corrected_unc_stacked, _ = (
            self.predictor_corrector_iteration(
                flux_stacked.values,
                flux_stat_unc_stacked.values,
                energies.values,
            )
        )

        # Convert back to DataArrays with stacked dimensions
        corrected_flux_da = xr.DataArray(
            corrected_flux_stacked,
            dims=flux_stacked.dims,
            coords=flux_stacked.coords,
        )
        corrected_unc_da = xr.DataArray(
            corrected_unc_stacked,
            dims=flux_stat_unc_stacked.dims,
            coords=flux_stat_unc_stacked.coords,
        )

        # Unstack back to original dimensions
        if spatial_dims:
            corrected_flux_da = corrected_flux_da.unstack("flux_pixel")
            corrected_unc_da = corrected_unc_da.unstack("flux_pixel")

            # Ensure dimension order matches input
            corrected_flux_da = corrected_flux_da.transpose(*flux.dims)
            corrected_unc_da = corrected_unc_da.transpose(*flux_stat_unc.dims)
        else:
            # Remove dummy pixel dimension
            corrected_flux_da = corrected_flux_da.squeeze("flux_pixel")
            corrected_unc_da = corrected_unc_da.squeeze("flux_pixel")

        return corrected_flux_da, corrected_unc_da


def add_spacecraft_velocity_to_pset(
    pset: xr.Dataset,
) -> xr.Dataset:
    """
    Calculate and add spacecraft velocity data to pointing set dataset.

    Parameters
    ----------
    pset : xr.Dataset
        Pointing set dataset to be updated. Must contain "epoch" coordinate
        and "epoch_delta" data variable.

    Returns
    -------
    pset_processed : xarray.Dataset
        Pointing set dataset with spacecraft velocity data added.

    Notes
    -----
    Adds the following DataArrays to input dataset:
    - "sc_velocity": Spacecraft velocity vector (km/s) with dims ["x_y_z"]
    - "sc_direction_vector": Spacecraft velocity unit vector with dims ["x_y_z"]
    """
    # Hi and Lo need to use different methods for computing the Pointing
    # midpoint time.
    if pset.attrs["Logical_source"].startswith("imap_hi"):
        # Compute ephemeris time (J2000 seconds) of PSET midpoint
        # epoch contains Pointing start time, and epoch_delta indicates the total
        # duration of the Pointing
        # For Hi, epoch_delta is the duration of the Pointing in nanoseconds
        pointing_duration_ns = pset["epoch_delta"].values[0]
    elif pset.attrs["Logical_source"].startswith("imap_lo"):
        # For Lo, compute the pointing duration using pointing start/end MET times
        pointing_duration_ns = (
            pset["pointing_end_met"].values[0] - pset["pointing_start_met"].values[0]
        ) * 1e9
    else:
        raise NotImplementedError(
            f"add_spacecraft_velocity_to_pset does not support PSETs with "
            f"Logical_source: {pset.attrs['Logical_source']}"
        )
    et = ttj2000ns_to_et(pset["epoch"].values[0] + pointing_duration_ns / 2)

    # Get spacecraft state in HAE frame
    sc_state = geometry.imap_state(et, ref_frame=geometry.SpiceFrame.IMAP_HAE)
    sc_velocity_vector = sc_state[3:6]

    # Store spacecraft velocity as DataArray
    pset["sc_velocity"] = xr.DataArray(
        sc_velocity_vector, dims=[CoordNames.CARTESIAN_VECTOR.value]
    )

    # Calculate spacecraft speed and direction
    sc_velocity_km_per_sec = np.linalg.norm(pset["sc_velocity"], axis=-1, keepdims=True)
    pset["sc_direction_vector"] = pset["sc_velocity"] / sc_velocity_km_per_sec

    return pset


def _add_cartesian_look_direction(pset: xr.Dataset) -> xr.Dataset:
    """
    Calculate and add look direction vectors to pointing set dataset.

    Parameters
    ----------
    pset : xarray.Dataset
        Pointing set dataset to be updated. Must contain "hae_longitude" and
        "hae_latitude" data variables.

    Returns
    -------
    pset_processed : xarray.Dataset
        Pointing set dataset with look direction vectors added.

    Notes
    -----
    Adds the following DataArray to input dataset:
    - "look_direction": Cartesian unit vectors with dims [...spatial_dims, "x_y_z"]
    """
    longitudes = pset["hae_longitude"]
    latitudes = pset["hae_latitude"]

    # Stack spherical coordinates (r=1 for unit vectors, azimuth, elevation)
    spherical_coords = np.stack(
        [
            np.ones_like(longitudes),  # r = 1 for unit vectors
            longitudes,  # azimuth = longitude
            latitudes,  # elevation = latitude
        ],
        axis=-1,
    )

    # Convert to Cartesian coordinates and store as DataArray
    pset["look_direction"] = xr.DataArray(
        geometry.spherical_to_cartesian(spherical_coords),
        dims=[*longitudes.dims, CoordNames.CARTESIAN_VECTOR.value],
    )

    return pset


def _calculate_compton_getting_transform(
    pset: xr.Dataset,
    energy_hf: xr.DataArray,
) -> xr.Dataset:
    """
    Apply Compton-Getting transformation to compute ENA source directions.

    This implements the Compton-Getting velocity transformation to correct
    for the motion of the spacecraft through the heliosphere. The transformation
    accounts for the Doppler shift of ENA energies and the aberration of
    arrival directions.

    All calculations are performed using xarray DataArrays to preserve
    dimension information throughout the computation.

    Parameters
    ----------
    pset : xarray.Dataset
        Pointing set dataset with sc_velocity, sc_direction_vector, and
        look_direction already added.
    energy_hf : xr.DataArray
        ENA energies in the heliosphere frame in eV.

    Returns
    -------
    pset : xarray.Dataset
        Pointing set dataset with Compton-Getting related variables added.

    Notes
    -----
    The algorithm is based on the "Appendix A. The IMAP-Lo Mapping Algorithms"
    document.
    Adds the following DataArrays to input dataset:
    - "energy_sc": ENA energies in spacecraft frame (eV)
    - "energy_hf": ENA energies in the heliosphere frame (eV)
    - "ram_mask": Mask indicating whether ENA source direction is from the ram
      direction.
    Updates the following DataArrays in input dataset:
    - "hae_longitude": ENA source longitudes in heliosphere frame (degrees)
    - "hae_latitude": ENA source latitudes in heliosphere frame (degrees)
    """
    # Store heliosphere frame energies
    pset["energy_hf"] = energy_hf

    # Calculate spacecraft speed
    sc_velocity_km_per_sec = np.linalg.norm(pset["sc_velocity"], axis=-1, keepdims=True)

    # Calculate dot product between look directions and spacecraft direction vector
    # Use Einstein summation for efficient vectorized dot product
    sc_direction_vector = pset["sc_velocity"] / sc_velocity_km_per_sec
    dot_product = xr.DataArray(
        np.einsum(
            "...i,...i->...",
            pset["look_direction"],
            sc_direction_vector,
        ),
        dims=pset["look_direction"].dims[:-1],
    )

    # Calculate the kinetic energy of a hydrogen ENA traveling at spacecraft velocity
    # E_u = (1/2) * m * U_sc^2 (convert km/s to cm/s with 1.0e5 factor)
    energy_u = (
        0.5 * PROTON_MASS_GRAMS * (sc_velocity_km_per_sec * 1e5) ** 2 / ERG_PER_EV
    )

    # Note: Tim thinks that this approach seems backwards. Here, we are assuming
    #     that ENAs are observed in the heliosphere frame at the ESA energy levels.
    #     We then calculate the velocity that said ENAs would have in the spacecraft
    #     frame as well as the CG corrected energy level in the spacecraft frame.
    #     We then use this velocity to calculate and the velocity of the spacecraft
    #     to do the vector math which determines the ENA source direction in the
    #     heliosphere frame.
    #     The ENAs are in fact observed in the spacecraft frame at a known energy
    #     level in the spacecraft frame. Why don't we use that energy level to
    #     calculate the source direction in the spacecraft frame and then do the
    #     vector math to find the source direction in the heliosphere frame? We
    #     would also need to calculate the CG corrected ENA energy in the heliosphere
    #     frame and keep track of that when binning.

    # Calculate y values for each energy level (Equation 61)
    # y_k = sqrt(E^h_k / E^u)
    y = np.sqrt(pset["energy_hf"] / energy_u)

    # Velocity magnitude factor calculation (Equation 62)
    # x_k = (-êₛ · û_sc) + sqrt(y² + (êₛ · û_sc)² - 1)
    # dot_product = look_hat dot usc_hat = (-êₛ · û_sc)
    x = dot_product + np.sqrt(y**2 + dot_product**2 - 1)
    # Get the dimensions in the right order so that spatial is last
    x = x.transpose(dot_product.dims[0], y.dims[0], *dot_product.dims[1:])

    # Calculate ENA speed in the spacecraft frame
    # |v⃗_sc| = x_k * U_sc
    velocity_sc = x * sc_velocity_km_per_sec

    # Calculate the kinetic energy of the spacecraft
    # E_sc = (1/2) * M_p * v_sc² (convert km/s to cm/s with 1.0e5 factor)
    pset["energy_sc"] = 0.5 * PROTON_MASS_GRAMS * (velocity_sc * 1e5) ** 2 / ERG_PER_EV

    # Calculate the velocity vector in the spacecraft frame
    # v⃗_sc = -|v_sc| * êₛ (velocity direction is opposite to look direction)
    velocity_vector_sc = -1 * velocity_sc * pset["look_direction"]

    # Calculate the ENA velocity vector in the heliosphere frame
    # v⃗_helio = v⃗_sc + U⃗_sc (simple velocity addition)
    velocity_vector_helio = velocity_vector_sc + pset["sc_velocity"]

    # Convert to spherical coordinates to get ENA source directions
    # Look direction is opposite of ENA direction
    ena_source_direction_helio = geometry.cartesian_to_spherical(
        -1 * velocity_vector_helio.data
    )

    # Update the PSET hae_longitude and hae_latitude variables with the new
    # energy-dependent values.
    pset["hae_longitude"] = (
        pset["energy_sc"].dims,
        ena_source_direction_helio[..., 1],
    )
    pset["hae_latitude"] = (
        pset["energy_sc"].dims,
        ena_source_direction_helio[..., 2],
    )

    return pset


def calculate_ram_mask(pset: xr.Dataset) -> xr.Dataset:
    """
    Calculate the RAM mask using the input spacecraft velocity vector.

    The RAM mask is a boolean array with the same dimensions as what is stored
    in the "hae_longitude" and "hae_latitude" variables of the dataset.

    Parameters
    ----------
    pset : xarray.Dataset
        Pointing set dataset. The pset dataset is assumed to have valid
        "hae_longitude", "hae_latitude", and "sc_velocity" variables.

    Returns
    -------
    pset : xarray.Dataset
        Pointing set dataset with ram_mask variable added.
    """
    logger.debug(
        f"Calculating the RAM mask using input spacecraft direction "
        f"vector: {pset['sc_velocity'].values} and hae coordinates in "
        f"the dataset hae_longitude and hae_latitude variables."
    )
    longitude = pset["hae_longitude"]
    latitude = pset["hae_latitude"]
    spacecraft_velocity = pset["sc_velocity"].values
    spherical_coords = np.stack(
        [
            np.ones_like(longitude.values),
            longitude.values,
            latitude.values,
        ],
        axis=-1,
    )
    cartesian_source_direction = xr.DataArray(
        geometry.spherical_to_cartesian(spherical_coords),
        dims=[*longitude.dims, CoordNames.CARTESIAN_VECTOR.value],
    )
    # For ram/anti-ram filtering we can use the sign of the scalar projection
    # of the ENA source direction vector (-v⃗_ena) onto the spacecraft velocity
    # vector.
    # ram_mask = (-v⃗_ena · û_sc) >= 0
    # Use Einstein summation for efficient vectorized dot product
    ram_mask = (
        np.einsum("...i,...i->...", spacecraft_velocity, cartesian_source_direction)
        >= 0
    )
    pset["ram_mask"] = xr.DataArray(
        ram_mask,
        dims=longitude.dims,
    )

    return pset


def apply_compton_getting_correction(
    pset: xr.Dataset,
    energy_hf: xr.DataArray,
) -> xr.Dataset:
    """
    Apply Compton-Getting correction to a pointing set dataset.

    This function performs the Compton-Getting velocity transformation to correct
    ENA observations for the motion of the spacecraft through the heliosphere.
    The corrected coordinates represent the true source directions of the ENAs
    in the heliosphere frame.

    New variables are added to the dataset for the corrected coordinates and
    energies.

    All calculations are performed using xarray DataArrays to preserve dimension
    information throughout the computation.

    Parameters
    ----------
    pset : xarray.Dataset
        Pointing set dataset. Must contain the following coordinates:
          - epoch: start time of the pointing
        Must contain the following variables:
          - sc_velocity: velocity vector of the spacecraft in the HAE frame at
            the midpoint time of the pointing [km/s]. See the
            `add_spacecraft_velocity_to_pset` function.
          - hae_longitude: PSET bin longitudes in the HAE frame (degrees)
          - hae_latitude: PSET bin latitudes in the HAE frame (degrees)
    energy_hf : xr.DataArray
        ENA energies in the heliosphere frame in eV. Must be 1D with an
        energy dimension.

    Returns
    -------
    processed_dataset : xarray.Dataset
        Updated dataset object with Compton-Getting related variables added and
        hae_longitude and hae_latitude variables updated to contain energy-dependent
        cg-corrected values.

    Notes
    -----
    This function adds the following variables to the dataset:
    - "look_direction": Cartesian unit vectors of observation directions
    - "energy_hf": ENA energies in heliosphere frame (eV)
    - "energy_sc": ENA energies in spacecraft frame (eV)
    This function modifies the following variables in the dataset:
    - "hae_longitude": ENA source longitudes in heliosphere frame (degrees)
    - "hae_latitude": ENA source latitudes in heliosphere frame (degrees)
    """
    # Step 1: Calculate and add look direction vectors to pset
    processed_dataset = _add_cartesian_look_direction(pset)

    # Step 2: Apply Compton-Getting transformation
    processed_dataset = _calculate_compton_getting_transform(
        processed_dataset, energy_hf
    )

    return processed_dataset


def interpolate_map_flux_to_helio_frame(
    map_ds: xr.Dataset,
    esa_energies: xr.DataArray,
    helio_energies: xr.DataArray,
    vars_to_interpolate: list[str],
) -> xr.Dataset:
    """
    Interpolate flux from spacecraft frame to heliocentric frame energies.

    This implements the Compton-Getting interpolation step that transforms
    flux measurements from the spacecraft frame to the heliocentric frame.
    The algorithm follows these steps:
    1. For each spatial pixel and energy step, get the spacecraft energy
    2. Find bounding ESA energy channels for interpolation
    3. Perform power-law interpolation between bounding channels to spacecraft energy
    4. Apply energy scaling transformation to heliocentric frame

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset with `energy_sc` data variable containing the spacecraft
        frame energies for each spatial pixel and ESA energy step.
    esa_energies : xarray.DataArray
        The ESA nominal central energies. Any energy unit is acceptable as long
        as it is consistent with `helio_energies`.
    helio_energies : xarray.DataArray
        The heliocentric frame energies to interpolate to. Any energy unit is
        acceptable as long as it is consistent with `esa_energies`.
        In practice, these are the same as esa_energies.
    vars_to_interpolate : list[str]
        List of variables to perform interpolation on. This is just the base
        flux/intensity variable. It is assumed that the associated statistical
        uncertainty and systematic error variables are also present in the input
        dataset and will be interpolated as well. For example, if ["ena_intensity"]
        is input, then the variables "ena_intensity", "ena_intensity_stat_uncert",
        and "ena_intensity_sys_err" will be interpolated.

    Returns
    -------
    map_ds : xarray.Dataset
        Updated map dataset with interpolated heliocentric frame fluxes.
    """
    logger.info("Performing Compton-Getting interpolation to heliocentric frame")

    # Work with xarray DataArrays to handle arbitrary spatial dimensions
    energy_sc = map_ds["energy_sc"]

    # Step 1: Find bounding ESA energy indices for each position
    # Use np.searchsorted on flattened array, then reshape back
    esa_energy_vals = esa_energies.values
    energy_sc_flat = energy_sc.values.ravel()

    # Find right bound index for each element (vectorized)
    right_idx_flat = np.searchsorted(esa_energy_vals, energy_sc_flat, side="right")
    right_idx_flat = np.clip(right_idx_flat, 1, len(esa_energy_vals) - 1)
    left_idx_flat = right_idx_flat - 1

    # Reshape indices back to match energy_sc shape
    right_idx = right_idx_flat.reshape(energy_sc.shape)
    left_idx = left_idx_flat.reshape(energy_sc.shape)

    # Create DataArrays for indices with same dims as energy_sc
    # Note: we need to avoid coordinate name conflicts when using isel()
    # The energy dimension should be present in dims but not as a coordinate
    # since we're using these as indices into the energy dimension
    # Create coordinates dict without the energy coordinate
    coords_without_energy = {k: v for k, v in energy_sc.coords.items() if k != "energy"}

    right_idx_da = xr.DataArray(
        right_idx, dims=energy_sc.dims, coords=coords_without_energy
    )
    left_idx_da = xr.DataArray(
        left_idx, dims=energy_sc.dims, coords=coords_without_energy
    )

    # Get energy values at boundaries - select from esa_energies using indices
    energy_left = esa_energies.isel({"energy": left_idx_da})
    energy_right = esa_energies.isel({"energy": right_idx_da})

    for var_name in vars_to_interpolate:
        logger.debug(
            f"Interpolating {var_name}, {var_name}_stat_uncert, and "
            f"{var_name}_sys_err to heliocentric frame energies"
        )
        # Step 2: Extract flux values at bounding energy channels
        # Use xarray's advanced indexing to get fluxes at left and right indices
        intensity = map_ds[var_name]
        stat_unc = map_ds[f"{var_name}_stat_uncert"]
        sys_err = map_ds[f"{var_name}_sys_err"]
        flux_left = intensity.isel({"energy": left_idx_da})
        flux_right = intensity.isel({"energy": right_idx_da})
        stat_unc_left = stat_unc.isel({"energy": left_idx_da})
        stat_unc_right = stat_unc.isel({"energy": right_idx_da})
        sys_err_left = sys_err.isel({"energy": left_idx_da})

        # Step 3: Perform power-law interpolation to spacecraft energy
        # slope = log(f_right/f_left) / log(e_right/e_left)
        # flux_sc = f_left * (energy_sc / e_left)^slope
        with np.errstate(divide="ignore", invalid="ignore"):
            # Calculate slope for power-law interpolation
            slope = np.log(flux_right / flux_left) / np.log(energy_right / energy_left)

            # Interpolate flux using power-law
            flux_sc = flux_left * ((energy_sc / energy_left) ** slope)

            # Interpolation factor for uncertainty propagation (Equations 75 & 76)
            unc_factor = np.log(energy_sc / energy_left) / np.log(
                energy_right / energy_left
            )

            # Statistical uncertainty propagation (Equation 75):
            # δJ = J * sqrt((δJ_left/J_left)^2 * (1 + unc_factor^2)
            #               + unc_factor^2 * (δJ_right/J_right)^2)
            stat_unc_sc = flux_sc * np.sqrt(
                (stat_unc_left / flux_left) ** 2 * (1.0 + unc_factor**2)
                + unc_factor**2 * (stat_unc_right / flux_right) ** 2
            )

            # Systematic uncertainty propagation (Equation 76):
            # σJ^g = σJ^src_kref * (⟨E^s_kref⟩ / E^ESA_kref)^γ_kref * (E^h / ⟨E^s_kref⟩)
            # Systematic error scales proportionally with flux during power-law
            # interpolation
            sys_err_sc = sys_err_left * ((energy_sc / energy_left) ** slope)

        # Step 4: Energy scaling transformation (Liouville theorem)
        # flux_helio = flux_sc * (helio_energy / energy_sc)
        # Using xarray broadcasting, helio_energies will broadcast
        # along esa_energy_step
        with np.errstate(divide="ignore", invalid="ignore"):
            energy_ratio = helio_energies / energy_sc
            flux_helio = flux_sc * energy_ratio
            stat_unc_helio = stat_unc_sc * energy_ratio
            sys_err_helio = sys_err_sc * energy_ratio

        # Set any location where the value is not finite to NaN (converts +/-inf to NaN)
        flux_helio = flux_helio.where(np.isfinite(flux_helio), np.nan)
        stat_unc_helio = stat_unc_helio.where(np.isfinite(stat_unc_helio), np.nan)
        sys_err_helio = sys_err_helio.where(np.isfinite(sys_err_helio), np.nan)

        # Update the dataset with interpolated values
        map_ds[var_name] = flux_helio
        map_ds[f"{var_name}_stat_uncert"] = stat_unc_helio
        map_ds[f"{var_name}_sys_err"] = sys_err_helio

    return map_ds


def get_pset_directional_mask(
    pset_ds: xr.Dataset, direction: str
) -> xr.DataArray | None:
    """
    Get the boolean mask appropriate for the indicated ena direction.

    Parameters
    ----------
    pset_ds : xarray.Dataset
        PSET dataset. If spin_phase is "ram" or "anti", the dataset must contain
        a "ram_mask" variable.
    direction : str
        Map spin phase. Must be "ram", "anti", or "full".

    Returns
    -------
    pset_bin_mask : xarray.DataArray | None
        Boolean mask indicating which bins are in the indicated direction. If
        direction is "full", then None is returned.
    """
    if direction not in ["ram", "anti", "full"]:
        raise ValueError(
            f"Invalid direction string: {direction}. Must be 'ram', 'anti', or 'full'."
        )
    # Set the mask used to filter ram/anti-ram pixels
    pset_valid_mask = None  # Default to no mask (full spin)
    if direction == "ram":
        pset_valid_mask = pset_ds["ram_mask"]
        logger.debug(
            f"Using ram mask with shape: {pset_valid_mask.shape} "
            f"containing {np.prod(pset_valid_mask.shape)} pixels,"
            f"{np.sum(pset_valid_mask.values)} of which are True."
        )
    elif direction == "anti":
        pset_valid_mask = ~pset_ds["ram_mask"]
        logger.debug(
            f"Using anti-ram mask with shape: {pset_valid_mask.shape} "
            f"containing {np.prod(pset_valid_mask.shape)} pixels,"
            f"{np.sum(pset_valid_mask.values)} of which are True."
        )
    return pset_valid_mask
