"""L2 corrections common to multiple IMAP ENA instruments."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial
from scipy.constants import electron_volt, erg, proton_mass

from imap_processing.ena_maps.ena_maps import LoHiBasePointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.time import ttj2000ns_to_et

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
            Energy levels.
        gamma : np.ndarray
            Power-law slopes.

        Returns
        -------
        np.ndarray
            ESA transmission scale factors.
        """
        k = np.atleast_1d(k)
        gamma = np.atleast_1d(gamma)
        eta = np.empty_like(gamma)
        for i, esa_step in enumerate(k):
            eta[i] = self.polynomial_lookup[esa_step](gamma[i])
            # Negative transmissions get set to 1
            if eta[i] < 0:
                eta[i] = 1

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
            Array of differential fluxes [J_1, J_2, ..., J_7].
        energies : np.ndarray
            Array of energy levels [E_1, E_2, ..., E_7].
        uncertainties : np.ndarray, optional
            Array of flux uncertainties [δJ_1, δJ_2, ..., δJ_7].

        Returns
        -------
        gamma : np.ndarray
            Array of power-law slopes.
        delta_gamma : np.ndarray or None
            Array of uncertainty slopes (if uncertainties provided).
        """
        n_levels = len(fluxes)
        gamma = np.full(n_levels, 0, dtype=float)
        delta_gamma = (
            np.full(n_levels, 0, dtype=float) if uncertainties is not None else None
        )

        # Create an array of indices that can be used to create a padded array where
        # the padding duplicates the first element on the front and the last element
        # on the end of the array
        extended_inds = np.pad(np.arange(n_levels), 1, mode="edge")

        # Compute logs, setting non-positive fluxes to NaN
        log_fluxes = np.log(np.where(fluxes > 0, fluxes, np.nan))
        log_energies = np.log(energies)
        # Create extended arrays by repeating first and last values. This allows
        # for linear differencing to be used on the ends and central differencing
        # to be used on the interior of the array with a single vectorized equation.
        # Interior points use central differencing equation:
        #     gamma_k = ln(J_{k+1}/J_{k-1}) / ln(E_{k+1}/E_{k-1})
        # Left boundary uses linear forward differencing:
        #     gamma_k = ln(J_{k+1}/J_{k}) / ln(E_{k+1}/E_{k})
        # Right boundary uses linear backward differencing:
        #     gamma_k = ln(J_{k}/J_{k-1}) / ln(E_{k}/E_{k-1})
        log_extended_fluxes = log_fluxes[extended_inds]
        log_extended_energies = log_energies[extended_inds]

        # Extract the left and right log values to use in slope calculation
        left_log_fluxes = log_extended_fluxes[:-2]  # indices 0 to n_levels-1
        right_log_fluxes = log_extended_fluxes[2:]  # indices 2 to n_levels+1
        left_log_energies = log_extended_energies[:-2]
        right_log_energies = log_extended_energies[2:]

        # Compute power-law slopes for valid indices
        central_valid = np.isfinite(left_log_fluxes) & np.isfinite(right_log_fluxes)
        gamma[central_valid] = (
            (right_log_fluxes - left_log_fluxes)
            / (right_log_energies - left_log_energies)
        )[central_valid]

        # Compute uncertainty slopes
        if uncertainties is not None:
            with np.errstate(divide="ignore"):
                rel_unc_sq = (uncertainties / fluxes) ** 2
            extended_rel_unc_sq = rel_unc_sq[extended_inds]
            delta_gamma = np.sqrt(
                extended_rel_unc_sq[:-2] + extended_rel_unc_sq[2:]
            ) / (log_extended_energies[2:] - log_extended_energies[:-2])
            delta_gamma[~central_valid] = 0

        # Handle one-sided differencing for points where central differencing failed
        need_fallback = ~central_valid & np.isfinite(log_fluxes)
        # Exclude first and last points since they already use the correct
        # one-sided differencing
        interior_fallback = np.zeros_like(need_fallback, dtype=bool)
        interior_fallback[1:-1] = need_fallback[1:-1]

        if np.any(interior_fallback):
            indices = np.where(interior_fallback)[0]

            for k in indices:
                # For interior points: try forward first, then backward
                if k < n_levels - 1 and np.isfinite(log_fluxes[k + 1]):
                    gamma[k] = (log_fluxes[k + 1] - log_fluxes[k]) / (
                        log_energies[k + 1] - log_energies[k]
                    )

                    # Compute uncertainty slope using same differencing
                    if isinstance(delta_gamma, np.ndarray):
                        delta_gamma[k] = np.sqrt(rel_unc_sq[k + 1] + rel_unc_sq[k]) / (
                            log_energies[k + 1] - log_energies[k]
                        )

                elif k > 0 and np.isfinite(log_fluxes[k - 1]):
                    gamma[k] = (log_fluxes[k] - log_fluxes[k - 1]) / (
                        log_energies[k] - log_energies[k - 1]
                    )

                    # Compute uncertainty slope using same differencing
                    if isinstance(delta_gamma, np.ndarray):
                        delta_gamma[k] = np.sqrt(rel_unc_sq[k] + rel_unc_sq[k - 1]) / (
                            log_energies[k] - log_energies[k - 1]
                        )

        return gamma, delta_gamma

    def predictor_corrector_iteration(
        self,
        observed_fluxes: np.ndarray,
        observed_uncertainties: np.ndarray,
        energies: np.ndarray,
        max_iterations: int = 20,
        convergence_threshold: float = 0.005,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Estimate source fluxes using iterative predictor-corrector scheme.

        Implements the algorithm from Appendix A of the Mapping Algorithm Document.

        Parameters
        ----------
        observed_fluxes : np.ndarray
            Array of observed fluxes.
        observed_uncertainties : numpy.ndarray
            Array of observed uncertainties.
        energies : np.ndarray
            Array of energy levels.
        max_iterations : int, optional
            Maximum number of iterations, by default 20.
        convergence_threshold : float, optional
            RMS convergence criterion, by default 0.005 (0.5%).

        Returns
        -------
        source_fluxes : np.ndarray
            Final estimate of source fluxes.
        source_uncertainties : np.ndarray
            Final estimate of source uncertainties.
        n_iterations : int
            Number of iterations run.
        """
        n_levels = len(observed_fluxes)
        energy_levels = np.arange(n_levels) + 1

        # Initial power-law estimate from observed fluxes
        gamma_initial, _ = self.estimate_power_law_slope(observed_fluxes, energies)

        # Initial source flux estimate
        eta_initial = self.eta_esa(energy_levels, gamma_initial)
        source_fluxes_n = observed_fluxes / eta_initial

        for _iteration in range(max_iterations):
            # Store previous iteration
            source_fluxes_prev = source_fluxes_n.copy()

            # Predictor step
            gamma_pred, _ = self.estimate_power_law_slope(source_fluxes_n, energies)
            gamma_half = 0.5 * (gamma_initial + gamma_pred)

            # Predictor source flux estimate
            eta_half = self.eta_esa(energy_levels, gamma_half)
            source_fluxes_half = observed_fluxes / eta_half

            # Corrector step
            gamma_corr, _ = self.estimate_power_law_slope(source_fluxes_half, energies)
            gamma_n = 0.5 * (gamma_pred + gamma_corr)

            # Final source flux estimate for this iteration
            eta_final = self.eta_esa(energy_levels, gamma_n)
            source_fluxes_n = observed_fluxes / eta_final
            source_uncertainties = observed_uncertainties / eta_final

            # Check convergence
            ratios_sq = (source_fluxes_n / source_fluxes_prev) ** 2
            chi_n = np.sqrt(np.mean(ratios_sq)) - 1

            if chi_n < convergence_threshold:
                break

        return source_fluxes_n, source_uncertainties, _iteration + 1

    def apply_flux_correction(
        self, flux: np.ndarray, flux_stat_unc: np.ndarray, energies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply flux correction to observed fluxes.

        Iterative predictor-corrector scheme is run on each spatial pixel
        individually to correct fluxes and statistical uncertainties. This method
        is intended to be used with the unwrapped data in the ena_maps.AbstractSkyMap
        class or child classes.

        Parameters
        ----------
        flux : numpy.ndarray
            Input flux with shape (n_energy, n_spatial_pixels).
        flux_stat_unc : np.ndarray
            Statistical uncertainty for input fluxes. Shape must match the shape
            of flux.
        energies : numpy.ndarray
            Array of energy levels in units of eV or keV.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Corrected fluxes and flux uncertainties.
        """
        corrected_flux = np.empty_like(flux)
        corrected_flux_stat_unc = np.empty_like(flux_stat_unc)

        # loop over spatial pixels (last dimension)
        for i_pixel in range(flux.shape[-1]):
            corrected_flux[:, i_pixel], corrected_flux_stat_unc[:, i_pixel], _ = (
                self.predictor_corrector_iteration(
                    flux[:, i_pixel], flux_stat_unc[:, i_pixel], energies
                )
            )

        return corrected_flux, corrected_flux_stat_unc


def _add_spacecraft_velocity_to_pset(pset: LoHiBasePointingSet) -> None:
    """
    Calculate and add spacecraft velocity data to pointing set.

    Parameters
    ----------
    pset : LoHiBasePointingSet
        Pointing set object to be updated.

    Notes
    -----
    Adds the following DataArrays to pset.data:
    - "sc_velocity": Spacecraft velocity vector (km/s) with dims ["x_y_z"]
    - "sc_direction_vector": Spacecraft velocity unit vector with dims ["x_y_z"]
    """
    # Compute ephemeris time (J2000 seconds) of PSET midpoint time
    # TODO: Use the Pointing midpoint time. Epoch should be start time
    #     but use it until we can make Lo and Hi PSETs have a consistent
    #     variable to hold the midpoint time.
    et = ttj2000ns_to_et(pset.data["epoch"].values[0])
    # Get spacecraft state in HAE frame
    sc_state = geometry.imap_state(et, ref_frame=geometry.SpiceFrame.IMAP_HAE)
    sc_velocity_vector = sc_state[3:6]

    # Store spacecraft velocity as DataArray
    pset.data["sc_velocity"] = xr.DataArray(
        sc_velocity_vector, dims=[CoordNames.CARTESIAN_VECTOR.value]
    )

    # Calculate spacecraft speed and direction
    sc_velocity_km_per_sec = np.linalg.norm(
        pset.data["sc_velocity"], axis=-1, keepdims=True
    )
    pset.data["sc_direction_vector"] = pset.data["sc_velocity"] / sc_velocity_km_per_sec


def _add_cartesian_look_direction(pset: LoHiBasePointingSet) -> None:
    """
    Calculate and add look direction vectors to pointing set.

    Parameters
    ----------
    pset : LoHiBasePointingSet
        Pointing set object to be updated.

    Notes
    -----
    Adds the following DataArray to pset.data:
    - "look_direction": Cartesian unit vectors with dims [...spatial_dims, "x_y_z"]
    """
    longitudes = pset.data["hae_longitude"]
    latitudes = pset.data["hae_latitude"]

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
    pset.data["look_direction"] = xr.DataArray(
        geometry.spherical_to_cartesian(spherical_coords),
        dims=[*longitudes.dims, CoordNames.CARTESIAN_VECTOR.value],
    )


def _calculate_compton_getting_transform(
    pset: LoHiBasePointingSet,
    energy_hf: xr.DataArray,
) -> None:
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
    pset : LoHiBasePointingSet
        Pointing set object with sc_velocity, sc_direction_vector, and
        look_direction already added.
    energy_hf : xr.DataArray
        ENA energies in the heliosphere frame in eV.

    Notes
    -----
    The algorithm is based on the "Appendix A. The IMAP-Lo Mapping Algorithms"
    document.
    Adds the following DataArrays to pset.data:
    - "energy_sc": ENA energies in spacecraft frame (eV)
    - "ena_source_hae_longitude": ENA source longitudes in heliosphere frame (degrees)
    - "ena_source_hae_latitude": ENA source latitudes in heliosphere frame (degrees)
    """
    # Store heliosphere frame energies
    pset.data["energy_hf"] = energy_hf

    # Calculate spacecraft speed
    sc_velocity_km_per_sec = np.linalg.norm(
        pset.data["sc_velocity"], axis=-1, keepdims=True
    )

    # Calculate dot product between look directions and spacecraft direction vector
    # Use Einstein summation for efficient vectorized dot product
    dot_product = xr.DataArray(
        np.einsum(
            "...i,...i->...",
            pset.data["look_direction"],
            pset.data["sc_direction_vector"],
        ),
        dims=pset.data["look_direction"].dims[:-1],
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
    y = np.sqrt(pset.data["energy_hf"] / energy_u)

    # Velocity magnitude factor calculation (Equation 62)
    # x_k = (êₛ · û_sc) + sqrt(y² + (êₛ · û_sc)² - 1)
    x = dot_product + np.sqrt(y**2 + dot_product**2 - 1)

    # Calculate ENA speed in the spacecraft frame
    # |v⃗_sc| = x_k * U_sc
    velocity_sc = x * sc_velocity_km_per_sec

    # Calculate the kinetic energy in the spacecraft frame
    # E_sc = (1/2) * M_p * v_sc² (convert km/s to cm/s with 1.0e5 factor)
    pset.data["energy_sc"] = (
        0.5 * PROTON_MASS_GRAMS * (velocity_sc * 1e5) ** 2 / ERG_PER_EV
    )

    # Calculate the velocity vector in the spacecraft frame
    # v⃗_sc = |v_sc| * êₛ (velocity direction follows look direction)
    velocity_vector_sc = velocity_sc * pset.data["look_direction"]

    # Calculate the ENA velocity vector in the heliosphere frame
    # v⃗_helio = v⃗_sc - U⃗_sc (simple velocity addition)
    velocity_vector_helio = velocity_vector_sc - pset.data["sc_velocity"]

    # Convert to spherical coordinates to get ENA source directions
    ena_source_direction_helio = geometry.cartesian_to_spherical(
        velocity_vector_helio.data
    )

    # Update the PSET hae_longitude and hae_latitude variables with the new
    # energy-dependent values.
    pset.data["hae_longitude"] = (
        pset.data["energy_sc"].dims,
        ena_source_direction_helio[..., 1],
    )
    pset.data["hae_latitude"] = (
        pset.data["energy_sc"].dims,
        ena_source_direction_helio[..., 2],
    )

    # For ram/anti-ram filtering we can use the sign of the scalar projection
    # of the ENA source direction onto the spacecraft velocity vector.
    # ram_mask = (v⃗_helio · û_sc) >= 0
    ram_mask = (
        np.einsum(
            "...i,...i->...", velocity_vector_helio, pset.data["sc_direction_vector"]
        )
        >= 0
    )
    pset.data["ram_mask"] = xr.DataArray(
        ram_mask,
        dims=velocity_vector_helio.dims[:-1],
    )


def apply_compton_getting_correction(
    pset: LoHiBasePointingSet,
    energy_hf: xr.DataArray,
) -> None:
    """
    Apply Compton-Getting correction to a pointing set and update coordinates.

    This function performs the Compton-Getting velocity transformation to correct
    ENA observations for the motion of the spacecraft through the heliosphere.
    The corrected coordinates represent the true source directions of the ENAs
    in the heliosphere frame.

    The pointing set is modified in-place: new variables are added to the dataset
    for the corrected coordinates and energies, and the az_el_points attribute
    is updated to use the corrected coordinates for binning.

    All calculations are performed using xarray DataArrays to preserve dimension
    information throughout the computation.

    Parameters
    ----------
    pset : LoHiBasePointingSet
        Pointing set object containing HAE longitude/latitude coordinates.
    energy_hf : xr.DataArray
        ENA energies in the heliosphere frame in eV. Must be 1D with an
        energy dimension.

    Notes
    -----
    This function adds the following variables to the pointing set dataset:
    - "sc_velocity": Spacecraft velocity vector (km/s)
    - "sc_direction_vector": Spacecraft velocity unit vector
    - "look_direction": Cartesian unit vectors of observation directions
    - "energy_hf": ENA energies in heliosphere frame (eV)
    - "energy_sc": ENA energies in spacecraft frame (eV)
    - "ena_source_hae_longitude": ENA source longitudes in heliosphere frame (degrees)
    - "ena_source_hae_latitude": ENA source latitudes in heliosphere frame (degrees)

    The az_el_points attribute is updated to use the corrected coordinates,
    which will be used for subsequent binning operations.
    """
    # Step 1: Add spacecraft velocity and direction to pset
    _add_spacecraft_velocity_to_pset(pset)

    # Step 2: Calculate and add look direction vectors to pset
    _add_cartesian_look_direction(pset)

    # Step 3: Apply Compton-Getting transformation
    _calculate_compton_getting_transform(pset, energy_hf)

    # Step 4: Update az_el_points to use the corrected coordinates
    pset.update_az_el_points()
