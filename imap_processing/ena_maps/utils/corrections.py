"""L2 corrections common to multiple IMAP ENA instruments."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


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
