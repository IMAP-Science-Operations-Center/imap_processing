"""Culls Events for ULTRA L1b."""

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import (
    ImapAttitudeUltraFlags,
    ImapDEScatteringUltraFlags,
    ImapHkUltraFlags,
    ImapInstrumentUltraFlags,
    ImapRatesUltraFlags,
)
from imap_processing.spice.spin import get_spin_data
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_scattering_coefficients,
    get_scattering_thresholds,
)
from imap_processing.ultra.l1b.quality_flag_filters import DE_QUALITY_FLAG_FILTERS
from imap_processing.ultra.l1b.ultra_l1b_extended import get_spin_info
from imap_processing.ultra.l1c.l1c_lookup_utils import build_energy_bins

logger = logging.getLogger(__name__)

SPIN_DURATION = 15  # Default spin duration in seconds.

RateResult = namedtuple(
    "RateResult",
    [
        "unique_spins",
        "start_per_spin",
        "stop_per_spin",
        "coin_per_spin",
        "start_pulses",
        "stop_pulses",
        "coin_pulses",
    ],
)


def get_energy_histogram(
    spin_number: NDArray, energy: NDArray
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Compute a 2D histogram of the counts binned by energy and spin number.

    Parameters
    ----------
    spin_number : NDArray
        Spin number.
    energy : NDArray
        The particle energy.

    Returns
    -------
    hist : NDArray
        A 2D histogram array containing the
        count rate per spin at each energy bin.
    spin_edges : NDArray
        Edges of the spin number bins.
    counts : NDArray
        A 2D histogram array containing the
        counts per spin at each energy bin.
    mean_duration : float
        Mean duration of the spin.
    """
    spin_df = get_spin_data()

    unique_spin_number = np.unique(spin_number)
    spin_edges = np.append(unique_spin_number, unique_spin_number.max() + 1)

    # Counts per spin at each energy bin.
    hist, _ = np.histogramdd(
        sample=(energy, spin_number),
        bins=[UltraConstants.CULLING_ENERGY_BIN_EDGES, spin_edges],
    )

    counts = hist.copy()
    total_spin_duration = 0

    # Count rate per spin at each energy bin.
    for i in range(hist.shape[1]):
        matched_spins = spin_df.spin_number == unique_spin_number[i]
        if not np.any(matched_spins):
            # TODO: we might throw an exception here instead.
            logger.info(f"Unmatched spin number: {unique_spin_number[i]}")
            spin_duration = SPIN_DURATION  # Default to 15 seconds if no match found
        else:
            spin_duration = spin_df.spin_period_sec[
                spin_df.spin_number == unique_spin_number[i]
            ].values[0]
        hist[:, i] /= spin_duration
        total_spin_duration += spin_duration

    mean_duration = total_spin_duration / hist.shape[1]

    return hist, spin_edges, counts, mean_duration


def flag_attitude(
    spin_number: NDArray, aux_dataset: xr.Dataset
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Flag data based on attitude.

    Parameters
    ----------
    spin_number : NDArray
        Spin number at each direct event.
    aux_dataset : xarray.Dataset
        Auxiliary dataset.

    Returns
    -------
    quality_flags : NDArray
        Quality flags.
    spin_rates : NDArray
        Spin rates.
    spin_period : NDArray
        Spin period.
    spin_starttime : NDArray
        Spin start time.
    """
    spins = np.unique(spin_number)  # Get unique spins
    spin_df = get_spin_data()  # Load spin data

    spin_period = spin_df.loc[spin_df.spin_number.isin(spins), "spin_period_sec"]
    spin_starttime = spin_df.loc[spin_df.spin_number.isin(spins), "spin_start_met"]
    spin_phase_valid = spin_df.loc[spin_df.spin_number.isin(spins), "spin_phase_valid"]
    spin_period_valid = spin_df.loc[
        spin_df.spin_number.isin(spins), "spin_period_valid"
    ]
    spin_rates = 60 / spin_period  # 60 seconds in a minute
    bad_spin_rate_indices = (spin_rates < UltraConstants.CULLING_RPM_MIN) | (
        spin_rates > UltraConstants.CULLING_RPM_MAX
    )

    quality_flags = np.full(
        spins.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )
    quality_flags[bad_spin_rate_indices] |= ImapAttitudeUltraFlags.SPINRATE.value
    mismatch_indices = compare_aux_univ_spin_table(aux_dataset, spins, spin_df)
    quality_flags[mismatch_indices] |= ImapAttitudeUltraFlags.AUXMISMATCH.value

    # Spin phase validity flag
    phase_invalid_indices = spin_phase_valid == 0
    quality_flags[phase_invalid_indices] |= ImapAttitudeUltraFlags.SPINPHASE.value

    # Spin period validity flag
    period_invalid_indices = ~spin_period_valid
    quality_flags[period_invalid_indices] |= ImapAttitudeUltraFlags.SPINPERIOD.value

    return quality_flags, spin_rates, spin_period, spin_starttime


def flag_hk(spin_number: NDArray) -> NDArray:
    """
    Flag data based on hk.

    Parameters
    ----------
    spin_number : NDArray
        Spin number at each direct event.

    Returns
    -------
    quality_flags : NDArray
        Quality flags..
    """
    spins = np.unique(spin_number)  # Get unique spins
    quality_flags = np.full(spins.shape, ImapHkUltraFlags.NONE.value, dtype=np.uint16)

    return quality_flags


def flag_imap_instruments(spin_number: NDArray) -> NDArray:
    """
    Flag data based on other IMAP instruments.

    Parameters
    ----------
    spin_number : NDArray
        Spin number at each direct event.

    Returns
    -------
    quality_flags : NDArray
        Quality flags..
    """
    spins = np.unique(spin_number)  # Get unique spins
    quality_flags = np.full(
        spins.shape, ImapInstrumentUltraFlags.NONE.value, dtype=np.uint16
    )

    return quality_flags


def get_n_sigma(count_rates: NDArray, mean_duration: float, sigma: int = 6) -> NDArray:
    """
    Calculate the threshold for the HIGHRATES flag.

    Parameters
    ----------
    count_rates : NDArray
        A 2D histogram array containing the
        count rates per spin at each energy bin.
    mean_duration : float
        Mean duration of the spins.
    sigma : int (default=6)
        The number of sigma.

    Returns
    -------
    threshold : NDArray
        Threshold for applying HIGHRATES flag.
    """
    # Take the Sample Standard Deviation.
    sigma_per_energy = np.std(count_rates, axis=1, ddof=1)
    n_sigma_per_energy = sigma * sigma_per_energy
    mean_per_energy = np.mean(count_rates, axis=1)
    # Must have a HIGHRATES threshold of at least 3 counts per spin.
    threshold = np.maximum(mean_per_energy + n_sigma_per_energy, 3 / mean_duration)

    return threshold


def flag_rates(
    spin_number: NDArray, energy: NDArray, sigma: int = 6
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Flag data based on counts and negative energies.

    Parameters
    ----------
    spin_number : NDArray
        Spin number at each direct event.
    energy : NDArray
        Energy data.
    sigma : int (default=6)
        The number of sigma.

    Returns
    -------
    quality_flags : NDArray
        Quality flags.
    spin : NDArray
        Spin data.
    energy_bin_geometric_mean : NDArray
        Energy bin geometric mean.
    n_sigma_per_energy_reshape : NDArray
        N sigma per energy.
    """
    count_rates, _spin_edges, _counts, duration = get_energy_histogram(
        spin_number, energy
    )
    quality_flags = np.full(
        count_rates.shape, ImapRatesUltraFlags.NONE.value, dtype=np.uint16
    )

    threshold = get_n_sigma(count_rates, duration, sigma=sigma)

    bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_bin_geometric_mean = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    spin = np.unique(spin_number)

    # Indices where the counts exceed the threshold
    indices_n_sigma = count_rates > threshold[:, np.newaxis]
    quality_flags[indices_n_sigma] |= ImapRatesUltraFlags.HIGHRATES.value

    # Flags the first and last spin
    quality_flags[:, 0] |= ImapRatesUltraFlags.FIRSTSPIN.value
    quality_flags[:, -1] |= ImapRatesUltraFlags.LASTSPIN.value

    return quality_flags, spin, energy_bin_geometric_mean, threshold


def compare_aux_univ_spin_table(
    aux_dataset: xr.Dataset, spins: NDArray, spin_df: pd.DataFrame
) -> NDArray:
    """
    Compare the auxiliary and Universal Spin Table.

    Parameters
    ----------
    aux_dataset : xarray.Dataset
        Auxiliary dataset.
    spins : np.ndarray
        Array of spin numbers to compare.
    spin_df : pd.DataFrame
        Universal Spin Table.

    Returns
    -------
    mismatch_indices : np.ndarray
        Boolean array indicating which spins have mismatches.
    """
    # Identify valid spin matches
    univ_spins = spin_df["spin_number"].values
    aux_spins = aux_dataset["spinnumber"].values
    present_in_both = np.intersect1d(univ_spins, aux_spins)

    # Filter and align by spin number
    df_univ = spin_df.set_index("spin_number").loc[present_in_both]
    df_aux = (
        pd.DataFrame({field: aux_dataset[field].values for field in aux_dataset})
        .groupby("spinnumber", as_index=True)
        .first()
        .loc[present_in_both]
    )

    mismatch_indices = np.zeros(len(spins), dtype=bool)

    fields_to_compare = [
        ("timespinstart", "spin_start_sec_sclk"),
        ("timespinstartsub", "spin_start_subsec_sclk"),
        ("duration", "spin_period_sec"),
        ("timespindata", "spin_start_met"),
        ("spinperiod", "spin_period_sec"),
    ]

    # Compare fields
    mismatch = np.zeros(len(df_aux), dtype=bool)
    for aux_field, spin_field in fields_to_compare:
        mismatch |= df_aux[aux_field].values != df_univ[spin_field].values

    # Get spin numbers where mismatch is True
    mismatched_spin_numbers = present_in_both[mismatch]
    # Find indices in `spins` that correspond to these mismatched spins
    mismatch_indices[np.isin(spins, mismatched_spin_numbers)] = True

    # Also flag any spins not present in the intersection
    missing_spin_mask = ~np.isin(spins, present_in_both)
    mismatch_indices[missing_spin_mask] = True

    return mismatch_indices


def get_pulses_per_spin(aux: xr.Dataset, rates: xr.Dataset) -> RateResult:
    """
    Get the total number of pulses per spin.

    Parameters
    ----------
    aux : xr.Dataset
        Auxiliary dataset.
    rates : xr.Dataset
        Rates dataset.

    Returns
    -------
    unique_spins : NDArray
        Unique spin numbers.
    start_per_spin : NDArray
        Total start pulses per spin.
    stop_per_spin : NDArray
        Total stop pulses per spin.
    coin_per_spin : NDArray
        Total coincidence pulses per spin.
    start_pulses : NDArray
        Total start pulses.
    stop_pulses : NDArray
        Total stop pulses.
    coin_pulses : NDArray
        Total coincidence pulses.
    """
    spin_ds = get_spin_info(aux, rates["shcoarse"].values)
    spin_number = spin_ds["spin_number"].values

    # Top coin pulses
    top_coin_pulses = np.stack(
        [v for k, v in rates.items() if k.startswith("coin_t")], axis=1
    )
    max_top_coin_pulse = np.max(top_coin_pulses, axis=1)

    # Bottom coin pulses
    bottom_coin_pulses = np.stack(
        [v for k, v in rates.items() if k.startswith("coin_b")], axis=1
    )
    max_bottom_coin_pulse = np.max(bottom_coin_pulses, axis=1)

    # Top stop pulses
    top_stop_pulses = np.stack(
        [v for k, v in rates.items() if k.startswith("stop_t")], axis=1
    )
    max_top_stop_pulse = np.max(top_stop_pulses, axis=1)

    # Bottom stop pulses
    bottom_stop_pulses = np.stack(
        [v for k, v in rates.items() if k.startswith("stop_b")], axis=1
    )
    max_bottom_stop_pulse = np.max(bottom_stop_pulses, axis=1)

    stop_pulses = max_top_stop_pulse + max_bottom_stop_pulse
    start_pulses = rates["start_rf"] + rates["start_lf"]
    coin_pulses = max_top_coin_pulse + max_bottom_coin_pulse

    unique_spins, spin_idx = np.unique(spin_number, return_inverse=True)

    start_per_spin = np.bincount(spin_idx, weights=start_pulses)
    stop_per_spin = np.bincount(spin_idx, weights=stop_pulses)
    coin_per_spin = np.bincount(spin_idx, weights=coin_pulses)

    return RateResult(
        unique_spins=unique_spins,
        start_per_spin=start_per_spin,
        stop_per_spin=stop_per_spin,
        coin_per_spin=coin_per_spin,
        start_pulses=start_pulses,
        stop_pulses=stop_pulses,
        coin_pulses=coin_pulses,
    )


def flag_scattering(
    tof_energy: NDArray,
    theta: NDArray,
    phi: NDArray,
    ancillary_files: dict,
    sensor: str,
    quality_flags: NDArray,
) -> None:
    """
    Flag events where either theta or phi FWHM exceed the threshold or equal nan.

    Parameters
    ----------
    tof_energy : NDArray
        TOF energy for each event in keV.
    theta : NDArray
        Elevation angles in degrees.
    phi : NDArray
        Azimuth angles in degrees.
    ancillary_files : dict[Path]
        Ancillary files.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    quality_flags : NDArray
        Quality flags.
    """
    scattering_thresholds = get_scattering_thresholds(ancillary_files)
    _, _, energy_bin_geometric_means = build_energy_bins()
    energy_bin_inds = np.digitize(tof_energy, UltraConstants.PSET_ENERGY_BIN_EDGES)
    # Clip indices to valid range (events outside the energy bins get assigned
    # to the nearest bin. These events have already been flagged and
    # will be ignored in l1c)
    energy_bin_inds = np.clip(energy_bin_inds, 1, len(energy_bin_geometric_means))
    energy_geom_means = energy_bin_geometric_means[energy_bin_inds - 1]
    for (e_min, e_max), threshold in scattering_thresholds.items():
        event_mask = (tof_energy >= e_min) & (tof_energy < e_max)
        # Input the theta and phi values for the current energy range.
        # Returns a_theta_val, g_theta_val, a_phi_val, g_phi_val
        theta_coeffs, phi_coeffs = get_scattering_coefficients(
            theta[event_mask],
            phi[event_mask],
            lookup_tables=None,
            ancillary_files=ancillary_files,
            instrument_id=int(sensor[-2:]),
        )
        # FWHM_PHI = A_PHI * E^G_PHI
        # FWHM_THETA = A_THETA * E^G_THETA
        # Use the geometric mean of the energy bin for the scattering check
        fwhm_theta = (
            theta_coeffs[:, 0] * energy_geom_means[event_mask] ** theta_coeffs[:, 1]
        )
        fwhm_phi = phi_coeffs[:, 0] * energy_geom_means[event_mask] ** phi_coeffs[:, 1]
        is_nan = np.isnan(fwhm_theta) | np.isnan(fwhm_phi)
        quality_flags[np.where(event_mask)[0][is_nan]] |= (
            ImapDEScatteringUltraFlags.NAN_PHI_OR_THETA.value
        )

        theta_exceeds = fwhm_theta > threshold
        phi_exceeds = fwhm_phi > threshold
        either_exceeds = theta_exceeds | phi_exceeds

        # Set flags for events where either theta or phi FWHM exceed the threshold
        quality_flags[np.where(event_mask)[0][either_exceeds]] |= (
            ImapDEScatteringUltraFlags.ABOVE_THRESHOLD.value
        )


def get_de_rejection_mask(
    quality_scattering: NDArray,
    quality_outliers: NDArray,
    reject_scattering: bool = True,
) -> NDArray:
    """
    Create boolean mask where event is rejected due to relevant flags.

    Parameters
    ----------
    quality_scattering : NDArray
        Quality scattering flags.
    quality_outliers : NDArray
        Quality outliers flags.
    reject_scattering : bool
        Whether to reject based on scattering flags.

    Returns
    -------
    rejected : NDArray
        Rejected events where True = rejected.
    """
    # Bitmasks from the DE_QUALITY_FLAG_FILTERS
    scattering_mask = sum(
        flag.value for flag in DE_QUALITY_FLAG_FILTERS["quality_scattering"]
    )
    outliers_mask = sum(
        flag.value for flag in DE_QUALITY_FLAG_FILTERS["quality_outliers"]
    )
    if reject_scattering:
        # Boolean mask where event is rejected due to relevant flags
        rejected = ((quality_scattering & scattering_mask) != 0) | (
            (quality_outliers & outliers_mask) != 0
        )
    else:
        rejected = (quality_outliers & outliers_mask) != 0

    return rejected


def count_rejected_events_per_spin(
    spins: NDArray, quality_scattering: NDArray, quality_outliers: NDArray
) -> NDArray:
    """
    Count rejected events per spin based on DE_QUALITY_FLAG_FILTERS.

    Parameters
    ----------
    spins : NDArray
        Spins in which each direct event is within.
    quality_scattering : NDArray
        Quality scattering flags.
    quality_outliers : NDArray
        Quality outliers flags.

    Returns
    -------
    rejected_counts : NDArray
        Rejected counts per spin.
    """
    # Boolean mask where event is rejected due to relevant flags
    rejected = get_de_rejection_mask(quality_scattering, quality_outliers)

    # Unique spin numbers
    unique_spins = np.unique(spins)

    # Count rejected events per spin
    rejected_counts = np.array(
        [np.count_nonzero(rejected[spins == spin]) for spin in unique_spins], dtype=int
    )

    return rejected_counts
