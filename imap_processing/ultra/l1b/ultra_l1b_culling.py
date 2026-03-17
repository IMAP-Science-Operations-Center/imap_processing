"""Culls Events for ULTRA L1b."""

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import spiceypy as sp
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from imap_processing.quality_flags import (
    ImapAttitudeUltraFlags,
    ImapDEScatteringUltraFlags,
    ImapHkUltraFlags,
    ImapInstrumentUltraFlags,
    ImapRatesUltraFlags,
)
from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
)
from imap_processing.spice.spin import get_spin_data
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_scattering_coefficients,
    get_scattering_thresholds,
)
from imap_processing.ultra.l1b.quality_flag_filters import (
    DE_QUALITY_FLAG_FILTERS,
    ENERGY_DEPENDENT_SPIN_QUALITY_FLAG_FILTERS,
)
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

    spin_df = spin_df[spin_df.spin_number.isin(spins)]
    spin_period = spin_df["spin_period_sec"].values
    spin_starttime = spin_df["spin_start_met"].values
    spin_phase_valid = spin_df["spin_phase_valid"].values
    spin_period_valid = spin_df["spin_period_valid"].values
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


def get_energy_and_spin_dependent_rejection_mask(
    goodtimes_dataset: xr.Dataset,
    energy: np.ndarray,
    spin_number: np.ndarray,
) -> NDArray:
    """
    Create boolean mask where event is rejected due to relevant flags.

    Parameters
    ----------
    goodtimes_dataset : xr.Dataset
        Dataset containing valid spins and energy bin flags.
    energy : np.ndarray
        The particle energy at each direct event.
    spin_number : np.ndarray
        Spin number at each direct event.

    Returns
    -------
    rejected : NDArray
        Rejected events where True = rejected.
    """
    # Get the ebin flags for each energy bin from the goodtimes dataset.
    energy_range_edges = goodtimes_dataset["energy_range_edges"].values
    # Filter out fill values from energy_range_edges (negative or zero)
    energy_range_edges = energy_range_edges[energy_range_edges > 0]
    # Get the quality flag arrays "turned on" for energy dependent culling from the
    # goodtimes dataset.
    flag_arrays = [
        goodtimes_dataset[flag_name].values
        for flag_name in ENERGY_DEPENDENT_SPIN_QUALITY_FLAG_FILTERS
    ]
    # Initialize all events to not rejected
    rejected = np.zeros_like(energy, dtype=bool)
    ebin_flags = goodtimes_dataset["energy_range_flags"].values
    # Filter out fill values (0s) from energy_range_flags
    ebin_flags = ebin_flags[ebin_flags > 0]
    # Get the index of the spin number in the goodtimes dataset for each event
    # all spin numbers should be present in the goodtimes dataset since we have already
    # filtered any events that are not
    spin_idx = np.searchsorted(goodtimes_dataset.spin_number, spin_number)
    event_energy_bins: NDArray = (np.digitize(energy, energy_range_edges) - 1).astype(
        np.intp
    )
    in_valid_bin = (event_energy_bins >= 0) & (event_energy_bins < len(ebin_flags))
    # get the flags for each event
    event_flags = np.zeros_like(energy, dtype=np.uint16)
    event_flags[in_valid_bin] = ebin_flags[event_energy_bins[in_valid_bin]]
    for qf_array in flag_arrays:
        # select the quality flag for each event
        quality_flags_at_events = qf_array[spin_idx]
        # If that flag is "turned on" for the spin of that event, and the event is in
        # an energy bin that is flagged for culling, then we reject that event.
        rejected |= quality_flags_at_events & event_flags > 0

    logger.info(
        "Rejected %d events based on energy and spin dependent flags.", np.sum(rejected)
    )

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


def flag_low_voltage(
    spin_tbin_edges: NDArray,
    status_dataset: xr.Dataset,
    voltage_threshold: float = UltraConstants.LOW_VOLTAGE_CULL_THRESHOLD,
) -> NDArray:
    """
    Flag low voltage events.

    Parameters
    ----------
    spin_tbin_edges : NDArray
        Edges of the spin time bins.
    status_dataset : xarray.Dataset
        Status dataset containing voltage information.
    voltage_threshold : float
        Voltage threshold below which to flag low voltage events.

    Returns
    -------
    quality_flags : NDArray
        Boolean quality flags shaped (n_spin_bins,).
    """
    spin_bin_size = len(spin_tbin_edges) - 1
    # initialize all spins to have no low voltage flag
    quality_flags = np.zeros(spin_bin_size, dtype=bool)
    # Get the min voltage across both deflection plate at each epoch
    min_voltage = np.minimum(
        status_dataset["rightdeflection_v"].data,
        status_dataset["leftdeflection_v"].data,
    )
    # Get the indices where the min voltage is below the threshold
    low_voltage_inds = np.nonzero(min_voltage < voltage_threshold)[0]

    if not low_voltage_inds.size:
        return quality_flags

    low_voltage_times = status_dataset["shcoarse"].data[low_voltage_inds]
    # For each low voltage time, find the corresponding spin time
    lv_spin_inds = np.atleast_1d(
        np.searchsorted(spin_tbin_edges, low_voltage_times, side="right") - 1
    )
    # Ensure that the indices are within the valid range of spin groups
    valid_bin_inds = (lv_spin_inds >= 0) & (lv_spin_inds < spin_bin_size)
    lv_spin_inds = lv_spin_inds[valid_bin_inds]
    # For each low voltage ind, flag the corresponding flag
    quality_flags[lv_spin_inds] = True

    num_culled: int = np.sum(quality_flags)
    logger.info(
        f"Low voltage culling removed {num_culled} spin bins across all energy "
        f"channels. Voltage threshold: {voltage_threshold} V."
    )

    return quality_flags


def flag_high_energy(
    de_dataset: xr.Dataset,
    spin_tbin_edges: NDArray,
    energy_ranges: NDArray,
    mask: NDArray = None,
    energy_thresholds: np.ndarray = UltraConstants.HIGH_ENERGY_CULL_THRESHOLDS,
    sensor_id: int = 90,
) -> NDArray:
    """
    Flag high energy events.

    Parameters
    ----------
    de_dataset : xr.Dataset
        Direct event dataset.
    spin_tbin_edges : NDArray
        Edges of the spin time bins.
    energy_ranges : numpy.ndarray
        Array of energy range edges.
    mask : numpy.ndarray, optional
        Mask indicating which events to consider for high energy flagging
         (e.g., after low voltage culling). True indicates the spin bins that should
         NOT be considered for high energy flagging.
    energy_thresholds : numpy.ndarray
        Array of count thresholds for flagging high energy events corresponding to
         each energy range.
    sensor_id : int
        Sensor ID (e.g., 45 or 90).

    Returns
    -------
    quality_flags : numpy.ndarray
        Boolean quality flags shaped (n_energy_bins, n_spin_bins).
    """
    # expand energy thresholds to have shape (n_energy_bins, 1) for comparison with
    # the counts per spin
    energy_thresholds = energy_thresholds[:, np.newaxis]  # Shape (n_energy_bins, 1)
    cull_channel = UltraConstants.HIGH_ENERGY_CULL_CHANNEL
    n_energy_bins = len(energy_ranges) - 1
    if len(energy_thresholds) != n_energy_bins:
        raise ValueError(
            f"Length of energy_thresholds ({len(energy_thresholds)}) must match"
            f" the number of energy bins ({n_energy_bins})."
        )
    if cull_channel >= n_energy_bins:
        raise ValueError(
            f"HIGH_ENERGY_CULL_CHANNEL ({cull_channel}) is out of bounds"
            f" for {n_energy_bins} energy ranges."
        )

    # Initialize all spin bins to have no high energy flag
    spin_bin_size = len(spin_tbin_edges) - 1
    quality_flags = np.zeros((n_energy_bins, spin_bin_size), dtype=bool)
    # Get valid events and counts at each spin bin for the
    # designated culling channel.
    de_counts = get_valid_de_count_summary(
        de_dataset,
        energy_ranges,
        spin_tbin_edges,
        UltraConstants.HIGH_ENERGY_COMBINED_SPIN_BIN_RADIUS,
        sensor_id,
    )
    cull_channel_counts = de_counts[cull_channel]
    # flag spins where the counts in the cull channel exceed the threshold for that
    # energy range
    flagged = (
        cull_channel_counts[np.newaxis, :] >= energy_thresholds
    )  # (n_energy_bins, n_spin_bins)

    if mask is not None:
        quality_flags[:, ~mask] = flagged[:, ~mask]
    else:
        quality_flags = flagged

    num_culled: int = np.sum(quality_flags)
    logger.info(
        f"High energy culling removed {num_culled} spin bins across {n_energy_bins} "
        f"energy channels. Energy thresholds: {energy_thresholds.flatten()}, "
    )

    return quality_flags


def flag_statistical_outliers(
    de_dataset: xr.Dataset,
    spin_tbin_edges: NDArray,
    energy_ranges: NDArray,
    mask: NDArray,
    sensor_id: int = 90,
    n_iterations: int = UltraConstants.STAT_CULLING_N_ITER,
    std_threshold: float = UltraConstants.STAT_CULLING_STD_THRESHOLD,
    combine_flags_across_energy_bins: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Flag statistical outlier events based on count rates per spin.

    After low voltage and high energy spins have been flagged, there still appears to
    be some time dependency in the signal. This algorithm identifies those outliers.

    Iterative algorithm to identify areas consistent with Poisson statistics
        For each energy range:
        1. Flag where there are less than 3 bins with counts
        2. Calculate the mean (μ) and standard deviation (σ) of the counts in each bin.
        3. Find bins where the counts, c, yield |(c-μ)/σ|>3,  cull these bins
        4. Calculate ε=σ/√μ-1
        5. If ε is less than a threshold value (0.05 for now) stop iterating
        6. If number of iterations exceeds threshold (5 for now), stop iterating
        7. Return to step 1

    Parameters
    ----------
    de_dataset : xr.Dataset
        Direct event dataset.
    spin_tbin_edges : numpy.ndarray
        Edges of the spin time bins.
    energy_ranges : numpy.ndarray
        Array of energy range edges.
    mask : numpy.ndarray
        Mask indicating which events to consider for statistical outlier flagging.
        This should be a 2d boolean array of shape (n_energy_bins, n_spin_bins) where
        True indicates the spin bins that have been flagged in previous steps (e.g.,
        after low voltage and high energy culling) and should be excluded from the
        outlier flagging process.
    sensor_id : int
        Sensor ID (e.g., 45 or 90).
    n_iterations : int
        Maximum number of iterations to perform for outlier flagging.
    std_threshold : float
        Threshold for standard deviation difference from Poisson stats to determine
        convergence.
    combine_flags_across_energy_bins : bool
        Whether to link energy channels such that if a spin bin is flagged in any energy
        channel, it is flagged in all energy channels.

    Returns
    -------
    quality_stats : numpy.ndarray
        Quality flags for statistical outliers, shaped (n_energy_bins, n_spin_bins).
    convergence : numpy.ndarray
        Boolean array of shape (n_energy_bins,) indicating whether the outlier flagging
        converged for each energy bin.
    iterations : numpy.ndarray
        Array of shape (n_energy_bins,) indicating how many iterations were performed
        for each energy bin.
    std_diff : numpy.ndarray
        Array of shape (n_energy_bins,) containing the final standard deviation
         difference from Poisson stats for each energy bin.
    """
    # Initialize all spin bins to have no outlier flag
    spin_bin_size = len(spin_tbin_edges) - 1
    n_energy_bins = len(energy_ranges) - 1
    # make a copy of the current mask to avoid modifying the original mask passed in.
    # This contains flags from previous steps (e.g., after low voltage and high
    # energy culling) and will be updated iteratively to include the outlier flags as
    # well
    curr_mask = mask.copy()
    # Initialize quality_stats to keep track of which bins are flagged as outliers for
    # each energy bin
    quality_stats = np.zeros((n_energy_bins, spin_bin_size), dtype=bool)
    # Initialize a mask to keep track of spin bins that have been flagged across all
    # energy bins
    all_channel_mask = np.zeros(spin_bin_size, dtype=bool)
    # Initialize convergence array to keep track of poisson stats
    convergence = np.full(n_energy_bins, False)
    # Keep track of how many iterations we have done of flagging outliers and
    # recalculating stats per energy bin
    iterations = np.zeros(n_energy_bins)
    # keep track of the standard deviation difference from poisson stats per energy bin
    std_diff = np.zeros(n_energy_bins, dtype=float)
    count_summary = get_valid_de_count_summary(
        de_dataset, energy_ranges, spin_tbin_edges, sensor_id=sensor_id
    )  # shape (n_energy_bins, n_spin_bins)
    for e_idx in np.arange(n_energy_bins):
        good_mask = ~curr_mask[e_idx]  # spin bins that are not currently flagged
        for it in range(n_iterations):
            counts = count_summary[e_idx, good_mask]
            # Step 1. check if any energy bins have less than 3 spin bins with counts.
            # If so, flag all spins for that energy bin and skip to the next iteration
            if np.sum(counts > 0) < 3:
                quality_stats[e_idx] = True
                curr_mask[e_idx] = True
                convergence[e_idx] = True
                std_diff[e_idx] = -1
                break
            # Step 2. Check how close the data is to poisson stats
            std_ratio, outlier_mask = get_poisson_stats(counts)
            std_diff[e_idx] = std_ratio
            # Step 3. Flag bins where the count is more than 3 standard deviations from
            # the mean.
            outlier_inds = np.where(good_mask)[0][outlier_mask]
            # Set the quality flag to True for the outlier inds
            quality_stats[e_idx, outlier_inds] = True
            all_channel_mask[outlier_inds] = True
            good_mask[outlier_inds] = False
            iterations[e_idx] = it + 1
            # Check for convergence: if the standard deviation difference from
            # poisson stats is below the threshold, then we can stop iterating for this
            # energy bin
            if std_ratio < std_threshold:
                convergence[e_idx] = True
                break

    if combine_flags_across_energy_bins:
        # If true, then use the all_channel_mask for every energy channel.
        quality_stats[:] = all_channel_mask
        # Recalculate convergence with the combined mask.
        for e_idx in range(n_energy_bins):
            if not convergence[e_idx]:
                # Select counts that have not been flagged in any channel.
                counts = count_summary[e_idx, ~all_channel_mask]
                std_ratio, _ = get_poisson_stats(counts)
                if std_ratio < std_threshold:
                    convergence[e_idx] = True

    num_culled: int = np.sum(quality_stats)
    logger.info(
        f"Statistical culling removed {num_culled} spin bins across {n_energy_bins}"
        f" energy channels. Convergence: {convergence} after "
        f"{iterations} iterations."
    )

    return quality_stats, convergence, iterations, std_diff


def get_poisson_stats(counts: NDArray) -> tuple[float, NDArray]:
    """
    Calculate Poisson statistics for a given array of counts.

    For a perfect Poisson distribution, the standard deviation should equal
    the square root of the mean. The std_ratio measures how far the observed
    distribution deviates from this.

    Outliers are identified as bins where the counts deviate more than 3
    standard deviations from the mean.

    Parameters
    ----------
    counts : numpy.ndarray
        Array of counts per spin bin for a given energy range.

    Returns
    -------
    std_ratio : float
        Ratio of the observed standard deviation to the expected Poisson
        standard deviation.
    sub_mask : numpy.ndarray
        Boolean array of the same length as counts. True where a bin is
        a statistical outlier (more than 3 sigma from the mean).
    """
    std = np.std(counts)
    if std == 0:
        # If std is 0, then all counts are the same. In this case, we can consider
        # there to be no outliers and the distribution to perfectly match Poisson
        return 0, np.zeros_like(counts, dtype=bool)
    std_ratio = std / np.sqrt(np.mean(counts)) - 1
    sub_mask = np.abs((counts - np.mean(counts)) / std) > 3
    return std_ratio, sub_mask


def get_valid_de_count_summary(
    de_dataset: xr.Dataset,
    energy_ranges: NDArray,
    spin_tbin_edges: NDArray,
    combine_spin_bin_radius: int | None = None,
    sensor_id: int = 90,
) -> NDArray:
    """
    Get a summary of valid counts per energy range and spin bin.

    Parameters
    ----------
    de_dataset : xr.Dataset
        Direct event dataset.
    energy_ranges : numpy.ndarray
        Array of energy range edges.
    spin_tbin_edges : numpy.ndarray
        Array of spin time bin edges.
    combine_spin_bin_radius : int
        If not None, average counts across this many spin bins x 2 to get a smoother
        estimate of the counts per bin.
    sensor_id : int
        Sensor ID (e.g., 45 or 90).

    Returns
    -------
    counts : numpy.ndarray
        A 2D array of counts per energy range and spin bin for valid events.
    """
    valid_events = get_valid_events_per_energy_range(
        de_dataset, energy_ranges, UltraConstants.EARTH_ANGLE_45_THRESHOLD, sensor_id
    )
    counts = np.zeros((len(energy_ranges) - 1, len(spin_tbin_edges) - 1), dtype=float)

    for i in range(len(energy_ranges) - 1):
        counts[i, :], _ = np.histogram(
            de_dataset["de_event_met"].values[valid_events[i, :]], bins=spin_tbin_edges
        )

    if combine_spin_bin_radius is not None and combine_spin_bin_radius > 0:
        # Pad array along the spin bin axis to ensure sliding_window_view returns
        # an array of the correct shape.
        counts_padded = np.pad(
            counts,
            ((0, 0), (combine_spin_bin_radius, combine_spin_bin_radius)),
            mode="edge",
        )
        window_size = combine_spin_bin_radius * 2 + 1
        windows = sliding_window_view(counts_padded, window_shape=window_size, axis=1)
        counts = np.mean(windows, axis=-1)
    return counts


def get_valid_events_per_energy_range(
    de_dataset: xr.Dataset, energy_ranges: NDArray, earth_ang_45: float, sensor_id: int
) -> NDArray:
    """
    Get valid events per energy range.

    Parameters
    ----------
    de_dataset : xr.Dataset
        Direct event dataset.
    energy_ranges : numpy.ndarray
        Array of energy range edges.
    earth_ang_45 : float
        Earth angle to use for culling in ULTRA 45.
    sensor_id : int
        Sensor ID (e.g., 45 or 90).

    Returns
    -------
    valid_events_per_range : numpy.ndarray
        A boolean array of shape (n_energy_ranges, n_events).
    """
    event_energies = de_dataset["energy_spacecraft"].values
    valid_events = np.zeros((len(energy_ranges) - 1, len(event_energies)), dtype=bool)
    valid_outliers = de_dataset["quality_outliers"].values == 0
    valid_scattering = de_dataset["quality_scattering"].values == 0
    # TODO what about species non-proton? For those psets dont cull based on
    #   High energy?
    ebin = de_dataset["ebin"].values
    valid_ebin = np.isin(ebin, UltraConstants.TOFXPH_SPECIES_GROUPS["proton"])
    for i in range(len(energy_ranges) - 1):
        energy_mask = (event_energies >= energy_ranges[i]) & (
            event_energies < energy_ranges[i + 1]
        )
        if not np.any(energy_mask):
            continue
        # subset the dataset to events within the energy range
        de_dataset_subset = de_dataset.isel(epoch=energy_mask)
        valid_earth_angle = np.full(np.sum(energy_mask), True, dtype=bool)
        # For ultra45, also apply an Earth angle cut to remove times when
        # the Earth is in the field of view. ULTRA 90 does not require this since Earth
        # is always outside the field of view.
        if sensor_id == 45:
            valid_earth_angle = get_valid_earth_angle_events(
                de_dataset_subset, earth_ang_45
            )

        # Flag events at the valid energy ranges if they meet all the criteria for
        # valid events: not flagged as outliers, not flagged as scattering,
        # in a valid ebin, and (for ultra45) have a valid Earth angle.
        valid_events[i, energy_mask] = np.logical_and.reduce(
            [
                valid_outliers[energy_mask],
                valid_scattering[energy_mask],
                valid_ebin[energy_mask],
                valid_earth_angle,
            ]
        )

    return valid_events


def get_valid_earth_angle_events(
    de_dataset_subset: xr.Dataset,
    earth_ang_45: float = UltraConstants.EARTH_ANGLE_45_THRESHOLD,
) -> NDArray:
    """
    Get events where the particle look direction is outside the Earth keepout angle.

    Parameters
    ----------
    de_dataset_subset : xr.Dataset
        Subset of the direct event dataset. Should contain events within a single
        energy bin.
    earth_ang_45 : float
        Earth keepout angle threshold (in radians) for ULTRA 45 instrument.

    Returns
    -------
    valid_earth_angle_events : NDArray
        A boolean array indicating which events have Earth angle greater than the
        specified threshold.
    """
    velocity_dps_sc = de_dataset_subset["velocity_dps_sc"].values
    # Use the mean event time to compute the Earth unit vector since the spacecraft
    # position doesn't change significantly over the course of the energy bin.
    et = np.mean(de_dataset_subset["event_times"].values)
    # Compute the unit vector from IMAP to Earth in the DPS frame at the time of the
    # events.
    # call spkezr to get the state vector from Earth to IMAP in the IMAP_DPS frame
    body_state, _ = sp.spkezr(
        SpiceBody.EARTH.name,
        et,
        SpiceFrame.IMAP_DPS.name,
        "NONE",
        SpiceBody.IMAP.name,
    )
    position = body_state[:3]
    distance = np.linalg.norm(position)
    earth_unit_vector = position / distance
    # Calculate the magnitude of the velocity vector for each event
    particle_mag = np.linalg.norm(velocity_dps_sc, axis=1)
    # Normalize and flip to get where each particle is looking.
    unit_look_dirs = (
        -velocity_dps_sc / particle_mag[:, np.newaxis]
    )  # shape (n_events, 3)
    # Get cos(theta) between each particle look direction and Earth direction
    cos_sep = np.dot(unit_look_dirs, earth_unit_vector)  # shape (n_events,)
    # Clip cos_sep to the valid range of [-1, 1] to avoid numerical issues with arccos
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    sep_angle = np.arccos(cos_sep)
    # An event is valid if the separation angle between the particle look
    # direction and Earth direction is greater than the Earth angle limit
    # (i.e., the Earth is outside the field of view).
    return sep_angle > earth_ang_45


def get_energy_range_flags(energy_ranges_edges: NDArray) -> NDArray:
    """
    Get the energy bin flags for energy dependent culling.

    Parameters
    ----------
    energy_ranges_edges : NDArray
        Array of energy range edges.

    Returns
    -------
    energy_bin_flags : NDArray
        Energy bin flags.
    """
    num_bins = len(energy_ranges_edges) - 1
    if num_bins > 16:
        raise ValueError(
            f"Number of culling energy bins ({num_bins}) "
            f"cannot exceed 16 due to uint16 bit limitations."
        )
    return np.array([2**bit for bit in range(num_bins)], dtype=np.uint16)


def get_binned_energy_ranges(
    energy_bin_edges: list[tuple[float, float]],
    max_energy: float | None = UltraConstants.MAX_ENERGY_THRESHOLD,
) -> NDArray:
    """
    Create L1C energy ranges by grouping energy bins.

    Parameters
    ----------
    energy_bin_edges : list[tuple[float, float]]
        List of (start, stop) tuples for each energy bin.
    max_energy : float | None
        Maximum energy to include in the energy ranges. If None, don't set a max.

    Returns
    -------
    energy_range_edges : NDArray
        Array of bin edges. For N energy ranges, returns N+1 edge values.
        Range i spans from energy_range_edges[i] to energy_range_edges[i+1].
    """
    # Get indices for group starts
    group_start_inds = np.arange(
        UltraConstants.BASE_CULL_EBIN,
        len(energy_bin_edges),
        UltraConstants.N_CULL_EBINS,
    )
    energy_starts = [energy_bin_edges[i][0] for i in group_start_inds]
    # Append the stop energy of the last bin to cover the full range
    last_group_end_ind = min(
        group_start_inds[-1] + UltraConstants.N_CULL_EBINS, len(energy_bin_edges)
    )
    energy_ranges = np.append(
        energy_starts, energy_bin_edges[last_group_end_ind - 1][1]
    )
    if max_energy is not None:
        # get the first index where the energy range exceeds the max energy
        # exclude the last edge since it is the stop energy of the last range
        max_reached_idx = np.where(energy_ranges[:-1] > max_energy)[0]
        if np.any(energy_ranges[:-1] > max_energy):
            max_reached_idx = max_reached_idx[0]
        else:
            # if no energy range exceeds the max energy, return the original energy
            # ranges
            return energy_ranges
        # Merge all energy ranges above the max energy into a single range and set the
        # stop
        energy_ranges_lim = energy_ranges[
            : max_reached_idx + 2
        ].copy()  # include the first edge above max energy and the last edge
        # update the last bin to start at the first original edge above the max energy
        # and end at the last edge
        energy_ranges_lim[-2] = next(
            e[0] for e in energy_bin_edges if e[0] > max_energy
        )
        energy_ranges_lim[-1] = energy_ranges[-1]
        energy_ranges = energy_ranges_lim

    return energy_ranges


def get_binned_spins_edges(
    spins: NDArray,
    spin_periods: NDArray,
    spin_start_times: NDArray,
    spin_bin_size: int = UltraConstants.SPIN_BIN_SIZE,
) -> NDArray:
    """
    Create spin bins for grouping spins together.

    Parameters
    ----------
    spins : NDArray
        Unique spin numbers.
    spin_periods : NDArray
        Spin periods corresponding to the unique spin numbers.
    spin_start_times : NDArray
        Spin start times corresponding to the unique spin numbers.
    spin_bin_size : int
        Number of spins to group together for voltage flagging.

    Returns
    -------
    spin_tbin_edges : NDArray
        Spin time bin edges.
    """
    # Create bins based on the number of spins per bin
    # We will only use complete bins for culling so use integer division.
    n_spin_bins = len(spins) // spin_bin_size
    # Get the start time of each bin
    spin_tbin_edges = spin_start_times[::spin_bin_size][:n_spin_bins]
    if spin_tbin_edges.size == 0:
        # If there are no valid spin bins, return an array with a single edge at 0
        raise ValueError(
            f"No valid spin bins found for bin size: {spin_bin_size}"
            f" and number of spins: {len(spins)}."
        )
    # Append the last start time plus the spin period to account for low times
    # that occur after the last spin start time
    last_spin_idx = min(n_spin_bins * spin_bin_size - 1, len(spins) - 1)
    spin_tbin_edges = np.append(
        spin_tbin_edges, spin_start_times[last_spin_idx] + spin_periods[last_spin_idx]
    )
    return spin_tbin_edges


def expand_bin_flags_to_spins(
    n_spins: int, binned_quality_flags: NDArray, spin_bin_size: int
) -> NDArray:
    """
    Map binned spin flags back to individual spins.

    Parameters
    ----------
    n_spins : int
        Number of unique spin numbers.
    binned_quality_flags : NDArray
        Quality flags for each spin bin.
    spin_bin_size : int
        Number of spins that were grouped together for the binned quality flags.

    Returns
    -------
    quality_flags : NDArray
        Quality flags mapped to each individual spin.
    """
    quality_flags = np.full(n_spins, ImapRatesUltraFlags.NONE.value, dtype=np.uint16)
    # Repeat each binned flag for the number of spins in each bin
    repeated_flags = np.repeat(binned_quality_flags, spin_bin_size)
    if len(repeated_flags) > n_spins:
        logger.warning(
            f"Found incomplete spin bin at the end with"
            f" {len(repeated_flags) - n_spins} spins. These spins will be "
            f"ignored."
        )
        repeated_flags = repeated_flags[:n_spins]
    quality_flags[: len(repeated_flags)] = repeated_flags

    return quality_flags
