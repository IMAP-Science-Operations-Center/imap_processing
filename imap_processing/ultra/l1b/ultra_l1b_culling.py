"""Culls Events for ULTRA L1b."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.spice.spin import get_spin_data
from imap_processing.ultra.constants import UltraConstants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPIN_DURATION = 15  # Default spin duration in seconds.


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
    spin_edges = unique_spin_number.astype(np.uint16)
    spin_edges = np.append(spin_edges, spin_edges.max() + 1)

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

    return quality_flags, spin_rates, spin_period, spin_starttime


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
    sigma_per_energy = np.std(count_rates, axis=1)
    n_sigma_per_energy = sigma * sigma_per_energy
    mean_per_energy = np.mean(count_rates, axis=1)
    # Must have a HIGHRATES threshold of at least 3 counts per spin.
    threshold = np.maximum(mean_per_energy + n_sigma_per_energy, 3 / mean_duration)

    return threshold


def flag_spin(
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
    energy_midpoints : NDArray
        Energy midpoint data.
    n_sigma_per_energy_reshape : NDArray
        N sigma per energy.
    """
    count_rates, spin_edges, counts, duration = get_energy_histogram(
        spin_number, energy
    )
    quality_flags = np.full(
        count_rates.shape, ImapRatesUltraFlags.NONE.value, dtype=np.uint16
    )

    # Zero counts/spin/energy level
    quality_flags[counts == 0] |= ImapRatesUltraFlags.ZEROCOUNTS.value
    threshold = get_n_sigma(count_rates, duration, sigma=sigma)

    bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_midpoints = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    spin = np.unique(spin_number)

    # Indices where the counts exceed the threshold
    indices_n_sigma = count_rates > threshold[:, np.newaxis]
    quality_flags[indices_n_sigma] |= ImapRatesUltraFlags.HIGHRATES.value

    return quality_flags, spin, energy_midpoints, threshold


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
