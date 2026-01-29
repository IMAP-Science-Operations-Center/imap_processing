"""IMAP-Lo L1C Data Processing."""

import logging
from dataclasses import Field
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.utils.corrections import add_spacecraft_velocity_to_pset
from imap_processing.lo import lo_ancillary
from imap_processing.lo.l1b.lo_l1b import set_bad_or_goodtimes
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform_az_el,
    lo_instrument_pointing,
)
from imap_processing.spice.repoint import get_pointing_times
from imap_processing.spice.spin import get_spin_data, get_spin_number
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

N_ESA_ENERGY_STEPS = 7
N_SPIN_ANGLE_BINS = 3600
N_OFF_ANGLE_BINS = 40
# 1 time, 7 energy steps, 3600 spin angle bins, and 40 off angle bins
PSET_SHAPE = (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
PSET_DIMS = ["epoch", "esa_energy_step", "spin_angle", "off_angle"]
ESA_ENERGY_STEPS = np.arange(N_ESA_ENERGY_STEPS) + 1  # 1 to 7 inclusive
SPIN_ANGLE_BIN_EDGES = np.linspace(0, 360, N_SPIN_ANGLE_BINS + 1)
SPIN_ANGLE_BIN_CENTERS = (SPIN_ANGLE_BIN_EDGES[:-1] + SPIN_ANGLE_BIN_EDGES[1:]) / 2
OFF_ANGLE_BIN_EDGES = np.linspace(-2, 2, N_OFF_ANGLE_BINS + 1)
OFF_ANGLE_BIN_CENTERS = (OFF_ANGLE_BIN_EDGES[:-1] + OFF_ANGLE_BIN_EDGES[1:]) / 2

# Constants for statistical exposure time calculation
# Number of time samples per spin to capture all potential timesteps
N_SAMPLES_PER_SPIN = 4096
# Default number of representative spins to sample across the pointing
DEFAULT_N_REPRESENTATIVE_SPINS = 5
# Nominal Lo pivot angle in degrees
LO_NOMINAL_PIVOT_ANGLE = 90.0


class FilterType(str, Enum):
    """
    Enum for the filter types used in the PSET counts.

    The filter types are used to filter the L1B Direct Event dataset
    to only include the specified event types.
    """

    TRIPLES = "triples"
    DOUBLES = "doubles"
    HYDROGEN = "h"
    OXYGEN = "o"
    NONE = ""


def lo_l1c(sci_dependencies: dict, anc_dependencies: list) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L1B data into L1C CDF data products.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1C data product creation in xarray Datasets.
    anc_dependencies : list
        Ancillary files needed for L1C data product creation.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1c")

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1b_de" in sci_dependencies:
        logical_source = "imap_lo_l1c_pset"
        l1b_de = sci_dependencies["imap_lo_l1b_de"]
        l1b_goodtimes_only = filter_goodtimes(l1b_de, anc_dependencies)
        # TODO: Need to handle case where no good times are found
        # Set the pointing start and end times based on the first epoch
        pointing_start_met, pointing_end_met = get_pointing_times(
            ttj2000ns_to_met(l1b_goodtimes_only["epoch"][0].item())
        )

        pset = xr.Dataset(
            coords={"epoch": np.array([met_to_ttj2000ns(pointing_start_met)])},
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

        # pass-through of the pivot_angle from L1B DE
        pset["pivot_angle"] = l1b_de["pivot_angle"]

        # ESA mode needs to be added to L1B DE. Adding try statement
        # to avoid error until it's available in the dataset
        if "esa_mode" not in l1b_de:
            logging.debug(
                "ESA mode not found in L1B DE dataset. \
                Setting to default value of 0 for Hi-Res."
            )
            pset["esa_mode"] = xr.DataArray(
                np.array([0]),
                dims=["epoch"],
                attrs=attr_mgr.get_variable_attributes("esa_mode"),
            )
        else:
            pset["esa_mode"] = xr.DataArray(
                np.array([l1b_de["esa_mode"].values[0]]),
                dims=["epoch"],
                attrs=attr_mgr.get_variable_attributes("esa_mode"),
            )

        pset["pointing_start_met"] = xr.DataArray(
            np.array([pointing_start_met]),
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("pointing_start_met"),
        )
        pset["pointing_end_met"] = xr.DataArray(
            np.array([pointing_end_met]),
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("pointing_end_met"),
        )

        # Get the start and end spin numbers based on the pointing start and end MET
        pset["start_spin_number"] = xr.DataArray(
            [get_spin_number(pset["pointing_start_met"].item())],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("start_spin_number"),
        )
        pset["end_spin_number"] = xr.DataArray(
            [get_spin_number(pset["pointing_end_met"].item())],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("end_spin_number"),
        )

        # Set the counts
        pset["triples_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.TRIPLES
        )
        pset["doubles_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.DOUBLES
        )
        pset["h_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.HYDROGEN)
        pset["o_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.OXYGEN)

        # Read good-times for exposure time calculation
        goodtimes_df = lo_ancillary.read_ancillary_file(
            next(str(s) for s in anc_dependencies if "good-times" in str(s))
        )

        # Set the exposure time using statistical off-pointing sampling
        # with good-times filtering applied
        pset["exposure_time"] = calculate_exposure_times(
            pointing_start_met, pointing_end_met, goodtimes_df
        )

        # Set backgrounds
        (
            pset["h_background_rates"],
            pset["h_background_rates_stat_uncert"],
            pset["h_background_rates_sys_err"],
        ) = set_background_rates(
            pset["pointing_start_met"].item(),
            pset["pointing_end_met"].item(),
            FilterType.HYDROGEN,
            anc_dependencies,
            attr_mgr,
        )

        (
            pset["o_background_rates"],
            pset["o_background_rates_stat_uncert"],
            pset["o_background_rates_sys_err"],
        ) = set_background_rates(
            pset["pointing_start_met"].item(),
            pset["pointing_end_met"].item(),
            FilterType.OXYGEN,
            anc_dependencies,
            attr_mgr,
        )

        pset["hae_longitude"], pset["hae_latitude"] = set_pointing_directions(
            pset["epoch"].item(), attr_mgr, pset["pivot_angle"].values[0].item()
        )

    pset.attrs = attr_mgr.get_global_attributes(logical_source)

    pset = pset.assign_coords(
        {
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        }
    )

    # add the spacecraft velocity and direction
    pset = add_spacecraft_velocity_to_pset(pset)

    return [pset]


def filter_goodtimes(l1b_de: xr.Dataset, anc_dependencies: list) -> xr.Dataset:
    """
    Filter the L1B Direct Event dataset to only include good times.

    The good times are read from the sweep table ancillary file.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset.

    anc_dependencies : list
        Ancillary files needed for L1C data product creation.

    Returns
    -------
    l1b_de : xarray.Dataset
        Filtered L1B Direct Event dataset.
    """
    # the goodtimes are currently the only ancillary file needed for L1C processing
    goodtimes_table_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "good-times" in str(s))
    )

    esa_steps = l1b_de["esa_step"].values
    epochs = l1b_de["epoch"].values
    spin_bins = l1b_de["spin_bin"].values

    # Get array of bools for each epoch 1 = good time, 0 not good time
    goodtimes_mask = set_bad_or_goodtimes(
        goodtimes_table_df, epochs, esa_steps, spin_bins
    )

    # Filter the dataset using the mask
    filtered_epochs = l1b_de.sel(epoch=goodtimes_mask.astype(bool))

    return filtered_epochs


def get_triple_coincidences(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the triple coincidence events from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_triples : xarray.Dataset
        L1B Direct Event dataset with only triple coincidence events.
    """
    triple_types = ["111111", "111100", "111000"]
    triple_idx = np.nonzero(np.isin(de["coincidence_type"], triple_types))[0]
    de_triples = de.isel(epoch=triple_idx)

    return de_triples


def get_double_coincidences(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the double coincidence events from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_doubles : xarray.Dataset
        L1B Direct Event dataset with only double coincidence events.
    """
    double_types = [
        "110100",
        "110000",
        "101101",
        "101100",
        "101000",
        "100100",
        "100101",
        "100000",
        "011100",
        "011000",
        "010100",
        "010101",
        "010000",
        "001100",
        "001101",
        "001000",
    ]
    double_idx = np.nonzero(np.isin(de["coincidence_type"], double_types))[0]
    de_doubles = de.isel(epoch=double_idx)

    return de_doubles


def _get_peak_mask(
    de: xr.Dataset, peak_lows: list[int], peak_highs: list[int]
) -> np.ndarray:
    """
    Get a boolean mask for events within specified peak ranges.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.
    peak_lows : list[int]
        List of low peak values for each TOF.
    peak_highs : list[int]
        List of high peak values for each TOF.

    Returns
    -------
    peak_mask : numpy.ndarray
        Boolean mask indicating events within the specified peak ranges.
    """
    tof0_s = de["tof0"] + 0.5 * de["tof3"]
    tof1_s = de["tof1"] - 0.5 * de["tof3"]

    peak_mask = (
        (tof0_s >= peak_lows[0])
        & (tof0_s <= peak_highs[0])
        & (tof1_s >= peak_lows[1])
        & (tof1_s <= peak_highs[1])
        & (de["tof2"] >= peak_lows[2])
        & (de["tof2"] <= peak_highs[2])
    )

    return peak_mask


def _get_golden_triple_mask(de: xr.Dataset) -> np.ndarray:
    """
    Get a boolean mask for events within the golden triple coincidence types.

    A golden triple coincidence is only one of the possible triples-types, so
    we need to subset it separately from just triples.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    golden_triple_mask : numpy.ndarray
        Boolean mask indicating events within the golden triple coincidence types.
    """
    return de["coincidence_type"] == "111111"


def get_h_species(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the hydrogen species from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_h : xarray.Dataset
        L1B Direct Event dataset with only hydrogen species.
    """
    h_peak_low = [20, 10, 10]
    h_peak_high = [70, 50, 40]

    golden_triple_mask = _get_golden_triple_mask(de)
    h_peak_mask = _get_peak_mask(de, h_peak_low, h_peak_high)

    h_idx = np.nonzero((golden_triple_mask & h_peak_mask).values)[0]

    de_h = de.isel(epoch=h_idx)
    return de_h


def get_o_species(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the oxygen species from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_o : xarray.Dataset
        L1B Direct Event dataset with only oxygen species.
    """
    co_peak_low = [100, 60, 60]
    co_peak_high = [270, 150, 150]

    golden_triple_mask = _get_golden_triple_mask(de)
    o_peak_mask = _get_peak_mask(de, co_peak_low, co_peak_high)
    o_idx = np.nonzero((golden_triple_mask & o_peak_mask).values)[0]

    de_o = de.isel(epoch=o_idx)
    return de_o


def create_pset_counts(
    de: xr.Dataset, filter_type: FilterType = FilterType.NONE
) -> xr.DataArray:
    """
    Create the PSET counts for the L1B Direct Event dataset.

    The counts are created by binning the data into 3600 longitude bins,
    40 latitude bins, and 7 energy bins. The data is filtered to only
    include counts based on the specified filter: "triples", "doubles", "h", or "o".

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.
    filter_type : FilterType, optional
        The event type to include in the counts.
        Can be "triples", "doubles", "h", or "o".

    Returns
    -------
    counts : xarray.DataArray
        The counts for the specified filter.
    """
    match filter_type:
        case FilterType.TRIPLES:
            de_filtered = get_triple_coincidences(de)
        case FilterType.DOUBLES:
            de_filtered = get_double_coincidences(de)
        case FilterType.HYDROGEN:
            de_filtered = get_h_species(de)
        case FilterType.OXYGEN:
            de_filtered = get_o_species(de)
        case _:
            # if no filter is specified, use all data
            de_filtered = de

    # stack the filtered data into the 3D array
    data = np.column_stack(
        (
            de_filtered["esa_step"],
            de_filtered["spin_bin"],
            de_filtered["off_angle_bin"],
        )
    )
    # Create the histogram with 3600 longitude bins, 40 latitude bins, and 7 energy bins
    lon_edges = np.arange(3601)
    lat_edges = np.arange(41)
    energy_edges = np.arange(1, 9)

    hist, _edges = np.histogramdd(
        data,
        bins=[energy_edges, lon_edges, lat_edges],
    )

    # add a new axis of size 1 for the epoch
    hist = hist[np.newaxis, :, :, :]

    counts = xr.DataArray(
        data=hist.astype(np.int16),
        dims=PSET_DIMS,
    )

    return counts


def get_representative_spin_times(
    pointing_start_met: float,
    pointing_end_met: float,
    n_spins: int = DEFAULT_N_REPRESENTATIVE_SPINS,
) -> pd.DataFrame:
    """
    Get evenly-spaced representative spin times from the pointing period.

    Selects N spins distributed evenly across the middle 80% of the pointing
    duration (skipping the first and last 10%) by querying the spin table for
    spins at evenly-spaced MET times.

    Parameters
    ----------
    pointing_start_met : float
        The start MET time of the pointing.
    pointing_end_met : float
        The end MET time of the pointing.
    n_spins : int, optional
        Number of representative spins to select. Default is 5.

    Returns
    -------
    representative_spins : pandas.DataFrame
        DataFrame containing the spin table data for the selected representative
        spins, including columns: spin_number, spin_start_met, actual_spin_period.
    """
    spin_df = get_spin_data()

    # Filter spin table to only spins within the pointing period
    pointing_spins = spin_df[
        (spin_df["spin_start_met"] >= pointing_start_met)
        & (spin_df["spin_start_met"] < pointing_end_met)
    ]

    if len(pointing_spins) == 0:
        raise ValueError(
            f"No spins found in spin table for pointing period "
            f"[{pointing_start_met}, {pointing_end_met}]."
        )

    # Select evenly-spaced indices from the middle 80% of available spins
    # Skip first 10% and last 10% to avoid boundary effects
    total_spins = len(pointing_spins)
    start_fraction = 0.1
    end_fraction = 0.9
    start_idx = int(total_spins * start_fraction)
    end_idx = int(total_spins * end_fraction) - 1

    # Ensure we have valid indices
    start_idx = max(0, start_idx)
    end_idx = max(start_idx, min(end_idx, total_spins - 1))

    available_spins = end_idx - start_idx + 1
    if available_spins <= n_spins:
        # Use all available spins in the middle 80% if fewer than requested
        selected_indices = np.arange(start_idx, end_idx + 1)
    else:
        # Select evenly-spaced indices from the middle 80%
        selected_indices = np.linspace(start_idx, end_idx, n_spins, dtype=int)

    representative_spins = pointing_spins.iloc[selected_indices]

    logging.debug(
        f"Selected {len(representative_spins)} representative spins from "
        f"{total_spins} total spins in pointing period (using middle 80%)."
    )

    return representative_spins


def sample_boresight_bins(
    spin_start_met: float,
    spin_period: float,
    n_samples: int = N_SAMPLES_PER_SPIN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the Lo boresight look direction throughout a single spin.

    Generates evenly-spaced time samples within a spin period, computes the
    Lo boresight pointing direction in the IMAP_DPS frame, and returns the
    spin_angle and off_angle for each sample.

    Parameters
    ----------
    spin_start_met : float
        The MET time at the start of the spin.
    spin_period : float
        The duration of the spin in seconds.
    n_samples : int, optional
        Number of time samples within the spin. Default is 4096.

    Returns
    -------
    spin_angles : numpy.ndarray
        Array of spin angles (0-360 degrees) for each sample time.
    off_angles : numpy.ndarray
        Array of off angles (elevation from DPS equatorial plane) for each sample.
    """
    # Generate evenly-spaced sample times within the spin
    # Use the center of each time bin for sampling
    sample_fractions = (np.arange(n_samples) + 0.5) / n_samples
    sample_mets = spin_start_met + sample_fractions * spin_period

    # Convert MET times to ephemeris time for SPICE
    sample_ttj2000ns = met_to_ttj2000ns(sample_mets)
    sample_ets = ttj2000ns_to_et(sample_ttj2000ns)

    # Get the Lo boresight pointing in the DPS frame
    # lo_instrument_pointing returns (longitude, latitude) in degrees
    # longitude corresponds to spin_angle, latitude corresponds to off_angle
    # Use nominal pivot angle of 90 degrees which rotates boresight to point
    # approximately in the spacecraft spin plane (near-zero off-pointing)
    pointing = lo_instrument_pointing(
        sample_ets, LO_NOMINAL_PIVOT_ANGLE, SpiceFrame.IMAP_DPS
    )

    # Extract spin_angle (longitude) and off_angle (latitude)
    spin_angles = pointing[:, 0]
    off_angles = pointing[:, 1]

    # Ensure spin angles are in [0, 360) range
    spin_angles = np.mod(spin_angles, 360)

    return spin_angles, off_angles


def calculate_bin_weights(off_angles: np.ndarray) -> np.ndarray:
    """
    Calculate the probability weight for each off_angle bin.

    Bins all sampled off angles into the 40-bin grid and normalizes
    the counts to get probability weights that sum to 1. These weights
    are applied uniformly across all spin_angle bins since the spacecraft
    rotates evenly and we want smooth exposure across spin angles.

    Parameters
    ----------
    off_angles : numpy.ndarray
        Array of off angles (degrees) from all sampled times.

    Returns
    -------
    bin_weights : numpy.ndarray
        1D array of shape (N_OFF_ANGLE_BINS,) containing the probability
        weight for each off_angle bin. Weights sum to 1.0.
    """
    # Create 1D histogram of off_angles only
    bin_counts, _ = np.histogram(off_angles, bins=OFF_ANGLE_BIN_EDGES)

    # Normalize to get probability weights
    total_samples = len(off_angles)
    if total_samples > 0:
        bin_weights = bin_counts / total_samples
    else:
        # If no samples, return zero weights
        bin_weights = np.zeros(N_OFF_ANGLE_BINS, dtype=np.float32)

    return bin_weights


def create_goodtimes_fraction(
    goodtimes_df: pd.DataFrame,
    pointing_start_met: float,
    pointing_end_met: float,
) -> np.ndarray:
    """
    Create fractional weights for spin_angle bins and ESA steps based on good-times.

    The good-times ancillary file specifies which spin angle bins (in 6-degree
    resolution) and ESA energy steps are valid during specific time periods.
    This function calculates the fraction of the pointing duration that is
    covered by good-times for each (ESA step, spin_angle bin) combination.

    Parameters
    ----------
    goodtimes_df : pandas.DataFrame
        DataFrame containing the good-times ancillary data with columns:
        GoodTime_start, GoodTime_end, bin_start, bin_end, E-Step1 through E-Step7.
    pointing_start_met : float
        The start MET time of the pointing.
    pointing_end_met : float
        The end MET time of the pointing.

    Returns
    -------
    goodtimes_fraction : numpy.ndarray
        2D array of shape (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS) containing
        the fraction of pointing duration covered by good-times for each
        ESA step and spin angle bin. Values range from 0.0 to 1.0.
    """
    total_pointing_duration = pointing_end_met - pointing_start_met

    # Initialize as all zeros (no good time)
    goodtimes_fraction = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS), dtype=np.float32
    )

    if total_pointing_duration <= 0:
        logging.warning("Pointing duration is zero or negative.")
        return goodtimes_fraction

    # Filter good-times to only those overlapping with the pointing period
    pointing_goodtimes = goodtimes_df[
        (goodtimes_df["GoodTime_start"] < pointing_end_met)
        & (goodtimes_df["GoodTime_end"] > pointing_start_met)
    ]

    if len(pointing_goodtimes) == 0:
        logging.warning(
            f"No good-times found for pointing period "
            f"[{pointing_start_met}, {pointing_end_met}]. "
            "All exposure times will be zero."
        )
        return goodtimes_fraction

    # Process each good-time row and accumulate fractional coverage
    for _, row in pointing_goodtimes.iterrows():
        # Calculate the overlap between this good-time period and the pointing
        goodtime_start = max(row["GoodTime_start"], pointing_start_met)
        goodtime_end = min(row["GoodTime_end"], pointing_end_met)
        overlap_duration = goodtime_end - goodtime_start

        if overlap_duration <= 0:
            continue

        # Calculate fraction of pointing duration covered by this good-time
        time_fraction = overlap_duration / total_pointing_duration

        # Convert bin_start/bin_end from 6-degree to 0.1-degree resolution
        # bin_start and bin_end are in units of 6-degree bins (0-59), inclusive
        # We need to convert to 0.1-degree bins (0-3599)
        bin_start_6deg = int(row["bin_start"])
        bin_end_6deg = int(row["bin_end"])

        # Convert to 0.1-degree resolution (multiply by 60)
        # bin_end is inclusive, so add 1 after scaling for Python slice indexing
        spin_bin_start = bin_start_6deg * 60
        spin_bin_end = (bin_end_6deg + 1) * 60  # +1 because bin_end is inclusive

        # For each ESA step, accumulate the fractional coverage
        for esa_idx in range(N_ESA_ENERGY_STEPS):
            esa_step_col = f"E-Step{esa_idx + 1}"
            if row[esa_step_col] == 1:
                # Add this time fraction to the affected bins
                goodtimes_fraction[esa_idx, spin_bin_start:spin_bin_end] += (
                    time_fraction
                )

    # Clip to [0, 1] in case of overlapping good-time periods
    goodtimes_fraction = np.clip(goodtimes_fraction, 0.0, 1.0)

    # Calculate average coverage for logging
    avg_coverage = goodtimes_fraction.mean()
    logging.debug(
        f"Good-times coverage: average={100 * avg_coverage:.1f}%, "
        f"min={100 * goodtimes_fraction.min():.1f}%, "
        f"max={100 * goodtimes_fraction.max():.1f}%"
    )

    return goodtimes_fraction


def calculate_exposure_times(
    pointing_start_met: float,
    pointing_end_met: float,
    goodtimes_df: pd.DataFrame | None = None,
    n_representative_spins: int = DEFAULT_N_REPRESENTATIVE_SPINS,
) -> xr.DataArray:
    """
    Calculate exposure times using statistical off-pointing sampling.

    Samples the Lo boresight look direction across representative spins to
    determine which spin_angle × off_angle bins are observed. The total
    pointing duration is then distributed across bins proportionally to
    the observed probability weights. If good-times data is provided,
    exposure times are zeroed for invalid spin_angle/ESA step combinations.

    Parameters
    ----------
    pointing_start_met : float
        The start MET time of the pointing.
    pointing_end_met : float
        The end MET time of the pointing.
    goodtimes_df : pandas.DataFrame, optional
        DataFrame containing the good-times ancillary data. If provided,
        exposure times will be zeroed for invalid spin_angle bins and ESA steps.
    n_representative_spins : int, optional
        Number of representative spins to sample. Default is 5.

    Returns
    -------
    exposure_time : xarray.DataArray
        The exposure times for each (esa_energy_step, spin_angle, off_angle) bin.
        Shape is (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS).
    """
    # Calculate total pointing duration in seconds
    total_pointing_duration = pointing_end_met - pointing_start_met

    # Get representative spins from the pointing period
    representative_spins = get_representative_spin_times(
        pointing_start_met, pointing_end_met, n_representative_spins
    )

    # Collect all sampled spin angles and off angles across representative spins
    all_spin_angles = []
    all_off_angles = []

    for _, spin_row in representative_spins.iterrows():
        spin_start_met = spin_row["spin_start_met"]
        spin_period = spin_row["actual_spin_period"]

        spin_angles, off_angles = sample_boresight_bins(spin_start_met, spin_period)
        all_spin_angles.append(spin_angles)
        all_off_angles.append(off_angles)

    # Concatenate all samples
    all_spin_angles = np.concatenate(all_spin_angles)
    all_off_angles = np.concatenate(all_off_angles)

    # Log statistics about the sampled angles for debugging
    logging.debug(
        f"Sampled angles - spin_angle: min={all_spin_angles.min():.2f}, "
        f"max={all_spin_angles.max():.2f}, mean={all_spin_angles.mean():.2f}"
    )
    logging.debug(
        f"Sampled angles - off_angle: min={all_off_angles.min():.2f}, "
        f"max={all_off_angles.max():.2f}, mean={all_off_angles.mean():.2f}"
    )

    # Calculate bin probability weights for off_angle only
    # We use 1D histogram on off_angle because discrete spin sampling creates
    # artifacts, but the spacecraft rotates evenly so spin_angle exposure
    # should be uniform
    off_angle_weights = calculate_bin_weights(all_off_angles)

    # Calculate exposure time per ESA step
    # Divide by N_ESA_ENERGY_STEPS because each ESA step is only active
    # for 1/7 of the total pointing duration
    # Divide by N_SPIN_ANGLE_BINS to distribute uniformly across spin angles
    exposure_per_esa_step = total_pointing_duration / N_ESA_ENERGY_STEPS
    exposure_per_spin_bin = exposure_per_esa_step / N_SPIN_ANGLE_BINS

    # Apply off_angle weights: each spin_angle bin gets the same off_angle distribution
    # Shape: (N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
    exposure_per_bin = exposure_per_spin_bin * off_angle_weights[np.newaxis, :]

    # Broadcast exposure across ESA energy steps (each ESA step has the same
    # geometric exposure pattern, but only 1/7 of the total time)
    # Need to make a copy since we may modify it with good-times mask
    exposure_3d = np.broadcast_to(
        exposure_per_bin[np.newaxis, :, :],
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
    ).copy()

    # Apply good-times fraction if provided
    if goodtimes_df is not None:
        goodtimes_fraction = create_goodtimes_fraction(
            goodtimes_df, pointing_start_met, pointing_end_met
        )
        # Expand fraction to include off_angle dimension
        # (fraction is same for all off_angles)
        # goodtimes_fraction shape: (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS)
        # exposure_3d shape: (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
        exposure_3d = exposure_3d * goodtimes_fraction[:, :, np.newaxis]

        logging.debug(
            f"Applied good-times mask: exposure sum reduced from "
            f"{(exposure_per_bin.sum() * N_ESA_ENERGY_STEPS):.1f}s to "
            f"{exposure_3d.sum():.1f}s"
        )

    # Add epoch dimension
    exposure_4d = exposure_3d[np.newaxis, :, :, :]

    exposure_time = xr.DataArray(
        data=exposure_4d.astype(np.float32),
        dims=PSET_DIMS,
    )

    logging.debug(
        f"Calculated exposure times: total duration={total_pointing_duration:.1f}s, "
        f"sampled {len(representative_spins)} spins x {N_SAMPLES_PER_SPIN} samples, "
        f"exposure sum={exposure_per_bin.sum():.1f}s"
    )

    return exposure_time


def create_datasets(
    attr_mgr: ImapCdfAttributes, logical_source: str, data_fields: list[Field]
) -> xr.Dataset:
    """
    Create a dataset using the populated data classes.

    Parameters
    ----------
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.
    logical_source : str
        The logical source of the data product that's being created.
    data_fields : list[dataclasses.Field]
        List of Fields for data classes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all data product fields in xr.DataArray.
    """
    # TODO: Once L1B DE processing is implemented using the spin packet
    #  and relative L1A DE time to calculate the absolute DE time,
    #  this epoch conversion will go away and the time in the DE dataclass
    #  can be used direction
    epoch_converted_time = [1]

    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1c_pset":
        esa_energy_step = xr.DataArray(
            data=ESA_ENERGY_STEPS,
            name="esa_energy_step",
            dims=["esa_energy_step"],
            attrs=attr_mgr.get_variable_attributes("esa_energy_step"),
        )
        esa_energy_step_label = xr.DataArray(
            esa_energy_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )

        spin_angle = xr.DataArray(
            data=SPIN_ANGLE_BIN_CENTERS,
            name="spin_angle",
            dims=["spin_angle"],
            attrs=attr_mgr.get_variable_attributes("spin_angle"),
        )
        spin_angle_label = xr.DataArray(
            spin_angle.values.astype(str),
            name="spin_angle_label",
            dims=["spin_angle_label"],
            attrs=attr_mgr.get_variable_attributes("spin_angle_label"),
        )

        off_angle = xr.DataArray(
            data=OFF_ANGLE_BIN_CENTERS,
            name="off_angle",
            dims=["off_angle"],
            attrs=attr_mgr.get_variable_attributes("off_angle"),
        )
        off_angle_label = xr.DataArray(
            off_angle.values.astype(str),
            name="off_angle_label",
            dims=["off_angle_label"],
            attrs=attr_mgr.get_variable_attributes("off_angle_label"),
        )

        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "esa_energy_step": esa_energy_step,
                "esa_energy_step_label": esa_energy_step_label,
                "spin_angle": spin_angle,
                "spin_angle_label": spin_angle_label,
                "off_angle": off_angle,
                "off_angle_label": off_angle_label,
            },
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    # Loop through the data fields that were pulled from the
    # data class. These should match the field names given
    # to each field in the YAML attribute file
    for data_field in data_fields:
        field = data_field.name.lower()
        # Create a list of all the dimensions using the DEPEND_I keys in the
        # YAML attributes
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]

        # Create a data array for the current field and add it to the dataset
        # TODO: TEMPORARY. need to update to use l1b data once that's available.
        if field in [
            "pointing_start_met",
            "pointing_end_met",
            "esa_mode",
            "pivot_angle",
        ]:
            dataset[field] = xr.DataArray(
                data=[1],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        elif field == "exposure_time":
            dataset[field] = xr.DataArray(
                data=np.ones((1, 7, 3600, 40), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

        elif "rates" in field:
            dataset[field] = xr.DataArray(
                data=np.ones(PSET_SHAPE, dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                data=np.ones(PSET_SHAPE, dtype=np.int16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

    return dataset


def set_background_rates(
    pointing_start_met: float,
    pointing_end_met: float,
    species: FilterType,
    anc_dependencies: list,
    attr_mgr: ImapCdfAttributes,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Set the background rates for the specified species.

    The background rates are set to a constant value of 0.01 counts/s for all bins.

    Parameters
    ----------
    pointing_start_met : float
        The start MET time of the pointing.
    pointing_end_met : float
        The end MET time of the pointing.
    species : FilterType
        The species to set the background rates for. Can be "h" or "o".
    anc_dependencies : list
        Ancillary files needed for L1C data product creation.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the L1C attributes.

    Returns
    -------
    background_rates : tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple containing:
        - The background rates for the specified species.
        - The statistical uncertainties for the background rates.
        - The systematic errors for the background rates.
    """
    if species not in {FilterType.HYDROGEN, FilterType.OXYGEN}:
        raise ValueError(f"Species must be 'h' or 'o', but got {species.value}.")

    bg_rates = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )
    bg_stat_uncert = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )
    bg_sys_err = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )

    # read in the background rates from ancillary file
    if species == FilterType.HYDROGEN:
        background_df = lo_ancillary.read_ancillary_file(
            next(str(s) for s in anc_dependencies if "hydrogen-background" in str(s))
        )
    else:
        background_df = lo_ancillary.read_ancillary_file(
            next(str(s) for s in anc_dependencies if "oxygen-background" in str(s))
        )

    # find to the rows for the current pointing
    pointing_bg_df = background_df[
        (background_df["GoodTime_start"] >= pointing_start_met)
        & (background_df["GoodTime_end"] <= pointing_end_met)
    ]

    # convert the bin start and end resolution from 6 degrees to .1 degrees
    pointing_bg_df["bin_start"] = pointing_bg_df["bin_start"] * 60
    # The last bin end in the file is 0, which means 60 degrees. This is
    # converted to 0.1 degree resolution of 3600
    pointing_bg_df["bin_end"] = pointing_bg_df["bin_end"] * 60
    pointing_bg_df.loc[pointing_bg_df["bin_end"] == 0, "bin_end"] = 3600
    # for each row in the bg ancillary file for this pointing
    for _, row in pointing_bg_df.iterrows():
        bin_start = int(row["bin_start"])
        bin_end = int(row["bin_end"])
        # for each energy step, set the background rate and uncertainty
        for esa_step in range(0, 7):
            value = row[f"E-Step{esa_step + 1}"]
            if row["rate/sigma"] == "rate":
                bg_rates[esa_step, bin_start:bin_end, :] = value
            elif row["rate/sigma"] == "sigma":
                bg_sys_err[esa_step, bin_start:bin_end, :] = value
            else:
                raise ValueError("Unknown background type in ancillary file.")
    # set the background rates, uncertainties, and systematic errors
    bg_rates_data = xr.DataArray(
        data=bg_rates[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(f"{species.value}_background_rates"),
    )
    bg_stat_uncert_data = xr.DataArray(
        data=bg_stat_uncert[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_stat_uncert"
        ),
    )
    bg_sys_err_data = xr.DataArray(
        data=bg_sys_err[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_sys_err"
        ),
    )

    return bg_rates_data, bg_stat_uncert_data, bg_sys_err_data


def set_pointing_directions(
    epoch: float,
    attr_mgr: ImapCdfAttributes,
    pivot_angle: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Set the pointing directions for the given epoch.

    The pointing directions are calculated by transforming Spin and off angles
    to HAE longitude and latitude using SPICE. This returns the HAE longitude and
    latitude as (3600, 40) arrays for each the latitude and longitude.

    Parameters
    ----------
    epoch : float
        The epoch time in TTJ2000ns.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the L1C attributes.
    pivot_angle : float
        The pivot angle in degrees.
        Off-angles are adjusted relative to this pivot angle before transformation.

    Returns
    -------
    hae_longitude : xr.DataArray
        The HAE longitude for each spin and off angle bin.
    hae_latitude : xr.DataArray
        The HAE latitude for each spin and off angle bin.
    """
    et = ttj2000ns_to_et(epoch)
    # create a meshgrid of spin and off angles using the bin centers
    spin, off = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    # off_angles need to account for the pivot_angle
    off += 90 - pivot_angle
    dps_az_el = np.stack([spin, off], axis=-1)

    # Transform from DPS Az/El to HAE lon/lat
    hae_az_el = frame_transform_az_el(
        et, dps_az_el, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_HAE, degrees=True
    )

    return xr.DataArray(
        data=hae_az_el[np.newaxis, :, :, 0].astype(np.float64),
        dims=["epoch", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes("hae_longitude"),
    ), xr.DataArray(
        data=hae_az_el[np.newaxis, :, :, 1].astype(np.float64),
        dims=["epoch", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes("hae_latitude"),
    )
