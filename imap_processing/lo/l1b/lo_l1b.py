"""IMAP-Lo L1B Data Processing."""

import logging
from dataclasses import Field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo import lo_ancillary
from imap_processing.lo.l1b.tof_conversions import (
    TOF0_CONV,
    TOF1_CONV,
    TOF2_CONV,
    TOF3_CONV,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_latitudinal,
    frame_transform,
    get_spacecraft_to_instrument_spin_phase_offset,
    lo_instrument_pointing,
)
from imap_processing.spice.repoint import (
    get_pointing_mid_time,
    get_pointing_times,
    interpolate_repoint_data,
)
from imap_processing.spice.spin import (
    get_spin_data,
    get_spin_number,
    interpolate_spin_data,
)
from imap_processing.spice.time import (
    epoch_to_fractional_doy,
    et_to_utc,
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Centralized field definitions to avoid repetition across functions
# -------------------------------------------------------------------
# spin-bin fields (count fields used in multiple places)
SPIN_BIN_6_FIELDS = [
    "h_counts",
    "o_counts",
    "tof0_tof1_counts",
    "tof0_tof2_counts",
    "tof1_tof2_counts",
    "silver_triple_counts",
]

SPIN_BIN_60_FIELDS = [
    "start_a_counts",
    "start_c_counts",
    "stop_b0_counts",
    "stop_b3_counts",
    "tof0_counts",
    "tof1_counts",
    "tof2_counts",
    "tof3_counts",
    "disc_tof0_counts",
    "disc_tof1_counts",
    "disc_tof2_counts",
    "disc_tof3_counts",
    "pos0_counts",
    "pos1_counts",
    "pos2_counts",
    "pos3_counts",
]

# Mapping from L1A field names to L1B count field names used in initialize_all_rates
SPIN_BIN_6_L1A_TO_L1B = {
    "hydrogen": "h_counts",
    "oxygen": "o_counts",
    "tof0_tof1": "tof0_tof1_counts",
    "tof0_tof2": "tof0_tof2_counts",
    "tof1_tof2": "tof1_tof2_counts",
    "silver": "silver_triple_counts",
}

SPIN_BIN_60_L1A_TO_L1B = {
    "start_a": "start_a_counts",
    "start_c": "start_c_counts",
    "stop_b0": "stop_b0_counts",
    "stop_b3": "stop_b3_counts",
    "tof0_count": "tof0_counts",
    "tof1_count": "tof1_counts",
    "tof2_count": "tof2_counts",
    "tof3_count": "tof3_counts",
    "disc_tof0": "disc_tof0_counts",
    "disc_tof1": "disc_tof1_counts",
    "disc_tof2": "disc_tof2_counts",
    "disc_tof3": "disc_tof3_counts",
    "pos0": "pos0_counts",
    "pos1": "pos1_counts",
    "pos2": "pos2_counts",
    "pos3": "pos3_counts",
}

# Count-field -> rate-field mappings used by calculate_histogram_rates
SPIN_BIN_6_COUNT_TO_RATE = {
    "h_counts": "h_rates",
    "o_counts": "o_rates",
    "tof0_tof1_counts": "tof0_tof1_rates",
    "tof0_tof2_counts": "tof0_tof2_rates",
    "tof1_tof2_counts": "tof1_tof2_rates",
    "silver_triple_counts": "silver_triple_rates",
}

SPIN_BIN_60_COUNT_TO_RATE = {
    "start_a_counts": "start_a_rates",
    "start_c_counts": "start_c_rates",
    "stop_b0_counts": "stop_b0_rates",
    "stop_b3_counts": "stop_b3_rates",
    "tof0_counts": "tof0_rates",
    "tof1_counts": "tof1_rates",
    "tof2_counts": "tof2_rates",
    "tof3_counts": "tof3_rates",
    "disc_tof0_counts": "disc_tof0_rates",
    "disc_tof1_counts": "disc_tof1_rates",
    "disc_tof2_counts": "disc_tof2_rates",
    "disc_tof3_counts": "disc_tof3_rates",
    "pos0_counts": "pos0_rates",
    "pos1_counts": "pos1_rates",
    "pos2_counts": "pos2_rates",
    "pos3_counts": "pos3_rates",
}

# Fields to include in the split hist/monitor rate datasets
HIST_RATE_FIELDS = [
    "h_rates",
    "o_rates",
    "h_counts",
    "o_counts",
    "esa_mode",
    "exposure_time_6deg",
    "spin_cycle",
]
MONITOR_RATE_FIELDS = [
    "tof0_tof1_rates",
    "tof0_tof2_rates",
    "tof1_tof2_rates",
    "silver_triple_rates",
    "start_a_rates",
    "start_c_rates",
    "stop_b0_rates",
    "stop_b3_rates",
    "tof0_rates",
    "tof1_rates",
    "tof2_rates",
    "tof3_rates",
    "disc_tof0_rates",
    "disc_tof1_rates",
    "disc_tof2_rates",
    "disc_tof3_rates",
    "pos0_rates",
    "pos1_rates",
    "pos2_rates",
    "pos3_rates",
    "esa_mode",
    "exposure_time_60deg",
    "exposure_time_6deg",
    "spin_cycle",
]
# -------------------------------------------------------------------
DE_CLOCK_TICK_S = 4.096e-3  # seconds per DE clock tick


def lo_l1b(
    sci_dependencies: dict, anc_dependencies: list, descriptor: str
) -> list[Path]:
    """
    Will process IMAP-Lo L1A data into L1B CDF data products.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.
    anc_dependencies : list
        List of ancillary file paths needed for L1B data product creation.
    descriptor : str
        Determines which datasets are produced.

    Returns
    -------
    created_file_paths : list[pathlib.Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1b")
    # create the attribute manager to access L1A fillval attributes
    attr_mgr_l1a = ImapCdfAttributes()
    attr_mgr_l1a.add_instrument_variable_attrs(instrument="lo", level="l1a")
    logger.info(f"\n Dependencies: {list(sci_dependencies.keys())}\n")

    datasets_to_return = []

    if descriptor == "badtimes":
        logger.info("\nProcessing IMAP-Lo L1B Bad Times...")
        badtimes_ds = create_badtimes_dataset()
        badtimes_ds.attrs = attr_mgr_l1b.get_global_attributes("imap_lo_l1b_badtimes")
        if len(badtimes_ds["epoch"]) > 0:
            # Only add the dataset if there are bad times added
            datasets_to_return.append(badtimes_ds)

    # if the dependencies are used to create Annotated Direct Events
    if descriptor == "de":
        logger.info("\nProcessing IMAP-Lo L1B Direct Events...")
        ds = l1b_de(sci_dependencies, anc_dependencies, attr_mgr_l1b, attr_mgr_l1a)
        datasets_to_return.append(ds)

    # If dependencies are used to create Histogram Rates
    if descriptor == "all-rates":
        logger.info("\nProcessing IMAP-Lo L1B Hist and Monitor Rates...")
        ds = l1b_allrates(sci_dependencies, anc_dependencies, attr_mgr_l1b)
        datasets_to_return.extend(ds)

    if descriptor == "derates":
        logger.info("\nProcessing IMAP-Lo L1B DE Rates...")
        ds = calculate_de_rates(sci_dependencies, anc_dependencies, attr_mgr_l1b)
        datasets_to_return.append(ds)

    if descriptor == "star":
        logger.info("\nProcessing IMAP-Lo L1B Star Sensor Profile...")
        ds = l1b_star(sci_dependencies, attr_mgr_l1b)
        datasets_to_return.append(ds)

    return datasets_to_return


def l1b_de(
    sci_dependencies: dict,
    anc_dependencies: list,
    attr_mgr_l1b: ImapCdfAttributes,
    attr_mgr_l1a: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Create the IMAP-Lo L1B Direct Events dataset.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.
    anc_dependencies : list
        List of ancillary file paths needed for L1B data product creation.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the global attributes.
    attr_mgr_l1a : ImapCdfAttributes
        Attribute manager used to get the variable attributes.

    Returns
    -------
    l1b_de : xr.Dataset
        The IMAP-Lo L1B Direct Events dataset.
    """
    logical_source = "imap_lo_l1b_de"
    # get the dependency dataset for l1b direct events
    l1a_de = sci_dependencies["imap_lo_l1a_de"]
    spin_data = sci_dependencies["imap_lo_l1a_spin"]
    l1b_nhk = sci_dependencies["imap_lo_l1b_nhk"]

    # Initialize the L1B DE dataset
    l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)
    # Get the pivot angle from the housekeeping dataset
    pivot_angle = _get_nearest_pivot_angle(l1b_de["epoch"].values[0], l1b_nhk)
    l1b_de["pivot_angle"] = xr.DataArray([pivot_angle], dims=["pivot_angle"])

    pointing_start_met, pointing_end_met = get_pointing_times(
        l1a_de["met"].values[0].item()
    )

    # Get the average spin durations for each epoch
    avg_spin_durations_per_cycle = get_avg_spin_durations_per_cycle(spin_data)
    # set the spin cycle for each direct event
    l1b_de = set_spin_cycle(pointing_start_met, l1a_de, l1b_de)

    # get the absolute met for each event
    l1b_de = set_event_met(l1a_de, l1b_de)
    # set the epoch for each event
    l1b_de = set_each_event_epoch(l1b_de)
    # Set the ESA mode for each direct event
    l1b_de = set_esa_mode(
        pointing_start_met, pointing_end_met, anc_dependencies, l1b_de
    )
    # Set the average spin duration for each direct event
    l1b_de = set_avg_spin_durations_per_event(
        l1a_de, l1b_de, avg_spin_durations_per_cycle
    )
    # calculate the TOF1 for golden triples
    # store in the l1a dataset to use in l1b calculations
    l1a_de = calculate_tof1_for_golden_triples(l1a_de)
    # set the coincidence type string for each direct event
    l1b_de = set_coincidence_type(l1a_de, l1b_de, attr_mgr_l1a)
    # convert the TOFs to engineering units
    l1b_de = convert_tofs_to_eu(l1a_de, l1b_de, attr_mgr_l1a, attr_mgr_l1b)
    # set the species for each direct event
    l1b_de = identify_species(l1b_de)
    # set the pointing direction for each direct event
    l1b_de = set_pointing_direction(l1b_de)
    # calculate and set the pointing bin based on the spin phase
    # pointing bin is 3600 x 40 bins
    l1b_de = set_pointing_bin(l1b_de)
    # set the badtimes
    l1b_de = set_bad_times(l1b_de, anc_dependencies)
    return l1b_de


def l1b_allrates(
    sci_dependencies: dict, anc_dependencies: list, attr_mgr_l1b: ImapCdfAttributes
) -> xr.Dataset:
    """
    Create the IMAP-Lo L1B Histogram Rates dataset.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.
    anc_dependencies : list
        List of ancillary file paths needed for L1B data product creation.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the global attributes.

    Returns
    -------
    [xr.Dataset, xr.Dataset]
        The IMAP-Lo L1B Histogram and Monitor Rates datasets.
    """
    datasets_to_return = []
    # get the dependency dataset for l1b histogram rates
    l1a_hist = sci_dependencies["imap_lo_l1a_histogram"]
    spin_data = sci_dependencies["imap_lo_l1a_spin"]
    # initialize the L1B Histogram Rates dataset from the L1A Histogram Rates
    # This carries over the epoch and count fields from L1A
    l1b_all_rates = initialize_all_rates(l1a_hist, attr_mgr_l1b)
    # set spin cycle and remove invalid spin ASCs
    l1b_all_rates = set_spin_cycle_from_spin_data(l1a_hist, l1b_all_rates, spin_data)

    pointing_start_met, pointing_end_met = get_pointing_times(
        ttj2000ns_to_met(l1a_hist["epoch"].values[0].item())
    )
    l1b_all_rates = set_esa_mode(
        pointing_start_met, pointing_end_met, anc_dependencies, l1b_all_rates
    )
    # resweep the histogram data
    l1b_all_rates, exposure_factor = resweep_histogram_data(
        l1b_all_rates, anc_dependencies
    )
    # Get the start and end times for each spin epoch
    acq_start, acq_end = convert_start_end_acq_times(spin_data)
    # Get the average spin durations for each epoch
    avg_spin_durations_per_cycle = get_avg_spin_durations_per_cycle(spin_data)
    l1b_all_rates = calculate_histogram_rates(
        l1b_all_rates,
        acq_start,
        acq_end,
        avg_spin_durations_per_cycle,
        exposure_factor,
    )

    l1b_hist_rates, l1b_monitor_rates = split_rate_dataset(l1b_all_rates, attr_mgr_l1b)
    datasets_to_return.extend([l1b_hist_rates, l1b_monitor_rates])

    return datasets_to_return


def initialize_l1b_de(
    l1a_de: xr.Dataset, attr_mgr_l1b: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Initialize the L1B DE dataset.

    Create an empty L1B DE dataset and copy over fields from the L1A DE that will
    not change during L1B processing.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the global attributes for the L1B DE dataset.
    logical_source : str
        The logical source of the direct event product.

    Returns
    -------
    l1b_de : xarray.Dataset
        The initialized L1B DE dataset.
    """
    l1b_de = xr.Dataset(
        attrs=attr_mgr_l1b.get_global_attributes(logical_source),
    )

    # Copy over fields from L1A DE that will not change in L1B processing
    l1b_de["pos"] = xr.DataArray(
        l1a_de["pos"].values,
        dims=["epoch"],
        # TODO: Add pos to YAML file
        # attrs=attr_mgr.get_variable_attributes("pos"),
    )
    l1b_de["mode_bit"] = xr.DataArray(
        l1a_de["mode"].values,
        dims=["epoch"],
        # TODO: Add mode to YAML file
        # attrs=attr_mgr.get_variable_attributes("mode"),
    )
    l1b_de["absent"] = xr.DataArray(
        l1a_de["coincidence_type"].values,
        dims=["epoch"],
        # TODO: Add absent to YAML file
        # attrs=attr_mgr.get_variable_attributes("absent"),
    )
    l1b_de["esa_step"] = xr.DataArray(
        l1a_de["esa_step"].values,
        dims=["epoch"],
        # TODO: Add esa_step to YAML file
        # attrs=attr_mgr.get_variable_attributes("esa_step"),
    )
    l1b_de["shcoarse"] = xr.DataArray(
        np.repeat(l1a_de["shcoarse"].values, l1a_de["de_count"].values),
        dims=["epoch"],
        # TODO: Add shcoarse to YAML file
        # attrs=attr_mgr.get_variable_attributes("shcoarse"),
    )

    return l1b_de


def set_esa_mode(
    pointing_start_met: float,
    pointing_end_met: float,
    anc_dependencies: list,
    l1b_science: xr.Dataset,
) -> xr.Dataset:
    """
    Set the ESA mode for each direct event or histogram.

    The ESA mode is determined from the sweep table for the time period of the pointing.

    Parameters
    ----------
    pointing_start_met : float
        Start time for the pointing in MET seconds.
    pointing_end_met : float
        End time for the pointing in MET seconds.
    anc_dependencies : list
        List of ancillary file paths.
    l1b_science : xarray.Dataset
        The L1B science dataset.

    Returns
    -------
    l1b_science : xr.Dataset
        The L1B science dataset with the ESA mode added.
    """
    # Read the sweep table from the ancillary files
    sweep_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "sweep-table" in str(s))
    )

    # Get the sweep table rows that correspond to the time period of the pointing
    pointing_sweep_df = sweep_df[
        (sweep_df["GoodTime_start"] >= pointing_start_met)
        & (sweep_df["GoodTime_start"] <= pointing_end_met)
    ]

    # Check that there is only one ESA mode in the sweep table for the pointing
    if len(pointing_sweep_df["ESA_Mode"].unique()) == 1:
        # Update the ESA mode strings to be 0 for HiRes and 1 for HiThr
        sweep_df["esa_mode"] = sweep_df["ESA_Mode"].map({"HiRes": 0, "HiThr": 1})
        # Get the ESA mode for the pointing
        esa_mode = sweep_df["esa_mode"].values[0]
        # Repeat the ESA mode for each direct event in the pointing
        esa_mode_array = np.repeat(esa_mode, len(l1b_science["epoch"]))
    else:
        raise ValueError("Multiple ESA modes found in sweep table for pointing.")

    l1b_science["esa_mode"] = xr.DataArray(
        esa_mode_array,
        dims=["epoch"],
        # TODO: Add esa_mode to YAML file
        # attrs=attr_mgr.get_variable_attributes("esa_mode"),
    )

    return l1b_science


def convert_start_end_acq_times(
    spin_data: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convert the start and end times from the spin data.

    The L1A spin data start and end acquisition times are stored in seconds and
    subseconds (microseconds). This function converts them to a single time in seconds.

    Parameters
    ----------
    spin_data : xarray.Dataset
        The L1A Spin dataset containing the start and end acquisition times.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing the start and end acquisition times as xarray DataArrays.
    """
    # Convert subseconds from microseconds to seconds
    acq_start = spin_data["acq_start_sec"] + spin_data["acq_start_subsec"] * 1e-6
    acq_end = spin_data["acq_end_sec"] + spin_data["acq_end_subsec"] * 1e-6
    return (acq_start, acq_end)


def get_avg_spin_durations_per_cycle(
    spin_data: xr.Dataset,
) -> xr.DataArray:
    """
    Get the average spin duration for each aggregated science cycle.

    Parameters
    ----------
    spin_data : xarray.Dataset
        The L1A Spin dataset.

    Returns
    -------
    avg_spin_durations : xarray.DataArray
        The average spin duration for each ASC.
    """
    acq_start = spin_data["acq_start_sec"] + spin_data["acq_start_subsec"] * 1e-6
    acq_end = spin_data["acq_end_sec"] + spin_data["acq_end_subsec"] * 1e-6
    # Get the avg spin duration for each spin epoch
    # We need to use the number of spins that were actually in the ASC
    # because there may be partial ASCs where only some of the spins were completed
    avg_spin_durations_per_cycle = (acq_end - acq_start) / spin_data["num_completed"]
    return avg_spin_durations_per_cycle


def set_spin_cycle(
    pointing_start_met: float, l1a_de: xr.Dataset, l1b_de: xr.Dataset
) -> xr.Dataset:
    """
    Set the spin cycle for each direct event.

    spin_cycle = spin_start + 7 + (esa_step - 1) * 2

    where spin_start is the spin number for the first spin
    in an Aggregated Science Cycle (ASC) and esa_step is the esa_step for a direct event

    The 28 spins in a spin epoch spans one ASC.

    Parameters
    ----------
    pointing_start_met : float
        The start time of the pointing in MET seconds.
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the spin cycle added for each direct event.
    """
    spin_start_num = get_spin_number(pointing_start_met)
    counts = l1a_de["de_count"].values
    # split the esa_steps into ASC groups
    de_asc_groups = np.split(l1a_de["esa_step"].values, np.cumsum(counts)[:-1])
    spin_cycle = []
    for esa_asc_group in de_asc_groups:
        # calculate the spin cycle for each DE in the ASC group
        # TODO: Add equation number in algorithm document when new version is
        #  available. Add to docstring as well
        spin_cycle.extend(spin_start_num + 7 + (esa_asc_group - 1) * 2)
        # increment the spin start number by 28 for the next ASC
        spin_start_num += 28

    l1b_de["spin_cycle"] = xr.DataArray(
        spin_cycle,
        dims=["epoch"],
        # TODO: Add spin cycle to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_cycle"),
    )

    return l1b_de


# TODO: The spin cycle function above needs to be updated for DEs. We cannot assume
#  there are 28 spins per ASC and we should calculate the spin start number based on the
#  corresponding L1A spin data Acq Start for the ASC. The implementation below should be
#  should be used for the DE rather than the above function, but in the interest of time
#  the below function is only hooked up to the histogram rates processing and should be
#  integrated into the DE processing in a later PR.
# TODO: Break up the invalid spin ASC removal and the code to find the closest DE/Hist
#  and spin ASCs into their own functions.
def set_spin_cycle_from_spin_data(
    l1a_science: xr.Dataset, l1b_science: xr.Dataset, spin_data: xr.Dataset
) -> xr.Dataset:
    """
    Set the spin cycle for each direct event using the L1A spin data.

    The spin cycle is the average spin for a given Aggregated Science Cycle
     in a given ESA Step.

    Parameters
    ----------
    l1a_science : xr.Dataset
        The L1A Histogram or Direct Event dataset.
    l1b_science : xr.Dataset
        The L1B Histogram Rate or Direct Event dataset.
    spin_data : xr.Dataset
        The L1A Spin dataset.

    Returns
    -------
    l1b_science : xr.Dataset
        The L1B science dataset with the spin cycle added for each direct event.
    """
    acq_start, _acq_end = convert_start_end_acq_times(spin_data)

    spin_met_per_asc = spin_data["shcoarse"].values.astype(np.float64)
    science_met_per_asc = ttj2000ns_to_met(l1a_science["epoch"]).astype(np.float64)

    science_to_spin_indices = match_science_to_spin_asc(
        science_met_per_asc, spin_met_per_asc
    )

    valid_mask = find_valid_asc(science_to_spin_indices, spin_data)

    # If none valid, return an empty/filtered dataset
    # (preserves dims & avoids misalignment)
    if not valid_mask.any():
        logger.warning(
            "No valid ASCs remain after filtering; returning empty epoch set"
        )
        return l1b_science.isel(epoch=[])

    # Filter the input datasets to only the valid ASCs so all subsequent arrays align
    l1a_valid = l1a_science.isel(epoch=valid_mask)
    l1b_valid = l1b_science.isel(epoch=valid_mask)

    # Use the valid closest indices to get the corresponding acq_start rows
    science_to_spin_indices_valid = science_to_spin_indices[valid_mask]
    closest_start_acq_per_asc = acq_start.isel(epoch=science_to_spin_indices_valid)

    # compute spin start number for each remaining ASC
    spin_start_num_per_asc = np.atleast_1d(get_spin_number(closest_start_acq_per_asc))
    spin_start_num_per_asc = spin_start_num_per_asc[:, None]  # (n_valid, 1)

    logical_src = l1a_science.attrs.get("Logical_source", "")
    if logical_src == "imap_lo_l1a_de":
        # For DE: expand per-event across ESA steps within each (valid) ASC
        counts = l1a_valid["de_count"].values
        spin_cycle = []
        for asc_idx, _count in enumerate(counts):
            esa_steps = l1a_valid["esa_step"].values[
                sum(counts[:asc_idx]) : sum(counts[: asc_idx + 1])
            ]
            spin_cycle.extend(
                spin_start_num_per_asc[asc_idx, 0] + 7 + (esa_steps - 1) * 2
            )
        spin_cycle = np.array(spin_cycle)
        l1b_valid["spin_cycle"] = xr.DataArray(spin_cycle, dims=["epoch"])
    elif logical_src == "imap_lo_l1a_histogram":
        # For histogram: keep 2D array (n_valid_epochs, esa_step)
        esa_steps = l1b_valid["esa_step"].values  # shape: (7,)
        spin_cycle = spin_start_num_per_asc + 7 + (esa_steps - 1) * 2
        l1b_valid["spin_cycle"] = xr.DataArray(spin_cycle, dims=["epoch", "esa_step"])
    else:
        raise ValueError(
            "set spin cycle called with unsupported dataset with "
            "Logical_source: {logical_src}"
        )

    return l1b_valid


def match_science_to_spin_asc(
    science_met_per_asc: xr.DataArray, spin_met_per_asc: xr.DataArray
) -> np.ndarray:
    """
    Compute the indices of the closest spin acquisition times for each science event.

    This function matches science data acquisition epochs to spin data acquisition
    epochs by finding the closest spin acquisition indices for each science data
    acquisition epoch. The result is an array where each element corresponds to the
    index of the closest spin data acquisition time for a given science event.

    Parameters
    ----------
    science_met_per_asc : xr.DataArray
        An array of science acquisition epochs in MET seconds.
    spin_met_per_asc : xr.DataArray
        An array of spin acquisition epochs in MET seconds.

    Returns
    -------
    science_to_spin_indices : np.ndarray
        Index of closest prior spin ASC for each science ASC.
        Set to -1 if no valid prior spin exists.
    """
    # Find the closest spin shcoarse for each science ASC
    # computes the index of the closest spin_met_per_asc for each science_met_per_asc
    # so the resulting array will be of length len(science_met_per_asc), one index per
    # ASC, but the value of each index will be the index of the closest spin data.
    science_to_spin_indices = np.abs(
        science_met_per_asc[:, None] - spin_met_per_asc
    ).argmin(axis=1)

    return science_to_spin_indices


def find_valid_asc(
    science_to_spin_indices: np.ndarray,
    spin_data: xr.Dataset,
) -> np.ndarray:
    """
    Find valid Aggregated Science Cycles by filtering invalid spin data.

    Parameters
    ----------
    science_to_spin_indices : np.ndarray
        Indices of closest spin acquisitions.
    spin_data : xr.Dataset
        The L1A Spin dataset.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask indicating valid ASCs.
    """
    # Apply each validation check independently on full arrays
    valid_indices = _check_valid_indices(science_to_spin_indices)
    valid_spin_count = _check_sufficient_spins(spin_data)[science_to_spin_indices]

    # Combine only these two masks:
    valid_mask = valid_indices & valid_spin_count

    total_invalid = (~valid_mask).sum()
    if total_invalid > 0:
        logger.info(f"Dropping {total_invalid} invalid ASCs total")

    return valid_mask


def _check_valid_indices(science_to_spin_indices: np.ndarray) -> np.ndarray:
    """
    Check that all matched spin indices are valid (non-negative).

    Parameters
    ----------
    science_to_spin_indices : np.ndarray
        Indices of closest spin acquisitions.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask where True indicates a valid index.
    """
    invalid_indices = science_to_spin_indices < 0
    if invalid_indices.any():
        logger.warning(f"Found {invalid_indices.sum()} ASCs with invalid spin indices")
    return ~invalid_indices


def _check_sufficient_spins(spin_data: xr.Dataset) -> np.ndarray:
    """
    Check that matched spin cycles have sufficient spins (28 completed).

    Parameters
    ----------
    spin_data : xr.Dataset
        The L1A Spin dataset containing num_completed field.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask where True indicates sufficient spins.
    """
    # Check if corresponding spin cycle has 28 spins
    valid_mask = spin_data["num_completed"].values == 28

    if (~valid_mask).any():
        logger.warning(f"Found {(~valid_mask).sum()} ASCs with fewer than 28 spins")

    return valid_mask


def get_spin_start_times(
    l1a_de: xr.Dataset,
) -> xr.DataArray:
    """
    Get the start time for the spin that each direct event is in.

    The resulting array of spin start times will be equal to the length of the direct
    events. If two direct events occurred in the same spin, then there will be repeating
    spin start times.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.

    Returns
    -------
    spin_start_time : np.ndarray
        The start time for the spin that each direct event is in.
    """
    # Get the actual spin start times from the spin data
    # Use the individual spin start times rather than calculating from ASC averages
    spin_start_times = interpolate_spin_data(l1a_de["shcoarse"].values)[
        "spin_start_met"
    ].values

    return spin_start_times


def set_event_met(
    l1a_de: xr.Dataset,
    l1b_de: xr.Dataset,
) -> xr.Dataset:
    """
    Get the event MET for each direct event.

    Each direct event is converted from a data number to engineering unit in seconds.
    time_from_start_of_spin = de_time * DE_CLOCK_TICK_S
    where de_time is the direct event time Data Number (DN).

    The direct event time is the time of direct event relative to the start of the spin.
    The event MET is the sum of the start time of the spin and the
    direct event EU time.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.
    l1b_de : xr.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xr.Dataset
        The L1B DE dataset with the event MET.
    """
    # get spin start times for each event
    spin_start_times = get_spin_start_times(l1a_de)

    # spin start + offset based on de_time ticks
    l1b_de["event_met"] = xr.DataArray(
        spin_start_times + l1a_de["de_time"].values * DE_CLOCK_TICK_S,
        dims=["epoch"],
        # attrs=attr_mgr.get_variable_attributes("epoch")
    )
    return l1b_de


def set_each_event_epoch(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the epoch for each direct event.

    Parameters
    ----------
    l1b_de : xr.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xr.Dataset
        The L1B DE dataset with the epoch set for each event.
    """
    l1b_de["epoch"] = xr.DataArray(
        met_to_ttj2000ns(l1b_de["event_met"].values),
        dims=["epoch"],
        # attrs=attr_mgr.get_variable_attributes("epoch")
    )
    return l1b_de


def set_avg_spin_durations_per_event(
    l1a_de: xr.Dataset, l1b_de: xr.Dataset, avg_spin_durations_per_cycle: xr.DataArray
) -> xr.DataArray:
    """
    Set the average spin duration for each direct event.

    The average spin duration for each cycle is repeated for the number of
    direct event counts in the cycle. For example, if there are two Aggregated
    Science Cycles with 2 events in the first cycle and 1 event in the second
    cycle and the average spin duration for each cycle is duration1, duration2,
    this will result in: [duration1, duration 1, duration2]

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.
    avg_spin_durations_per_cycle : xarray.DataArray
        The average spin duration for each spin epoch.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the average spin duration added.
    """
    # repeat the average spin durations for each cycle based on the direct event count
    # to get an average spin duration for each direct event. This will be used in L1C
    # to calculate the exposure time for each direct event.
    l1b_de["avg_spin_durations"] = xr.DataArray(
        np.repeat(avg_spin_durations_per_cycle.values, l1a_de["de_count"]),
        dims=["epoch"],
    )
    return l1b_de


def calculate_tof1_for_golden_triples(l1a_de: xr.Dataset) -> xr.Dataset:
    """
    Calculate the TOF1 for golden triples.

    TOF1 is not transmitted for golden triples, but is recovered on the
    ground using the TOF0, TOF2, TOF3, and CKSUM values. The equation is:
    TOF1 = (TOF0 + TOF3 - TOF2 - CKSUM - left_cksm_bound) << 1

    where left_cksm_bound is the left checksum boundary value. This is a
    constant value that is not transmitted in the telemetry.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.

    Returns
    -------
    l1a_de : xr.Dataset
        The L1A DE dataset with the TOF1 calculated for golden triples.
    """
    for idx, coin_type in enumerate(l1a_de["coincidence_type"].values):
        # NOTE: mode bit of 1 is used to identify golden triple (event was compressed)
        if coin_type == 0 and l1a_de["mode"][idx] == 1:
            # Calculate TOF1
            # TOF1 equation requires values to be right bit shifted. These values were
            # originally right bit shifted when packed in the telemetry packet, but were
            # left bit shifted for the L1A product. Need to right bit shift them again
            # to apply the TOF1 equation
            tof0 = l1a_de["tof0"][idx] >> 1
            tof2 = l1a_de["tof2"][idx] >> 1
            tof3 = l1a_de["tof3"][idx] >> 1
            cksm = l1a_de["cksm"][idx] >> 1
            # TODO: will get left checksum boundary from LUT table when available
            left_cksm_bound = -21
            # Calculate TOF1, then left bit shift it to store it with the rest of the
            # left shifted L1A dataset data.
            l1a_de["tof1"][idx] = (tof0 + tof3 - tof2 - cksm - left_cksm_bound) << 1
    return l1a_de


def set_coincidence_type(
    l1a_de: xr.Dataset,
    l1b_de: xr.Dataset,
    attr_mgr_l1a: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Set the coincidence type for each direct event.

    The coincidence type is a string that indicates the type of coincidence
    for each direct event. The string is a combination of the following depending
    on whether the TOF or CKSM value is present (1) or absent (0) and the value
    of the mode for each direct event:
    "<TOF0><TOF1><TOF2><TOF3><CKSM><Mode>"

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.
    attr_mgr_l1a : ImapCdfAttributes
        Attribute manager used to get the fill values for the L1A DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the coincidence type added.
    """
    tof0_fill = attr_mgr_l1a.get_variable_attributes("tof0")["FILLVAL"]
    tof0_mask = (l1a_de["tof0"].values != tof0_fill).astype(int)
    tof1_fill = attr_mgr_l1a.get_variable_attributes("tof1")["FILLVAL"]
    tof1_mask = (l1a_de["tof1"].values != tof1_fill).astype(int)
    tof2_fill = attr_mgr_l1a.get_variable_attributes("tof2")["FILLVAL"]
    tof2_mask = (l1a_de["tof2"].values != tof2_fill).astype(int)
    tof3_fill = attr_mgr_l1a.get_variable_attributes("tof3")["FILLVAL"]
    tof3_mask = (l1a_de["tof3"].values != tof3_fill).astype(int)
    cksm_fill = attr_mgr_l1a.get_variable_attributes("cksm")["FILLVAL"]
    cksm_mask = (l1a_de["cksm"].values != cksm_fill).astype(int)

    coincidence_type = [
        f"{tof0_mask[i]}{tof1_mask[i]}{tof2_mask[i]}{tof3_mask[i]}{cksm_mask[i]}{l1a_de['mode'].values[i]}"
        for i in range(len(l1a_de["direct_events"]))
    ]

    l1b_de["coincidence_type"] = xr.DataArray(
        coincidence_type,
        dims=["epoch"],
        # TODO: Add coincidence_type to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_cycle"),
    )

    return l1b_de


def convert_tofs_to_eu(
    l1a_de: xr.Dataset,
    l1b_de: xr.Dataset,
    attr_mgr_l1a: ImapCdfAttributes,
    attr_mgr_l1b: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Convert the TOFs to engineering units.

    The TOFs are converted from data numbers (DN) to engineering units (EU) using the
    following equation:
    TOF_EU = C0 + C1 * TOF_DN

    where C0 and C1 are the conversion coefficients for each TOF.

    This equation is applied to all four TOFs (TOF0, TOF1, TOF2, TOF3).

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.
    attr_mgr_l1a : ImapCdfAttributes
        Attribute manager used to get the fill values for the L1A DE dataset.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the fill values for the L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the TOFs converted to engineering units.
    """
    tof_fields = ["tof0", "tof1", "tof2", "tof3"]
    tof_conversions = [TOF0_CONV, TOF1_CONV, TOF2_CONV, TOF3_CONV]

    # Loop through the TOF fields and convert them to engineering units
    for tof, conv in zip(tof_fields, tof_conversions, strict=False):
        # Get the fill value for the L1A and L1B TOF
        fillval_1a = attr_mgr_l1a.get_variable_attributes(tof)["FILLVAL"]
        fillval_1b = attr_mgr_l1b.get_variable_attributes(tof)["FILLVAL"]
        # Create a mask for the TOF
        mask = l1a_de[tof] != fillval_1a
        # Convert the DN TOF to EU and add the EU TOF to the dataset.
        # If the TOF is not present, set it to the fill value for the L1B TOF data.
        tof_eu = np.where(
            mask,
            conv.C0 + conv.C1 * l1a_de[tof],
            fillval_1b,
        )
        l1b_de[tof] = xr.DataArray(
            tof_eu,
            dims=["epoch"],
            attrs=attr_mgr_l1b.get_variable_attributes(tof),
        )

    return l1b_de


def identify_species(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Identify the species for each direct event.

    The species are determined using the U_PAC 7-13kV range table with the TOF2 value.
    Each event is set to "H" for Hydrogen, "O" for Oxygen, or "U" for Unknown.

    See the species identification section in the Lo algorithm document for more
    information on the ranges used to identify the species.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the species identified.
    """
    # Define upper and lower ranges for Hydrogen and Oxygen
    # Table defined in 9.3.4.4 of the Lo algorithm document
    # UNH-IMAP-Lo-27850-6002-Data-Product-Algorithms-v9_&_IMAP-LoMappingAlgorithm
    # The ranges are used for U_PAC voltages 7-12kV. Lo does not expect to use
    # voltages outside of that range.
    range_hydrogen = (13, 40)
    range_oxygen = (75, 200)

    # Initialize the species array with U for Unknown
    species = np.full(len(l1b_de["epoch"]), "U")

    tof2 = l1b_de["tof2"]
    # Check for range Hydrogen using the TOF2 value
    mask_h = (tof2 >= range_hydrogen[0]) & (tof2 <= range_hydrogen[1])
    species[mask_h] = "H"

    # Check for range Oxygen using the TOF2 value
    mask_oxygen = (tof2 >= range_oxygen[0]) & (tof2 <= range_oxygen[1])
    species[mask_oxygen] = "O"

    # Add species to the dataset
    l1b_de["species"] = xr.DataArray(
        species,
        dims=["epoch"],
        # TODO: Add to yaml
        # attrs=attr_mgr.get_variable_attributes("species"),
    )

    return l1b_de


def set_bad_times(l1b_de: xr.Dataset, anc_dependencies: list) -> xr.Dataset:
    """
    Set the bad times for each direct event.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B DE dataset.
    anc_dependencies : list
        List of ancillary file paths.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the bad times added.
    """
    badtimes_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "bad-times" in str(s))
    )

    esa_steps = l1b_de["esa_step"].values
    epochs = l1b_de["epoch"].values
    spin_bins = l1b_de["spin_bin"].values

    badtimes = set_bad_or_goodtimes(badtimes_df, epochs, esa_steps, spin_bins)

    # 1 = badtime, 0 = not badtime
    l1b_de["badtimes"] = xr.DataArray(
        badtimes,
        dims=["epoch"],
        # TODO: Add to yaml
        # attrs=attr_mgr.get_variable_attributes("bad_times"),
    )

    return l1b_de


def set_bad_or_goodtimes(
    times_df: pd.DataFrame,
    epochs: np.ndarray,
    esa_steps: np.ndarray,
    spin_bins: np.ndarray,
) -> np.ndarray:
    """
    Find the good/bad time flags for each epoch based on the provided times DataFrame.

    Parameters
    ----------
    times_df : pd.DataFrame
        Good or Bad times dataframe containing time ranges and corresponding flags.
    epochs : np.ndarray
        Array of epochs in TTJ2000ns format.
    esa_steps : np.ndarray
        Array of ESA steps corresponding to each epoch.
    spin_bins : np.ndarray
        Array of spin bins corresponding to each epoch.

    Returns
    -------
    time_flags : np.ndarray
        Array of time good or bad time flags for each epoch.
    """
    if "BadTime_start" in times_df.columns and "BadTime_end" in times_df.columns:
        times_start = met_to_ttj2000ns(times_df["BadTime_start"])
        times_end = met_to_ttj2000ns(times_df["BadTime_end"])
    elif "GoodTime_start" in times_df.columns and "GoodTime_end" in times_df.columns:
        times_start = met_to_ttj2000ns(times_df["GoodTime_start"])
        times_end = met_to_ttj2000ns(times_df["GoodTime_end"])
    else:
        raise ValueError("DataFrame must contain either BadTime or GoodTime columns.")

    # Create masks for time and bin ranges using broadcasting
    # the bin_start and bin_end are 6 degree bins and need to be converted to
    # 0.1 degree bins to align with the spin_bins, so multiply by 60
    time_mask = (epochs[:, None] >= times_start) & (epochs[:, None] <= times_end)
    # The ancillary file binning uses 0-59 for the 6 degree bins, so add 1 to bin_end
    # so the upper bound is inclusive of the full bin range.
    bin_mask = (spin_bins[:, None] >= times_df["bin_start"].values * 60) & (
        spin_bins[:, None] < (times_df["bin_end"].values + 1) * 60
    )

    # Combined mask for epochs that fall within the time and bin ranges
    combined_mask = time_mask & bin_mask

    # Get the time flags for each epoch's esa_step from matching rows
    time_flags = np.zeros(len(epochs), dtype=int)
    for epoch_idx in range(len(epochs)):
        matching_rows = np.where(combined_mask[epoch_idx])[0]
        if len(matching_rows) > 0:
            # Use the first matching row
            row_idx = matching_rows[0]
            esa_step = esa_steps[epoch_idx]
            if f"E-Step{esa_step}" in times_df.columns:
                time_flags[epoch_idx] = times_df[f"E-Step{esa_step}"].iloc[row_idx]

    return time_flags


def set_pointing_direction(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the pointing direction for each direct event.

    The pointing direction is determined using the SPICE instrument pointing
    function. The pointing direction are two 1D vectors in units of degrees
    for longitude and latitude sharing the same epoch dimension.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the pointing direction added.
    """
    # Get the pointing bin for each DE
    et = ttj2000ns_to_et(l1b_de["epoch"])
    # get the direction in HAE coordinates
    direction = lo_instrument_pointing(
        et, l1b_de["pivot_angle"].values[0], SpiceFrame.IMAP_HAE, cartesian=True
    )

    # TODO: Need to ask Lo what to do if a latitude is outside of the
    # +/-2 degree range. Is that possible?
    l1b_de["hae_x"] = xr.DataArray(
        direction[:, 0],
        dims=["epoch"],
        # TODO: Add direction_lon to YAML file
        # attrs=attr_mgr.get_variable_attributes("hae_x"),
    )

    l1b_de["hae_y"] = xr.DataArray(
        direction[:, 1],
        dims=["epoch"],
        # TODO: Add direction_lat to YAML file
        # attrs=attr_mgr.get_variable_attributes("hae_y"),
    )

    l1b_de["hae_z"] = xr.DataArray(
        direction[:, 2],
        dims=["epoch"],
        # TODO: Add direction_lat to YAML file
        # attrs=attr_mgr.get_variable_attributes("hae_z"),
    )

    return l1b_de


def set_pointing_bin(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the pointing bin for each direct event.

    The pointing bins are defined as 3600 bins for spin and 40 bins for off angle.
    Each bin is 0.1 degrees. The bins are defined as follows:
    Longitude bins: -180 to 180 degrees
    Latitude bins: -2 to 2 degrees

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the pointing bins added.
    """
    x = l1b_de["hae_x"]
    y = l1b_de["hae_y"]
    z = l1b_de["hae_z"]
    # Convert from HAE to DPS coordinates
    dps_xyz = frame_transform(
        ttj2000ns_to_et(l1b_de["epoch"]),
        np.column_stack((x, y, z)),
        SpiceFrame.IMAP_HAE,
        SpiceFrame.IMAP_DPS,
        allow_spice_noframeconnect=True,
    )
    # convert the pointing direction to latitudinal coordinates
    direction = cartesian_to_latitudinal(dps_xyz)
    # first column: radius (Not needed)
    # second column: longitude
    lons = direction[:, 1]
    # shift to 0-360 range (spin-phase 0 should be in bin 0)
    lons = (lons + 360) % 360
    # third column: latitude
    lats = direction[:, 2]

    # Define bin edges
    # 3600 bins, 0.1° each
    lon_bins = np.linspace(0, 360, 3601)
    # 40 bins, 0.1° each
    lat_bins = np.linspace(-2, 2, 41)

    # put the lons and lats into bins
    # shift to 0-based index
    lon_bins = np.digitize(lons, lon_bins) - 1
    lat_bins = np.digitize(lats, lat_bins) - 1

    l1b_de["spin_bin"] = xr.DataArray(
        lon_bins,
        dims=["epoch"],
        # TODO: Add pointing_bin_lon to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_bin"),
    )

    l1b_de["off_angle_bin"] = xr.DataArray(
        lat_bins,
        dims=["epoch"],
        # TODO: Add point_bin_lat to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_bin"),
    )

    return l1b_de


# TODO: This is going to work differently when I sample data.
#  The data_fields input is temporary.
def create_datasets(
    attr_mgr: ImapCdfAttributes,
    logical_source: str,
    data_fields: list[Field],
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
    epoch_converted_time = met_to_ttj2000ns([0, 1, 2])

    # Create a data array for the epoch time
    # TODO: might need to update the attrs to use new YAML file
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1b_de":
        direction_vec = xr.DataArray(
            data=[0, 1, 2],
            name="direction_vec",
            dims=["direction_vec"],
            attrs=attr_mgr.get_variable_attributes("direction_vec"),
        )

        direction_vec_label = xr.DataArray(
            data=direction_vec.values.astype(str),
            name="direction_vec_label",
            dims=["direction_vec_label"],
            attrs=attr_mgr.get_variable_attributes("direction_vec_label"),
        )

        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "direction_vec": direction_vec,
                "direction_vec_label": direction_vec_label,
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
        # TODO: TEMPORARY. need to update to use l1a data once that's available.
        #  Won't need to check for the direction field when I have sample data either.
        if field == "direction":
            dataset[field] = xr.DataArray(
                [[0, 0, 1], [0, 1, 0], [0, 0, 1]],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        #  The data type will be set in the data class when that's created
        elif field in ["tof0", "tof1", "tof2", "tof3"]:
            dataset[field] = xr.DataArray(
                [np.float16(1), np.float16(1), np.float16(1)],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                [1, 1, 1], dims=dims, attrs=attr_mgr.get_variable_attributes(field)
            )

    return dataset


def create_badtimes_dataset() -> xr.Dataset:
    """
    Create a badtimes dataset using the spin products.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all badtimes data product fields in xr.DataArray.
    """
    logger.info("Creating badtimes dataset")
    try:
        spin_df = get_spin_data()
    except ValueError:
        logger.warning("No spin data found. Skipping badtimes dataset creation.")
        # Return an empty dataset with the expected badtimes fields (zero-length)
        empty_epoch = xr.DataArray(
            data=np.array([], dtype=np.int64), name="epoch", dims=["epoch"]
        )
        empty_ds = xr.Dataset(coords={"epoch": empty_epoch})

        empty_ds["yyyymmdd"] = xr.DataArray(
            data=np.array([], dtype=np.int32), dims=["epoch"]
        )
        empty_ds["BadTime_start"] = xr.DataArray(
            data=np.array([], dtype=np.int64), dims=["epoch"]
        )
        empty_ds["BadTime_end"] = xr.DataArray(
            data=np.array([], dtype=np.int64), dims=["epoch"]
        )
        empty_ds["bin_start"] = xr.DataArray(
            data=np.array([], dtype=np.uint8), dims=["epoch"]
        )
        empty_ds["bin_end"] = xr.DataArray(
            data=np.array([], dtype=np.uint8), dims=["epoch"]
        )

        empty_ds["esa_step"] = xr.DataArray(
            data=np.arange(1, 8, dtype=np.uint8),
            name="esa_step",
            dims=["esa_step"],
        )
        empty_ds["badtime_flag"] = xr.DataArray(
            data=np.empty((0, len(empty_ds["esa_step"])), dtype=np.uint8),
            dims=["epoch", "esa_step"],
        )

        empty_ds["Comment"] = xr.DataArray(
            data=np.array([], dtype=object), dims=["epoch"]
        )

        return empty_ds

    # All spins with thruster firings are bad times
    thruster_data = spin_df[spin_df["thruster_firing"]]
    logger.info("Number of thruster firings found: %d", len(thruster_data))
    thruster_ds = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                data=met_to_ttj2000ns(thruster_data["spin_start_met"]),
                name="epoch",
                dims=["epoch"],
            )
        },
    )
    thruster_ds["yyyymmdd"] = xr.DataArray(
        data=thruster_data["spin_start_utc"]
        .str.replace("-", "")
        .str.slice(0, 8)
        .values.astype(int),
        dims=["epoch"],
    )
    thruster_ds["BadTime_start"] = xr.DataArray(
        data=thruster_data["spin_start_sec_sclk"].values,
        dims=["epoch"],
    )
    thruster_ds["BadTime_end"] = thruster_ds["BadTime_start"] + thruster_data[
        "spin_period_sec"
    ].values.astype(int)
    thruster_ds["bin_start"] = xr.DataArray(
        data=np.zeros(len(thruster_ds["epoch"]), dtype=np.uint8),
        dims=["epoch"],
    )
    thruster_ds["bin_end"] = xr.DataArray(
        data=np.full(len(thruster_ds["epoch"]), 59, dtype=np.uint8),
        dims=["epoch"],
    )
    thruster_ds["esa_step"] = xr.DataArray(
        data=np.arange(1, 8, dtype=np.uint8),
        name="esa_step",
        dims=["esa_step"],
    )
    thruster_ds["badtime_flag"] = xr.DataArray(
        data=np.ones(
            (len(thruster_ds["epoch"]), len(thruster_ds["esa_step"])), dtype=np.uint8
        ),
        dims=["epoch", "esa_step"],
    )
    thruster_ds["Comment"] = xr.DataArray(
        data=np.full(len(thruster_ds["epoch"]), "Thruster Firing", dtype=object),
        dims=["epoch"],
    )

    # TODO: Merge with other datasets if/when those are created
    return thruster_ds


def initialize_all_rates(
    l1a_hist: xr.Dataset, attr_mgr_l1b: ImapCdfAttributes
) -> xr.Dataset:
    """
    Initialize the L1B histogram rates dataset.

    Parameters
    ----------
    l1a_hist : xr.Dataset
        The L1A histogram rates dataset.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the L1B histogram rates dataset attributes.

    Returns
    -------
    l1b_all_rates : xr.Dataset
        The initialized L1B histogram and monitor rates dataset.
    """
    l1b_all_rates = xr.Dataset(
        coords={
            "epoch": xr.DataArray(l1a_hist["epoch"].values, dims=["epoch"]),
            "esa_step": l1a_hist["esa_step"],
            "spin_bin_6": xr.DataArray(
                l1a_hist["azimuth_6"].values,
                dims=["spin_bin_6"],
            ),
            "spin_bin_60": xr.DataArray(
                l1a_hist["azimuth_60"].values,
                dims=["spin_bin_60"],
            ),
        },
    )
    # Use centralized mappings for field definitions
    for l1a_field, l1b_field in SPIN_BIN_6_L1A_TO_L1B.items():
        l1b_all_rates[l1b_field] = xr.DataArray(
            l1a_hist[l1a_field].values,
            dims=["epoch", "esa_step", "spin_bin_6"],
        )

    for l1a_field, l1b_field in SPIN_BIN_60_L1A_TO_L1B.items():
        l1b_all_rates[l1b_field] = xr.DataArray(
            l1a_hist[l1a_field].values,
            dims=["epoch", "esa_step", "spin_bin_60"],
        )

    return l1b_all_rates


def resweep_histogram_data(
    l1b_histrates: xr.Dataset,
    anc_dependencies: list,
) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    """
    Correct energy steps in histogram data based on sweep and LUT tables.

    Returns the updated dataset and a 3D array of reswept counts
    (epoch, azimuth, esa_step) indicating how many original steps were reswept into
    each final step.

    Parameters
    ----------
    l1b_histrates : xr.Dataset
        The L1B histogram rates dataset.
    anc_dependencies : list
        List of ancillary file paths.

    Returns
    -------
    l1b_histrates : xr.Dataset
        The updated L1B histogram rates dataset with reswept counts.
    exposure_factor : dict[str, np.ndarray]
        Dictionary mapping bin types to their 3D exposure factor arrays
        (epoch, esa_step, azimuth) indicating how many ESA steps were
        reswept during resweeping.
    """
    epochs = l1b_histrates["epoch"].values
    energy_mapping = _get_esa_level_indices(epochs, anc_dependencies=anc_dependencies)

    # initialize the reswept counts arrays
    for field in SPIN_BIN_6_FIELDS + SPIN_BIN_60_FIELDS:
        reswept = np.zeros_like(l1b_histrates[field].values)
        # Place potentially multiple esa_steps into the same energy level bin
        np.add.at(
            reswept,
            (slice(None), energy_mapping, slice(None)),
            l1b_histrates[field].values,
        )
        l1b_histrates[field].values = reswept

    # Calculate exposure factors for each bin type
    exposure_factor_6deg = np.zeros_like(l1b_histrates["h_counts"].values, dtype=int)
    exposure_factor_60deg = np.zeros_like(
        l1b_histrates["start_a_counts"].values, dtype=int
    )
    np.add.at(exposure_factor_6deg, (slice(None), energy_mapping, slice(None)), 1)
    np.add.at(exposure_factor_60deg, (slice(None), energy_mapping, slice(None)), 1)

    # Create a dictionary to hold exposure factors for both bin types
    exposure_factors = {}
    exposure_factors["6deg"] = exposure_factor_6deg
    exposure_factors["60deg"] = exposure_factor_60deg

    return l1b_histrates, exposure_factors


def calculate_histogram_rates(
    l1b_histrates: xr.Dataset,
    acq_start: xr.DataArray,
    acq_end: xr.DataArray,
    avg_spin_durations_per_cycle: xr.DataArray,
    exposure_factors: dict[str, np.ndarray],
) -> xr.Dataset:
    """
    Calculate histogram rates by dividing reswept counts by exposure time.

    For each epoch in l1b_histrates, this function finds the corresponding
    spin interval, calculates the exposure time for each bin type,
    and divides the counts by the exposure time. The exposure time is scaled
    by the number of ESA steps that were reswept during resweeping.

    Parameters
    ----------
    l1b_histrates : xr.Dataset
        The L1B histogram rates dataset containing reswept counts.
    acq_start : xr.DataArray
        Start times for each spin cycle in MET seconds.
    acq_end : xr.DataArray
        End times for each spin cycle in MET seconds.
    avg_spin_durations_per_cycle : xr.DataArray
        Average spin duration for each cycle in seconds.
    exposure_factors : dict[str, np.ndarray]
        Dictionary mapping bin types to their 3D exposure factor arrays
        (epoch, esa_step, azimuth) indicating how many ESA steps were
        reswept during resweeping.

    Returns
    -------
    l1b_histrates : xr.Dataset
        The L1B histogram rates dataset with rates calculated.
    """
    epochs_ttj2000 = l1b_histrates["epoch"].values
    epochs_met = ttj2000ns_to_met(epochs_ttj2000)

    # Match each histogram epoch to its corresponding spin cycle
    closest_spin_idx = np.abs(epochs_met[:, None] - acq_start.values).argmin(axis=1)

    # Get spin durations for each epoch
    spin_durations = avg_spin_durations_per_cycle.values[closest_spin_idx]

    # Calculate exposure time for 6-degree bins (60 bins per spin)
    exposure_time_6deg = spin_durations / 60
    # Calculate effective exposure time with broadcasting
    effective_exposure_6deg = (
        exposure_time_6deg[:, None, None] * exposure_factors["6deg"]
    )

    # Calculate exposure time for 60-degree bins (6 bins per spin)
    exposure_time_60deg = spin_durations / 6
    effective_exposure_60deg = (
        exposure_time_60deg[:, None, None] * exposure_factors["60deg"]
    )

    # Process all fields
    # Process 6-degree bin fields
    for count_field, rate_field in SPIN_BIN_6_COUNT_TO_RATE.items():
        counts = l1b_histrates[count_field].values  # (epoch, esa_step, spin_bin_6)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(
                effective_exposure_6deg > 0, counts / effective_exposure_6deg, 0
            )

        l1b_histrates[rate_field] = xr.DataArray(
            rates,
            dims=l1b_histrates[count_field].dims,
        )

        l1b_histrates["exposure_time_6deg"] = xr.DataArray(
            effective_exposure_6deg,
            dims=["epoch", "esa_step", "spin_bin_6"],
        )

    # Process 60-degree bin fields
    for count_field, rate_field in SPIN_BIN_60_COUNT_TO_RATE.items():
        counts = l1b_histrates[count_field].values  # (epoch, esa_step, spin_bin_60)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(
                effective_exposure_60deg > 0, counts / effective_exposure_60deg, 0
            )

        l1b_histrates[rate_field] = xr.DataArray(
            rates,
            dims=l1b_histrates[count_field].dims,
        )

        l1b_histrates["exposure_time_60deg"] = xr.DataArray(
            effective_exposure_60deg,
            dims=["epoch", "esa_step", "spin_bin_60"],
        )

    return l1b_histrates


def calculate_de_rates(
    sci_dependencies: dict,
    anc_dependencies: list,
    attr_mgr_l1b: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Calculate direct event rates histograms.

    The histograms are per ASC (28 spins), so we need to
    regroup the individual DEs from the l1b_de dataset into
    their associated ASC and then bin them by ESA / spin bin.

    Parameters
    ----------
    sci_dependencies : dict
        The science dependencies for the derates product.
    anc_dependencies : list
        List of ancillary file paths.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the L1B derates dataset attributes.

    Returns
    -------
    l1b_derates : xr.Dataset
        Dataset containing DE rates histograms.
    """
    l1b_de = sci_dependencies["imap_lo_l1b_de"]
    l1a_spin = sci_dependencies["imap_lo_l1a_spin"]
    # Set the asc_start for each DE by removing the average spin cycle
    # which is a function of esa_step (see set_spin_cycle function)
    # spin_cycle is an average over esa steps and spins per asc, so finding
    # the "average" spin that an esa step occurred at.
    asc_start = l1b_de["spin_cycle"] - (7 + (l1b_de["esa_step"] - 1) * 2)

    # Get unique ASC values and create a mapping from asc_start to index
    unique_asc, unique_idx, asc_idx = np.unique(
        asc_start.values, return_index=True, return_inverse=True
    )
    num_asc = len(unique_asc)

    # Pre-extract arrays for faster access (avoid repeated xarray indexing)
    esa_step_idx = l1b_de["esa_step"].values - 1  # Convert to 0-based index
    # Convert spin_bin from 0.1 degree bins to 6 degree bins for coarse histograms
    spin_bin = l1b_de["spin_bin"].values // 60
    species = l1b_de["species"].values
    coincidence_type = l1b_de["coincidence_type"].values

    if len(anc_dependencies) == 0:
        logger.warning("No ancillary dependencies provided, using linear stepping.")
        energy_step_mapping = np.arange(7)
    else:
        # An array mapping esa step index to esa level for resweeping
        energy_step_mapping = _get_esa_level_indices(asc_start, anc_dependencies)

    # exposure time shape: (num_asc, num_esa_steps)
    exposure_time = np.zeros((num_asc, 7), dtype=float)
    # exposure_time_6deg = 4 * avg_spin_per_asc / 60
    # 4 sweeps per ASC (28 / 7) in 60 bins
    asc_avg_spin_durations = 4 * l1b_de["avg_spin_durations"].data[unique_idx] / 60
    np.add.at(
        exposure_time,
        (slice(None), energy_step_mapping),
        asc_avg_spin_durations[:, np.newaxis],
    )

    # Create output arrays
    output_shape = (num_asc, 7, 60)
    h_counts = np.zeros(output_shape)
    o_counts = np.zeros(output_shape)
    triple_counts = np.zeros(output_shape)
    double_counts = np.zeros(output_shape)

    # Species masks
    h_mask = species == "H"
    o_mask = species == "O"

    # Coincidence type masks
    triple_types = ["111111", "111100", "111000"]
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
    triple_mask = np.isin(coincidence_type, triple_types)
    double_mask = np.isin(coincidence_type, double_types)

    # Vectorized histogramming using np.add.at with full index arrays
    np.add.at(h_counts, (asc_idx[h_mask], esa_step_idx[h_mask], spin_bin[h_mask]), 1)
    np.add.at(o_counts, (asc_idx[o_mask], esa_step_idx[o_mask], spin_bin[o_mask]), 1)
    np.add.at(
        triple_counts,
        (asc_idx[triple_mask], esa_step_idx[triple_mask], spin_bin[triple_mask]),
        1,
    )
    np.add.at(
        double_counts,
        (asc_idx[double_mask], esa_step_idx[double_mask], spin_bin[double_mask]),
        1,
    )

    ds = xr.Dataset(
        coords={
            # ASC start time in TTJ2000ns
            "epoch": l1a_spin["epoch"],
            "esa_step": np.arange(7),
            "spin_bin": np.arange(60),
        },
    )
    ds["h_counts"] = xr.DataArray(
        h_counts,
        dims=["epoch", "esa_step", "spin_bin"],
    )
    ds["o_counts"] = xr.DataArray(
        o_counts,
        dims=["epoch", "esa_step", "spin_bin"],
    )
    ds["triple_counts"] = xr.DataArray(
        triple_counts,
        dims=["epoch", "esa_step", "spin_bin"],
    )
    ds["double_counts"] = xr.DataArray(
        double_counts,
        dims=["epoch", "esa_step", "spin_bin"],
    )
    ds["exposure_time"] = xr.DataArray(
        exposure_time,
        dims=["epoch", "esa_step"],
    )
    ds["h_rates"] = ds["h_counts"] / ds["exposure_time"]
    ds["o_rates"] = ds["o_counts"] / ds["exposure_time"]
    ds["triple_rates"] = ds["triple_counts"] / ds["exposure_time"]
    ds["double_rates"] = ds["double_counts"] / ds["exposure_time"]

    # (N, 7)
    unique_asc = xr.DataArray(unique_asc, dims=["epoch"])
    ds["spin_cycle"] = unique_asc + 7 + (ds["esa_step"] - 1) * 2

    # TODO: Add badtimes
    ds["badtime"] = xr.zeros_like(ds["epoch"], dtype=int)

    ds["pivot_angle"] = l1b_de["pivot_angle"]

    pointing_start_met, pointing_end_met = get_pointing_times(
        ttj2000ns_to_met(ds["epoch"].values[0].item())
    )
    ds = set_esa_mode(pointing_start_met, pointing_end_met, anc_dependencies, ds)

    ds.attrs = attr_mgr_l1b.get_global_attributes("imap_lo_l1b_derates")
    ds["epoch"].attrs = attr_mgr_l1b.get_variable_attributes("epoch")

    return ds


def _get_nearest_pivot_angle(epoch: int, ds_nhk: xr.Dataset) -> float:
    """
    Get the nearest pivot angle for the given epoch from the NHK dataset.

    Parameters
    ----------
    epoch : int
        The epoch in TTJ2000ns format.
    ds_nhk : xr.Dataset
        The NHK dataset containing pivot angle information.

    Returns
    -------
    pivot_angle : float
        The nearest pivot angle for the given epoch.
    """
    return ds_nhk["pcc_cumulative_cnt_pri"].sel(epoch=epoch, method="nearest").item()


def _get_esa_level_indices(epochs: np.ndarray, anc_dependencies: list) -> np.ndarray:
    """
    Get the ESA level indices (reswept indices) for the given epochs.

    This will always return a 7-element array mapping the original ESA step
    indices (0-6) to the true ESA levels after resweeping. i.e. we could have
    taken two measurements in a row at the same energy level, so the mapping
    would be [0, 0, 1, 1, 2, 2, 3] potentially. The nominal stepping is
    [0, 1, 2, 3, 4, 5, 6].

    Parameters
    ----------
    epochs : np.ndarray
        Array of epochs in TTJ2000ns format.
    anc_dependencies : list
        List of ancillary file paths.

    Returns
    -------
    esa_level_indices : np.ndarray
        Array of ESA level indices for each epoch.
    """
    # The sweep table contains the mapping of dates to the LUT table which shows how
    # the ESA steps should be reswept.
    sweep_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "sweep-table" in str(s))
    )
    lut_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "esa-mode-lut" in str(s))
    )

    # Get the time information to compare the epochs to the sweep table dates
    sweep_dates = sweep_df["Date"].astype(str)
    # Get only the date portion of the epoch string for comparison with the sweep table
    # NOTE: We only use the first epoch here since the LUT mapping should be
    #       constant through the entire dataset
    epoch_date_only = et_to_utc(ttj2000ns_to_et(epochs[0])).split("T")[0]

    # Get the matching sweep table entry for the epoch date and its LUT table index
    matching_sweep = sweep_df[sweep_dates == epoch_date_only]
    # if the epoch date is not in the sweep table, raise an error
    if len(matching_sweep) == 0:
        raise ValueError(f"No sweep table entry found for date {epoch_date_only}")

    unique_lut_tables = matching_sweep["LUT_table"].unique()

    # There should only be one unique LUT table for each date
    if len(unique_lut_tables) != 1:
        logger.warning(
            f"Multiple LUT tables found for epoch {epoch_date_only}, "
            f"but found tables {unique_lut_tables}."
        )

    # Get the LUT entries for the identified LUT index
    lut_table_idx = unique_lut_tables[0]
    lut_entries = lut_df[lut_df["Tbl_Idx"] == lut_table_idx].copy()

    # If there are no LUT entries for the identified LUT table, log a warning
    # and return the default mapping
    if len(lut_entries) == 0:
        logger.warning(
            f"No LUT entries for epoch {epoch_date_only}. Looking"
            f"for table index {lut_table_idx}."
        )
        return np.arange(7)

    # Sort the LUT entries by E-Step_Idx to ensure correct mapping order
    lut_entries = lut_entries.sort_values("E-Step_Idx")

    # TODO: It seems like this is also given to us in the main sweep table
    #       Can we just take the last 7 entries of the sweep table for that
    #       date and use those values instead of this extra work with the
    #       separate LUT ancillary file?
    energy_step_mapping = np.zeros(7, dtype=int)
    # Loop through the LUT entries and populate the mapping
    for _, row in lut_entries.iterrows():
        # Original ESA step index is 1-based, convert to 0-based
        esa_idx = int(row["E-Step_Idx"]) - 1
        true_esa_step = int(row["E-Step_lvl"]) - 1
        # Populate the mapping
        energy_step_mapping[esa_idx] = true_esa_step

    return energy_step_mapping


def split_rate_dataset(
    l1b_all_rates: xr.Dataset, attr_mgr_l1b: ImapCdfAttributes
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Split the L1B all rates dataset into histogram rates and monitor rates datasets.

    Parameters
    ----------
    l1b_all_rates : xr.Dataset
        The L1B all rates dataset containing both histogram and monitor rates.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the L1B histogram and monitor rates dataset
        attributes.

    Returns
    -------
    l1b_hist_rates : xr.Dataset
        The L1B histogram rates dataset.
    l1b_monitor_rates : xr.Dataset
        The L1B monitor rates dataset.
    """
    # Use centralized lists for fields to include in split datasets
    l1b_hist_rates = l1b_all_rates[HIST_RATE_FIELDS]
    l1b_hist_rates.attrs = attr_mgr_l1b.get_global_attributes("imap_lo_l1b_histrates")
    l1b_monitor_rates = l1b_all_rates[MONITOR_RATE_FIELDS]
    l1b_monitor_rates.attrs = attr_mgr_l1b.get_global_attributes(
        "imap_lo_l1b_monitorrates"
    )

    return l1b_hist_rates, l1b_monitor_rates


# ============================================================================
# Star Sensor L1B Processing Functions
# ============================================================================


def filter_valid_star_records(
    l1a_star: xr.Dataset,
    min_count: int = 700,
    time_window_offset: float = 0.0,
    time_window_duration: float | None = None,
) -> np.ndarray:
    """
    Create boolean mask for valid star sensor records.

    Records are valid if:
    1. COUNT >= min_count (default 700, per algorithm Section 5)
    2. Within specified time window (if provided)
    3. Not during a repoint maneuver

    Parameters
    ----------
    l1a_star : xr.Dataset
        L1A star sensor dataset containing 'shcoarse' (MET seconds) and 'count'.
    min_count : int
        Minimum acceptable COUNT value (default: 700).
    time_window_offset : float
        Time offset in seconds from first record (default: 0.0).
    time_window_duration : float | None
        Duration of valid time window in seconds (None = no filter, default).

    Returns
    -------
    valid_mask : np.ndarray
        Boolean array indicating valid records.
    """
    # Section 5: Acceptance Criteria - COUNT >= 700
    count_mask = l1a_star["count"].values >= min_count

    # shcoarse is already in MET seconds
    shcoarse_sec = l1a_star["shcoarse"].values.astype(np.float64)

    # Section 2.2: Time window filter (if specified)
    if time_window_duration is not None:
        t0 = shcoarse_sec[0]
        time_mask = (shcoarse_sec >= (t0 + time_window_offset)) & (
            shcoarse_sec <= (t0 + time_window_offset + time_window_duration)
        )
        valid_mask = count_mask & time_mask
    else:
        valid_mask = count_mask

    # Filter out repoint maneuvers
    repoint_df = interpolate_repoint_data(shcoarse_sec)
    # Exclude times where repoint_in_progress is True
    repoint_mask = ~repoint_df["repoint_in_progress"].values
    valid_mask = valid_mask & repoint_mask

    n_valid = valid_mask.sum()
    n_total = len(valid_mask)
    logger.info(
        f"Star sensor valid records: {n_valid}/{n_total} "
        f"({100 * n_valid / n_total:.1f}%)"
    )

    return valid_mask


def calculate_star_sensor_profile_for_group(
    data: np.ndarray,
    counts: np.ndarray,
    end_bins_to_exclude: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate averaged star sensor amplitude profile for a group of records.

    Parameters
    ----------
    data : np.ndarray
        Star sensor data array, shape (n_records, 720).
    counts : np.ndarray
        Count values for each record, shape (n_records,).
    end_bins_to_exclude : int
        Number of bins to exclude from end of each row of data (default: 2).

    Returns
    -------
    avg_amplitude : np.ndarray
        Average amplitude in mV per bin, shape (720,).
    count_per_bin : np.ndarray
        Number of samples accumulated per bin, shape (720,).
    """
    if len(data) == 0:
        return np.full(720, np.nan, dtype=np.float64), np.zeros(720, dtype=np.int32)

    # Determine valid bin ranges for each record
    use_edge_exclusion = (end_bins_to_exclude > 0) & (counts > end_bins_to_exclude)
    end_bins = np.where(
        use_edge_exclusion,
        np.minimum(counts - end_bins_to_exclude, 720),
        np.minimum(counts, 720),
    )

    # Create mask for valid bins: shape (n_records, 720)
    bin_indices = np.arange(720)
    valid_bin_mask = bin_indices[None, :] < end_bins[:, None]

    # Apply mask and sum across all records
    masked_data = np.where(valid_bin_mask, data, 0)
    sum_array = masked_data.sum(axis=0).astype(np.float64)
    count_array = valid_bin_mask.sum(axis=0).astype(np.int32)

    # Compute average amplitude per bin
    avg_amplitude = np.full(720, np.nan, dtype=np.float64)
    mask = count_array > 0
    avg_amplitude[mask] = sum_array[mask] / count_array[mask]

    return avg_amplitude, count_array


def calculate_star_sensor_profiles_by_group(
    l1a_star: xr.Dataset,
    sampling_cadence: float,
    spin_period: float,
    group_size: int = 64,
    start_angle_offset: float = 62.0,
    end_bins_to_exclude: int = 2,
    min_count_threshold: int = 700,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate averaged star sensor amplitude profiles for groups of records.

    Groups L1A star sensor records into chunks of `group_size` and calculates
    an averaged profile for each group.

    Parameters
    ----------
    l1a_star : xr.Dataset
        L1A star sensor data.
    sampling_cadence : float
        Sampling period in milliseconds (ifb_data_interval).
    spin_period : float
        Spin period in seconds.
    group_size : int
        Number of records per group (default: 64).
    start_angle_offset : float
        Starting angle offset in degrees (default: 62.0 = 90° - 28°).
    end_bins_to_exclude : int
        Number of ending bins to exclude from each average (default: 2).
    min_count_threshold : int
        Minimum COUNT value for valid record (default: 700).

    Returns
    -------
    spin_angle : np.ndarray
        Spin angles in degrees [0-360], shape (720,).
    group_mets : np.ndarray
        Start MET for each group, shape (n_groups,).
    avg_amplitudes : np.ndarray
        Average amplitude in mV per bin per group, shape (n_groups, 720).
    counts_per_bin : np.ndarray
        Number of samples accumulated per bin per group, shape (n_groups, 720).
    """
    # Get valid record mask
    valid_mask = filter_valid_star_records(
        l1a_star, min_count_threshold, time_window_offset=0.0, time_window_duration=None
    )

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    # Calculate spin angles (same for all groups)
    deg_per_bin = 360.0 * (sampling_cadence / 1000.0) / spin_period
    bin_indices = np.arange(720)
    sample_centers = (bin_indices + 0.5) * deg_per_bin
    spin_angle = (start_angle_offset + sample_centers) % 360.0

    if n_valid == 0:
        logger.warning(
            "No valid star sensor records found. Returning empty profile with FILLVAL."
        )
        return (
            spin_angle,
            np.array([], dtype=np.int64),
            np.empty((0, 720), dtype=np.float64),
            np.empty((0, 720), dtype=np.int32),
        )

    # Keep valid data using xarray selection
    l1a_star = l1a_star.isel(epoch=valid_indices)

    # Calculate number of groups (include partial groups)
    n_groups = (n_valid + group_size - 1) // group_size
    last_group_size = n_valid % group_size

    logger.info(
        f"Processing {n_valid} valid records into {n_groups} groups of {group_size}"
    )
    if last_group_size != 0:
        logger.debug(f"Last group contains {last_group_size} records (partial group)")

    # Assign group labels to the dataset for xarray groupby operations
    group_labels = np.repeat(np.arange(n_groups), group_size)[:n_valid]
    l1a_star = l1a_star.assign_coords(group=("epoch", group_labels))

    # Extract first MET for each group using xarray groupby
    group_mets = l1a_star["shcoarse"].groupby("group").first().values.astype(np.int64)

    # Initialize output arrays
    avg_amplitudes = np.zeros((n_groups, 720), dtype=np.float64)
    counts_per_bin = np.zeros((n_groups, 720), dtype=np.int32)

    # Process each group using xarray groupby
    for group_label, group_data in l1a_star.groupby("group"):
        # Calculate profile for this group
        avg_amp, count_arr = calculate_star_sensor_profile_for_group(
            group_data["data"].values, group_data["count"].values, end_bins_to_exclude
        )

        avg_amplitudes[group_label] = avg_amp
        counts_per_bin[group_label] = count_arr

    return spin_angle, group_mets, avg_amplitudes, counts_per_bin


def get_sampling_cadence_from_nhk(l1b_nhk: xr.Dataset) -> float:
    """
    Extract ifb_data_interval from NHK dataset.

    The sampling cadence is already in engineering units after L1B processing.
    Formula applied in XTCE: ifb_data_interval = 13.3344 + 0.06945 * DN

    Parameters
    ----------
    l1b_nhk : xr.Dataset
        L1B NHK dataset with derived values (engineering units).

    Returns
    -------
    sampling_cadence : float
        Average sampling cadence in milliseconds.
    """
    if "ifb_data_interval" not in l1b_nhk:
        raise KeyError(
            "ifb_data_interval field not found in L1B NHK dataset. "
            "Cannot calculate sampling cadence."
        )

    # Get mean value across all epochs (should be relatively constant)
    sampling_cadence = float(l1b_nhk["ifb_data_interval"].values.mean())

    logger.info(f"Star sensor sampling cadence from NHK: {sampling_cadence:.3f} ms")
    return sampling_cadence


def l1b_star(
    sci_dependencies: dict,
    attr_mgr_l1b: ImapCdfAttributes,
    group_size: int = 64,
) -> xr.Dataset:
    """
    Create the IMAP-Lo L1B Star Sensor dataset.

    Creates averaged spin profiles from L1A star sensor data, computing
    the average amplitude per spin angle bin for each group of records.
    Each group contains `group_size` consecutive valid records.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager for L1B dataset metadata.
    group_size : int
        Number of records to average per group (default: 64).

    Returns
    -------
    l1b_star_ds : xr.Dataset
        L1B star sensor dataset with spin_angle, avg_amplitude, count_per_bin,
        and time range metadata. Each epoch corresponds to a group of records.
    """
    logical_source = "imap_lo_l1b_prostar"
    l1a_star = sci_dependencies["imap_lo_l1a_star"]
    l1b_nhk = sci_dependencies["imap_lo_l1b_nhk"]
    spin_data = sci_dependencies["imap_lo_l1a_spin"]

    # Get sampling cadence from NHK
    sampling_cadence = get_sampling_cadence_from_nhk(l1b_nhk)

    # Get spin duration from spin data
    avg_spin_durations = get_avg_spin_durations_per_cycle(spin_data)
    spin_duration = float(avg_spin_durations.mean().values)
    logger.info(f"Using spin duration from spin data: {spin_duration:.6f} s")

    # TODO: Read from ancillary config file when available
    lo_angle_offset = 2.0
    sc_to_inst_angle_offset = (
        360 * get_spacecraft_to_instrument_spin_phase_offset(SpiceFrame.IMAP_LO)
        + lo_angle_offset
    )
    end_bins_to_exclude = 2
    min_count_threshold = 700

    # Calculate profiles for each 64-spin group
    (
        spin_angle,
        group_mets,
        avg_amplitudes,
        counts_per_bin,
    ) = calculate_star_sensor_profiles_by_group(
        l1a_star,
        sampling_cadence,
        spin_duration,
        group_size=group_size,
        start_angle_offset=sc_to_inst_angle_offset,
        end_bins_to_exclude=end_bins_to_exclude,
        min_count_threshold=min_count_threshold,
    )

    # Get global epoch times from L1A data for start_doy and end_doy
    global_start_epoch = l1a_star["epoch"].values[0]
    global_end_epoch = l1a_star["epoch"].values[-1]

    # Create dataset with spin_angle as coordinate and multiple epochs
    group_epochs = met_to_ttj2000ns(group_mets)
    l1b_star_ds = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                group_epochs,
                dims=["epoch"],
                attrs=attr_mgr_l1b.get_variable_attributes("epoch"),
            ),
            "spin_angle": xr.DataArray(
                spin_angle,
                dims=["spin_angle"],
                attrs=attr_mgr_l1b.get_variable_attributes(
                    "spin_angle", check_schema=False
                ),
            ),
        },
        attrs=attr_mgr_l1b.get_global_attributes(logical_source),
    )

    # Add spin_angle_bin as a variable (original bin indices)
    l1b_star_ds["spin_angle_bin"] = xr.DataArray(
        np.arange(720, dtype=np.uint16),
        dims=["spin_angle"],
        attrs=attr_mgr_l1b.get_variable_attributes(
            "spin_angle_bin", check_schema=False
        ),
    )

    l1b_star_ds["met"] = xr.DataArray(
        group_mets,
        dims=["epoch"],
        attrs=attr_mgr_l1b.get_variable_attributes("met"),
    )

    l1b_star_ds["avg_amplitude"] = xr.DataArray(
        avg_amplitudes,
        dims=["epoch", "spin_angle"],
        attrs=attr_mgr_l1b.get_variable_attributes("avg_amplitude"),
    )

    l1b_star_ds["count_per_bin"] = xr.DataArray(
        counts_per_bin,
        dims=["epoch", "spin_angle"],
        attrs=attr_mgr_l1b.get_variable_attributes("count_per_bin"),
    )

    # Sort the dataset by spin_angle
    l1b_star_ds = l1b_star_ds.sortby("spin_angle")

    # Add pointing mid time (MET) as a scalar value
    # Use the first epoch to determine which pointing we're in
    first_met = l1a_star["shcoarse"].values[0]
    pointing_mid_met = get_pointing_mid_time(first_met)

    # Add global start and end day of year as scalar values
    start_doy = epoch_to_fractional_doy(global_start_epoch)
    end_doy = epoch_to_fractional_doy(global_end_epoch)

    # Add processing parameters as metadata
    l1b_star_ds.attrs["start_doy"] = start_doy
    l1b_star_ds.attrs["end_doy"] = end_doy
    l1b_star_ds.attrs["pointing_mid_met"] = pointing_mid_met
    l1b_star_ds.attrs["sampling_cadence_ms"] = sampling_cadence
    l1b_star_ds.attrs["spin_duration_sec"] = spin_duration
    l1b_star_ds.attrs["lo_angle_offset_deg"] = lo_angle_offset
    l1b_star_ds.attrs["end_bins_excluded"] = end_bins_to_exclude
    l1b_star_ds.attrs["min_count_threshold"] = min_count_threshold
    l1b_star_ds.attrs["group_size"] = group_size

    logger.info(
        f"L1B star sensor dataset created successfully with {len(group_epochs)} groups"
    )

    return l1b_star_ds
