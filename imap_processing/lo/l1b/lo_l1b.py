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
    instrument_pointing,
)
from imap_processing.spice.repoint import get_pointing_times
from imap_processing.spice.spin import get_spin_data, get_spin_number
from imap_processing.spice.time import (
    et_to_utc,
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

logger = logging.getLogger(__name__)


def lo_l1b(sci_dependencies: dict, anc_dependencies: list) -> list[Path]:
    """
    Will process IMAP-Lo L1A data into L1B CDF data products.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.
    anc_dependencies : list
        List of ancillary file paths needed for L1B data product creation.

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

    badtimes_ds = create_badtimes_dataset()
    if badtimes_ds.data_vars:
        # If it was an empty dataset, then we don't want to
        badtimes_ds.attrs = attr_mgr_l1b.get_global_attributes("imap_lo_l1b_badtimes")
        datasets_to_return.append(badtimes_ds)

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1a_de" in sci_dependencies and "imap_lo_l1a_spin" in sci_dependencies:
        logger.info("\nProcessing IMAP-Lo L1B Direct Events...")
        logical_source = "imap_lo_l1b_de"
        # get the dependency dataset for l1b direct events
        l1a_de = sci_dependencies["imap_lo_l1a_de"]
        spin_data = sci_dependencies["imap_lo_l1a_spin"]

        # Initialize the L1B DE dataset
        l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)
        pointing_start_met, pointing_end_met = get_pointing_times(
            l1a_de["met"].values[0].item()
        )
        # Get the start and end times for each spin epoch
        acq_start, acq_end = convert_start_end_acq_times(spin_data)
        # Get the average spin durations for each epoch
        avg_spin_durations_per_cycle = get_avg_spin_durations_per_cycle(
            acq_start, acq_end
        )
        # set the spin cycle for each direct event
        l1b_de = set_spin_cycle(pointing_start_met, l1a_de, l1b_de)
        # get spin start times for each event
        spin_start_time = get_spin_start_times(l1a_de, l1b_de, spin_data, acq_end)
        # get the absolute met for each event
        l1b_de = set_event_met(
            l1a_de, l1b_de, spin_start_time, avg_spin_durations_per_cycle
        )
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
        datasets_to_return.append(l1b_de)

    # If dependencies are used to create Histogram Rates
    if (
        "imap_lo_l1a_histogram" in sci_dependencies
        and "imap_lo_l1a_spin" in sci_dependencies
    ):
        logger.info("\nProcessing IMAP-Lo L1B Histogram Rates...")
        logical_source = "imap_lo_l1b_histrates"
        # get the dependency dataset for l1b histogram rates
        l1a_hist = sci_dependencies["imap_lo_l1a_histogram"]
        spin_data = sci_dependencies["imap_lo_l1a_spin"]
        # initialize the L1B Histogram Rates dataset from the L1A Histogram Rates
        # This carries over the epoch and count fields from L1A
        l1b_histrates = initialize_l1b_histrates(l1a_hist, attr_mgr_l1b, logical_source)
        # set spin cycle and remove invalid spin ASCs
        l1b_histrates = set_spin_cycle_from_spin_data(
            l1a_hist, l1b_histrates, spin_data
        )

        pointing_start_met, pointing_end_met = get_pointing_times(
            ttj2000ns_to_met(l1a_hist["epoch"].values[0].item())
        )
        l1b_histrates = set_esa_mode(
            pointing_start_met, pointing_end_met, anc_dependencies, l1b_histrates
        )
        # resweep the histogram data
        l1b_histrates, exposure_factor = resweep_histogram_data(
            l1b_histrates, anc_dependencies
        )
        # Get the start and end times for each spin epoch
        acq_start, acq_end = convert_start_end_acq_times(spin_data)
        # Get the average spin durations for each epoch
        avg_spin_durations_per_cycle = get_avg_spin_durations_per_cycle(
            acq_start, acq_end
        )
        l1b_histrates = calculate_histogram_rates(
            l1b_histrates,
            acq_start,
            acq_end,
            avg_spin_durations_per_cycle,
            exposure_factor,
        )
        datasets_to_return.append(l1b_histrates)

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
    acq_start: xr.DataArray, acq_end: xr.DataArray
) -> xr.DataArray:
    """
    Get the average spin duration for each spin epoch.

    Parameters
    ----------
    acq_start : xarray.DataArray
        The start acquisition times for each spin epoch.
    acq_end : xarray.DataArray
        The end acquisition times for each spin epoch.

    Returns
    -------
    avg_spin_durations : xarray.DataArray
        The average spin duration for each spin epoch.
    """
    # Get the avg spin duration for each spin epoch
    # There are 28 spins per epoch (1 aggregated science cycle)
    avg_spin_durations_per_cycle = (acq_end - acq_start) / 28
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
    l1a_de: xr.Dataset, l1b_de: xr.Dataset, spin_data: xr.Dataset, acq_end: xr.DataArray
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
    l1b_de : xr.Dataset
        The L1B DE dataset.
    spin_data : xr.Dataset
        The L1A Spin dataset.
    acq_end : xr.DataArray
        The end acquisition times for each spin ASC.

    Returns
    -------
    spin_start_time : xr.DataArray
        The start time for the spin that each direct event is in.
    """
    # Get the MET times for each individual direct event
    # l1a_de["met"] has one value per time epoch, but we need one per direct event
    de_met = np.repeat(l1a_de["met"], l1a_de["de_count"])

    # Find the closest stop_acq for each direct event
    closest_stop_acq_indices = np.abs(de_met.values[:, None] - acq_end.values).argmin(
        axis=1
    )
    # There are 28 spins per epoch (1 aggregated science cycle)
    # Set the spin_cycle_num to the spin number relative to the
    # start of the ASC
    spin_cycle_num = l1b_de["spin_cycle"] % 28
    # Get the seconds portion of the start time for each spin
    start_sec_spins = spin_data["start_sec_spin"].values[
        closest_stop_acq_indices, spin_cycle_num
    ]
    # Get the subseconds portion of the spin start time and convert from
    # microseconds to seconds
    start_subsec_spins = (
        spin_data["start_subsec_spin"].values[closest_stop_acq_indices, spin_cycle_num]
        * 1e-6
    )

    # Combine the seconds and subseconds to get the start time for each spin
    spin_start_time = start_sec_spins + start_subsec_spins
    return xr.DataArray(spin_start_time)


def set_event_met(
    l1a_de: xr.Dataset,
    l1b_de: xr.Dataset,
    spin_start_time: xr.DataArray,
    avg_spin_durations: xr.DataArray,
) -> xr.Dataset:
    """
    Get the event MET for each direct event.

    Each direct event is converted from a data number to engineering unit in seconds.
    de_eu_time de_dn_time / 4096 * avg_spin_duration
    where de_time is the direct event time Data Number (DN) and avg_spin_duration
    is the average spin duration for the ASC that the event was measured in.

    The direct event time is the time of direct event relative to the start of the spin.
    The event MET is the sum of the start time of the spin and the
    direct event EU time.

    Parameters
    ----------
    l1a_de : xr.Dataset
        The L1A DE dataset.
    l1b_de : xr.Dataset
        The L1B DE dataset.
    spin_start_time : np.ndarray
        The start time for the spin that each direct event is in.
    avg_spin_durations : xr.DataArray
        The average spin duration for each epoch.

    Returns
    -------
    l1b_de : xr.Dataset
        The L1B DE dataset with the event MET.
    """
    counts = l1a_de["de_count"].values
    de_time_asc_groups = np.split(l1a_de["de_time"].values, np.cumsum(counts)[:-1])
    de_times_eu = []
    for i, de_time_asc in enumerate(de_time_asc_groups):
        # DE Time is 12 bit DN. The max possible value is 4095
        # divide by 4096 to get fraction of a spin duration
        de_times_eu.extend(de_time_asc / 4096 * avg_spin_durations[i].values)

    l1b_de["event_met"] = xr.DataArray(
        spin_start_time + de_times_eu,
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
        if coin_type == 0 and l1a_de["mode"][idx] == 0:
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
    direction = instrument_pointing(
        et, SpiceFrame.IMAP_LO_BASE, SpiceFrame.IMAP_HAE, cartesian=True
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
    # third column: latitude
    lats = direction[:, 2]

    # Define bin edges
    # 3600 bins, 0.1° each
    lon_bins = np.linspace(-180, 180, 3601)
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


def initialize_l1b_histrates(
    l1a_hist: xr.Dataset, attr_mgr_l1b: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Initialize the L1B histogram rates dataset.

    Parameters
    ----------
    l1a_hist : xr.Dataset
        The L1A histogram rates dataset.
    attr_mgr_l1b : ImapCdfAttributes
        Attribute manager used to get the L1B histogram rates dataset attributes.
    logical_source : str
        The logical source of the data product that's being created.

    Returns
    -------
    l1b_histrates : xr.Dataset
        The initialized L1B histogram rates dataset.
    """
    l1b_histrates = xr.Dataset(
        coords={
            "epoch": xr.DataArray(l1a_hist["epoch"].values, dims=["epoch"]),
            "esa_step": l1a_hist["esa_step"],
            "spin_bin_6": xr.DataArray(
                l1a_hist["azimuth_6"].values,
                dims=["spin_bin_6"],
            ),
        },
        attrs=attr_mgr_l1b.get_global_attributes(logical_source),
    )

    # l1b_histrates["epoch"] = xr.DataArray(
    #     l1a_hist["epoch"].values,
    #     dims=["epoch"],
    #     attrs=attr_mgr_l1b.get_variable_attributes("epoch"),
    # )
    # Copy over fields from L1A DE that will not change in L1B processing
    l1b_histrates["h_counts"] = xr.DataArray(
        l1a_hist["hydrogen"].values,
        dims=["epoch", "esa_step", "spin_bin_6"],
        # TODO: Add hydrogen to YAML file
        # attrs=attr_mgr.get_variable_attributes("hydrogen"),
    )
    l1b_histrates["o_counts"] = xr.DataArray(
        l1a_hist["oxygen"].values,
        dims=["epoch", "esa_step", "spin_bin_6"],
        # TODO: Add oxygen to YAML file
        # attrs=attr_mgr.get_variable_attributes("oxygen"),
    )

    return l1b_histrates


def resweep_histogram_data(
    l1b_histrates: xr.Dataset,
    anc_dependencies: list,
) -> tuple[xr.Dataset, np.ndarray]:
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
    exposure_factor : np.ndarray
        3D array of exposure factors (epoch, azimuth, esa_step) indicating how many
        ESA steps were reswept during resweeping.
    """
    epochs = l1b_histrates["epoch"].values
    energy_mapping = _get_esa_level_indices(epochs, anc_dependencies=anc_dependencies)

    # initialize the reswept counts arrays
    h_counts_reswept = np.zeros_like(l1b_histrates["h_counts"].values)
    o_counts_reswept = np.zeros_like(l1b_histrates["o_counts"].values)
    exposure_factor = np.zeros_like(h_counts_reswept, dtype=int)

    # Place potentially multiple esa_steps into the same energy level bin
    np.add.at(
        h_counts_reswept,
        (slice(None), energy_mapping, slice(None)),
        l1b_histrates["h_counts"].values,
    )
    np.add.at(
        o_counts_reswept,
        (slice(None), energy_mapping, slice(None)),
        l1b_histrates["o_counts"].values,
    )
    np.add.at(exposure_factor, (slice(None), energy_mapping, slice(None)), 1)
    l1b_histrates["h_counts"].values = h_counts_reswept
    l1b_histrates["o_counts"].values = o_counts_reswept
    l1b_histrates.attrs["energy_step_correction"] = (
        "Applied LUT table energy step mapping"
    )

    return l1b_histrates, exposure_factor


def calculate_histogram_rates(
    l1b_histrates: xr.Dataset,
    acq_start: xr.DataArray,
    acq_end: xr.DataArray,
    avg_spin_durations_per_cycle: xr.DataArray,
    exposure_factor: np.ndarray,
) -> xr.Dataset:
    """
    Calculate histogram rates by dividing reswept counts by exposure time.

    For each epoch in l1b_histrates, this function finds the corresponding
    spin interval, calculates the exposure time for 6-degree bins,
    and divides the counts by the exposure time. The exposure time is scaled
    by the number of ESA steps that were reswept during resweeping.

    Parameters
    ----------
    l1b_histrates : xr.Dataset
        The L1B histogram rates dataset containing reswept h_counts and o_counts.
    acq_start : xr.DataArray
        Start times for each spin cycle in MET seconds.
    acq_end : xr.DataArray
        End times for each spin cycle in MET seconds.
    avg_spin_durations_per_cycle : xr.DataArray
        Average spin duration for each cycle in seconds.
    exposure_factor : np.ndarray
        3D array of exposure factors (epoch, azimuth, esa_step) indicating how many
        ESA steps were reswept during resweeping.

    Returns
    -------
    l1b_histrates : xr.Dataset
        Updated dataset with h_rates and o_rates added.
    """
    epochs = l1b_histrates["epoch"].values
    h_counts = l1b_histrates["h_counts"].values
    o_counts = l1b_histrates["o_counts"].values

    h_rates = np.zeros_like(h_counts, dtype=float)
    o_rates = np.zeros_like(o_counts, dtype=float)
    num_azimuth = h_counts.shape[2]
    exposure_times = np.zeros((len(epochs), 7, num_azimuth), dtype=float)

    # Calculate rates for each epoch
    for epoch_idx, epoch in enumerate(epochs):
        # Find the spin cycle that contains the current epoch
        spin_cycle_mask = (epoch >= met_to_ttj2000ns(acq_start.values)) & (
            epoch <= met_to_ttj2000ns(acq_end.values)
        )
        spin_cycle_indices = np.nonzero(spin_cycle_mask)[0]

        # If no matching spin cycle is found, log a warning and set rates to NaN
        if len(spin_cycle_indices) == 0:
            logger.warning(f"Epoch {epoch_idx} not found in any spin_cycle interval")
            h_rates[epoch_idx] = np.nan
            o_rates[epoch_idx] = np.nan
            continue

        spin_cycle_idx = spin_cycle_indices[0]
        # Calculate the base exposure time for the spin cycle in minutes
        base_exposure_time = (
            4 * avg_spin_durations_per_cycle.values[spin_cycle_idx] / 60
        )
        # Scale the exposure time by the exposure factor from resweeping
        scaled_exposure = base_exposure_time * exposure_factor[epoch_idx, ...]
        # Avoid division by zero by setting zero exposure times to NaN
        exposure_times[epoch_idx, ...] = scaled_exposure
        with np.errstate(divide="ignore"):
            h_rates[epoch_idx, ...] = h_counts[epoch_idx, ...] / scaled_exposure
            o_rates[epoch_idx, ...] = o_counts[epoch_idx, ...] / scaled_exposure

    l1b_histrates["exposure_time"] = xr.DataArray(
        exposure_times,
        dims=["epoch", "esa_step", "spin_bin_6"],
    )
    l1b_histrates["h_rates"] = xr.DataArray(
        h_rates,
        dims=l1b_histrates["h_counts"].dims,
    )
    l1b_histrates["o_rates"] = xr.DataArray(
        o_rates,
        dims=l1b_histrates["o_counts"].dims,
    )

    return l1b_histrates


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
        raise ValueError(
            f"Expected exactly 1 unique LUT_table value for date {epoch_date_only},"
            f" but found {len(unique_lut_tables)}: {unique_lut_tables}"
        )

    # Get the LUT entries for the identified LUT index
    lut_table_idx = unique_lut_tables[0]
    lut_entries = lut_df[lut_df["Tbl_Idx"] == lut_table_idx].copy()

    # If there are no LUT entries for the identified LUT table, log a warning
    # and return the default mapping
    if len(lut_entries) == 0:
        logger.warning(f"No LUT entries found for table index {lut_table_idx}")
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
