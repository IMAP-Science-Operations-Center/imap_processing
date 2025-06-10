"""IMAP-Lo L1B Data Processing."""

import logging
from dataclasses import Field
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1b.tof_conversions import (
    TOF0_CONV,
    TOF1_CONV,
    TOF2_CONV,
    TOF3_CONV,
)
from imap_processing.spice.geometry import SpiceFrame, instrument_pointing
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_et

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lo_l1b(dependencies: dict) -> list[Path]:
    """
    Will process IMAP-Lo L1A data into L1B CDF data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of datasets needed for L1B data product creation in xarray Datasets.

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
    logger.info(f"\n Dependencies: {list(dependencies.keys())}\n")
    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1a_de" in dependencies and "imap_lo_l1a_spin" in dependencies:
        logger.info("\nProcessing IMAP-Lo L1B Direct Events...")
        logical_source = "imap_lo_l1b_de"
        # get the dependency dataset for l1b direct events
        l1a_de = dependencies["imap_lo_l1a_de"]
        spin_data = dependencies["imap_lo_l1a_spin"]

        # Initialize the L1B DE dataset
        l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)
        # Get the start and end times for each spin epoch
        acq_start, acq_end = convert_start_end_acq_times(spin_data)
        # Get the average spin durations for each epoch
        avg_spin_durations_per_cycle = get_avg_spin_durations_per_cycle(
            acq_start, acq_end
        )
        # get spin angle (0 - 360 degrees) for each DE
        spin_angle = get_spin_angle(l1a_de)
        # calculate and set the spin bin based on the spin angle
        # spin bins are 0 - 60 bins
        l1b_de = set_spin_bin(l1b_de, spin_angle)
        # set the spin cycle for each direct event
        l1b_de = set_spin_cycle(l1a_de, l1b_de)
        # get spin start times for each event
        spin_start_time = get_spin_start_times(l1a_de, l1b_de, spin_data, acq_end)
        # get the absolute met for each event
        l1b_de = set_event_met(
            l1a_de, l1b_de, spin_start_time, avg_spin_durations_per_cycle
        )
        # set the epoch for each event
        l1b_de = set_each_event_epoch(l1b_de)
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
        # set the badtimes
        l1b_de = set_bad_times(l1b_de)
        # set the pointing direction for each direct event
        l1b_de = set_pointing_direction(l1b_de)
        # calculate and set the pointing bin based on the spin phase
        # pointing bin is 3600 x 40 bins
        l1b_de = set_pointing_bin(l1b_de)

    return [l1b_de]


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
    l1b_de["mode"] = xr.DataArray(
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


def get_spin_angle(l1a_de: xr.Dataset) -> Union[np.ndarray[np.float64], Any]:
    """
    Get the spin angle (0 - 360 degrees) for each DE.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.

    Returns
    -------
    spin_angle : np.ndarray
        The spin angle for each DE.
    """
    de_times = l1a_de["de_time"].values
    # DE Time is 12 bit DN. The max possible value is 4096
    spin_angle = np.array(de_times / 4096 * 360, dtype=np.float64)
    return spin_angle


def set_spin_bin(l1b_de: xr.Dataset, spin_angle: np.ndarray) -> xr.Dataset:
    """
    Set the spin bin (0 - 60 bins) for each Direct Event where each bin is 6 degrees.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B Direct Event dataset.
    spin_angle : np.ndarray
        The spin angle (0-360 degrees) for each Direct Event.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the spin bin added.
    """
    # Get the spin bin for each DE
    # Spin bins are 0 - 60 where each bin is 6 degrees
    spin_bin = (spin_angle // 6).astype(int)
    l1b_de["spin_bin"] = xr.DataArray(
        spin_bin,
        dims=["epoch"],
        # TODO: Add spin angle to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_bin"),
    )
    return l1b_de


def set_spin_cycle(l1a_de: xr.Dataset, l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the spin cycle for each direct event.

    spin_cycle = spin_start + 7 + (esa_step - 1) * 2

    where spin_start is the spin number for the first spin
    in an Aggregated Science Cycle (ASC) and esa_step is the esa_step for a direct event

    The 28 spins in a spin epoch spans one ASC.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        The L1A DE dataset.
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the spin cycle added for each direct event.
    """
    counts = l1a_de["de_count"].values
    # split the esa_steps into ASC groups
    de_asc_groups = np.split(l1a_de["esa_step"].values, np.cumsum(counts)[:-1])
    spin_cycle = []
    for i, esa_asc_group in enumerate(de_asc_groups):
        # TODO: Spin Number does not reset for each pointing. Need to figure out
        #  how to retain this information across days
        # increment the spin_start by 28 after each aggregated science cycle
        spin_start = i * 28
        # calculate the spin cycle for each DE in the ASC group
        # TODO: Add equation number in algorithm document when new version is
        # available. Add to docstring as well
        spin_cycle.extend(spin_start + 7 + (esa_asc_group - 1) * 2)

    l1b_de["spin_cycle"] = xr.DataArray(
        spin_cycle,
        dims=["epoch"],
        # TODO: Add spin cycle to YAML file
        # attrs=attr_mgr.get_variable_attributes("spin_cycle"),
    )

    return l1b_de


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
    met = l1a_de["met"].values
    # Find the closest stop_acq for each shcoarse
    closest_stop_acq_indices = np.abs(met[:, None] - acq_end.values).argmin(axis=1)
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
    for tof, conv in zip(tof_fields, tof_conversions):
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


def set_bad_times(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the bad times for each direct event.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        The L1B DE dataset.

    Returns
    -------
    l1b_de : xarray.Dataset
        The L1B DE dataset with the bad times added.
    """
    # Initialize all times as not bad for now
    # TODO: Update to set badtimes based on criteria that
    #  will be defined in the algorithm document
    # 1 = badtime, 0 = not badtime
    l1b_de["badtimes"] = xr.DataArray(
        np.zeros(len(l1b_de["epoch"]), dtype=int),
        dims=["epoch"],
        # TODO: Add to yaml
        # attrs=attr_mgr.get_variable_attributes("bad_times"),
    )

    return l1b_de


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

    direction = instrument_pointing(et, SpiceFrame.IMAP_LO_BASE, SpiceFrame.IMAP_DPS)
    # TODO: Need to ask Lo what to do if a latitude is outside of the
    # +/-2 degree range. Is that possible?
    l1b_de["direction_lon"] = xr.DataArray(
        direction[:, 0],
        dims=["epoch"],
        # TODO: Add direction_lon to YAML file
        # attrs=attr_mgr.get_variable_attributes("direction_lon"),
    )

    l1b_de["direction_lat"] = xr.DataArray(
        direction[:, 1],
        dims=["epoch"],
        # TODO: Add direction_lat to YAML file
        # attrs=attr_mgr.get_variable_attributes("direction_lat"),
    )

    return l1b_de


def set_pointing_bin(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Set the pointing bin for each direct event.

    The pointing bins are defined as 3600 bins for longitude and 40 bins for latitude.
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
    # First column: latitudes
    lats = l1b_de["direction_lat"]
    # Second column: longitudes
    lons = l1b_de["direction_lon"]

    # Define bin edges
    # 3600 bins, 0.1° each
    lon_bins = np.linspace(-180, 180, 3601)
    # 40 bins, 0.1° each
    lat_bins = np.linspace(-2, 2, 41)

    # put the lons and lats into bins
    # shift to 0-based index
    lon_bins = np.digitize(lons, lon_bins) - 1
    lat_bins = np.digitize(lats, lat_bins) - 1

    l1b_de["pointing_bin_lon"] = xr.DataArray(
        lon_bins,
        dims=["epoch"],
        # TODO: Add pointing_bin_lon to YAML file
        # attrs=attr_mgr.get_variable_attributes("pointing_bin_lon"),
    )

    l1b_de["pointing_bin_lat"] = xr.DataArray(
        lat_bins,
        dims=["epoch"],
        # TODO: Add point_bin_lat to YAML file
        # attrs=attr_mgr.get_variable_attributes("pointing_bin_lat"),
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
