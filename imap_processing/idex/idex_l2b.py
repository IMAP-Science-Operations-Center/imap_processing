"""
Perform IDEX L2b Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l1b import idex_l2a
    from imap_processing.idex.idex_l1b import idex_l2b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_raw_20231218_v001.pkts"
    l0_file_hk = "imap_processing/tests/idex/imap_idex_l0_raw_20250108_v001.pkts"
    l1a_data = PacketParser(l0_file).data[0]
    evt_data = PacketParser(l0_file_hk).data[0]
    l1a_data, l1a_evt_data, l1b_evt_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l1a_data = idex_l2a(l1b_data)
    l2b_data = idex_l2b(l2a_data, [evt_data])
    write_cdf(l2b_data)
"""

import collections
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from imap_processing.idex.idex_constants import (
    FG_TO_KG,
    SECONDS_IN_DAY,
    IDEXEvtAcquireCodes,
)
from imap_processing.idex.idex_utils import get_idex_attrs
from imap_processing.spice.time import epoch_to_doy, et_to_datetime64, ttj2000ns_to_et

logger = logging.getLogger(__name__)
# Bin edges
MASS_BIN_EDGES = np.array(
    [
        6.31e-17,
        1.00e-16,
        1.58e-16,
        2.51e-16,
        3.98e-16,
        6.31e-16,
        1.00e-15,
        1.58e-15,
        2.51e-15,
        3.98e-15,
        1.00e-14,
    ]
)
CHARGE_BIN_EDGES = np.array(
    [
        1.00e-01,
        3.16e-01,
        1.00e00,
        3.16e00,
        1.00e01,
        3.16e01,
        1.00e02,
        3.16e02,
        1.00e03,
        3.16e03,
        1.00e04,
    ]
)
SPIN_PHASE_BIN_EDGES = np.array([0, 90, 180, 270, 360])


def idex_l2b(
    l2a_datasets: list[xr.Dataset], evt_datasets: list[xr.Dataset]
) -> xr.Dataset:
    """
    Will process IDEX l2a data to create l2b data products.

    Parameters
    ----------
    l2a_datasets : list[xarray.Dataset]
        IDEX L2a datasets to process.
    evt_datasets : list[xarray.Dataset]
        List of IDEX housekeeping event message datasets.

    Returns
    -------
    l2b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2B processing on datasets: "
        f"{[ds.attrs['Logical_source'] for ds in l2a_datasets]}"
    )

    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l2b")
    evt_dataset = xr.concat(evt_datasets, dim="epoch")

    # Concat all the l2a datasets together
    l2a_dataset = xr.concat(l2a_datasets, dim="epoch")
    epoch_doy_unique = np.unique(epoch_to_doy(l2a_dataset["epoch"].data))
    counts_by_charge, counts_by_mass, daily_epoch = compute_counts_by_charge_and_mass(
        l2a_dataset, epoch_doy_unique
    )
    # Get science acquisition percentage for each day
    daily_on_percentage = get_science_acquisition_on_percentage(evt_dataset)
    rate_by_charge, rate_by_mass, rate_quality_flags = compute_rates_by_charge_and_mass(
        counts_by_charge, counts_by_mass, epoch_doy_unique, daily_on_percentage
    )
    # Create l2b Dataset
    charge_bins = np.arange(len(CHARGE_BIN_EDGES))
    mass_bins = np.arange(len(CHARGE_BIN_EDGES))
    spin_phase_bins = np.arange(len(SPIN_PHASE_BIN_EDGES) - 1)
    epoch = xr.DataArray(
        name="epoch",
        data=daily_epoch,
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    vars = {
        "impact_day_of_year": xr.DataArray(
            name="impact_day_of_year",
            data=epoch_doy_unique,
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes("impact_day_of_year"),
        ),
        "rate_calculation_quality_flags": xr.DataArray(
            name="rate_calculation_quality_flags",
            data=rate_quality_flags,
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes("rate_calculation_quality_flags"),
        ),
        "charge_labels": xr.DataArray(
            name="impact_charge_labels",
            data=charge_bins.astype(str),
            dims="impact_charge_bins",
            attrs=idex_attrs.get_variable_attributes(
                "charge_labels", check_schema=False
            ),
        ),
        "spin_phase_labels": xr.DataArray(
            name="spin_phase_labels",
            data=spin_phase_bins.astype(str),
            dims="spin_phase_bins",
            attrs=idex_attrs.get_variable_attributes(
                "spin_phase_labels", check_schema=False
            ),
        ),
        "mass_labels": xr.DataArray(
            name="mass_labels",
            data=mass_bins.astype(str),
            dims="mass_bins",
            attrs=idex_attrs.get_variable_attributes("mass_labels", check_schema=False),
        ),
        "impact_charge_bins": xr.DataArray(
            name="impact_charge_bins",
            data=charge_bins,
            dims="impact_charge_bins",
            attrs=idex_attrs.get_variable_attributes(
                "impact_charge_bins", check_schema=False
            ),
        ),
        "mass_bins": xr.DataArray(
            name="mass_bins",
            data=mass_bins,
            dims="mass_bins",
            attrs=idex_attrs.get_variable_attributes("mass_bins", check_schema=False),
        ),
        "spin_phase_bins": xr.DataArray(
            name="spin_phase_bins",
            data=spin_phase_bins,
            dims="spin_phase_bins",
            attrs=idex_attrs.get_variable_attributes(
                "spin_phase_bins", check_schema=False
            ),
        ),
        "counts_by_charge": xr.DataArray(
            name="counts_by_charge",
            data=counts_by_charge.astype(np.int64),
            dims=("epoch", "charge_bins", "spin_phase_bins"),
            attrs=idex_attrs.get_variable_attributes("counts_by_charge"),
        ),
        "counts_by_mass": xr.DataArray(
            name="counts_by_mass",
            data=counts_by_mass.astype(np.int64),
            dims=("epoch", "mass_bins", "spin_phase_bins"),
            attrs=idex_attrs.get_variable_attributes("counts_by_mass"),
        ),
        "rate_by_charge": xr.DataArray(
            name="rate_by_charge",
            data=rate_by_charge,
            dims=("epoch", "charge_bins", "spin_phase_bins"),
            attrs=idex_attrs.get_variable_attributes("rate_by_charge"),
        ),
        "rate_by_mass": xr.DataArray(
            name="rate_by_mass",
            data=rate_by_mass,
            dims=("epoch", "mass_bins", "spin_phase_bins"),
            attrs=idex_attrs.get_variable_attributes("rate_by_mass"),
        ),
    }
    l2b_dataset = xr.Dataset(
        coords={"epoch": epoch},
        data_vars=vars,
        attrs=idex_attrs.get_global_attributes("imap_idex_l2b_sci"),
    )
    # Copy longitude and latitude from the l2a dataset
    l2b_dataset["longitude"] = l2a_dataset["longitude"].copy()
    l2b_dataset["latitude"] = l2a_dataset["latitude"].copy()

    logger.info("IDEX L2B science data processing completed.")

    return l2b_dataset


def compute_counts_by_charge_and_mass(
    l2a_dataset: xr.Dataset, epoch_doy_unique: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dust event counts by charge and mass by spin phase per day.

    Parameters
    ----------
    l2a_dataset : xarray.Dataset
        Combined IDEX L2a datasets.
    epoch_doy_unique : np.ndarray
        Unique days of year corresponding to the epochs in the dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Two 3D arrays containing counts by charge or mass, and by spin phase for each
        dataset, and a 1D array of daily epoch values.
    """
    # Initialize arrays to hold counts.
    # There should be 4 spin phase bins, 11 charge bins, and 11 mass bins.
    # The first bin for charge and mass is for values below the first bin edge.
    counts_by_charge = np.zeros(
        (len(epoch_doy_unique), len(CHARGE_BIN_EDGES), len(SPIN_PHASE_BIN_EDGES) - 1),
    )
    counts_by_mass = np.zeros(
        (len(epoch_doy_unique), len(MASS_BIN_EDGES), len(SPIN_PHASE_BIN_EDGES) - 1),
    )
    daily_epoch = np.zeros(len(epoch_doy_unique))
    for i in range(len(epoch_doy_unique)):
        doy = epoch_doy_unique[i]
        # Get the indices for the current day
        current_day_indices = np.where(epoch_to_doy(l2a_dataset["epoch"].data) == doy)[
            0
        ]
        # Set the epoch for the current day to be the mean epoch of the day.
        daily_epoch[i] = np.mean(l2a_dataset["epoch"].data[current_day_indices])
        mass_vals = l2a_dataset["target_low_dust_mass_estimate"].data[
            current_day_indices
        ]
        charge_vals = l2a_dataset["target_low_impact_charge"].data[current_day_indices]
        spin_phase_angles = l2a_dataset["spin_phase"].data[current_day_indices]
        # Convert units
        mass_vals = FG_TO_KG * np.array(mass_vals)
        # Bin masses
        binned_mass = np.array(np.digitize(mass_vals, bins=MASS_BIN_EDGES))
        # Bin charges
        binned_charge = np.array(np.digitize(charge_vals, bins=CHARGE_BIN_EDGES))
        # Bin spin phases
        binned_spin_phase = bin_spin_phases(spin_phase_angles)
        # If the values in the array are beyond the bounds of bins, 0 or len(bins) it is
        # returned as such. In this case, the desired result is to place the values
        # beyond the last bin into the last bin and keep the values below the first bin.
        binned_charge[binned_charge == len(CHARGE_BIN_EDGES)] = (
            len(CHARGE_BIN_EDGES) - 1
        )
        binned_mass[binned_mass == len(MASS_BIN_EDGES)] = len(MASS_BIN_EDGES) - 1

        # TODO use np.histogramdd to compute the counts by charge and mass.
        # Count dust events for each spin phase and mass bin or charge bin.
        for mass_bin, charge_bin, spin_phase_bin in zip(
            binned_mass, binned_charge, binned_spin_phase
        ):
            counts_by_mass[i, mass_bin, spin_phase_bin] += 1
            counts_by_charge[i, charge_bin, spin_phase_bin] += 1

    return counts_by_charge, counts_by_mass, daily_epoch


def compute_rates_by_charge_and_mass(
    counts_by_charge: np.ndarray,
    counts_by_mass: np.ndarray,
    epoch_doy: np.ndarray,
    daily_on_percentage: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dust event counts rates by charge and mass by spin phase for each day.

    Parameters
    ----------
    counts_by_charge : np.ndarray
        3D array containing counts by charge and spin phase for each dataset.
    counts_by_mass : np.ndarray
        3D array containing counts by mass and spin phase for each dataset.
    epoch_doy : np.ndarray
        Unique days of year corresponding to the epochs in the dataset.
    daily_on_percentage : dict
        Percentage of time science acquisition was on for each doy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Two 3D arrays containing counts rates by charge or mass, and by spin phase for
        each dataset and the quality flags for each epoch.
    """
    # Initialize arrays to hold rates.
    rate_by_charge = np.full(counts_by_charge.shape, -1.0)
    rate_by_mass = np.full(counts_by_mass.shape, -1.0)
    # Initialize an array to hold quality flags for each epoch. A quality flag of 0
    # indicates that there was no science acquisition data for that epoch, and the rate
    # is not valid. A quality flag of 1 indicates that the rate is valid.
    rate_quality_flags = np.ones(epoch_doy.shape, dtype=np.uint8)

    # Get percentages in order of epoch_doy. Log any missing days.
    epoch_doy_percent_on = np.array(
        [daily_on_percentage.get(doy, -1) for doy in epoch_doy]
    )

    missing_doy_uptimes_inds = np.where(epoch_doy_percent_on == -1)[0]
    if np.any(missing_doy_uptimes_inds):
        rate_quality_flags[missing_doy_uptimes_inds] = 0
        logger.warning(
            f"Missing science acquisition uptime percentages for day(s) of"
            f" year: {epoch_doy[missing_doy_uptimes_inds]}."
        )
    # Compute rates
    # Create a boolean mask for DOYs that have a non-zero percentage of science
    # acquisition time.
    non_zero_inds = np.where(epoch_doy_percent_on > 0)[0]
    # Compute rates only for days with non-zero science acquisition percentage
    rate_by_charge[non_zero_inds] = counts_by_charge[non_zero_inds] / (
        0.01
        * epoch_doy_percent_on[non_zero_inds, np.newaxis, np.newaxis]
        * SECONDS_IN_DAY
    )
    rate_by_mass[non_zero_inds] = counts_by_mass[non_zero_inds] / (
        0.01
        * epoch_doy_percent_on[non_zero_inds, np.newaxis, np.newaxis]
        * SECONDS_IN_DAY
    )

    return rate_by_charge, rate_by_mass, rate_quality_flags


def bin_spin_phases(spin_phases: xr.DataArray) -> np.ndarray:
    """
    Bin spin phase angles into 4 quadrants: [315°-45°,45°-135°,135°-225°, 225°-315°].

    Parameters
    ----------
    spin_phases : xarray.DataArray
        Spacecraft spin phase angles. Expected to be integers in the range [0, 360).

    Returns
    -------
    numpy.ndarray
        Spin phases binned into quadrants.
    """
    if np.any(spin_phases < 0) or np.any(spin_phases >= 360):
        logger.warning(
            f"Spin phase angles, {spin_phases.data} are outside of the expected spin "
            f"phase angle range, [0, 360)."
        )
    # Shift spin phases by +45° so that the first bin starts at 0°.
    # Use mod to wrap values > 360 to 0.
    shifted_spin_phases = (spin_phases + 45) % 360
    # Use np.digitize to find the bin index for each spin phase.
    bin_indices = np.digitize(shifted_spin_phases, SPIN_PHASE_BIN_EDGES, right=False)
    # Shift bins to be zero-based.
    bin_indices -= 1
    return np.asarray(bin_indices)


def get_science_acquisition_timestamps(
    evt_dataset: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the science acquisition start and stop times and messages from the event data.

    Parameters
    ----------
    evt_dataset : xarray.Dataset
        Contains IDEX event message data.

    Returns
    -------
    event_logs : np.ndarray
        Array containing science acquisition start and stop events messages.
    event_timestamps : np.ndarray
        Array containing science acquisition start and stop timestamps.
    event_values : np.ndarray
        Array containing values indicating if the event is a start (1) or
        stop (0).
    """
    # Sort the event dataset by the epoch time. Drop duplicates
    evt_dataset = evt_dataset.sortby("epoch").drop_duplicates("epoch")
    # First find indices of the state change events
    sc_indices = np.where(evt_dataset["elid_evtpkt"].data == "SCI_STE")[0]
    event_logs = []
    event_timestamps = []
    event_values = []
    # Get the values of the state change events
    val1 = (
        evt_dataset["el1par_evtpkt"].data[sc_indices] << 8
        | evt_dataset["el2par_evtpkt"].data[sc_indices]
    )
    val2 = (
        evt_dataset["el3par_evtpkt"].data[sc_indices] << 8
        | evt_dataset["el4par_evtpkt"].data[sc_indices]
    )
    epochs = evt_dataset["epoch"][sc_indices].data
    # Now the state change values and check if it is either a science
    # acquisition start or science acquisition stop event.
    for v1, v2, epoch in zip(val1, val2, epochs):
        # An "acquire" start will have val1=ACQSETUP and val2=ACQ
        # An "acquire" stop will have val1=ACQ and val2=CHILL
        if (v1, v2) == (IDEXEvtAcquireCodes.ACQSETUP, IDEXEvtAcquireCodes.ACQ):
            event_logs.append("SCI state change: ACQSETUP to ACQ")
            event_timestamps.append(epoch)
            event_values.append(1)
        elif (v1, v2) == (IDEXEvtAcquireCodes.ACQ, IDEXEvtAcquireCodes.CHILL):
            event_logs.append("SCI state change: ACQ to CHILL")
            event_timestamps.append(epoch)
            event_values.append(0)

    logger.info(
        f"Found science acquisition events: {event_logs} at times: {event_timestamps}"
    )
    return (
        np.asarray(event_logs),
        np.asarray(event_timestamps),
        np.asarray(event_values),
    )


def get_science_acquisition_on_percentage(evt_dataset: xr.Dataset) -> dict:
    """
    Calculate the percentage of time science acquisition was occurring for each day.

    Parameters
    ----------
    evt_dataset : xarray.Dataset
        Contains IDEX event message data.

    Returns
    -------
    dict
        Percentages of time the instrument was in science acquisition mode for each day
         of year.
    """
    # Get science acquisition start and stop times
    evt_logs, evt_time, evt_values = get_science_acquisition_timestamps(evt_dataset)
    # Track total and 'on' durations per day
    daily_totals: collections.defaultdict = defaultdict(timedelta)
    daily_on: collections.defaultdict = defaultdict(timedelta)
    # Convert epoch event times to datetime
    dates = et_to_datetime64(ttj2000ns_to_et(evt_time)).astype(datetime)
    # Simulate an event at the start of the first day.
    start_of_first_day = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
    # Assume that the state at the start of the day is the opposite of what the first
    # state is.
    state_at_start = 0 if evt_values[0] == 1 else 1
    dates = np.insert(dates, 0, start_of_first_day)
    evt_values = np.insert(evt_values, 0, state_at_start)
    for i in range(len(dates)):
        start = dates[i]
        state = evt_values[i]
        if i == len(dates) - 1:
            # If this is the last event, set the "end" value the end of the day.
            end = (start + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            # Otherwise, use the next event time as the end time.
            end = dates[i + 1]

        # Split time span by day boundaries
        current = start
        while current < end:
            next_day = (current + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            segment_end = min(end, next_day)
            duration = segment_end - current
            doy = current.timetuple().tm_yday
            daily_totals[doy] += duration
            # If the state is 1, add to the 'on' duration for that day
            if state == 1:
                daily_on[doy] += duration
            current = segment_end

    # Calculate the percentage of time science acquisition was on for each day
    percent_on_times = {}
    for doy in sorted(daily_totals.keys()):
        total = daily_totals[doy].total_seconds()
        on_time = daily_on[doy].total_seconds()
        pct_on = (on_time / total) * 100 if total > 0 else 0
        percent_on_times[doy] = pct_on

    return percent_on_times
