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
    l2b_and_l2c_datasets = idex_l2b(l2a_data, [evt_data])
    write_cdf(l2b_and_l2c_datasets[0])
    write_cdf(l2b_and_l2c_datasets[1])
"""

import collections
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import SkyTilingType
from imap_processing.ena_maps.utils.spatial_utils import AzElSkyGrid
from imap_processing.idex.idex_constants import (
    FG_TO_KG,
    IDEX_EVENT_REFERENCE_FRAME,
    IDEX_SPACING_DEG,
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

# Get the rectangular map grid with the specified spacing
SKY_GRID = AzElSkyGrid(IDEX_SPACING_DEG)
LON_BINS_EDGES = SKY_GRID.az_bin_edges
LAT_BINS_EDGES = SKY_GRID.el_bin_edges


def idex_l2b(
    l2a_datasets: list[xr.Dataset], evt_datasets: list[xr.Dataset]
) -> list[xr.Dataset]:
    """
    Will process IDEX l2a data to create l2b and l2c data products.

    IDEX L2B processing creates L2b and L2c at the same time because L2c needs no
    additional dependencies and is a natural extension of L2b processing.

    Parameters
    ----------
    l2a_datasets : list[xarray.Dataset]
        IDEX L2a datasets to process.
    evt_datasets : list[xarray.Dataset]
        List of IDEX housekeeping event message datasets.

    Returns
    -------
    list[xarray.Dataset]
        The``xarray`` datasets containing the l2b and l2c science data and supporting
        metadata.
    """
    logger.info(
        "Running IDEX L2B and L2C processing on L2a datasets. NOTE: L2C datasets are "
        "processed at the same time as L2B datasets because L2C needs no additional "
        "dependencies."
    )
    # create the attribute manager for this data level
    idex_l2b_attrs = get_idex_attrs("l2b")
    idex_l2c_attrs = get_idex_attrs("l2c")
    evt_dataset = xr.concat(evt_datasets, dim="epoch")

    # Concat all the l2a datasets together
    l2a_dataset = xr.concat(l2a_datasets, dim="epoch")
    epoch_doy_unique = np.unique(epoch_to_doy(l2a_dataset["epoch"].data))
    (
        counts_by_charge,
        counts_by_mass,
        counts_by_charge_map,
        counts_by_mass_map,
        daily_epoch,
    ) = compute_counts_by_charge_and_mass(l2a_dataset, epoch_doy_unique)
    # Get science acquisition percentage for each day
    daily_on_percentage = get_science_acquisition_on_percentage(evt_dataset)
    (
        rate_by_charge,
        rate_by_mass,
        rate_by_charge_map,
        rate_by_mass_map,
        rate_quality_flags,
    ) = compute_rates_by_charge_and_mass(
        counts_by_charge,
        counts_by_mass,
        counts_by_charge_map,
        counts_by_mass_map,
        epoch_doy_unique,
        daily_on_percentage,
    )
    # Create l2b Dataset
    charge_bin_means = np.sqrt(CHARGE_BIN_EDGES[:-1] * CHARGE_BIN_EDGES[1:])
    mass_bin_means = np.sqrt(MASS_BIN_EDGES[:-1] * MASS_BIN_EDGES[1:])
    spin_phase_means = (SPIN_PHASE_BIN_EDGES[:-1] + SPIN_PHASE_BIN_EDGES[1:]) / 2

    # Define xarrays that are shared between l2b and l2c
    epoch = xr.DataArray(
        name="epoch",
        data=daily_epoch,
        dims="epoch",
        attrs=idex_l2b_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    common_vars = {
        "impact_day_of_year": xr.DataArray(
            name="impact_day_of_year",
            data=epoch_doy_unique,
            dims="epoch",
            attrs=idex_l2b_attrs.get_variable_attributes("impact_day_of_year"),
        ),
        "charge_labels": xr.DataArray(
            name="impact_charge_labels",
            data=charge_bin_means.astype(str),
            dims="impact_charge",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "charge_labels", check_schema=False
            ),
        ),
        "mass_labels": xr.DataArray(
            name="mass_labels",
            data=mass_bin_means.astype(str),
            dims="mass",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "mass_labels", check_schema=False
            ),
        ),
        "impact_charge": xr.DataArray(
            name="impact_charge",
            data=charge_bin_means,
            dims="impact_charge",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "impact_charge", check_schema=False
            ),
        ),
        "mass": xr.DataArray(
            name="mass",
            data=mass_bin_means,
            dims="mass",
            attrs=idex_l2b_attrs.get_variable_attributes("mass", check_schema=False),
        ),
    }
    l2b_vars = common_vars | {
        "spin_phase": xr.DataArray(
            name="spin_phase",
            data=spin_phase_means,
            dims="spin_phase",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "spin_phase", check_schema=False
            ),
        ),
        "spin_phase_labels": xr.DataArray(
            name="spin_phase_labels",
            data=spin_phase_means.astype(str),
            dims="spin_phase",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "spin_phase_labels", check_schema=False
            ),
        ),
        "rate_calculation_quality_flags": xr.DataArray(
            name="rate_calculation_quality_flags",
            data=rate_quality_flags,
            dims="epoch",
            attrs=idex_l2b_attrs.get_variable_attributes(
                "rate_calculation_quality_flags"
            ),
        ),
        "counts_by_charge": xr.DataArray(
            name="counts_by_charge",
            data=counts_by_charge.astype(np.int64),
            dims=("epoch", "impact_charge", "spin_phase"),
            attrs=idex_l2b_attrs.get_variable_attributes("counts_by_charge"),
        ),
        "counts_by_mass": xr.DataArray(
            name="counts_by_mass",
            data=counts_by_mass.astype(np.int64),
            dims=("epoch", "mass", "spin_phase"),
            attrs=idex_l2b_attrs.get_variable_attributes("counts_by_mass"),
        ),
        "rate_by_charge": xr.DataArray(
            name="rate_by_charge",
            data=rate_by_charge,
            dims=("epoch", "impact_charge", "spin_phase"),
            attrs=idex_l2b_attrs.get_variable_attributes("rate_by_charge"),
        ),
        "rate_by_mass": xr.DataArray(
            name="rate_by_mass",
            data=rate_by_mass,
            dims=("epoch", "mass", "spin_phase"),
            attrs=idex_l2b_attrs.get_variable_attributes("rate_by_mass"),
        ),
    }
    l2c_vars = common_vars | {
        "rectangular_lon_pixel_label": xr.DataArray(
            name="rectangular_lon_pixel_label",
            data=SKY_GRID.az_bin_midpoints.astype(str),
            dims="rectangular_lon_pixel",
            attrs=idex_l2c_attrs.get_variable_attributes(
                "rectangular_lon_pixel_label", check_schema=False
            ),
        ),
        "rectangular_lat_pixel_label": xr.DataArray(
            name="rectangular_lat_pixel_label",
            data=SKY_GRID.el_bin_midpoints.astype(str),
            dims="rectangular_lat_pixel",
            attrs=idex_l2c_attrs.get_variable_attributes(
                "rectangular_lat_pixel_label", check_schema=False
            ),
        ),
        "rectangular_lon_pixel": xr.DataArray(
            name="rectangular_lon_pixel",
            data=SKY_GRID.az_bin_midpoints,
            dims="rectangular_lon_pixel",
            attrs=idex_l2c_attrs.get_variable_attributes(
                "rectangular_lon_pixel", check_schema=False
            ),
        ),
        "rectangular_lat_pixel": xr.DataArray(
            name="rectangular_lat_pixel",
            data=SKY_GRID.el_bin_midpoints,
            dims="rectangular_lat_pixel",
            attrs=idex_l2c_attrs.get_variable_attributes(
                "rectangular_lat_pixel", check_schema=False
            ),
        ),
        "counts_by_charge_map": xr.DataArray(
            name="counts_by_charge_map",
            data=counts_by_charge_map.astype(np.int64),
            dims=(
                "epoch",
                "impact_charge",
                "rectangular_lon_pixel",
                "rectangular_lat_pixel",
            ),
            attrs=idex_l2c_attrs.get_variable_attributes("counts_by_charge_map"),
        ),
        "counts_by_mass_map": xr.DataArray(
            name="counts_by_mass_map",
            data=counts_by_mass_map.astype(np.int64),
            dims=(
                "epoch",
                "mass",
                "rectangular_lon_pixel",
                "rectangular_lat_pixel",
            ),
            attrs=idex_l2c_attrs.get_variable_attributes("counts_by_mass_map"),
        ),
        "rate_by_charge_map": xr.DataArray(
            name="rate_by_charge_map",
            data=rate_by_charge_map,
            dims=(
                "epoch",
                "impact_charge",
                "rectangular_lon_pixel",
                "rectangular_lat_pixel",
            ),
            attrs=idex_l2c_attrs.get_variable_attributes("rate_by_charge_map"),
        ),
        "rate_by_mass_map": xr.DataArray(
            name="rate_by_mass_map",
            data=rate_by_mass_map,
            dims=(
                "epoch",
                "mass",
                "rectangular_lon_pixel",
                "rectangular_lat_pixel",
            ),
            attrs=idex_l2c_attrs.get_variable_attributes("rate_by_mass_map"),
        ),
    }

    l2b_dataset = xr.Dataset(
        coords={"epoch": epoch},
        data_vars=l2b_vars,
        attrs=idex_l2b_attrs.get_global_attributes("imap_idex_l2b_sci"),
    )
    l2c_dataset = xr.Dataset(
        coords={"epoch": epoch},
        data_vars=l2c_vars,
    )
    # Add map attributes
    map_attrs = {
        "sky_tiling_type": SkyTilingType.RECTANGULAR.value,
        "Spacing_degrees": str(IDEX_SPACING_DEG),
        "Spice_reference_frame": IDEX_EVENT_REFERENCE_FRAME.name,
    } | idex_l2c_attrs.get_global_attributes("imap_idex_l2c_sci-rectangular")

    l2c_dataset.attrs.update(map_attrs)

    logger.info("IDEX L2B and L2C science data processing completed.")

    return [l2b_dataset, l2c_dataset]


def compute_counts_by_charge_and_mass(
    l2a_dataset: xr.Dataset, epoch_doy_unique: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dust counts by charge and mass by spin phase or lon and lat per day.

    Parameters
    ----------
    l2a_dataset : xarray.Dataset
        Combined IDEX L2a datasets.
    epoch_doy_unique : np.ndarray
        Unique days of year corresponding to the epochs in the dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Two 3D arrays containing counts by charge or mass, and by spin phase for each
        dataset, Two 4D arrays containing counts by charge or mass, and by lon and lat
        for each dataset, and a 1D array of daily epoch values.
    """
    # Initialize lists to hold counts.
    counts_by_charge = []
    counts_by_mass = []
    counts_by_charge_map = []
    counts_by_mass_map = []
    daily_epoch = np.zeros(len(epoch_doy_unique), dtype=np.float64)
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
        # Make sure longitude values are in the range [0, 360)
        longitude = np.mod(l2a_dataset["longitude"].data[current_day_indices], 360)
        latitude = l2a_dataset["latitude"].data[current_day_indices]
        # Convert units
        mass_vals = FG_TO_KG * np.atleast_1d(mass_vals)
        # Bin spin phases
        binned_spin_phase = bin_spin_phases(spin_phase_angles)
        # Clip arrays to ensure that the values are within the valid range of bins.
        # Latitude should be binned with the right edge included. 90 is a valid latitude
        latitude = np.clip(latitude, -90, 90)
        mass_vals = np.clip(mass_vals, MASS_BIN_EDGES[0], MASS_BIN_EDGES[-1])
        charge_vals = np.clip(charge_vals, CHARGE_BIN_EDGES[0], CHARGE_BIN_EDGES[-1])

        counts_by_mass.append(
            np.histogramdd(
                np.column_stack([mass_vals, binned_spin_phase]),
                bins=[MASS_BIN_EDGES, np.arange(5)],
            )[0]
        )
        counts_by_charge.append(
            np.histogramdd(
                np.column_stack([charge_vals, binned_spin_phase]),
                bins=[CHARGE_BIN_EDGES, np.arange(5)],
            )[0]
        )
        counts_by_mass_map.append(
            np.histogramdd(
                np.column_stack([mass_vals, longitude, latitude]),
                bins=[MASS_BIN_EDGES, LON_BINS_EDGES, LAT_BINS_EDGES],
            )[0]
        )
        counts_by_charge_map.append(
            np.histogramdd(
                np.column_stack([charge_vals, longitude, latitude]),
                bins=[CHARGE_BIN_EDGES, LON_BINS_EDGES, LAT_BINS_EDGES],
            )[0]
        )

    return (
        np.stack(counts_by_charge),
        np.stack(counts_by_mass),
        np.stack(counts_by_charge_map),
        np.stack(counts_by_mass_map),
        daily_epoch,
    )


def compute_rates(
    counts: np.ndarray, epoch_doy_percent_on: np.ndarray, non_zero_inds: np.ndarray
) -> np.ndarray:
    """
    Compute the count rates given the percent uptime of IDEX.

    Parameters
    ----------
    counts : np.ndarray
        Count values for the dust events.
    epoch_doy_percent_on : np.ndarray
        Percentage of time science acquisition was on for each day of the year.
    non_zero_inds : np.ndarray
        Indices of the days with non-zero science acquisition percentage.

    Returns
    -------
    np.ndarray
        Count rates.
    """
    while len(epoch_doy_percent_on.shape) < len(counts.shape):
        epoch_doy_percent_on = np.expand_dims(epoch_doy_percent_on, axis=-1)

    return counts[non_zero_inds] / (
        0.01 * epoch_doy_percent_on[non_zero_inds] * SECONDS_IN_DAY
    )


def compute_rates_by_charge_and_mass(
    counts_by_charge: np.ndarray,
    counts_by_mass: np.ndarray,
    counts_by_charge_map: np.ndarray,
    counts_by_mass_map: np.ndarray,
    epoch_doy: np.ndarray,
    daily_on_percentage: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dust event counts rates by charge and mass by spin phase for each day.

    Parameters
    ----------
    counts_by_charge : np.ndarray
        3D array containing counts by charge and spin phase for each dataset.
    counts_by_mass : np.ndarray
        3D array containing counts by mass and lon and lat for each dataset.
    counts_by_charge_map : np.ndarray
        4D array containing counts by charge and lon and lat for each dataset.
    counts_by_mass_map : np.ndarray
        4D array containing counts by mass and spin phase for each dataset.
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
    rate_by_charge_map = np.full(counts_by_charge_map.shape, -1.0)
    rate_by_mass_map = np.full(counts_by_mass_map.shape, -1.0)
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
    rate_by_charge[non_zero_inds] = compute_rates(
        counts_by_charge, epoch_doy_percent_on, non_zero_inds
    )
    rate_by_mass[non_zero_inds] = compute_rates(
        counts_by_mass, epoch_doy_percent_on, non_zero_inds
    )
    rate_by_charge_map[non_zero_inds] = compute_rates(
        counts_by_charge_map, epoch_doy_percent_on, non_zero_inds
    )
    rate_by_mass_map[non_zero_inds] = compute_rates(
        counts_by_mass_map, epoch_doy_percent_on, non_zero_inds
    )

    return (
        rate_by_charge,
        rate_by_mass,
        rate_by_charge_map,
        rate_by_mass_map,
        rate_quality_flags,
    )


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
    # Use mod to wrap values >= 360 to 0.
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
    for v1, v2, epoch in zip(val1, val2, epochs, strict=False):
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
    _evt_logs, evt_time, evt_values = get_science_acquisition_timestamps(evt_dataset)
    if len(evt_time) == 0:
        logger.warning(
            "No science acquisition events found in event dataset. Returning empty "
            "uptime percentages. All rate variables will be set to -1."
        )
        return {}
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
