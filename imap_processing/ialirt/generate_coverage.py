"""Coverage time for each station."""

import logging

import numpy as np

from imap_processing.ialirt.constants import STATIONS
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)


def generate_coverage(
    start_time: str,
    outages: dict | None = None,
    dsn: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Build the output dictionary containing coverage time for each station.

    Parameters
    ----------
    start_time : str
        Start time in UTC.
    outages : dict, optional
        Dictionary of outages for each station.
    dsn : dict, optional
        Dictionary of Deep Space Network (DSN) stations.

    Returns
    -------
    coverage_dict: dict
        Coverage for each station.
    """
    duration_seconds = 24 * 60 * 60  # 86400 seconds in 24 hours
    time_step = 3600  # 1 hr in seconds

    stations = {
        "Kiel": STATIONS["Kiel"],
    }
    coverage_dict = {}

    start_et_input = str_to_et(start_time)
    stop_et_input = start_et_input + duration_seconds

    time_range = np.arange(start_et_input, stop_et_input, time_step)
    total_visible_mask = np.zeros(time_range.shape, dtype=bool)

    # Precompute DSN outage mask for non-DSN stations
    dsn_outage_mask = np.zeros(time_range.shape, dtype=bool)
    if dsn:
        for dsn_contacts in dsn.values():
            for start, end in dsn_contacts:
                start_et = str_to_et(start)
                end_et = str_to_et(end)
                dsn_outage_mask |= (time_range >= start_et) & (time_range <= end_et)

    for station_name, (lon, lat, alt, min_elevation) in stations.items():
        azimuth, elevation = calculate_azimuth_and_elevation(lon, lat, alt, time_range)
        visible = elevation > min_elevation

        if outages and station_name in outages:
            for start, end in outages[station_name]:
                start_et = str_to_et(start)
                end_et = str_to_et(end)
                visible[(time_range >= start_et) & (time_range <= end_et)] = False

        # DSN contacts block other stations
        visible[dsn_outage_mask] = False
        total_visible_mask |= visible
        time_utc = et_to_utc(time_range[visible], format_str="ISOC")

        coverage_dict[f"{station_name}_time"] = time_utc

    # --- DSN Stations ---
    if dsn:
        for dsn_station, contacts in dsn.items():
            dsn_visible_mask = np.zeros(time_range.shape, dtype=bool)
            for start, end in contacts:
                start_et = str_to_et(start)
                end_et = str_to_et(end)
                dsn_visible_mask |= (time_range >= start_et) & (time_range <= end_et)

            # Apply DSN outages if present
            if outages and dsn_station in outages:
                for start, end in outages[dsn_station]:
                    start_et = str_to_et(start)
                    end_et = str_to_et(end)
                    dsn_visible_mask[
                        (time_range >= start_et) & (time_range <= end_et)
                    ] = False

            total_visible_mask |= dsn_visible_mask
            coverage_dict[f"{dsn_station}_time"] = et_to_utc(
                time_range[dsn_visible_mask], format_str="ISOC"
            )

    # Total coverage percentage
    total_coverage_percent = (
        np.count_nonzero(total_visible_mask) / time_range.size
    ) * 100
    coverage_dict["total_coverage_percent"] = total_coverage_percent

    all_stations = list(stations.keys()) + (list(dsn.keys()) if dsn else [])
    logger.info(
        f"Calculated station time coverage for stations: {', '.join(all_stations)}."
    )

    return coverage_dict
