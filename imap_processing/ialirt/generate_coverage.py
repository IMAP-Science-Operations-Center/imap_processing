"""Coverage time for each station."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from imap_processing.ialirt.constants import STATIONS, StationProperties
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)

ALL_STATIONS = [
    "Kiel",
    "DSS-24",
    "DSS-25",
    "DSS-26",
    "DSS-34",
    "DSS-35",
    "DSS-36",
    "DSS-53",
    "DSS-54",
    "DSS-55",
    "DSS-56",
    "DSS-74",
    "DSS-75",
]


def parse_uksa_schedule_xlsx(xlsx_path: Path) -> list[tuple[str, str]]:
    """
    Parse the UKSA (GHY-6) availability sheet and return a list of contacts.

    Parameters
    ----------
    xlsx_path : Path
        Path to the UKSA (GHY-6) availability sheet.

    Returns
    -------
    contacts : list[tuple[str, str]]
        Available contacts for UKSA (GHY-6) availability sheet.
    """
    data = pd.read_excel(xlsx_path)

    # Import start and stop times.
    start_dt = (
        data["Date"]
        + pd.to_timedelta(
            data["GHY-6 Start Availability Times  (5degrees) (UTC)"].astype(str)
        )
    ).to_numpy("datetime64[s]")

    stop_dt = (
        data["Date"]
        + pd.to_timedelta(
            data["GHY-6 Stop Availability Times  (5degrees) (UTC)"].astype(str)
        )
    ).to_numpy("datetime64[s]")

    # Indicates whether or not setup or teardown should be taken from contact window.
    notes = data["Short due to existing booking "].fillna("")

    truncate_setup = (
        notes.eq("Yes- setup needs to be included with the window")
        | notes.eq("Yes- setup and teardown needs to be included with the window")
    ).to_numpy()

    truncate_teardown = (
        notes.eq("Yes- tear down needs to be included within the window")
        | notes.eq("Yes- setup and teardown needs to be included with the window")
    ).to_numpy()

    setup_time = data["Setup time"].iloc[0]
    teardown_time = data["Tear down time"].iloc[0]

    setup_seconds = setup_time.hour * 3600 + setup_time.minute * 60 + setup_time.second
    teardown_seconds = (
        teardown_time.hour * 3600 + teardown_time.minute * 60 + teardown_time.second
    )

    setup_delta = np.timedelta64(setup_seconds, "s")
    teardown_delta = np.timedelta64(teardown_seconds, "s")

    # Apply adjustments
    start_dt[truncate_setup] += setup_delta
    stop_dt[truncate_teardown] -= teardown_delta

    # Format to strings with ms, append Z
    start_str = np.datetime_as_string(start_dt, unit="ms")
    stop_str = np.datetime_as_string(stop_dt, unit="ms")

    return list(zip(start_str, stop_str, strict=False))


def create_schedule_mask(
    station: StationProperties, time_range: np.ndarray
) -> np.ndarray:
    """
    Create a boolean mask based on the static daily operating schedule.

    Parameters
    ----------
    station : StationProperties
        Ground station configuration.
    time_range : np.ndarray
        Array of ephemeris time (ET) values corresponding to the
        coverage time.

    Returns
    -------
    schedule_mask : np.ndarray
        Boolean array True is operating window.
    """
    if station.schedule_start is None and station.schedule_end is None:
        return np.ones(time_range.shape, dtype=bool)

    utc_times = et_to_utc(time_range, format_str="ISOC")
    utc_dt = utc_times.astype("datetime64[s]")

    # seconds since midnight (UTC), vectorized
    sec_of_day = (utc_dt - utc_dt.astype("datetime64[D]")) / np.timedelta64(1, "s")

    schedule_mask = np.ones(time_range.shape, dtype=bool)

    if station.schedule_start is not None:
        start_sec = (
            station.schedule_start.hour * 3600
            + station.schedule_start.minute * 60
            + station.schedule_start.second
        )
        schedule_mask &= sec_of_day >= start_sec

    if station.schedule_end is not None:
        end_sec = (
            station.schedule_end.hour * 3600
            + station.schedule_end.minute * 60
            + station.schedule_end.second
        )
        schedule_mask &= sec_of_day <= end_sec

    return schedule_mask


def generate_coverage(  # noqa: PLR0912
    start_time: str,
    outages: dict | None = None,
    dsn: dict | None = None,
    uksa: list | None = None,
) -> tuple[dict, dict]:
    """
    Build the output dictionary containing coverage and outage time for each station.

    Parameters
    ----------
    start_time : str
        Start time in UTC.
    outages : dict, optional
        Dictionary of outages for each station.
    dsn : dict, optional
        Dictionary of Deep Space Network (DSN) stations.
    uksa : list, optional
        List of UKSA contacts.

    Returns
    -------
    coverage_dict : dict
        Visibility times per station.
    outage_dict : dict
        Outage times per station.
    """
    duration_seconds = 24 * 60 * 60  # 86400 seconds in 24 hours
    time_step = 5 * 60  # 5 min in seconds

    stations = {
        "Kiel": STATIONS["Kiel"],
    }
    coverage_dict = {}
    outage_dict = {}

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

    for station_name, station in stations.items():
        _azimuth, elevation = calculate_azimuth_and_elevation(
            station.longitude,
            station.latitude,
            station.altitude,
            time_range,
            obsref="IAU_EARTH",
        )
        visible = elevation > station.min_elevation_deg

        schedule_mask = create_schedule_mask(station, time_range)
        visible &= schedule_mask

        outage_mask = np.zeros(time_range.shape, dtype=bool)
        if outages and station_name in outages:
            for start, end in outages[station_name]:
                start_et = str_to_et(start)
                end_et = str_to_et(end)
                outage_mask |= (time_range >= start_et) & (time_range <= end_et)

        visible[outage_mask] = False
        # DSN contacts block other stations
        visible[dsn_outage_mask] = False
        total_visible_mask |= visible

        coverage_dict[station_name] = et_to_utc(time_range[visible], format_str="ISOC")
        outage_dict[station_name] = et_to_utc(
            time_range[outage_mask], format_str="ISOC"
        )

    # --- DSN Stations ---
    if dsn:
        for dsn_station, contacts in dsn.items():
            dsn_visible_mask = np.zeros(time_range.shape, dtype=bool)
            for start, end in contacts:
                start_et = str_to_et(start)
                end_et = str_to_et(end)
                dsn_visible_mask |= (time_range >= start_et) & (time_range <= end_et)

            # Apply DSN outages if present
            outage_mask = np.zeros(time_range.shape, dtype=bool)
            if outages and dsn_station in outages:
                for start, end in outages[dsn_station]:
                    start_et = str_to_et(start)
                    end_et = str_to_et(end)
                    outage_mask |= (time_range >= start_et) & (time_range <= end_et)

            dsn_visible_mask[outage_mask] = False
            total_visible_mask |= dsn_visible_mask

            coverage_dict[f"{dsn_station}"] = et_to_utc(
                time_range[dsn_visible_mask], format_str="ISOC"
            )
            outage_dict[f"{dsn_station}"] = et_to_utc(
                time_range[outage_mask], format_str="ISOC"
            )
    if uksa:
        uksa_visible_mask = np.zeros(time_range.shape, dtype=bool)
        for start, end in uksa:
            start_et = str_to_et(start)
            end_et = str_to_et(end)
            uksa_visible_mask |= (time_range >= start_et) & (time_range <= end_et)
        total_visible_mask |= uksa_visible_mask
        coverage_dict["UKSA"] = et_to_utc(
            time_range[uksa_visible_mask], format_str="ISOC"
        )

    # Total coverage percentage
    total_coverage_percent = (
        np.count_nonzero(total_visible_mask) / time_range.size
    ) * 100
    coverage_dict["total_coverage_percent"] = total_coverage_percent

    # Ensure all stations are present in both dicts
    for station_name in ALL_STATIONS:
        coverage_dict.setdefault(station_name, np.array([], dtype="<U23"))
        outage_dict.setdefault(station_name, np.array([], dtype="<U23"))

    return coverage_dict, outage_dict


def format_coverage_summary(
    coverage_dict: dict, outage_dict: dict, start_time: str
) -> dict:
    """
    Build the output dictionary containing coverage time for each station.

    Parameters
    ----------
    coverage_dict : dict
        Coverage for each station, keyed by station name with arrays of UTC times.
    outage_dict : dict
        Outage times for each station, keyed by station name with arrays of UTC times.
    start_time : str
        Start time in UTC.

    Returns
    -------
    output_dict : dict
        Formatted coverage summary.
    """
    # Include all known stations,
    # plus any new ones that appear in coverage_dict.
    all_stations = ALL_STATIONS + [
        station
        for station in coverage_dict.keys()
        if station not in ALL_STATIONS and station != "total_coverage_percent"
    ]

    duration_seconds = 24 * 60 * 60  # 86400 seconds in 24 hours
    time_step = 5 * 60  # 5 min in seconds

    start_et_input = str_to_et(start_time)
    stop_et_input = start_et_input + duration_seconds

    time_range = np.arange(start_et_input, stop_et_input, time_step)
    all_times = et_to_utc(time_range, format_str="ISOC")

    data_rows = []
    for time in all_times:
        row = {"time": time}
        for station in all_stations:
            visible_times = coverage_dict.get(station, [])
            outage_times = outage_dict.get(station, [])
            if time in outage_times:
                row[station] = "X"
            elif time in visible_times:
                row[station] = "1"
            else:
                row[station] = "0"
        data_rows.append(row)

    output_dict = {
        "summary": "I-ALiRT Coverage Summary",
        "generated": start_time,
        "time_format": "UTC (ISOC)",
        "stations": all_stations,
        "total_coverage_percent": round(coverage_dict["total_coverage_percent"], 1),
        "data": data_rows,
    }

    return output_dict
