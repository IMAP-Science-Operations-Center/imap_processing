"""Packet ingest times and rates for each station."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

STATIONS = ["Kiel"]


def packets_created(start_file_creation: datetime, lines: list) -> dict:
    """
    Find timestamps and rates when packets were ingested based on log lines.

    Parameters
    ----------
    start_file_creation : datetime
        File creation time of last file minus 48 hrs.
    lines : list
        All lines of log files.

    Returns
    -------
    station_dict : dict
        Timestamps and rates when packets were ingested.
    """
    station_dict: dict[str, dict[str, list[Any]]] = {
        station: {"last_data_received": [], "rate_kbps": []}
        for station in list(STATIONS)
    }

    station_year: dict[str, int] = {
        station: start_file_creation.year for station in station_dict
    }
    prev_doy: dict[str, int | None] = {station: None for station in station_dict}

    for line in lines:
        # If line begins with a digit and the station is present.
        if line.split()[0].isdigit() and line.split()[1] in STATIONS:
            # Get bps rate.
            rate = float(line.split()[-1])
            # Get last data received.
            data_last_received = line.split()[2]
            # Get day of year.
            doy = int(data_last_received[:3])
            # Get station.
            station = line.split()[1]

            # Handle end of year rollover
            prev = prev_doy[station]

            if prev is not None and doy < prev:
                station_year[station] += 1

            prev_doy[station] = doy

            dt = (
                datetime.strptime(
                    f"{station_year[station]}/{data_last_received}",
                    "%Y/%j-%H:%M:%S",
                )
                .replace(tzinfo=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            station_dict[station]["last_data_received"].append(dt)
            station_dict[station]["rate_kbps"].append(rate)

    return station_dict


def format_ingest_data(last_filename: str, log_lines: list) -> dict:
    """
    Format packet ingest times and rates from log file.

    Parameters
    ----------
    last_filename : str
        Log file that is last chronologically.
    log_lines : list[str]
        Combined lines from all log files (assumed already sorted by time).

    Returns
    -------
    realtime_summary : dict
        Structured output with packet receipt info per station.

    Notes
    -----
    Example output:
    {
      "summary": "I-ALiRT Real-time Ingest Summary",
      "generated": "2025-08-07T21:36:09Z",
      "time_format": "UTC (ISOC)",
      "time_range": [
        "2025-01-21T09:50:58Z",
        "2025-01-21T09:55:58Z"
      ],
        "Kiel": {"last_data_received": ["2025-01-21T09:50:58Z", "2025-01-21T09:51:58Z"],
        "rate_kbps": [2.0, 2.0]}
    }
    """
    # File creation time.
    last_timestamp_str = last_filename.split(".")[2]
    last_timestamp_str = last_timestamp_str.replace("_", ":")
    end_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S")

    # File is created every 5 minutes.
    start_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S") - timedelta(
        minutes=5
    )

    # Parse file.
    station_dict = packets_created(start_of_time, log_lines)

    realtime_summary: dict[str, Any] = {
        "summary": "I-ALiRT Real-time Ingest Summary",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_format": "UTC (ISOC)",
        "time_range": [
            start_of_time.isoformat(),
            end_of_time.isoformat(),
        ],  # Overall time range of the data
        **station_dict,
    }

    logger.info(f"Created ingest files for {realtime_summary['time_range']}")

    return realtime_summary
