"""Packet ingest times for each station."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from imap_processing.ialirt.constants import STATIONS

logger = logging.getLogger(__name__)

ALL_STATIONS = [*STATIONS.keys()]


def find_tcp_connections(
    start_file_creation: datetime,
    end_file_creation: datetime,
    lines: list,
    partner: str,
) -> list:
    """
    Find tcp connection time ranges for ground station from log lines.

    Parameters
    ----------
    start_file_creation : datetime
        File creation time of first file minus 1 hr.
    end_file_creation : datetime
        File creation time of last file.
    lines : list
        All lines of log files.
    partner : str
        Ground station partner.

    Returns
    -------
    connection_ranges : list
        List of (start, end) representing TCP connection windows.
    """
    connection_ranges = []
    current_start = None

    for line in lines:
        if f"{partner} antenna partner connection is up." in line:
            timestamp = line.split(" ")[0]
            current_start = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
        elif f"{partner} antenna partner connection is down!" in line:
            timestamp = line.split(" ")[0]
            end_time = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
            if current_start is None:
                connection_ranges.append((start_file_creation, end_time))
            else:
                connection_ranges.append((current_start, end_time))
                current_start = None

    if current_start is not None:
        connection_ranges.append((current_start, end_file_creation))

    return connection_ranges


def packets_created(lines: list) -> list:
    """
    Find timestamps when packets were created based on log lines.

    Parameters
    ----------
    lines : list
        All lines of log files.

    Returns
    -------
    packet_times : list
        List of datetime objects when packets were finalized.
    """
    packet_times = []

    for line in lines:
        if "Renamed iois_1_packets" in line:
            timestamp_str = line.split(" ")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y/%j-%H:%M:%S.%f")
            packet_times.append(timestamp)

    return packet_times


def format_ingest_data(
    first_filename: str, last_filename: str, all_lines: list
) -> dict:
    """
    Format TCP connection and packet ingest data from multiple log files.

    Parameters
    ----------
    first_filename : str
        Log file that is first chronologically.
    last_filename : str
        Log file that is last chronologically.
    all_lines : list[str]
        Combined lines from all log files (assumed already sorted by time).

    Returns
    -------
    formatted : dict
        Structured output with TCP connection windows per station
        and global packet ingest timestamps.

    Notes
    -----
    Example output:
    {
      "summary": "I-ALiRT Real-time Ingest Summary",
      "generated": "2025-08-07T21:36:09Z",
      "time_format": "UTC (ISOC)",
      "stations": [
        "Kiel"
      ],
      "time_range": [
        "2025-07-30T23:00:00",
        "2025-07-31T02:00:00"
      ],
      "packet_ingest": [
        "2025-07-31T00:00:00",
        "2025-07-31T02:01:00"
      ],
      "tcp": {
        "Kiel": [
          {
            "start": "2025-07-30T23:00:00",
            "end": "2025-07-31T00:15:00"
          },
          {
            "start": "2025-07-31T02:00:00",
            "end": "2025-07-31T02:00:00"
          }
        ]
      }
    }
    """
    # File creation time minus 1 hr.
    first_timestamp_str = first_filename.split(".")[2]
    first_timestamp_str = first_timestamp_str.replace("_", ":")
    start_of_time = datetime.strptime(
        first_timestamp_str, "%Y-%jT%H:%M:%S"
    ) - timedelta(hours=1)

    # File creation time.
    last_timestamp_str = last_filename.split(".")[2]
    last_timestamp_str = last_timestamp_str.replace("_", ":")
    end_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S")

    formatted: dict[str, Any] = {
        "summary": "I-ALiRT Real-time Ingest Summary",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_format": "UTC (ISOC)",
        "stations": ALL_STATIONS,
        "time_range": [
            start_of_time.isoformat(),
            end_of_time.isoformat(),
        ],  # Overall time range of the data
        "packet_ingest": [],  # Global packet ingest times
        "tcp": {
            station: [] for station in ALL_STATIONS
        },  # Per-station TCP connection windows
    }

    # TCP connection data for each station
    for station in ALL_STATIONS:
        tcp_ranges = find_tcp_connections(
            start_of_time, end_of_time, all_lines, station
        )
        formatted["tcp"][station] = [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in tcp_ranges
        ]

    # Global packet ingest timestamps
    packet_times = packets_created(all_lines)
    formatted["packet_ingest"] = [pkt_time.isoformat() for pkt_time in packet_times]

    logger.info(f"Created ingest files for {formatted['time_range']}")

    return formatted
