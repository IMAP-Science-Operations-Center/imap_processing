"""Packet ingest and tcp connection times for each station."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from imap_processing.ialirt.constants import STATIONS

logger = logging.getLogger(__name__)


def find_tcp_connections(  # noqa: PLR0912
    start_file_creation: datetime,
    end_file_creation: datetime,
    lines: list,
    realtime_summary: dict,
) -> dict:
    """
    Find tcp connection time ranges for ground station from log lines.

    Parameters
    ----------
    start_file_creation : datetime
        File creation time of last file minus 48 hrs.
    end_file_creation : datetime
        File creation time of last file.
    lines : list
        All lines of log files.
    realtime_summary : dict
        Input dictionary containing ingest parameters.

    Returns
    -------
    realtime_summary : dict
        Output dictionary with tcp connection info.
    """
    current_starts: dict[str, datetime | None] = {}
    partners_opened = set()

    for line in lines:
        # Note if this line appears.
        if "Opened raw record file" in line:
            station = line.split("Opened raw record file for ")[1].split(
                " antenna_partner"
            )[0]
            partners_opened.add(station)

        if "antenna partner connection is" not in line:
            continue

        timestamp_str = line.split(" ")[0]
        msg = " ".join(line.split(" ")[1:])
        station = msg.split(" antenna")[0]

        if station not in realtime_summary["connection_times"]:
            realtime_summary["connection_times"][station] = []
        if station not in realtime_summary["stations"]:
            realtime_summary["stations"].append(station)

        timestamp = datetime.strptime(timestamp_str, "%Y/%j-%H:%M:%S.%f")

        if f"{station} antenna partner connection is up." in line:
            current_starts[station] = timestamp

        elif f"{station} antenna partner connection is down!" in line:
            start = current_starts.get(station)
            if start is not None:
                realtime_summary["connection_times"][station].append(
                    {
                        "start": datetime.isoformat(start),
                        "end": datetime.isoformat(timestamp),
                    }
                )
                current_starts[station] = None
            else:
                # No matching "up"
                realtime_summary["connection_times"][station].append(
                    {
                        "start": datetime.isoformat(start_file_creation),
                        "end": datetime.isoformat(timestamp),
                    }
                )
                current_starts[station] = None

    # Handle hanging "up" at the end of file
    for station, start in current_starts.items():
        if start is not None:
            realtime_summary["connection_times"][station].append(
                {
                    "start": datetime.isoformat(start),
                    "end": datetime.isoformat(end_file_creation),
                }
            )

    # Handle stations with only "Opened raw record file" (no up/down)
    for station in partners_opened:
        if not realtime_summary["connection_times"][station]:
            realtime_summary["connection_times"][station].append(
                {
                    "start": datetime.isoformat(start_file_creation),
                    "end": datetime.isoformat(end_file_creation),
                }
            )

    # Filter out connection windows that are completely outside the time window
    for station in realtime_summary["connection_times"]:
        realtime_summary["connection_times"][station] = [
            window
            for window in realtime_summary["connection_times"][station]
            if datetime.fromisoformat(window["end"]) >= start_file_creation
            and datetime.fromisoformat(window["start"]) <= end_file_creation
        ]

    return realtime_summary


def packets_created(start_file_creation: datetime, lines: list) -> list:
    """
    Find timestamps when packets were created based on log lines.

    Parameters
    ----------
    start_file_creation : datetime
        File creation time of last file minus 48 hrs.
    lines : list
        All lines of log files.

    Returns
    -------
    packet_times : list
        List of datetime objects when packets were created.
    """
    packet_times = []

    for line in lines:
        if "Renamed iois_1_packets" in line:
            timestamp_str = line.split(" ")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y/%j-%H:%M:%S.%f")
            # Possible that data extends further than 48 hrs in the past.
            if timestamp >= start_file_creation:
                packet_times.append(timestamp)

    return packet_times


def format_ingest_data(last_filename: str, log_lines: list) -> dict:
    """
    Format TCP connection and packet ingest data from multiple log files.

    Parameters
    ----------
    last_filename : str
        Log file that is last chronologically.
    log_lines : list[str]
        Combined lines from all log files (assumed already sorted by time).

    Returns
    -------
    realtime_summary : dict
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
      "connection_times": {
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

    where time_range is the overall time range of the data,
    packet_ingest contains timestamps when packets were finalized,
    and tcp contains connection windows for each station.
    """
    # File creation time.
    last_timestamp_str = last_filename.split(".")[2]
    last_timestamp_str = last_timestamp_str.replace("_", ":")
    end_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S")

    # File creation time of last file minus 48 hrs.
    start_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S") - timedelta(
        hours=48
    )

    realtime_summary: dict[str, Any] = {
        "summary": "I-ALiRT Real-time Ingest Summary",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_format": "UTC (ISOC)",
        "stations": list(STATIONS),
        "time_range": [
            start_of_time.isoformat(),
            end_of_time.isoformat(),
        ],  # Overall time range of the data
        "packet_ingest": [],  # Global packet ingest times
        "connection_times": {
            station: [] for station in list(STATIONS)
        },  # Per-station TCP connection windows
    }

    # TCP connection data for each station
    realtime_summary = find_tcp_connections(
        start_of_time, end_of_time, log_lines, realtime_summary
    )

    # Global packet ingest timestamps
    packet_times = packets_created(start_of_time, log_lines)
    realtime_summary["packet_ingest"] = [
        pkt_time.isoformat() for pkt_time in packet_times
    ]

    logger.info(f"Created ingest files for {realtime_summary['time_range']}")

    return realtime_summary
