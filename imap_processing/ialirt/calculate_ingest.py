"""Packet ingest and tcp connection times for each station."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from imap_processing.ialirt.constants import STATIONS

logger = logging.getLogger(__name__)


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

    Notes:
    ID  Description   LastDataRcvd  ConnectionTime  Rate (kbps)
    2  tlmrelay      001-00:00:00  020-19:57:14    0.0
    10  Kiel          021-09:57:58  021-08:40:39    2.0
    """
    dict = {
        station: {
            "last_data_received": [],
            "rate_kbps": [],
        }
        for station in list(STATIONS) + ["tlmrelay"]
    }

    year = start_file_creation.year
    in_rate_table = False

    for line in lines:
        if "Rate (kbps)" in line:
            in_rate_table = True
            continue
        if (in_rate_table and "tlmrelay" in line) or any(
            station in line for station in STATIONS
        ):
            rate = float(line.split()[-1])
            data_last_received = line.split()[2]
            station = line.split()[1]
            dt = datetime.strptime(f"{year}/{data_last_received}", "%Y/%j-%H:%M:%S", )
            dict[station]["last_data_received"].append(dt)
            dict[station]["rate_kbps"].append(rate)

    return dict


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
    }

    where time_range is the overall time range of the data,
    packet_ingest contains timestamps when packets were finalized,
    and tcp contains connection windows for each station.
    """
    # File creation time.
    last_timestamp_str = last_filename.split(".")[2]
    last_timestamp_str = last_timestamp_str.replace("_", ":")
    end_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S")

    # File creation time.
    start_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S") - timedelta(
        minutes=5
    )

    realtime_summary: dict[str, Any] = {
        "summary": "I-ALiRT Real-time Ingest Summary",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_format": "UTC (ISOC)",
        "time_range": [
            start_of_time.isoformat(),
            end_of_time.isoformat(),
        ],  # Overall time range of the data
        "packet_ingest": [],  # Global packet ingest times
    }

    # Global packet ingest timestamps
    packet_times = packets_created(start_of_time, log_lines)
    realtime_summary["packet_ingest"] = [
        pkt_time.isoformat() for pkt_time in packet_times
    ]

    logger.info(f"Created ingest files for {realtime_summary['time_range']}")

    return realtime_summary
