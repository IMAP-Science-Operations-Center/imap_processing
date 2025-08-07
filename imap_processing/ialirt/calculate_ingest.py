"""Coverage time for each station."""

import logging

from datetime import datetime, timedelta

from imap_processing.ialirt.constants import STATIONS
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)

ALL_STATIONS = [*STATIONS.keys()]


def find_tcp_connections(first_filename, last_filename, lines, partner):
    """
    Find connection time ranges for Kiel ground station from log lines.

    Returns
    -------
    List of (start, end) datetime tuples representing TCP connection windows.
    """
    connection_ranges = []
    current_start = None

    # File creation time minus 1 hr.
    first_timestamp_str = first_filename.split(".")[2]
    first_timestamp_str = first_timestamp_str.replace("_", ":")
    start_of_time = datetime.strptime(first_timestamp_str, "%Y-%jT%H:%M:%S") - timedelta(hours=1)

    # File creation time.
    last_timestamp_str = last_filename.split(".")[2]
    last_timestamp_str = last_timestamp_str.replace("_", ":")
    end_of_time = datetime.strptime(last_timestamp_str, "%Y-%jT%H:%M:%S")

    for line in lines:
        if f"{partner} antenna partner connection is up." in line:
            timestamp = line.split(" ")[0]
            current_start = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
        elif f"{partner} antenna partner connection is down!" in line:
            timestamp = line.split(" ")[0]
            end_time = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
            if current_start is None:
                connection_ranges.append((start_of_time, end_time))
            else:
                connection_ranges.append((current_start, end_time))
                current_start = None

    if current_start is not None:
        connection_ranges.append((current_start, end_of_time))

    return connection_ranges


def packets_created(lines):
    """
    Find timestamps when packets were created based on log lines.

    Returns
    -------
    List of datetime objects when packets were finalized.
    """
    packet_times = []

    for line in lines:
        if "Renamed iois_1_packets" in line:
            timestamp_str = line.split(" ")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y/%j-%H:%M:%S.%f")
            packet_times.append(timestamp)

    return packet_times


def format_ingest_data(first_filename, last_filename, all_lines):
    """
    Format TCP connection and packet ingest data from multiple log files.

    Parameters
    ----------
    first_filename : str
        Log file that is first chronologically.
    first_filename : str
        Log file that is first chronologically.
    all_lines : list[str]
        Combined lines from all log files (assumed already sorted by time).

    Returns
    -------
    dict
        Structured output with TCP connection windows per station
        and global packet ingest timestamps.
    """

    formatted = {
        "packet_ingest": [],  # Global packet ingest times
        "tcp": {},            # Per-station TCP connection windows
    }

    # TCP connection data for each station
    for station in ALL_STATIONS:
        tcp_ranges = find_tcp_connections(first_filename, last_filename, all_lines, station)
        formatted["tcp"][station] = [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in tcp_ranges
        ]

    # Global packet ingest timestamps
    packet_times = packets_created(all_lines)
    formatted["packet_ingest"] = [
        pkt_time.isoformat() for pkt_time in packet_times
    ]

    return formatted
