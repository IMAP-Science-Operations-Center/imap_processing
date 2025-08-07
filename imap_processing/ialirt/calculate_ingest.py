"""Coverage time for each station."""

import logging

from datetime import datetime, timedelta

from imap_processing.ialirt.constants import STATIONS
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)

ALL_STATIONS = [*STATIONS.keys()]


def find_tcp_connections(filename, lines, partner):
    """
    Find connection time ranges for Kiel ground station from log lines.

    Returns
    -------
    List of (start, end) datetime tuples representing TCP connection windows.
    """
    connection_ranges = []
    current_start = None

    timestamp_str = filename.split(".")[2]
    timestamp_str = timestamp_str.replace("_", ":")
    base_time = datetime.strptime(timestamp_str, "%Y-%jT%H:%M:%S")
    start_of_hour = base_time.replace(minute=0, second=0, microsecond=0)
    end_of_hour = start_of_hour + timedelta(hours=1)

    for line in lines:
        if f"{partner} antenna partner connection is up." in line:
            timestamp = line.split(" ")[0]
            current_start = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
        elif f"{partner} antenna partner connection is down!" in line:
            timestamp = line.split(" ")[0]
            end_time = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
            if current_start is None:
                connection_ranges.append((start_of_hour, end_time))
            else:
                connection_ranges.append((current_start, end_time))
                current_start = None

    if current_start is not None:
        connection_ranges.append((current_start, end_of_hour))

    return connection_ranges

