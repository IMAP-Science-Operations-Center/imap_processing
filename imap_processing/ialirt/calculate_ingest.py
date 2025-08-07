"""Coverage time for each station."""

import logging

from datetime import datetime

from imap_processing.ialirt.constants import STATIONS
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)

# TODO: get a list of all potential DSN stations.
ALL_STATIONS = [*STATIONS.keys(), "DSS-24", "DSS-25", "DSS-26",
                "DSS-34", "DSS-35", "DSS-36", "DSS-53", "DSS-54",
                "DSS-55", "DSS-56", "DSS-74", "DSS-75"]


def find_tcp_connections(filename, lines):
    """
    Find connection time ranges for Kiel ground station from log lines.

    Returns
    -------
    List of (start, end) datetime tuples representing TCP connection windows.
    """
    connection_ranges = []
    current_start = None

    # Extract start of hour from filename: "flight_iois_1.log.2025-212T16_55_27.531613"
    timestamp_str = filename.split(".")[2]  # "2025-212T16_55_27"
    timestamp_str = timestamp_str.replace("_", ":")  # "2025-212T16:55:27"
    base_time = datetime.strptime(timestamp_str, "%Y-%jT%H:%M:%S")
    start_of_hour = base_time.replace(minute=0, second=0, microsecond=0)

    for line in lines:
        if "Kiel antenna partner connection is up." in line:
            timestamp = line.split(" ")[0]
            current_start = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
        elif "Kiel antenna partner connection is down!" in line:
            timestamp = line.split(" ")[0]
            end_time = datetime.strptime(timestamp, "%Y/%j-%H:%M:%S.%f")
            if current_start is None:
                connection_ranges.append((start_of_hour, end_time))
            else:
                connection_ranges.append((current_start, end_time))
                current_start = None

    return connection_ranges

