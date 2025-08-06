"""Coverage time for each station."""

import logging

import numpy as np

from imap_processing.ialirt.constants import STATIONS
from imap_processing.ialirt.process_ephemeris import calculate_azimuth_and_elevation
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)

# TODO: get a list of all potential DSN stations.
ALL_STATIONS = [*STATIONS.keys(), "DSS-55", "DSS-56", "DSS-74", "DSS-75"]


def find_tcp_connections(lines):

    return

