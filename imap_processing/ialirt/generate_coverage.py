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
) -> dict[str, np.ndarray]:
    """
    Build the output dictionary containing coverage time for each station.

    Parameters
    ----------
    start_time : str
        Start time in UTC.

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

    for station_name, (lon, lat, alt, min_elevation) in stations.items():
        azimuth, elevation = calculate_azimuth_and_elevation(lon, lat, alt, time_range)
        visible = elevation > min_elevation
        total_visible_mask |= visible
        time_utc = et_to_utc(time_range[visible], format_str="ISOC")

        coverage_dict[f"{station_name}_time"] = time_utc

    # Total coverage percentage
    total_coverage_percent = (
        np.count_nonzero(total_visible_mask) / time_range.size
    ) * 100
    coverage_dict["total_coverage_percent"] = total_coverage_percent

    logger.info(
        f"Calculated station time coverage for stations: {', '.join(stations.keys())}."
    )

    return coverage_dict
