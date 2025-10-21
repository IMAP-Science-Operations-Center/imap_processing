"""
Find azimuth (degrees), elevation (degrees), and doppler shift (Hz).

Based on ephemeris data and ground station location (longitude, latitude, altitude).

Reference: https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging
import typing
from datetime import datetime, timedelta

import numpy as np
import spiceypy
from numpy import ndarray

from imap_processing.ialirt.constants import STATIONS
from imap_processing.spice.geometry import SpiceBody, SpiceFrame, imap_state
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)


def latitude_longitude_to_ecef(
    longitude: float, latitude: float, altitude: float
) -> ndarray:
    """
    Convert geodetic coordinates to rectangular coordinates.

    Earth-Centered, Earth-Fixed (ECEF) coordinates are a Cartesian coordinate system
    with an origin at the center of the Earth.

    Parameters
    ----------
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian, negative to west.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative to south.
    altitude : float
        Altitude in kilometers.

    Returns
    -------
    rect_coords : ndarray
        Rectangular coordinates in kilometers.
    """
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)

    # Retrieve Earth's radii from SPICE
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.bod
    # (url cont.) vrd
    radii = spiceypy.bodvrd("EARTH", "RADII", 3)[1]
    equatorial_radius = radii[0]  # Equatorial radius in km
    polar_radius = radii[2]  # Polar radius in km
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

    # Convert geodetic coordinates to rectangular coordinates
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.georec
    rect_coords = spiceypy.georec(
        longitude_radians, latitude_radians, altitude, equatorial_radius, flattening
    )

    return rect_coords


@typing.no_type_check
def calculate_azimuth_and_elevation(
    longitude: float,
    latitude: float,
    altitude: float,
    observation_time: float | np.ndarray,
    target: str = SpiceBody.IMAP.name,
    obsref: str = "ITRF93",
) -> tuple:
    """
    Calculate azimuth and elevation.

    Parameters
    ----------
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative
        to south.
    altitude : float
        Altitude in kilometers.
    observation_time : float or np.ndarray
        Time at which the state of the target relative to the observer
        is to be computed. Expressed as ephemeris time, seconds past J2000 TDB.
    target : str (Optional)
        The target body. Default is "IMAP".
    obsref : str (Optional)
        Body-fixed, body-centered reference frame wrt
        observer's center.

    Returns
    -------
    azimuth : np.ndarray
        Azimuth in degrees.
    elevation : np.ndarray
        Elevation in degrees.
    """
    ground_station_position_ecef = latitude_longitude_to_ecef(
        longitude, latitude, altitude
    )

    if not isinstance(observation_time, np.ndarray):
        observation_time = [observation_time]

    azimuth = []
    elevation = []

    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.azlcpo
    for timestamp in observation_time:
        azel_results = spiceypy.azlcpo(
            method="Ellipsoid",  # Only method supported
            target=target,  # target ephemeris object
            et=timestamp,  # time of observation
            abcorr="LT+S",  # Aberration correction
            azccw=False,  # Azimuth measured clockwise from the positive y-axis
            elplsz=True,  # Elevation increases from the XY plane toward +Z
            obspos=ground_station_position_ecef,  # observer pos. to center of motion
            obsctr="EARTH",  # Name of the center of motion
            obsref=obsref,  # Body-fixed, body-centered reference frame wrt
            # observer's center
        )
        azimuth.append(np.rad2deg(azel_results[0][1]))
        elevation.append(np.rad2deg(azel_results[0][2]))

    return np.asarray(azimuth), np.asarray(elevation)


@typing.no_type_check
def calculate_doppler(
    longitude: float,
    latitude: float,
    altitude: float,
    observation_time: float | np.ndarray,
) -> float | ndarray[float]:
    """
    Calculate the doppler velocity.

    Notes about the spkezr function (wrapped in imap_state):
    The function returns the state of the target (state) and the light time.
    The first three components of state represent the x-, y- and z-components of the
    target's position; the last three components form the corresponding velocity vector.

    Parameters
    ----------
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative
        to south.
    altitude : float
        Altitude in kilometers.
    observation_time : float or np.ndarray
        Time at which the state of the target relative to the observer
        is to be computed. Expressed as ephemeris time, seconds past J2000 TDB.

    Returns
    -------
    doppler : float or np.ndarray[float]
        Doppler velocity in kilometers per second.
    """
    ground_station_position_ecef = latitude_longitude_to_ecef(
        longitude, latitude, altitude
    )

    # find position and velocity relative to the center of the earth using spice spkezr
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.spkezr
    state = imap_state(
        et=observation_time,
        ref_frame=SpiceFrame.ITRF93,
        abcorr="LT+S",
        observer=SpiceBody.EARTH,
    )
    # shifting position by subtracting ground station location relative to the center
    # of the earth
    state[..., 0:3] -= ground_station_position_ecef
    # calculate radial velocity
    doppler = np.sum(state[..., 3:6] * state[..., 0:3], axis=-1) / np.linalg.norm(
        state[..., 0:3], axis=-1
    )

    return np.asarray(doppler)


def build_output(
    longitude: float,
    latitude: float,
    altitude: float,
    time_endpoints: tuple[str, str],
    time_step: float = 60,
) -> dict[str, np.ndarray]:
    """
    Build the output dictionary containing time, azimuth, elevation, and doppler.

    Parameters
    ----------
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian, negative to west.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative to south.
    altitude : float
        Altitude in kilometers.
    time_endpoints : tuple[str, str]
        Start and stop times in UTC.
    time_step : float
        Seconds between data points. Default is 60.

    Returns
    -------
    output_dict: dict[str, np.ndarray]
        Keys are time, azimuth, elevation and doppler. Values are calculated for every
        timestamp between start_utc_input and stop_utc_input, spaced by time_step.
    """
    output_dict: dict[str, np.ndarray] = {}

    start_et_input = str_to_et(time_endpoints[0])
    stop_et_input = str_to_et(time_endpoints[1])
    time_range = np.arange(start_et_input, stop_et_input, time_step)

    # For now, assume that kernel management will be handled by ensure_spice
    azimuth, elevation = calculate_azimuth_and_elevation(
        longitude, latitude, altitude, time_range, obsref="ITRF93"
    )

    output_dict["time"] = et_to_utc(time_range, format_str="ISOC")
    output_dict["azimuth"] = np.round(azimuth, 6)
    output_dict["elevation"] = np.round(elevation, 6)
    output_dict["doppler"] = np.round(
        calculate_doppler(longitude, latitude, altitude, time_range), 6
    )

    logger.info(
        f"Calculated azimuth, elevation and doppler for time range from "
        f"{start_et_input} to {stop_et_input}."
    )

    return output_dict


def generate_text_files(station: str, day: str) -> list[str]:
    """
    Generate a pointing schedule text file and return it as a list of strings.

    Parameters
    ----------
    station : str
        Station name.
    day : str
        The day for which to generate a pointing schedule, in ISO format.
        Ex: "2025-08-11".

    Returns
    -------
    lines : list[str]
        A list of strings that makeup the lines of a pointing schedule file.
    """
    station_properties = STATIONS[station]

    day_as_datetime = datetime.fromisoformat(day)
    time_endpoints = (
        datetime.strftime(day_as_datetime, "%Y-%m-%d %H:%M:%S"),
        datetime.strftime(day_as_datetime + timedelta(days=1), "%Y-%m-%d %H:%M:%S"),
    )
    output_dict = build_output(
        station_properties[0],
        station_properties[1],
        station_properties[2],
        time_endpoints,
    )

    lines = [
        f"Station: {station}\n",
        "Target: IMAP\n",
        f"Creation date (UTC): {datetime.utcnow()}\n",
        f"Start time: {time_endpoints[0]}\n",
        f"End time: {time_endpoints[1]}\n",
        "Cadence (sec): 60\n\n",
        "Date/Time"
        + "Azimuth".rjust(29)
        + "Elevation".rjust(17)
        + "Doppler".rjust(15)
        + "\n",
        "(UTC)" + "(deg.)".rjust(33) + "(deg.)".rjust(16) + "(km/s)".rjust(16) + "\n",
    ]

    length = len(output_dict["time"])
    for i in range(length):
        lines.append(
            f"{output_dict['time'][i]}"
            + f"{output_dict['azimuth'][i]}".rjust(16)
            + f"{output_dict['elevation'][i]}".rjust(16)
            + f"{output_dict['doppler'][i]}".rjust(15)
            + "\n"
        )

    return lines
