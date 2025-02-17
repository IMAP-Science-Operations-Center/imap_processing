"""
Find azimuth (degrees), elevation (degrees), and doppler shift (Hz).

Based on ephemeris data and ground station location (longitude, latitude, altitude).

Reference: https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging
import typing
from typing import Union

import numpy as np
import spiceypy
from numpy import ndarray

from imap_processing.spice.geometry import SpiceBody, SpiceFrame, imap_state
from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)


@typing.no_type_check
@ensure_spice
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
@ensure_spice
def calculate_azimuth_and_elevation(
    longitude: float,
    latitude: float,
    altitude: float,
    observation_time: Union[float, np.ndarray],
    target: str = SpiceBody.IMAP.name,
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
            obsref="IAU_EARTH",  # Body-fixed, body-centered reference frame wrt
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
    observation_time: Union[float, np.ndarray],
) -> Union[float, ndarray[float]]:
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
    time_step: float,
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
        Seconds between data points.

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
        longitude, latitude, altitude, time_range
    )

    output_dict["time"] = et_to_utc(time_range, format_str="ISOC")
    output_dict["azimuth"] = azimuth
    output_dict["elevation"] = elevation
    output_dict["doppler"] = calculate_doppler(
        longitude, latitude, altitude, time_range
    )

    logger.info(
        f"Calculated azimuth, elevation and doppler for time range from "
        f"{start_et_input} to {stop_et_input}."
    )

    return output_dict
