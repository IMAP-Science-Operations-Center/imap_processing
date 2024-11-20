"""
Find azimuth (degrees), elevation (degrees), and doppler shift (Hz).

Based on ephemeris data and ground station location (longitude, latitude, altitude).

Reference: https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging
import typing

import numpy as np
import spiceypy as spice
from numpy import ndarray

from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.time import et_to_utc, str_to_et

# Logger setup
logger = logging.getLogger(__name__)


def calculate_doppler() -> int:
    """
    Calculate the doppler shift. Placeholder for now.

    Returns
    -------
    doppler : int
        Doppler shift. Currently a throwaway value.
    """
    return 1


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
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.
    # (url cont.) bodvrd
    radii = spice.bodvrd("EARTH", "RADII", 3)[1]
    equatorial_radius = radii[0]  # Equatorial radius in km
    polar_radius = radii[2]  # Polar radius in km
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

    # Convert geodetic coordinates to rectangular coordinates
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.
    # (url cont.) georec
    rect_coords = spice.georec(
        longitude_radians, latitude_radians, altitude, equatorial_radius, flattening
    )

    return rect_coords


@typing.no_type_check
@ensure_spice
def calculate_azimuth_and_elevation(
    longitude: float,
    latitude: float,
    altitude: float,
    observation_time: float,
    target: str = "IMAP",
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
    observation_time : float
        Time at which the state of the target relative to the observer
        is to be computed. Expressed as ephemeris time, seconds past J2000 TDB.
    target : str (Optional)
        The target body. Default is "IMAP".

    Returns
    -------
    azimuth : float
        Azimuth in degrees.
    elevation : float
        Elevation in degrees.
    """
    observer_position_ecef = latitude_longitude_to_ecef(longitude, latitude, altitude)

    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.
    # (url cont.) azlcpo
    azel_results = spice.azlcpo(
        method="Ellipsoid",  # Only method supported
        target=target,  # target ephemeris object
        et=observation_time,  # time of observation
        abcorr="LT+S",  # Aberration correction
        azccw=False,  # Azimuth measured clockwise from the positive y-axis
        # TODO: why not clockwise?
        elplsz=True,  # Elevation increases from the XY plane toward +Z
        obspos=observer_position_ecef,  # observer position relative to center of motion
        obsctr="EARTH",  # Name of the center of motion
        obsref="IAU_EARTH",  # Body-fixed, body-centered reference frame wrt observer's
        # center
    )

    # codespell:ignore convrt
    azimuth = spice.convrt(azel_results[0][1], "RADIANS", "DEGREES")
    elevation = spice.convrt(azel_results[0][2], "RADIANS", "DEGREES")
    # TODO: potentially use the velocity components returned from azlcpo to calculate
    # TODO: doppler

    return azimuth, elevation


def build_output(
    longitude: float,
    latitude: float,
    altitude: float,
    time_endpoints: tuple[str, str],
    time_step: float,
) -> dict[str, list]:
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
    output_dict: dict[str, list]
        Keys are time, azimuth, elevation and doppler. Values are calculated for every
        timestamp between start_utc_input and stop_utc_input, spaced by time_step.
    """
    output_dict: dict[str, list] = {
        "time": [],
        "azimuth": [],
        "elevation": [],
        "doppler": [],
    }

    start_et_input = str_to_et(time_endpoints[0])
    stop_et_input = str_to_et(time_endpoints[1])

    # For now, assume that kernel management will be handled by ensure spice

    for obs_time in np.arange(start_et_input, stop_et_input, time_step):
        azimuth, elevation = calculate_azimuth_and_elevation(
            longitude, latitude, altitude, obs_time
        )

        output_dict["time"].append(et_to_utc(obs_time, format_str="ISOC"))
        output_dict["azimuth"].append(azimuth)
        output_dict["elevation"].append(elevation)
        output_dict["doppler"].append(calculate_doppler())

    logger.info(
        f"Calculated azimuth, elevation and doppler for time range from "
        f"{start_et_input} to {stop_et_input}."
    )

    return output_dict
