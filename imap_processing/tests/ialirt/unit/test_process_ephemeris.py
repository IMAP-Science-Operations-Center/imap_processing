"""Test processEphemeris functions."""

import numpy as np
import pytest

from imap_processing.ialirt import process_ephemeris
from imap_processing.spice.time import str_to_et


def test_calculate_doppler():
    """
    Test the calculate_doppler() function.
    """
    assert process_ephemeris.calculate_doppler(805794429.1837295) == 1


@pytest.mark.external_kernel()
def test_latitude_longitude_to_ecef(furnish_kernels):
    """
    Test the latitude_longitude_to_ecef() function.

    Test data is from https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/
    georec_c.html.
    """
    longitude = 118.0  # degrees
    latitude = 30.0  # degrees
    altitude = 0.0  # kilometers

    kernels = ["pck00011.tpc"]
    with furnish_kernels(kernels):
        rect_coords = process_ephemeris.latitude_longitude_to_ecef(
            longitude, latitude, altitude
        )

    np.testing.assert_allclose(
        rect_coords, [-2595.359123, 4881.160589, 3170.373523], atol=1e-6
    )


@pytest.mark.external_kernel()
def test_calculate_azimuth_and_elevation(furnish_kernels):
    """
    Test the calculate_azimuth_and_elevation() function.
    """
    longitude = -71.41  # longitude in degrees
    latitude = -33.94  # latitude in degrees
    altitude = 0.157  # altitude in kilometers
    # test single observation time
    observation_time = 805794429.1837295  # "2025-07-14T19:46:00.000"

    kernels = [
        "pck00011.tpc",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        azimuth_result, elevation_result = (
            process_ephemeris.calculate_azimuth_and_elevation(
                longitude, latitude, altitude, observation_time
            )
        )
    assert azimuth_result, elevation_result is not None

    # test array of observation times
    time_endpoints = ("2026 SEP 22 00:00:00", "2026 SEP 22 23:59:59")
    time_interval = int(1e3)  # seconds between data points
    observation_time = np.arange(
        str_to_et(time_endpoints[0]), str_to_et(time_endpoints[1]), time_interval
    )
    with furnish_kernels(kernels):
        azimuth_result, elevation_result = (
            process_ephemeris.calculate_azimuth_and_elevation(
                longitude, latitude, altitude, observation_time
            )
        )
    assert len(azimuth_result) == len(observation_time)


@pytest.mark.external_kernel()
def test_build_output(furnish_kernels):
    """
    Test the build_output() function.
    """
    # Example usage for Longovilo-Hacienda, Chile
    # https://www.findlatitudeandlongitude.com/l/LONGOVILO%2C+CHILE/
    longitude = -71.41  # longitude in degrees
    latitude = -33.94  # latitude in degrees
    altitude = 0.157  # altitude in kilometers
    time_endpoints = ("2026 SEP 22 00:00:00", "2026 SEP 22 23:59:59")
    time_interval = int(1e3)  # seconds between data points

    kernels = [
        "naif0012.tls",
        "pck00011.tpc",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        output_dict = process_ephemeris.build_output(
            longitude, latitude, altitude, time_endpoints, time_interval
        )

    for key_name in ["azimuth", "elevation", "time", "doppler"]:
        assert key_name in output_dict.keys()
        assert len(output_dict[key_name]) == len(
            np.arange(
                str_to_et(time_endpoints[0]),
                str_to_et(time_endpoints[1]),
                time_interval,
            )
        )
