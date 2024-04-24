import os
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from matplotlib.colors import Normalize
from pvlib.solarposition import get_solarposition

from tools.spice.spice_examples_partners import (
    calculate_azimuth_elevation,
    get_kernels,
    latitude_longitude_to_ecef,
)


def plot_test_data(data_dict):
    """Plot test data."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())

    ax[0].plot(data_dict["time_array"], data_dict["azimuth_array"], label="IMAP")
    ax[0].plot(
        data_dict["time_array"],
        data_dict["solar_azimuth_array"],
        label="SUN",
        marker="*",
    )
    ax[0].plot(
        data_dict["time_array"], data_dict["test_azimuth_array"], label="Test Data: SUN"
    )
    ax[0].set_ylabel("Azimuth (degrees)")
    ax[0].legend()

    ax[1].plot(data_dict["time_array"], data_dict["elevation_array"], label="IMAP")
    ax[1].plot(
        data_dict["time_array"],
        data_dict["solar_elevation_array"],
        label="SUN",
        marker="*",
    )
    ax[1].plot(
        data_dict["time_array"],
        data_dict["test_elevation_array"],
        label="Test Data: SUN",
    )
    ax[1].set_ylabel("Elevation (degrees)")
    ax[1].legend()

    plt.gcf().autofmt_xdate()
    plt.show()


def plot_color_data(data_dict):
    """Plot colorbar data."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    date_format = "%Y %b %d %H:%M:%S"
    datetime_objects = [
        datetime.strptime(date_str, date_format) for date_str in data_dict["time_array"]
    ]
    times = mdates.date2num(datetime_objects)

    norm = Normalize(vmin=min(times), vmax=max(times))
    cmap = plt.get_cmap("coolwarm")

    sc = ax.scatter(
        data_dict["azimuth_array"],
        data_dict["elevation_array"],
        c=times,
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel("Azimuth (Degrees)")
    ax.set_ylabel("Elevation (Degrees)")
    ax.set_title("Azimuth vs. Elevation Colored by Time")
    ax.grid(True)

    cbar = plt.colorbar(
        sc, ax=ax, orientation="vertical", format=mdates.DateFormatter("%H:%M")
    )
    cbar.ax.set_ylabel("Time of Day (UTC)")

    plt.subplots_adjust(hspace=0.4)
    plt.gcf().autofmt_xdate()
    plt.show()


def test_latitude_longitude_to_ecef():
    """Tests latitude_longitude_to_ecef function."""

    # Test data from
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/georec_c.html

    kernel_directory = Path(__file__).parents[0]
    kernel = os.path.join(
        kernel_directory, "../test_data/spice_partner_example/pck00010.tpc"
    )

    # Set a geodetic position
    longitude = 118.0  # degrees
    latitude = 30.0  # degrees
    altitude = 0.0  # kilometers

    with spice.KernelPool(kernel):
        rect_coords = latitude_longitude_to_ecef(longitude, latitude, altitude)

    assert np.round(rect_coords[0], 6) == -2595.359123
    assert np.round(rect_coords[1], 6) == 4881.160589
    assert np.round(rect_coords[2], 6) == 3170.373523


def test_calculate_azimuth_elevation():
    """Tests calculate_azimuth_elevation function."""

    kernel_directory = Path(__file__).parents[1] / "test_data" / "spice_partner_example"

    # Example usage for Longovilo-Hacienda, Chile
    # https://www.findlatitudeandlongitude.com/l/LONGOVILO%2C+CHILE/
    longitude = -71.41  # longitude in degrees
    latitude = -33.94  # latitude in degrees
    altitude = 0.157  # altitude in kilometers
    time_interval = int(1e3)  # seconds between data points

    kernels = get_kernels(kernel_directory)

    (
        time_array,
        azimuth_array,
        elevation_array,
        range_array,
        test_azimuth_array,
        test_elevation_array,
        solar_azimuth_array,
        solar_elevation_array,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    with spice.KernelPool(kernels):
        start_et_input = spice.utc2et("2026 SEP 22 00:00:00")
        stop_et_input = spice.utc2et("2026 SEP 22 23:59:59")

        for time in range(int(start_et_input), int(stop_et_input), time_interval):
            azimuth, elevation, data_range = calculate_azimuth_elevation(
                longitude, latitude, altitude, time
            )
            solar_azimuth, solar_elevation, solar_range = calculate_azimuth_elevation(
                longitude, latitude, altitude, time, target="SUN"
            )

            time_utc = spice.et2utc(time, "C", 0)
            test = get_solarposition(time_utc, latitude, longitude, altitude)

            time_array.append(time_utc)
            azimuth_array.append(azimuth)
            elevation_array.append(elevation)
            range_array.append(data_range)
            solar_azimuth_array.append(solar_azimuth)
            solar_elevation_array.append(solar_elevation)
            test_azimuth_array.append(test["azimuth"].values[0])
            test_elevation_array.append(test["elevation"].values[0])

    dict = {
        "time_array": time_array,
        "azimuth_array": azimuth_array,
        "elevation_array": elevation_array,
        "solar_azimuth_array": solar_azimuth_array,
        "solar_elevation_array": solar_elevation_array,
        "test_azimuth_array": test_azimuth_array,
        "test_elevation_array": test_elevation_array,
    }

    plot_test_data(dict)
    plot_color_data(dict)

    assert np.allclose(
        range_array, 1.5e6, atol=1e5
    ), "Values are not within 1e5 km of 1.5 million km"
