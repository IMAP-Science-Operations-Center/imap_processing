"""Find azimuth (degrees), elevation (degrees), and range (km).

Data from the IMAP satellite defined in ephemeris kernel
IMAP_launch20250429_1D.bsp.

Reference: https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import spiceypy as spice
from matplotlib.ticker import ScalarFormatter

# Logger setup
logger = logging.getLogger(__name__)


def get_coverage(kernel_directory, start_et_input, stop_et_input):
    """Decide if input time range falls within ephemeris timerange.

    Parameters
    ----------
    kernel_directory: str
        Directory of the spk kernel.
    start_et_input: float
        Start time in ET.
    stop_et_input: float
        End time in ET.

    NOTE: spk_kernel refers to the kernel containing
    ephemeris data (extension .bsp).
    """
    # Find the time coverage of the ephemeris data
    # provided in the kernel using the IMAP spacecraft ID: -43
    cover = spice.spkcov(kernel_directory, -43)
    start_time_et = cover[0]
    stop_time_et = cover[1]

    # Convert the coverage start and stop times from ET to UTC
    start_time_utc = spice.et2utc(start_time_et, "C", 0)
    stop_time_utc = spice.et2utc(stop_time_et, "C", 0)

    # Print the coverage start and stop times
    print(f"Coverage Start: {start_time_utc}")
    print(f"Coverage Stop: {stop_time_utc}")

    if (start_et_input < start_time_et) or (stop_et_input > stop_time_et):
        raise ValueError("User input time is outside of " "the coverage interval.")


def get_kernels(kernel_directory):
    """Make a list of kernels in the directory.

    Parameters
    ----------
    kernel_directory: str
        Directory of the spk kernel.

    Returns
    -------
    kernels: list
        List of kernels in the directory.
    """
    kernels = []
    acceptable_extensions = [".tsc", ".tls", ".tf", ".bsp", ".ck", ".tpc"]

    for file in os.listdir(kernel_directory):
        if any(file.endswith(ext) for ext in acceptable_extensions):
            file_path = os.path.join(kernel_directory, file)
            kernels.append(file_path)
        else:
            print(f"File {file} is not a valid SPICE kernel.")

    return kernels


def latitude_longitude_to_ecef(longitude, latitude, altitude):
    """
    Convert geodetic coordinates to rectangular coordinates.

    Earth-Centered, Earth-Fixed (ECEF) coordinates are a
    Cartesian coordinate system with an origin at the center of the Earth.

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

    Returns
    -------
    rect_coords : list
        Rectangular coordinates in kilometers.
    """
    latitude_radians = spice.convrt(latitude, "DEGREES", "RADIANS")
    longitude_radians = spice.convrt(longitude, "DEGREES", "RADIANS")

    # Retrieve Earth's radii from SPICE
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.bodvrd
    radii = spice.bodvrd("EARTH", "RADII", 3)[1]
    equatorial_radius = radii[0]  # Equatorial radius in km
    polar_radius = radii[2]  # Polar radius in km
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

    # Convert geodetic coordinates to rectangular coordinates
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.georec
    rect_coords = spice.georec(
        longitude_radians, latitude_radians, altitude, equatorial_radius, flattening
    )

    return rect_coords


def calculate_azimuth_elevation(longitude, latitude, altitude, et, target="IMAP"):
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
    et : int
        Time at which the state of the target relative to the observer
        is to be computed. `et' is expressed as seconds past J2000 TDB.
    target : str (Optional)
        The target body. Default is "IMAP".

    Returns
    -------
    azimuth : float
        Azimuth in degrees.
    elevation : float
        Elevation in degrees.
    range_to_target : float
        Range to the target in kilometers.
    """
    observer_position_ecef = latitude_longitude_to_ecef(longitude, latitude, altitude)

    method = "Ellipsoid"  # Only method supported
    abcorr = "None"  # Aberration correction
    azccw = False  # Azimuth measured clockwise from the positive y-axis
    elplsz = True  # Elevation increases from the XY plane toward +Z
    obsctr = "EARTH"  # Name of the center of motion
    obsref = "IAU_EARTH"  # Body-fixed, body-centered reference frame wrt obsctr

    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.azlcpo
    azel_results = spice.azlcpo(
        method,
        target,
        et,
        abcorr,
        azccw,
        elplsz,
        observer_position_ecef,
        obsctr,
        obsref,
    )

    azimuth = spice.convrt(azel_results[0][1], "RADIANS", "DEGREES")
    elevation = spice.convrt(azel_results[0][2], "RADIANS", "DEGREES")
    range_to_target = azel_results[0][0]  # km

    return azimuth, elevation, range_to_target


def plot_data(data_file):
    """Plot data from the output file.

    Parameters
    ----------
    data_file: str
        Path to the data file.
    """
    data = pd.read_csv(data_file, header=0, parse_dates=True, index_col="time_utc")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    date_str = data.index[0].strftime("%Y-%m-%d")
    fig.suptitle(f"Data for {date_str}", fontsize=16, y=1.02)

    # Plot for Azimuth
    axs[0].plot(data.index, data["azimuth"], label="Azimuth", color="blue")
    axs[0].set_ylabel("Degrees")
    axs[0].set_title("Azimuth over Time")
    axs[0].grid(True)

    # Plot for Elevation
    axs[1].plot(data.index, data["elevation"], label="Elevation", color="green")
    axs[1].set_ylabel("Degrees")
    axs[1].set_title("Elevation over Time")
    axs[1].grid(True)

    # Plot for Range to Target
    axs[2].plot(data.index, data["range"], label="Range", color="red")
    axs[2].set_ylabel("km")
    axs[2].set_title("Range to Target over Time")
    axs[2].grid(True)
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Adjust the y-axis format
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(True)
    axs[2].yaxis.set_major_formatter(formatter)

    plt.subplots_adjust(top=0.95)
    plt.show()


def main():
    """Use main function."""
    # Example usage for Longovilo-Hacienda, Chile
    # https://www.findlatitudeandlongitude.com/l/LONGOVILO%2C+CHILE/
    # Modify as needed
    longitude = -71.41  # longitude in degrees
    latitude = -33.94  # latitude in degrees
    altitude = 0.157  # altitude in kilometers
    start_utc_input = "2026 SEP 22 00:00:00"
    stop_utc_input = "2026 SEP 22 23:59:59"
    time_interval = int(1e3)  # seconds between data points

    # Define the path to the kernel directory relative to this script
    kernel_directory = Path(__file__).parents[0]
    kernels = get_kernels(kernel_directory)

    # Load spice kernels and perform calculations.
    with (
        spice.KernelPool(kernels),
        open(
            "../tests/test_data/spice_partner_example/spice_output_data.txt", "w"
        ) as file,
    ):
        # Write the header line to the output file
        file.write("time_et,time_utc,azimuth,elevation,range\n")

        start_et_input = spice.utc2et(start_utc_input)
        stop_et_input = spice.utc2et(stop_utc_input)

        # Get time coverage from ephemeris kernel
        get_coverage(
            str(kernel_directory / "IMAP_launch20250429_1D.bsp"),
            start_et_input,
            stop_et_input,
        )

        # Get azimuth (degrees) and elevation (degrees) and
        # range (km) to the target for each time step
        for time in range(int(start_et_input), int(stop_et_input), time_interval):
            # Convert to UTC time for readability
            time_utc = spice.et2utc(time, "C", 0)

            azimuth, elevation, range_to_target = calculate_azimuth_elevation(
                longitude, latitude, altitude, time
            )

            azimuth, elevation, range_to_target = (
                f"{azimuth:.4f}",
                f"{elevation:.4f}",
                f"{range_to_target:.4f}",
            )

            # Format data to be written to text file
            data_line = (
                f"{time}, {time_utc}, {azimuth}, " f"{elevation}, {range_to_target}\n"
            )

            file.write(data_line)

    print("Data written to spice_output_data.txt")
    plot_data("../tests/test_data/spice_partner_example/spice_output_data.txt")


if __name__ == "__main__":
    main()
