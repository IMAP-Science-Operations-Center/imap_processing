"""Mock expected data for use in some tests."""

import numpy as np
import spiceypy as spice
import xarray as xr

from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins

DEFAULT_SPACING_DEG_L1C = 0.5


def mock_l1c_pset_product(
    spacing_deg: float = DEFAULT_SPACING_DEG_L1C,
    stripe_center_lon: int = 0,
    timestr: str = "2025-01-01T00:00:00",
    head: str = "45",
):
    """Mock the L1C PSET product.

    Will be an xarray.Dataset with at least the variables and shapes:
    counts: (num_lat_bins, num_lon_bins, num_energy_bins)
    exposure_time: (num_lat_bins, num_lon_bins)
    sensitivity: (num_lat_bins, num_lon_bins, num_energy_bins)

    and the coordinate variables:
    latitude: (num_lat_bins)
    longitude: (num_lon_bins)
    energy: (determined by build_energy_bins function)
    head: Either '45' or '90'. Default is '45'.

    as well as the epoch (assumed to be a single time for each product).
    """
    num_lat_bins = int(180 / spacing_deg)
    num_lon_bins = int(360 / spacing_deg)
    stripe_center_lon_bin = int(stripe_center_lon / spacing_deg)

    _, energy_bin_midpoints = build_energy_bins()
    energy_bin_midpoints = energy_bin_midpoints[1:]
    num_energy_bins = len(energy_bin_midpoints)

    grid_shape = (num_lon_bins, num_lat_bins, num_energy_bins)

    def get_binomial_counts(distance_scaling, lon_bin, central_lon_bin):
        # Note, this is not quite correct, as it won't wrap around at 720
        distance_lon_bin = np.abs(lon_bin - central_lon_bin)
        return np.random.binomial(
            n=50,
            p=np.maximum(1 - (distance_lon_bin / distance_scaling), 0.01),
        )

    counts = np.fromfunction(
        lambda lon_bin, lat_bin, energy_bin: get_binomial_counts(
            distance_scaling=20,
            lon_bin=lon_bin,
            central_lon_bin=stripe_center_lon_bin,
        )
        / 1,
        # If you want to reduce counts at the poles, divide by this instead:
        # (
        #     np.exp(1E-4 * (lat_bin - (num_lat_bins // 2))**2)
        # ),
        shape=grid_shape,
    )

    exposure_time = np.zeros(grid_shape[:2]) + 0.1
    if head == "90":
        exposure_time[
            stripe_center_lon_bin : stripe_center_lon_bin + int(20 / spacing_deg),
            :,
        ] = 1
    else:
        counts[
            stripe_center_lon_bin : stripe_center_lon_bin + int(70 / spacing_deg),
            : int(90 / spacing_deg),
            :,
        ] *= 0.1
        counts = counts.astype(int)
        exposure_time[
            stripe_center_lon_bin : stripe_center_lon_bin + int(70 / spacing_deg),
            : int(90 / spacing_deg),
        ] = 1

    sensitivity = np.ones(grid_shape)

    pset_product = xr.Dataset(
        {
            "counts": (
                ["azimuth_bin_center", "elevation_bin_center", "energy_bin_center"],
                counts,
            ),
            "exposure_time": (
                ["azimuth_bin_center", "elevation_bin_center"],
                exposure_time,
            ),
            "sensitivity": (
                ["azimuth_bin_center", "elevation_bin_center", "energy_bin_center"],
                sensitivity,
            ),
            "epoch": ensure_spice(spice.str2et, time_kernels_only=True)(timestr),
        },
        coords={
            "azimuth_bin_center": np.arange(0 + spacing_deg / 2, 360, spacing_deg),
            "elevation_bin_center": np.arange(-90 + spacing_deg / 2, 90, spacing_deg),
            "energy_bin_center": energy_bin_midpoints,
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l1c_{head}sensor-pset_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            )
        },
    )

    return pset_product
