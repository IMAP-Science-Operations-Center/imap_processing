"""Mock expected data for use in some tests."""

import astropy_healpix.healpy as hp
import numpy as np
import spiceypy as spice
import xarray as xr

from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.time import str_to_et
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins

DEFAULT_RECT_SPACING_DEG_L1C = 0.5
DEFAULT_HEALPIX_NSIDE_L1C = 128


def mock_l1c_pset_product_rectangular(
    spacing_deg: float = DEFAULT_RECT_SPACING_DEG_L1C,
    stripe_center_lon: int = 0,
    timestr: str = "2025-01-01T00:00:00",
    head: str = "45",
) -> xr.Dataset:
    """
    Mock the L1C PSET product with recognizable but unrealistic counts.

    This is not meant to perfectly mimic the real data, but to provide a
    recognizable structure for L2 testing purposes.
    Function will produce an xarray.Dataset with at least the variables and shapes:
    counts: (1 epoch, num_energy_bins, num_lon_bins, num_lat_bins)
    exposure_time: (num_lon_bins, num_lat_bins)
    sensitivity: (1 epoch, num_energy_bins, num_lon_bins, num_lat_bins)

    and the coordinate variables:
    the epoch (assumed to be a single time for each product).
    energy: (determined by build_energy_bins function)
    longitude: (num_lon_bins)
    latitude: (num_lat_bins)

    While not a coordinate, PSETs can also be distinguished by the 'head' attribute.
    head: Either '45' or '90'. Default is '45'.

    The counts are generated along a stripe, centered at a given longitude.
    This stripe can be thought of as a 'vertical' line if the lon/az axis is plotted
    as the x-axis and the lat/el axis is plotted as the y-axis. See the figure below.

    ^  Elevation/Latitude
    |
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    |    000000000000002468642000000000000000000000000000000
    --------------------------------------------------------->
    Azimuth/Longitude ->

    Fig. 1: Example of the '90' sensor head stripe

    Parameters
    ----------
    spacing_deg : float, optional
        The bin spacing in degrees (default is 0.5 degrees).
    stripe_center_lon : int, optional
        The center longitude of the stripe in degrees (default is 0).
    timestr : str, optional
        The time string for the epoch (default is "2025-01-01T00:00:00").
    head : str, optional
        The sensor head (either '45' or '90') (default is '45').
    """
    num_lat_bins = int(180 / spacing_deg)
    num_lon_bins = int(360 / spacing_deg)
    stripe_center_lon_bin = int(stripe_center_lon / spacing_deg)

    _, energy_bin_midpoints = build_energy_bins()
    num_energy_bins = len(energy_bin_midpoints)

    # 1 epoch x num_energy_bins x num_lon_bins x num_lat_bins
    grid_shape = (1, num_energy_bins, num_lon_bins, num_lat_bins)

    def get_binomial_counts(distance_scaling, lon_bin, central_lon_bin):
        # Note, this is not quite correct, as it won't wrap around at 360 degrees
        # but it's all meant to provide a recognizable pattern for testing
        distance_lon_bin = np.abs(lon_bin - central_lon_bin)

        rng = np.random.default_rng(seed=42)
        return rng.binomial(
            n=50,
            p=np.maximum(1 - (distance_lon_bin / distance_scaling), 0.01),
        )

    counts = np.fromfunction(
        lambda epoch, energy_bin, lon_bin, lat_bin: get_binomial_counts(
            distance_scaling=20,
            lon_bin=lon_bin,
            central_lon_bin=stripe_center_lon_bin,
        ),
        shape=grid_shape,
    )

    exposure_time = np.zeros(grid_shape[2:]) + 0.1
    if head == "90":
        exposure_time[
            stripe_center_lon_bin : stripe_center_lon_bin + int(20 / spacing_deg),
            :,
        ] = 1
    else:
        exposure_time[
            stripe_center_lon_bin : stripe_center_lon_bin + int(70 / spacing_deg),
            : int(90 / spacing_deg),
        ] = 1

    counts = counts.astype(int)
    sensitivity = np.ones(grid_shape)

    # Determine the epoch, which is TT time in nanoseconds since J2000 epoch
    tdb_et = str_to_et(timestr)
    tt_j2000ns = (
        ensure_spice(spice.unitim, time_kernels_only=True)(tdb_et, "ET", "TT") * 1e9
    )

    pset_product = xr.Dataset(
        {
            "counts": (
                [
                    "epoch",
                    "energy_bin_center",
                    "longitude_bin_center",
                    "latitude_bin_center",
                ],
                counts,
            ),
            "exposure_time": (
                ["longitude_bin_center", "latitude_bin_center"],
                exposure_time,
            ),
            "sensitivity": (
                [
                    "epoch",
                    "energy_bin_center",
                    "longitude_bin_center",
                    "latitude_bin_center",
                ],
                sensitivity,
            ),
        },
        coords={
            "epoch": [
                tt_j2000ns,
            ],
            "energy_bin_center": energy_bin_midpoints,
            "longitude_bin_center": np.arange(0 + spacing_deg / 2, 360, spacing_deg),
            "latitude_bin_center": np.arange(-90 + spacing_deg / 2, 90, spacing_deg),
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l1c_{head}sensor-pset_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            )
        },
    )

    return pset_product


def mock_l1c_pset_product_healpix(  # noqa: PLR0913
    nside: int = DEFAULT_HEALPIX_NSIDE_L1C,
    stripe_center_lat: int = 0,
    width_scale: float = 10.0,
    counts_scaling_params: tuple[int, float] = (100, 0.01),
    peak_exposure: float = 1000.0,
    timestr: str = "2025-01-01T00:00:00",
    head: str = "45",
) -> xr.Dataset:
    """
    Mock the L1C PSET product with recognizable but unrealistic counts.

    This is not meant to perfectly mimic the real data, but to provide a
    recognizable structure for L2 testing purposes.
    Function will produce an xarray.Dataset with at least the variables and shapes:
    counts: (1 epoch, num_energy_bins, num_lon_bins, num_lat_bins)
    exposure_time: (num_lon_bins, num_lat_bins)
    sensitivity: (1 epoch, num_energy_bins, num_lon_bins, num_lat_bins)

    and the coordinate variables:
    the epoch (assumed to be a single time for each product).
    energy: (determined by build_energy_bins function)
    longitude: (num_lon_bins)
    latitude: (num_lat_bins)

    While not a coordinate, PSETs can also be distinguished by the 'head' attribute.
    head: Either '45' or '90'. Default is '45'.

    The counts are generated along a stripe, centered at a given latitude.
    This stripe can be thought of as a 'vertical' line if the lon/az axis is plotted
    as the x-axis and the lat/el axis is plotted as the y-axis. See the figure below.

    ^  Elevation/Latitude
    |
    |                   00000000000000000000
    |               0000000000000000000000000000
    |           0000000000000000000000000000000000000
    |        0000000000000000000000000000000000000000000
    |      00000000000000000000000000000000000000000000000
    |     0000000000000000000000000000000000000000000000000
    |    222222222222222222222222222222222222222222222222222
    |    444444444444444444444444444444444444444444444444444
    |    666666666666666666666666666666666666666666666666666
    |     4444444444444444444444444444444444444444444444444
    |      22222222222222222222222222222222222222222222222
    |        0000000000000000000000000000000000000000000
    |           0000000000000000000000000000000000000
    |               0000000000000000000000000000
    |                   00000000000000000000
    --------------------------------------------------------->
    Azimuth/Longitude ->

    Fig. 1: Example of the '90' sensor head stripe on a HEALPix grid

    """
    _, energy_bin_midpoints = build_energy_bins()
    num_energy_bins = len(energy_bin_midpoints)
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix)
    exposure_time = np.zeros(npix)

    # Get latitude for each healpix pixel
    pix_indices = np.arange(npix)
    lon_pix, lat_pix = hp.pix2ang(nside, pix_indices, lonlat=True)

    counts = np.zeros(shape=(num_energy_bins, npix))

    # Calculate probability based on distance from target latitude
    lat_diff = np.abs(lat_pix - stripe_center_lat)
    prob_scaling_factor = counts_scaling_params[1] * np.exp(
        -(lat_diff**2) / (2 * width_scale**2)
    )
    # Generate counts using binomial distribution

    rng = np.random.default_rng(seed=42)
    counts = np.array(
        [
            rng.binomial(n=counts_scaling_params[0], p=prob_scaling_factor)
            for _ in range(num_energy_bins)
        ]
    )

    # Generate exposure times using gaussian distribution
    exposure_time = peak_exposure * prob_scaling_factor

    # Ensure counts are integers
    counts = counts.astype(int)
    # add an epoch dimension
    counts = np.expand_dims(counts, axis=0)
    sensitivity = np.ones_like(counts)

    # Determine the epoch, which is TT time in nanoseconds since J2000 epoch
    tdb_et = str_to_et(timestr)
    tt_j2000ns = (
        ensure_spice(spice.unitim, time_kernels_only=True)(tdb_et, "ET", "TT") * 1e9
    )

    pset_product = xr.Dataset(
        {
            "counts": (
                [
                    "epoch",
                    "energy_bin_center",
                    "healpix_pixel_index",
                ],
                counts,
            ),
            "exposure_time": (
                ["healpix_pixel_index"],
                exposure_time,
            ),
            "sensitivity": (
                [
                    "epoch",
                    "energy_bin_center",
                    "healpix_pixel_index",
                ],
                sensitivity,
            ),
            "longitude_bin_center": (
                ["healpix_pixel_index"],
                lon_pix,
            ),
            "latitude_bin_center": (
                ["healpix_pixel_index"],
                lat_pix,
            ),
        },
        coords={
            "epoch": [
                tt_j2000ns,
            ],
            "energy_bin_center": energy_bin_midpoints,
            "healpix_pixel_index": pix_indices,
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l1c_{head}sensor-pset_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            )
        },
    )

    return pset_product
