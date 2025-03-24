"""Mock expected data for use in some tests."""

import astropy_healpix.healpy as hp
import numpy as np
import spiceypy as spice
import xarray as xr

from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.time import str_to_et
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins

DEFAULT_RECT_SPACING_DEG_L1C = 0.5
DEFAULT_HEALPIX_NSIDE_L1C = 128


def mock_l1c_pset_product_rectangular(  # noqa: PLR0913
    spacing_deg: float = DEFAULT_RECT_SPACING_DEG_L1C,
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
    This stripe can be thought of as a 'horizontal' line if the lon/az axis is plotted
    as the x-axis and the lat/el axis is plotted as the y-axis. See the figure below.

    ^  Elevation/Latitude
    |
    |    000000000000000000000000000000000000000000000000000     |
    |    000000000000000000000000000000000000000000000000000     |
    |    000000000000000000000000000000000000000000000000000     |
    |    000000000000000000000000000000000000000000000000000     |
    |    000000000000000000000000000000000000000000000000000      \
    |    222222222222222222222222222222222222222222222222222       \
    |    444444444444444444444444444444444444444444444444444        \
    |    666666666666666666666666666666666666666666666666666         |
    |    444444444444444444444444444444444444444444444444444        /
    |    222222222222222222222222222222222222222222222222222       /
    |    000000000000000000000000000000000000000000000000000      /
    |    000000000000000000000000000000000000000000000000000     |
    --------------------------------------------------------->
    Azimuth/Longitude ->

    Fig. 1: Example of the '90' sensor head stripe

    Parameters
    ----------
    spacing_deg : float, optional
        The bin spacing in degrees (default is 0.5 degrees).
    stripe_center_lat : int, optional
        The center latitude of the stripe in degrees (default is 0).
    width_scale : float, optional
        The width of the stripe in degrees (default is 20 degrees).
    counts_scaling_params : tuple[int, float], optional
        The parameters for the binomial distribution of counts (default is (100, 0.01)).
        The 0th element is the number of trials to draw,
        the 1st element scales the probability of success for each trial.
    peak_exposure : float, optional
        The peak exposure time (default is 1000.0).
    timestr : str, optional
        The time string for the epoch (default is "2025-01-01T00:00:00").
    head : str, optional
        The sensor head (either '45' or '90') (default is '45').
    """
    num_lat_bins = int(180 / spacing_deg)
    num_lon_bins = int(360 / spacing_deg)
    stripe_center_lat_bin = int((stripe_center_lat + 90) / spacing_deg)

    _, energy_bin_midpoints = build_energy_bins()
    num_energy_bins = len(energy_bin_midpoints)

    # 1 epoch x num_energy_bins x num_lon_bins x num_lat_bins
    grid_shape = (1, num_energy_bins, num_lon_bins, num_lat_bins)

    def get_binomial_counts(distance_scaling, lat_bin, central_lat_bin):
        # Note, this is not quite correct, as it won't wrap around at 360 degrees
        # but it's all meant to provide a recognizable pattern for testing
        distance_lat_bin = np.abs(lat_bin - central_lat_bin)

        rng = np.random.default_rng(seed=42)
        return rng.binomial(
            n=counts_scaling_params[0],
            p=np.maximum(
                1 - (distance_lat_bin / distance_scaling), counts_scaling_params[1]
            ),
        )

    counts = np.fromfunction(
        lambda epoch, energy_bin, lon_bin, lat_bin: get_binomial_counts(
            distance_scaling=width_scale,
            lat_bin=lat_bin,
            central_lat_bin=stripe_center_lat_bin,
        ),
        shape=grid_shape,
    )

    # exposure_time should be a gaussian distribution centered on the stripe
    # with a width of 20 degrees
    exposure_time = np.zeros(grid_shape[2:])
    exposure_time = np.fromfunction(
        lambda lon_bin, lat_bin: np.exp(
            -((lat_bin - stripe_center_lat_bin) ** 2) / (2 * width_scale**2)
        ),
        shape=grid_shape[2:],
    )
    exposure_time /= exposure_time.max()
    exposure_time *= peak_exposure
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
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                    CoordNames.ELEVATION_L1C.value,
                ],
                counts,
            ),
            "exposure_time": (
                [
                    CoordNames.TIME.value,
                    CoordNames.AZIMUTH_L1C.value,
                    CoordNames.ELEVATION_L1C.value,
                ],
                np.expand_dims(exposure_time, axis=0),
            ),
            "sensitivity": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                    CoordNames.ELEVATION_L1C.value,
                ],
                sensitivity,
            ),
        },
        coords={
            CoordNames.TIME.value: [
                tt_j2000ns,
            ],
            CoordNames.ENERGY.value: energy_bin_midpoints,
            CoordNames.AZIMUTH_L1C.value: np.arange(
                0 + spacing_deg / 2, 360, spacing_deg
            ),
            CoordNames.ELEVATION_L1C.value: np.arange(
                -90 + spacing_deg / 2, 90, spacing_deg
            ),
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

    See the docstring for mock_l1c_pset_product_rectangular for more details about
    the structure of the dataset.
    The rectangular and Healpix mocked datasets are very similar in structure, though
    the actual values at a given latitude and longitude may be different. This is only
    meant to provide a recognizable structure for L2 testing purposes.

    The counts are generated along a stripe, centered at a given latitude.
    This stripe can be thought of as a 'vertical' line if the lon/az axis is plotted
    as the x-axis and the lat/el axis is plotted as the y-axis. See the figure below.

    ^  Elevation/Latitude
    |
    |                   00000000000000000000                     |
    |               0000000000000000000000000000                 |
    |           0000000000000000000000000000000000000            |
    |        0000000000000000000000000000000000000000000         |
    |      00000000000000000000000000000000000000000000000       |
    |     0000000000000000000000000000000000000000000000000       \
    |    222222222222222222222222222222222222222222222222222       \
    |    444444444444444444444444444444444444444444444444444        \
    |    666666666666666666666666666666666666666666666666666         |
    |     4444444444444444444444444444444444444444444444444         /
    |      22222222222222222222222222222222222222222222222         /
    |        0000000000000000000000000000000000000000000          /
    |           0000000000000000000000000000000000000            |
    |               0000000000000000000000000000                 |
    |                   00000000000000000000                     |
    --------------------------------------------------------->
    Azimuth/Longitude ->

    Fig. 1: Example of the '90' sensor head stripe on a HEALPix grid

    Parameters
    ----------
    nside : int, optional
        The HEALPix nside parameter (default is 128).
    stripe_center_lat : int, optional
        The center latitude of the stripe in degrees (default is 0).
    width_scale : float, optional
        The width of the stripe in degrees (default is 10 degrees).
    counts_scaling_params : tuple[int, float], optional
        The parameters for the binomial distribution of counts (default is (100, 0.01)).
        The 0th element is the number of trials to draw,
        the 1st element scales the probability of success for each trial.
    peak_exposure : float, optional
        The peak exposure time (default is 1000.0).
    timestr : str, optional
        The time string for the epoch (default is "2025-01-01T00:00:00").
    head : str, optional
        The sensor head (either '45' or '90') (default is '45').
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
    exposure_time = peak_exposure * (prob_scaling_factor / prob_scaling_factor.max())

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
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                counts,
            ),
            "exposure_time": (
                [CoordNames.HEALPIX_INDEX.value],
                exposure_time,
            ),
            "sensitivity": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                sensitivity,
            ),
            CoordNames.AZIMUTH_L1C.value: (
                [CoordNames.HEALPIX_INDEX.value],
                lon_pix,
            ),
            CoordNames.ELEVATION_L1C.value: (
                [CoordNames.HEALPIX_INDEX.value],
                lat_pix,
            ),
        },
        coords={
            CoordNames.TIME.value: [
                tt_j2000ns,
            ],
            CoordNames.ENERGY.value: energy_bin_midpoints,
            CoordNames.HEALPIX_INDEX.value: pix_indices,
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l1c_{head}sensor-pset_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            )
        },
    )

    return pset_product
