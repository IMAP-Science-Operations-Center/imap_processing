"""IMAP Ultra utils for spatial binning and grid creation."""

import typing

import numpy as np
from numpy.typing import NDArray


def build_spatial_bins(
    az_spacing: float = 0.5,
    el_spacing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial bin boundaries for azimuth and elevation.

    Parameters
    ----------
    az_spacing : float, optional
        The azimuth bin spacing in degrees (default is 0.5 degrees).
    el_spacing : float, optional
        The elevation bin spacing in degrees (default is 0.5 degrees).

    Returns
    -------
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values.
    az_bin_midpoints : np.ndarray
        Array of azimuth bin midpoint values.
    el_bin_midpoints : np.ndarray
        Array of elevation bin midpoint values.
    """
    # Azimuth bins from 0 to 360 degrees.
    az_bin_edges = np.arange(0, 360 + az_spacing, az_spacing)
    az_bin_midpoints = az_bin_edges[:-1] + az_spacing / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + el_spacing, el_spacing)
    el_bin_midpoints = el_bin_edges[:-1] + el_spacing / 2  # Midpoints between edges

    return az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints


def build_solid_angle_map(
    spacing: float, input_degrees: bool = True, output_degrees: bool = False
) -> NDArray:
    """
    Build a solid angle map for a given spacing in degrees.

    Parameters
    ----------
    spacing : float
        The bin spacing in the specified units.
    input_degrees : bool, optional
        If True, the input spacing is in degrees
        (default is True for radians).
    output_degrees : bool, optional
        If True, the output solid angle map is in square degrees
        (default is False for steradians).

    Returns
    -------
    solid_angle_grid : np.ndarray
        The solid angle map grid in steradians (default) or square degrees.
        First index is latitude/el, second index is longitude/az.
    """
    if input_degrees:
        spacing = np.deg2rad(spacing)

    if spacing <= 0:
        raise ValueError("Spacing must be positive valued, non-zero.")

    if not np.isclose((np.pi / spacing) % 1, 0):
        raise ValueError("Spacing must divide evenly into pi radians.")

    latitudes = np.arange(-np.pi / 2, np.pi / 2 + spacing, step=spacing)
    sine_latitudes = np.sin(latitudes)
    delta_sine_latitudes = np.diff(sine_latitudes)
    solid_angle_by_latitude = np.abs(spacing * delta_sine_latitudes)

    solid_angle_grid = np.repeat(
        solid_angle_by_latitude[:, np.newaxis], (2 * np.pi) / spacing, axis=1
    )

    if output_degrees:
        solid_angle_grid *= (180 / np.pi) ** 2

    return solid_angle_grid


def build_az_el_grid(
    spacing: float,
    input_degrees: bool = False,
    output_degrees: bool = False,
    centered_azimuth: bool = False,
    centered_elevation: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Build a 2D grid of azimuth and elevation angles.

    Azimuth and Elevation values represent the center of each grid cell,
    so the grid is offset by half the spacing.

    Parameters
    ----------
    spacing : float
        Spacing of the grid in degrees if `input_degrees` is True, else radians.
    input_degrees : bool, optional
        Whether the spacing is specified in degrees and must be converted to radians,
        by default False (indicating radians).
    output_degrees : bool, optional
        Whether the output azimuth and elevation angles should be in degrees,
        by default False (indicating radians).
    centered_azimuth : bool, optional
        Whether the azimuth grid should be centered around 0 degrees/0 radians,
        i.e. from -pi to pi radians, by default False, indicating 0 to 2pi radians.
        If True, the azimuth grid will be from -pi to pi radians.
    centered_elevation : bool, optional
        Whether the elevation grid should be centered around 0 degrees/0 radians,
        i.e. from -pi/2 to pi/2 radians, by default True.
        If False, the elevation grid will be from 0 to pi radians.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        - The evenly spaced, 1D range of azimuth angles
        e.g.(0, 0.5, 1, ..., 359.5) deg.
        - The evenly spaced, 1D range of elevation angles
        e.g.(-90, -89.5, ..., 89.5) deg.
        - The 2D grid of azimuth angles (azimuths for each elevation).
        This grid will be constant along the elevation (0th) axis.
        - The 2D grid of elevation angles (elevations for each azimuth).
        This grid will be constant along the azimuth (1st) axis.

    Raises
    ------
    ValueError
        If the spacing is not positive or does not divide evenly into pi radians.
    """
    if input_degrees:
        spacing = np.deg2rad(spacing)

    if spacing <= 0:
        raise ValueError("Spacing must be positive valued, non-zero.")

    if not np.isclose((np.pi / spacing) % 1, 0):
        raise ValueError("Spacing must divide evenly into pi radians.")

    el_range = np.arange(spacing / 2, np.pi, spacing)
    az_range = np.arange(spacing / 2, 2 * np.pi, spacing)
    if centered_azimuth:
        az_range = az_range - np.pi
    if centered_elevation:
        el_range = el_range - np.pi / 2

    # Reverse the elevation range so that the grid is in the order
    # defined by the Ultra prototype code (`build_dps_grid.m`).
    el_range = el_range[::-1]

    az_grid, el_grid = np.meshgrid(az_range, el_range)

    if output_degrees:
        az_range = np.rad2deg(az_range)
        el_range = np.rad2deg(el_range)
        az_grid = np.rad2deg(az_grid)
        el_grid = np.rad2deg(el_grid)

    return az_range, el_range, az_grid, el_grid


@typing.no_type_check
def rewrap_even_spaced_el_az_grid(
    raveled_values: NDArray,
    shape: typing.Optional[tuple[int]] = None,
    extra_axis: bool = False,
) -> NDArray:
    """
    Take an unwrapped (raveled) 1D array and reshapes it into a 2D el/az grid.

    Assumes the following must be true of the original grid:
    1. Grid was evenly spaced in angular space,
    2. Grid had the same spacing in both azimuth and elevation.
    3. Elevation is the 0th axis (and extends a total of 180 degrees),
    4. Azimuth is the 1st axis (and extends a total of 360 degrees).
    5. The grid was raveled in Fortran (F) order.

    Parameters
    ----------
    raveled_values : NDArray
        1D array of values to be reshaped into a 2D grid.
    shape : tuple[int], optional
        The shape of the original grid, if known, by default None.
        If None, the shape will be inferred from the size of the input array.
    extra_axis : bool, optional
        If True, input is a 2D array with latter axis being 'extra', non-spatial axis.
        This axis (e.g. energy bins) will be preserved in the reshaped grid.

    Returns
    -------
    NDArray
        The reshaped 2D grid of values.

    Raises
    ------
    ValueError
        If the input is not a 1D array or 2D array with an extra axis.
    """
    if raveled_values.ndim not in (1, 2) or (
        raveled_values.ndim == 2 and not extra_axis
    ):
        raise ValueError("Input must be a 1D array or 2D array with extra axis.")

    # We can infer the shape if its evenly spaced and 2D
    if not shape:
        spacing_deg = 1 / np.sqrt(raveled_values.shape[0] / (360 * 180))
        shape = (int(180 // spacing_deg), int(360 // spacing_deg))

    if extra_axis:
        shape = (shape[0], shape[1], raveled_values.shape[1])
    return raveled_values.reshape(shape, order="F")
