"""IMAP utils for spatial binning and az/el grid creation."""

from __future__ import annotations

import typing

import numpy as np
from numpy.typing import NDArray


def build_spatial_bins(
    az_spacing_deg: float = 0.5,
    el_spacing_deg: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial bin boundaries for azimuth and elevation.

    Input angles in degrees for consistency with map inputs,
    output angles in radians for internal use.

    Parameters
    ----------
    az_spacing_deg : float, optional
        The azimuth bin spacing in degrees (default is 0.5 degrees).
    el_spacing_deg : float, optional
        The elevation bin spacing in degrees (default is 0.5 degrees).

    Returns
    -------
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values in radians.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values in radians.
    az_bin_midpoints : np.ndarray
        Array of azimuth bin midpoint values in radians.
    el_bin_midpoints : np.ndarray
        Array of elevation bin midpoint values in radians.
    """
    # Azimuth bins from 0 to 360 degrees.
    az_bin_edges = np.arange(0, 360 + az_spacing_deg, az_spacing_deg)
    az_bin_midpoints = az_bin_edges[:-1] + az_spacing_deg / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + el_spacing_deg, el_spacing_deg)
    el_bin_midpoints = el_bin_edges[:-1] + el_spacing_deg / 2  # Midpoints between edges

    # Convert all angles to radians and return them
    return (
        np.deg2rad(az_bin_edges),
        np.deg2rad(el_bin_edges),
        np.deg2rad(az_bin_midpoints),
        np.deg2rad(el_bin_midpoints),
    )


def build_solid_angle_map(
    spacing_deg: float,
) -> NDArray:
    """
    Build a solid angle map in steradians for a given spacing in degrees.

    Parameters
    ----------
    spacing_deg : float
        The bin spacing in degrees.

    Returns
    -------
    solid_angle_grid : np.ndarray
        The solid angle map grid in steradians.
        First index is latitude/el, second index is longitude/az.
    """
    # Degrees are the preferred input units of angle, given map definitions,
    # but we'll convert to radians for internal calculations and output steradians.
    spacing = np.deg2rad(spacing_deg)

    if spacing <= 0:
        raise ValueError("Spacing must be positive valued, non-zero.")

    if not np.isclose((np.pi / spacing) % 1, 0):
        raise ValueError("Spacing must divide evenly into pi radians.")

    latitudes = np.arange(-np.pi / 2, np.pi / 2 + spacing, step=spacing)
    sine_latitudes = np.sin(latitudes)
    delta_sine_latitudes = np.diff(sine_latitudes)
    solid_angle_by_latitude = np.abs(spacing * delta_sine_latitudes)

    # Order ensures agreement with build_az_el_grid's order of tiling az/el grid.
    solid_angle_grid = np.repeat(
        solid_angle_by_latitude[np.newaxis, :], (2 * np.pi) / spacing, axis=0
    )

    return solid_angle_grid


# Ignore linting rule to allow for 6 unrelated args
def build_az_el_grid(  # noqa: PLR0913
    spacing: float,
    input_degrees: bool = False,
    output_degrees: bool = False,
    centered_azimuth: bool = False,
    centered_elevation: bool = True,
    reversed_elevation: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Build a 2D grid of azimuth and elevation angles, and their 1D bin edges.

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
    reversed_elevation : bool, optional
        Whether the elevation grid should be reversed, by default False.
        If False, the elevation grid will be from -pi/2 to pi/2 radians (-90 to 90 deg).
        If True, the elevation grid will be from pi/2 to -pi/2 radians (90 to -90 deg).

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
        - The evenly spaced, 1D range of azimuth angles
        e.g.(0.5, 1, ..., 359.5) deg.
        - The evenly spaced, 1D range of elevation angles
        e.g.(-89.5, ..., 89.5) deg.
        - The 2D grid of azimuth angles (azimuths for each elevation).
        This grid will be constant along the elevation (0th) axis.
        - The 2D grid of elevation angles (elevations for each azimuth).
        This grid will be constant along the azimuth (1st) axis.
        - The 1D bin edges for azimuth angles.
        e.g. if spacing=1 deg:
        az_bin_edges = [0, 1, 2, ..., 359, 360] deg.
        - The 1D bin edges for elevation angles.
        e.g. if spacing=1 deg:
        el_bin_edges = [-90, -89, -88, ..., 89, 90] deg.

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

    # Create the bin edges for azimuth and elevation.
    # E.g. for spacing=1, az_bin_edges = [0, 1, 2, ..., 359, 360] deg.
    el_bin_edges = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / spacing) + 1)
    az_bin_edges = np.linspace(0, 2 * np.pi, int(2 * np.pi / spacing) + 1)

    # Create the 2D grid of azimuth and elevation angles at center of each bin.
    # These ranges are offset by half the spacing and are
    # one element shorter than the bin edges.
    el_range = np.arange(spacing / 2, np.pi, spacing)
    az_range = np.arange(spacing / 2, 2 * np.pi, spacing)
    if centered_azimuth:
        az_range = az_range - np.pi
    if centered_elevation:
        el_range = el_range - np.pi / 2

    # Reverse the elevation range so that the grid is in the order
    # defined by the Ultra prototype code (`build_dps_grid.m`).
    if reversed_elevation:
        el_range = el_range[::-1]
        el_bin_edges = el_bin_edges[::-1]

    # Deriving our az/el grids with indexing "ij" allows for
    # ravel_multi_index to work correctly with 1D digitized indices in each az and el,
    # using the same ravel order ('C' or 'F') as the grid points were unwrapped.
    az_grid, el_grid = np.meshgrid(az_range, el_range, indexing="ij")

    if output_degrees:
        az_range = np.rad2deg(az_range)
        el_range = np.rad2deg(el_range)
        az_grid = np.rad2deg(az_grid)
        el_grid = np.rad2deg(el_grid)
        az_bin_edges = np.rad2deg(az_bin_edges)
        el_bin_edges = np.rad2deg(el_bin_edges)

    return az_range, el_range, az_grid, el_grid, az_bin_edges, el_bin_edges


@typing.no_type_check
def rewrap_even_spaced_az_el_grid(
    raveled_values: NDArray,
    shape: tuple[int] | None = None,
    order: typing.Literal["C"] | typing.Literal["F"] = "C",
) -> NDArray:
    """
    Take an unwrapped (raveled) 1D array and reshapes it into a 2D az/el grid.

    Assumes the following must be true of the original grid:
    1. Grid was evenly spaced in angular space,
    2. Grid had the same spacing in both azimuth and elevation.
    3. Azimuth is axis 0 (and extends a total of 360 degrees).
    4. Elevation is axis 1 (and extends a total of 180 degrees),

    Parameters
    ----------
    raveled_values : NDArray
        1D array of values to be reshaped into a 2D grid.
    shape : tuple[int], optional
        The shape of the original grid, if known, by default None.
        If None, the shape will be inferred from the size of the input array.
    order : {'C', 'F'}, optional
        The order in which to rewrap the values, by default 'C'.

    Returns
    -------
    NDArray
        The reshaped 2D grid of values.

    Raises
    ------
    ValueError
        If the input is not a 1D array or 2D array with an 'extra' non-spatial axis.
    """
    if raveled_values.ndim > 2:
        raise ValueError(
            "Input must be a 1D array or 2D array with only one spatial axis as axis 0."
        )

    # We can infer the shape if its evenly spaced and 2D
    if not shape:
        spacing_deg = 1 / np.sqrt(raveled_values.shape[0] / (360 * 180))
        shape = (int(360 // spacing_deg), int(180 // spacing_deg))

    if raveled_values.ndim == 2:
        shape = (shape[0], shape[1], raveled_values.shape[1])
    return raveled_values.reshape(shape, order=order)


class AzElSkyGrid:
    """
    Representation of a 2D grid of azimuth and elevation angles covering the sky.

    All angles are stored internally in radians.
    Azimuth is within the range [0, 2*pi) radians,
    elevation is within the range [-pi/2, pi/2) radians.

    Parameters
    ----------
    spacing_deg : float, optional
        Spacing of the grid in degrees, by default 0.5.
    reversed_elevation : bool, optional
        Whether the elevation grid should be reversed, by default False.
        If False, the elevation grid will be from -pi/2 to pi/2 radians (-90 to 90 deg).
        If True, the elevation grid will be from pi/2 to -pi/2 radians (90 to -90 deg).

    Raises
    ------
    ValueError
        If the spacing is not positive or does not divide evenly into pi radians.
    """

    def __init__(
        self,
        spacing_deg: float = 0.5,
        reversed_elevation: bool = False,
    ) -> None:
        # Store grid properties
        self.reversed_elevation = reversed_elevation

        # Internally, work in radians, regardless of desired output units
        self.spacing = np.deg2rad(spacing_deg)

        # Ensure valid grid spacing (positive, divides evenly into pi radians)
        if self.spacing <= 0:
            raise ValueError("Spacing must be positive valued, non-zero.")

        if not np.isclose((np.pi / self.spacing) % 1, 0):
            raise ValueError("Spacing must divide evenly into pi radians.")

        # build_spacial_bins creates the bin edges and centers for azimuth and elevation
        # E.g. for spacing=1, az_bin_edges = [0, 1, 2, ..., 359, 360] deg.
        # However returned values are in radians.
        (
            self.az_bin_edges,
            self.el_bin_edges,
            self.az_bin_midpoints,
            self.el_bin_midpoints,
        ) = build_spatial_bins(az_spacing_deg=spacing_deg, el_spacing_deg=spacing_deg)

        # If desired, reverse the elevation range so that the grid is in the order
        # defined by the Ultra prototype code (`build_dps_grid.m`).
        if self.reversed_elevation:
            self.el_bin_midpoints = self.el_bin_midpoints[::-1]
            self.el_bin_edges = self.el_bin_edges[::-1]

        # Deriving our az/el grids with indexing "ij" allows for ravel_multi_index
        # to work correctly with 1D digitized indices in each az and el,
        # using the same ravel order ('C' or 'F') as the grid points were unwrapped.
        self.az_grid, self.el_grid = np.meshgrid(
            self.az_bin_midpoints, self.el_bin_midpoints, indexing="ij"
        )

        # Keep track of number of points on the grid
        self.grid_shape = self.az_grid.shape
        self.grid_size = self.az_grid.size

    def __repr__(self) -> str:
        """
        Return a string representation of the AzElSkyGrid.

        Returns
        -------
        str
            A string representation of the AzElSkyGrid.
        """
        return (
            f"AzElSkyGrid with a spacing of {self.spacing:.4e} radians. "
            f"{self.grid_shape} Grid."
        )
