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

    Input/output angles in degrees.

    Parameters
    ----------
    az_spacing_deg : float, optional
        The azimuth bin spacing in degrees (default is 0.5 degrees).
    el_spacing_deg : float, optional
        The elevation bin spacing in degrees (default is 0.5 degrees).

    Returns
    -------
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values in degrees.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values in degrees.
    az_bin_midpoints : np.ndarray
        Array of azimuth bin midpoint values in degrees.
    el_bin_midpoints : np.ndarray
        Array of elevation bin midpoint values in degrees.
    """
    # Azimuth bins from 0 to 360 degrees.
    az_bin_edges = np.arange(0, 360 + az_spacing_deg, az_spacing_deg)
    az_bin_midpoints = az_bin_edges[:-1] + az_spacing_deg / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + el_spacing_deg, el_spacing_deg)
    el_bin_midpoints = el_bin_edges[:-1] + el_spacing_deg / 2  # Midpoints between edges

    return (
        az_bin_edges,
        el_bin_edges,
        az_bin_midpoints,
        el_bin_midpoints,
    )


def build_solid_angle_map(
    spacing_deg: float,
) -> NDArray:
    """
    Build a solid angle map in steradians for a given spacing in degrees.

    NOTE: This function works in radians internally and returns steradians, while other
    functions in this module work in degrees. Expressing solid angles in steradians
    is the preferred unit for ENA Maps.

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

    proposed_number_of_lat_bins = 180 / spacing_deg
    number_of_lat_bins = round(180 / spacing_deg)
    number_of_lon_bins = 2 * number_of_lat_bins
    if not np.isclose(proposed_number_of_lat_bins, number_of_lat_bins):
        raise ValueError("Spacing must divide evenly into pi radians.")

    latitude_edges = np.linspace(
        -np.pi / 2, np.pi / 2, num=number_of_lat_bins + 1, endpoint=True
    )
    sine_latitude_edges = np.sin(latitude_edges)
    delta_sine_latitudes = np.diff(sine_latitude_edges)
    solid_angle_by_latitude = np.abs(spacing * delta_sine_latitudes)

    # Order ensures agreement with build_az_el_grid's order of tiling az/el grid.
    solid_angle_grid = np.repeat(
        solid_angle_by_latitude[np.newaxis, :], number_of_lon_bins, axis=0
    )

    return solid_angle_grid


@typing.no_type_check
def rewrap_even_spaced_az_el_grid(
    raveled_values: NDArray,
    grid_shape: tuple[int] | None = None,
    order: typing.Literal["C"] | typing.Literal["F"] = "C",
) -> NDArray:
    """
    Take an unwrapped (raveled) 1D array and reshapes it into a 2D az/el grid.

    In the input, unwrapped grid, the spatial axis is the final (-1) axis.
    In the output, the spatial axes are the -2 (azimuth) and -1 (elevation) axes.

    Assumes the following must be true of the original grid:
    1. Grid was evenly spaced in angular space,
    2. Grid had the same spacing in both azimuth and elevation.
    3. Azimuth is the first spatial axis (and extends a total of 360 degrees).
    4. Elevation is the second spatial axis (and extends a total of 180 degrees).

    Parameters
    ----------
    raveled_values : NDArray
        1D array of values to be reshaped into a 2D grid.
    grid_shape : tuple[int], optional
        The shape of the original grid, if known, by default None.
        If None, the shape will be inferred from the size of the input array.
    order : {'C', 'F'}, optional
        The order in which to rewrap the values, by default 'C'.

    Returns
    -------
    NDArray
        The reshaped 2D grid of values with (azimuth, elevation) as the final 2 axes.
    """
    # We can infer the shape if its evenly spaced and 2D
    if not grid_shape:
        spacing_deg = 1 / np.sqrt(raveled_values.shape[-1] / (360 * 180))
        grid_shape = (int(360 // spacing_deg), int(180 // spacing_deg))

    if raveled_values.ndim == 1:
        array_shape = grid_shape
    else:
        array_shape = (*raveled_values.shape[:-1], *grid_shape)
    return raveled_values.reshape(array_shape, order=order)


class AzElSkyGrid:
    """
    Representation of a 2D grid of azimuth and elevation angles covering the sky.

    All angles are stored internally in degrees.
    Azimuth is within the range [0, 360) degrees,
    elevation is within the range [-90, 90) degrees.

    Parameters
    ----------
    spacing_deg : float, optional
        Spacing of the grid in degrees, by default 0.5.
    reversed_elevation : bool, optional
        Whether the elevation grid should be reversed, by default False.
        If False, the elevation grid will be from -90 to 90 deg.
        If True, the elevation grid will be from 90 to -90 deg.

    Raises
    ------
    ValueError
        If the spacing is not positive or does not divide evenly into 180 degrees.
    """

    def __init__(
        self,
        spacing_deg: float = 0.5,
        reversed_elevation: bool = False,
    ) -> None:
        # Store grid properties
        self.reversed_elevation = reversed_elevation

        # Internally, work in degrees
        self.spacing_deg = spacing_deg

        # Ensure valid grid spacing (positive, divides evenly into 180 degrees)
        if self.spacing_deg <= 0:
            raise ValueError("Spacing must be positive valued, non-zero.")

        if not np.isclose((180 / self.spacing_deg) % 1, 0):
            raise ValueError("Spacing must divide evenly into 180 degrees.")

        # build_spacial_bins creates the bin edges and centers for azimuth and elevation
        # E.g. for spacing=1, az_bin_edges = [0, 1, 2, ..., 359, 360] deg.
        (
            self.az_bin_edges,
            self.el_bin_edges,
            self.az_bin_midpoints,
            self.el_bin_midpoints,
        ) = build_spatial_bins(
            az_spacing_deg=self.spacing_deg, el_spacing_deg=self.spacing_deg
        )

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
            f"AzElSkyGrid with a spacing of {self.spacing_deg:.4e} degrees. "
            f"{self.grid_shape} Grid."
        )
