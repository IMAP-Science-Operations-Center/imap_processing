"""IMAP utils for spatial binning and az/el grid creation."""

from __future__ import annotations

import typing

import numpy as np
from numpy.typing import NDArray


def build_spatial_bins(
    az_spacing: float = 0.5,
    el_spacing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial bin boundaries for azimuth and elevation (input/output angles in deg).

    Parameters
    ----------
    az_spacing : float, optional
        The azimuth bin spacing in degrees (default is 0.5 degrees).
    el_spacing : float, optional
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
    az_bin_edges = np.arange(0, 360 + az_spacing, az_spacing)
    az_bin_midpoints = az_bin_edges[:-1] + az_spacing / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + el_spacing, el_spacing)
    el_bin_midpoints = el_bin_edges[:-1] + el_spacing / 2  # Midpoints between edges

    return az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints


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


@typing.no_type_check
def rewrap_even_spaced_az_el_grid(
    raveled_values: NDArray,
    shape: tuple[int] | None = None,
    extra_axis: bool = False,
    order: typing.Literal["C"] | typing.Literal["F"] = "C",
) -> NDArray:
    """
    Take an unwrapped (raveled) 1D array and reshapes it into a 2D el/az grid.

    Assumes the following must be true of the original grid:
    1. Grid was evenly spaced in angular space,
    2. Grid had the same spacing in both azimuth and elevation.
    3. Elevation is the 0th axis (and extends a total of 180 degrees),
    4. Azimuth is the 1st axis (and extends a total of 360 degrees).

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
    order : {'C', 'F'}, optional
        The order in which to rewrap the values, by default 'C'.

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
    return raveled_values.reshape(shape, order=order)


class AzElSkyGrid:
    """
    Representation of a 2D grid of azimuth and elevation angles covering the sky.

    All angles are stored internally in radians, but can be accessed in degrees, by
    appending "_degrees" to the attribute name. For example, `spacing` in radians
    can be accessed as `spacing_degrees` in degrees.

    Parameters
    ----------
    spacing_deg : float, optional
        Spacing of the grid in degrees, by default 0.5.
    centered_azimuth : bool, optional
        Whether the azimuth grid should be centered around 0 degrees/0 radians,
        i.e. from -pi to pi radians, by default False.
        If True, the azimuth grid will be from -pi to pi radians.
    centered_elevation : bool, optional
        Whether the elevation grid should be centered around 0 degrees/0 radians,
        i.e. from -pi/2 to pi/2 radians, by default True.
        If False, the elevation grid will be from 0 to pi radians.
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
        centered_azimuth: bool = False,
        centered_elevation: bool = True,
        reversed_elevation: bool = False,
    ) -> None:
        # Store grid properties
        self.centered_azimuth = centered_azimuth
        self.centered_elevation = centered_elevation
        self.reversed_elevation = reversed_elevation

        # Internally, work in radians, regardless of desired output units
        # If angular_units == deg, conversion will be done at the end
        self.spacing = np.deg2rad(spacing_deg)

        # Ensure valid grid spacing (positive, divides evenly into pi radians)
        if self.spacing <= 0:
            raise ValueError("Spacing must be positive valued, non-zero.")

        if not np.isclose((np.pi / self.spacing) % 1, 0):
            raise ValueError("Spacing must divide evenly into pi radians.")

        # build_spacial_bins creates the bin edges and centers for azimuth and elevation
        # E.g. for spacing=1, az_bin_edges = [0, 1, 2, ..., 359, 360] deg.
        (az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints) = (
            build_spatial_bins(az_spacing=spacing_deg, el_spacing=spacing_deg)
        )

        # Store the bin edges and midpoints in radians
        self.az_bin_edges = np.deg2rad(az_bin_edges)
        self.el_bin_edges = np.deg2rad(el_bin_edges)
        self.az_bin_midpoints = np.deg2rad(az_bin_midpoints)
        self.el_bin_midpoints = np.deg2rad(el_bin_midpoints)

        # By default, build_spacial_bins creates bins from az=0->360 and el=-90->90.
        if centered_azimuth:
            self.az_bin_midpoints = self.az_bin_midpoints - np.pi
        if not centered_elevation:
            self.el_bin_midpoints = self.el_bin_midpoints + np.pi / 2

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

        # Create degree property attributes for all radian attributes
        self.__init_properties__()

    def __init_properties__(self) -> None:
        """Automatically generate degree properties for all radian attributes."""
        for name in [
            "spacing",
            "az_bin_midpoints",
            "el_bin_midpoints",
            "az_bin_edges",
            "el_bin_edges",
            "az_grid",
            "el_grid",
        ]:
            setattr(
                self.__class__,
                f"{name}_degrees",
                property(lambda self, n=name: np.rad2deg(getattr(self, n))),  # type: ignore[misc]
            )

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
