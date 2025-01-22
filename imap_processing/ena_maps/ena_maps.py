"""Define classes for handling pointing sets and maps for ENA data."""

import logging
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ena_maps import map_utils
from imap_processing.spice import geometry
from imap_processing.ultra.utils import spatial_utils

logger = logging.getLogger(__name__)


class TilingType(Enum):
    """Enumeration of the types of tiling used in the ENA maps."""

    RECTANGULAR = "Rectangular"
    HEALPIX = "Healpix"
    ABSTRACT = "Abstract"


# Define the pointing set classes
class PointingSet(ABC):
    """
    Abstract class to contain pointing set (PSET) data in the context of ENA maps.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the pointing set data.
    reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set.
    """

    @abstractmethod
    def __init__(self, dataset: xr.Dataset, reference_frame: geometry.SpiceFrame):
        self.reference_frame = reference_frame
        self.num_points = 0
        self.az_el_points = np.zeros((self.num_points, 2))
        self.data = xr.Dataset()

    @abstractmethod
    def project_to_frame(
        self, out_frame: geometry.SpiceFrame, event_time: float | None = None
    ) -> None:
        """
        Project the pointing set to a new frame (placeholder).

        Parameters
        ----------
        out_frame : geometry.SpiceFrame
            The frame to which the pointing set should be projected.
        event_time : float
            The event time at which to project the pointing set into the new frame.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the pointing set.

        Returns
        -------
        str
            String representation of the pointing set.
        """
        return f"{self.__class__} PointingSet(reference_frame={self.reference_frame})"


class UltraPointingSet(PointingSet):
    """
    PSET object specifically for ULTRA data, nominally at Level 1C.

    Parameters
    ----------
    l1c_dataset : xr.Dataset | pathlib.Path | str
        L1c xarray dataset containing the pointing set data or the path to the dataset.
        Currently, the dataset is expected to be in a rectangular grid,
        with data_vars indexed along the coordinates:
            - 'epoch' : time value (1 value per PSET)
            - 'azimuth_bin_center' : azimuth bin center values
            - 'elevation_bin_center' : elevation bin center values
        Some data_vars may additionally be indexed by energy bin;
        however, only the spatial axes are used in this class.
    reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set. Default is IMAP_DPS.
    order : {'C', 'F'}, optional
        The order of the grid to be used in any raveling processes.
        Default is 'F' (Fortran-style ordering).

    Raises
    ------
    ValueError
        If the azimuth or elevation bin centers do not match the constructed grid.
        Or if the azimuth or elevation bin spacing is not uniform.
    ValueError
        If multiple epochs are found in the dataset.
    """

    def __init__(
        self,
        l1c_dataset: xr.Dataset | pathlib.Path | str,
        reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
        order: typing.Literal["C"] | typing.Literal["F"] = "F",
    ):
        # History of reference frames to which the pointing set has been projected
        # Current frame is the last element in the list, accessed as @property
        self.reference_frame_history = [
            reference_frame,
        ]

        # Read in the data and store the xarray dataset as data attr
        if isinstance(l1c_dataset, (str, pathlib.Path)):
            self.path: str | None = str(l1c_dataset)
            self.data = xr.open_dataset(l1c_dataset)
        elif isinstance(l1c_dataset, xr.Dataset):
            self.path = None
            self.data = l1c_dataset

        # A PSET must have a single epoch
        self.epoch = self.data["epoch"].values
        if len(np.unique(self.epoch)) > 1:
            raise ValueError("Multiple epochs found in the dataset.")

        # The rest of the constructor handles the rectangular grid
        # aspects of the Ultra PSET.
        # NOTE: This may be changed to Healpix tessellation in the future

        self.tiling_type = TilingType.RECTANGULAR

        # Ensure 1D axes grids are uniformly spaced,
        # then set spacing based on data's azimuth bin spacing.
        az_bin_delta = np.diff(self.data["azimuth_bin_center"])
        el_bin_delta = np.diff(self.data["elevation_bin_center"])
        if not np.allclose(az_bin_delta, az_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Azimuth bin spacing is not uniform.")
        if not np.allclose(el_bin_delta, el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Elevation bin spacing is not uniform.")
        if not np.isclose(az_bin_delta[0], el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError(
                "Azimuth and elevation bin spacing do not match: "
                f"az {az_bin_delta[0]} != el {el_bin_delta[0]}."
            )
        self.spacing_deg = az_bin_delta[0]

        # Build the azimuth and elevation grids and check that
        # the 1D axes match the dataset.
        (
            az_axis_bin_centers_input,
            el_axis_bin_centers_input,
            az_grid_input,
            el_grid_input,
            az_bin_edges_input,
            el_bin_edges_input,
        ) = spatial_utils.build_az_el_grid(
            spacing=self.spacing_deg,
            input_degrees=True,
            output_degrees=False,
            centered_azimuth=False,
            centered_elevation=True,
        )

        for dim, constructed_bins in zip(
            ["azimuth", "elevation"],
            [az_axis_bin_centers_input, el_axis_bin_centers_input],
        ):
            if not np.allclose(
                # TODO: Consider removing the sort and the inversion in spatial_utils
                # The constructed el bins may be inverted (+90deg -> -90deg),
                # so compare a sorted version of the bins
                sorted(np.rad2deg(constructed_bins)),
                self.data[f"{dim}_bin_center"],
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    f"{dim} bin centers do not match."
                    f"Constructed: {np.rad2deg(constructed_bins)}"
                    f"Dataset: {self.data[f'{dim}_bin_center']}"
                )

        # Unwrap the az, el grids to series of points tiling the sky and combine them
        # into shape (number of points in tiling of the sky, 2) where
        # column 0 (az_el_points[:, 0]) is the azimuth of that point and
        # column 1 (az_el_points[:, 1]) is the elevation of that point.
        self.az_el_points = np.column_stack(
            (az_grid_input.ravel(order=order), el_grid_input.ravel(order=order))
        )
        self.num_points = self.az_el_points.shape[0]

        # Must also store the bin edges for the pointing set to allow for "pull" method
        # of index matching
        self.az_bin_edges = az_bin_edges_input
        self.el_bin_edges = el_bin_edges_input

    @property
    def reference_frame(self) -> geometry.SpiceFrame:
        """
        Return the current reference frame of the pointing set.

        Returns
        -------
        geometry.SpiceFrame
            The current reference frame of the pointing set: the frame in which its
            azimuth and elevation points are defined.
        """
        return self.reference_frame_history[-1]

    def project_to_frame(
        self, out_frame: geometry.SpiceFrame, event_time: float | None = None
    ) -> None:
        """
        Project the pointing set to a new frame at a given event time.

        Parameters
        ----------
        out_frame : geometry.SpiceFrame
            The frame to which the pointing set should be projected.
        event_time : float, optional
            The event time at which to project the pointing set.
            Default is None, in which case the epoch of the pointing set is used.

        Notes
        -----
        This method modifies the pointing set in place, updating the
        reference frame and the azimuth and elevation points.
        It also appends the new reference frame to the reference_frame_history.
        """
        if event_time is None:
            event_time = self.epoch

        # Check if the frame is already in the desired frame
        if self.reference_frame == out_frame:
            logger.info(f"Pointing set is already in frame {out_frame}.")
            return

        logger.info(
            f"Projecting pointing set from reference frame {self.reference_frame}"
            f"to frame {out_frame} at event time {event_time}."
        )

        # Project the pixel centers to the new frame
        self.az_el_points = geometry.frame_transform_az_el(
            et=event_time,
            az_el=self.az_el_points,
            from_frame=self.reference_frame,
            to_frame=out_frame,
            degrees=False,
        )

        self.reference_frame_history.append(out_frame)

    def __repr__(self) -> str:
        """
        Return a string representation of the UltraPointingSet.

        Returns
        -------
        str
            String representation of the UltraPointingSet.
        """
        return (
            f"UltraPointingSet\n\t(reference_frame="
            f"{self.reference_frame}, epoch={self.epoch}, "
            f"num_points={self.num_points})"
        )


# Define the Map classes
class AbstractMap(ABC):
    """Abstract base class to contain map data in the context of ENA maps."""

    @abstractmethod
    def __init__(self) -> None:
        self.tiling_type = TilingType.ABSTRACT

    @abstractmethod
    def match_pset_coords_to_indices(self, pointing_set: PointingSet) -> None:
        """
        Match pointing set coordinates to map indices (placeholder).

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set to match to the map grid.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the map.

        Returns
        -------
        str
            String representation of the map.
        """
        return f"{self.__class__} Map(tiling_type={self.tiling_type}.)"


class RectangularMap(AbstractMap):
    """
    Map which tiles the sky with a 2D rectangular grid of azimuth/elevation pixels.

    NOTE: Internally, the map is stored as a 1D array of pixels.

    Parameters
    ----------
    spacing_deg : float
        The spacing of the rectangular grid in degrees.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.
    order : {'C', 'F'}, optional
        The order of the grid to be used in any raveling processes.
        Default is 'F' (Fortran-style ordering).
    """

    def __init__(
        self,
        spacing_deg: float,
        spice_frame: geometry.SpiceFrame,
        order: typing.Literal["C"] | typing.Literal["F"] = "F",
    ):
        # Define the core properties of the map
        self.tiling_type = TilingType.RECTANGULAR
        self.spacing_deg = spacing_deg
        self.reference_frame = spice_frame
        self.order = order

        # Build the azimuth and elevation grids and the solid angle grid
        (
            self.az_axis_bin_centers,
            self.el_axis_bin_centers,
            self.az_grid,
            self.el_grid,
            self.az_bin_edges,
            self.el_bin_edges,
        ) = spatial_utils.build_az_el_grid(
            spacing=self.spacing_deg,
            input_degrees=True,
            output_degrees=False,
            centered_azimuth=False,
            centered_elevation=True,
        )
        self.solid_angle_grid = spatial_utils.build_solid_angle_map(
            spacing=self.spacing_deg, input_degrees=True, output_degrees=False
        )

        # Unwrap the az, el, solid angle grids to series of points tiling the sky
        az_points = self.az_grid.ravel(order=self.order)
        el_points = self.el_grid.ravel(order=self.order)
        self.az_el_points = np.column_stack((az_points, el_points))
        self.num_points = self.az_el_points.shape[0]
        self.solid_angle_points = self.solid_angle_grid.ravel(order=self.order)

        # Initialize empty data dictionary to store map data
        self.data_dict: dict[str, NDArray] = {}

    def match_pset_coords_to_indices(self, pointing_set: PointingSet) -> NDArray:
        """
        Get indices of map grid corresponding to az/el points in the pointing set.

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set to match to the map grid.
            Must contain the azimuth and elevation of all points in the attribute
            az_el_points, which is a 2D array of shape (num_points, 2).

        Returns
        -------
        NDArray
            Indices of the map grid corresponding to each points in the pointing set.
            1D array of shape (num_points,) where each element is the index of the
            raveled map grid.

        Notes
        -----
        Currently only supports the "push" method of index matching,
        where the pointing set is projected to the map frame and binned to the edges
        of the map grid. This method ensures all counts are captured in the map, but
        does not ensure that all pixels in the map receive data.
        The Ultra instrument team has determined that all pixels in the map must
        receive data, at least for calculating flux. So we will need to use the "pull"
        method of index matching, where the map grid is projected to the pointing set
        frame and binned to the edges of the pointing set grid.
        # TODO: Implement the "pull" method of index matching.
        """
        if pointing_set.reference_frame != self.reference_frame:
            pointing_set.project_to_frame(self.reference_frame)

        az_indices = np.digitize(pointing_set.az_el_points[:, 0], self.az_bin_edges) - 1
        el_indices = np.digitize(pointing_set.az_el_points[:, 1], self.el_bin_edges) - 1
        flat_indices = np.ravel_multi_index(
            multi_index=(az_indices, el_indices),
            dims=(len(self.az_axis_bin_centers), len(self.el_axis_bin_centers)),
            order=self.order,
        )
        return flat_indices

    def project_pset_values_to_map(
        self, pointing_set: PointingSet, value_keys: list[str] | None = None
    ) -> None:
        """
        Project a pointing set's values to the map grid.

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set containing the values to project to the map.
        value_keys : list[str], optional
            The keys of the values to project to the map.
            data_vars named each key must be present, and of the same dimensionality in
            each pointing set which is to be projected to the map.
            Default is None, in which case all data_vars in the pointing set are used.

        Raises
        ------
        ValueError
            If a value key is not found in the pointing set.
        """
        if value_keys is None:
            value_keys = pointing_set.data.data_vars.keys()

        for value_key in value_keys:
            if value_key not in pointing_set.data.data_vars:
                raise ValueError(f"Value key {value_key} not found in pointing set.")

        # Determine the indices of the map grid that correspond to the pointing set
        matched_indices = self.match_pset_coords_to_indices(pointing_set)
        for value_key in value_keys:
            # If multiple spatial axes present
            # (i.e (az, el) for rectangular coordinate PSET),
            # flatten them in the values array to match the raveled indices
            raveled_pset_data = np.reshape(
                np.array(pointing_set.data[value_key]),
                (pointing_set.num_points, -1),
                order=self.order,
            )

            if value_key not in self.data_dict:
                # Initialize the map data array if it doesn't exist (values start at 0)
                output_shape = (self.num_points, *raveled_pset_data.shape[1:])
                self.data_dict[value_key] = np.zeros(output_shape)

            pointing_projected_values = map_utils.bin_single_array_at_indices(
                value_array=raveled_pset_data,
                projection_grid_shape=(
                    len(self.az_axis_bin_centers),
                    len(self.el_axis_bin_centers),
                ),
                projection_indices=matched_indices,
            )
            self.data_dict[value_key] += pointing_projected_values

    def __repr__(self) -> str:
        """
        Return a string representation of the RectangularMap.

        Returns
        -------
        str
            String representation of the RectangularMap.
        """
        return (
            "RectangularMap\n\t(reference_frame="
            f"{self.reference_frame.name} ({self.reference_frame.value}), "
            f"spacing_deg={self.spacing_deg}, num_points={self.num_points})"
        )


# TODO:
# Add pulling index matching in match_pset_coords_to_indices

# TODO:
# Add ability to push some, pull other indices in project_pset_values_to_map
# dict{"value_key": (array, "push"| "pull")}

# TODO:
# Check units of time which will be read in. Do we need to add j2000ns_to_j2000s?
