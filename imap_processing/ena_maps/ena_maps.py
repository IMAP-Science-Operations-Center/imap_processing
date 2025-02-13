"""Define classes for handling pointing sets and maps for ENA data."""

from __future__ import annotations

import logging
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ena_maps.utils import map_utils, spatial_utils
from imap_processing.spice import geometry

logger = logging.getLogger(__name__)


# Define ravel order for unravelling, multi-indexing of rectangular grids.
# Must be done as explicit typing Literal for mypy to accept it.
RAVEL_ORDER = typing.cast(typing.Literal["C", "F"], "C")


class SkyTilingType(Enum):
    """Enumeration of the types of tiling used in the ENA maps."""

    RECTANGULAR = "Rectangular"
    HEALPIX = "Healpix"


class IndexMatchMethod(Enum):
    """
    Enumeration of the types of index matching methods used in the ENA sky maps.

    Notes
    -----
    Index matching is the process of determining which pixels in a map grid correspond
    to which pixels in a pointing set grid. The Ultra instrument team has determined
    that they must support two methods of index matching for rectangular grid maps:

    **Push Method**

    The "push" method takes each pixel in a pointing set and transforms its coordinates
    to the frame of the map, then determines into which pixel in the map grid the
    projected pointing set pixel falls.
    This method ensures that all pointing set pixels (and thus all counts) are
    captured in the map, but does not ensure that all pixels in the map receive data.

    **Pull Method**

    The "pull" method takes each pixel in the map grid and transforms its coordinates
    to the frame of the pointing set, then determines into which pixel in the
    pointing set grid the projected map pixel falls.
    This method ensures that all pixels in the map receive data, but can result in
    some pointing set pixels not being captured in the map, and others being captured
    multiple times.
    """

    PUSH = "Push"
    PULL = "Pull"


def match_coords_to_indices(
    spatial_object_input_frame: PointingSet | AbstractSkyMap,
    spatial_object_output_frame: PointingSet | AbstractSkyMap,
    event_time: float | None = None,
) -> NDArray:
    """
    Find the output indices corresponding to each input coord between 2 spatial objects.

    First, the pixel center coordinates of the input spatial object are
    transformed from the Spice coordinate frame of the input object to their
    corresponding coordinates in the Spice frame of the output object.
    Then, the transformed pixel centers are matched to the 1D indices of the spatial
    pixels in the output frame, either in an unwrapped rectangular grid or a Healpix
    tessellation of the sky.

    This function always "pushes" the pixels of the input object to corresponding pixels
    in the output object's unwrapped rectangular grid or healpix tessellation;
    however, by swapping the input and output objects, one can apply the "pull" method
    of index  matching.

    Parameters
    ----------
    spatial_object_input_frame : PointingSet | AbstractSkyMap
        An object containing 1D spatial pixel centers in azimuth and elevation,
        which will be matched to 1D indices of spatial pixels in the output frame.
        Must contain the Spice frame in which the pixel centers are defined.
    spatial_object_output_frame : PointingSet | AbstractSkyMap
        The object containing a grid or tessellation of spatial pixels
        into which the input spatial pixel centers will 'land', and be matched to
        corresponding pixel 1D indices in the output frame.
    event_time : float, optional
        The event time at which to project the input spatial object to the output frame.
        This can be manually specified, e.g., for converting between Maps which do not
        contain an epoch value.
        The default value is None, in which case the event time of the PointingSet
        object is used.

    Returns
    -------
    flat_indices_input_grid_output_frame : NDArray
        1D array of pixel indices of the output object corresponding to each pixel in
        the input object. The length of the array is equal to the number of pixels in
        the input object, and may contain 0 or >1 occurrences of the same output index.

    Raises
    ------
    ValueError
        If both input and output objects are PointingSet objects.
    ValueError
        If the event time is not specified and both objects are SkyMaps.
    NotImplementedError
        If the output tiling type is HEALPIX. Will be implemented in the future.
    ValueError
        If the tiling type of the output frame is not RECTANGULAR or HEALPIX.
    """
    if isinstance(spatial_object_input_frame, PointingSet) and isinstance(
        spatial_object_output_frame, PointingSet
    ):
        raise ValueError("Cannot match indices between two PointingSet objects.")

    # If event_time is not specified, use event_time of the PointingSet, if present.
    if event_time is None:
        if isinstance(spatial_object_input_frame, PointingSet):
            event_time = spatial_object_input_frame.data["epoch"].values
        elif isinstance(spatial_object_output_frame, PointingSet):
            event_time = spatial_object_output_frame.data["epoch"].values
        else:
            raise ValueError(
                "Event time must be specified if both objects are SkyMaps."
            )

    obj1_az_el_points_frame1 = spatial_object_input_frame.az_el_points

    # If the two objects are not already in the same frame,
    # project the input pixel centers to the output frame.
    if (
        spatial_object_input_frame.spice_reference_frame
        is not spatial_object_output_frame.spice_reference_frame
    ):
        obj1_az_el_points_frame2 = geometry.frame_transform_az_el(
            et=event_time,
            az_el=obj1_az_el_points_frame1,
            from_frame=spatial_object_input_frame.spice_reference_frame,
            to_frame=spatial_object_output_frame.spice_reference_frame,
            degrees=False,
        )
    else:
        obj1_az_el_points_frame2 = obj1_az_el_points_frame1

    obj1_az_el_points_frame2 = geometry.frame_transform_az_el(
        et=event_time,
        az_el=obj1_az_el_points_frame1,
        from_frame=spatial_object_input_frame.spice_reference_frame,
        to_frame=spatial_object_output_frame.spice_reference_frame,
        degrees=False,
    )

    # The way indices are matched depends on the tiling type of the 2nd object
    if spatial_object_output_frame.tiling_type is SkyTilingType.RECTANGULAR:
        # To match to a rectangular grid, we need to digitize the transformed az, el
        # pixel centers onto the bin edges of the output frame's grid, then
        # use ravel_multi_index to get the 1D indices of the pixels in the output frame.
        az_indices = (
            np.digitize(
                obj1_az_el_points_frame2[:, 0],
                spatial_object_output_frame.sky_grid.az_bin_edges,
            )
            - 1
        )
        el_indices = (
            np.digitize(
                obj1_az_el_points_frame2[:, 1],
                spatial_object_output_frame.sky_grid.el_bin_edges,
            )
            - 1
        )
        flat_indices_input_grid_output_frame = np.ravel_multi_index(
            multi_index=(az_indices, el_indices),
            dims=(
                len(spatial_object_output_frame.sky_grid.az_bin_midpoints),
                len(spatial_object_output_frame.sky_grid.el_bin_midpoints),
            ),
            order=RAVEL_ORDER,
        )

    elif spatial_object_output_frame.tiling_type is SkyTilingType.HEALPIX:
        # To match to a Healpix tessellation, we need to use the healpy function ang2pix
        # which directly returns the index on the output frame's Healpix tessellation.
        """
        Leaving this as a placeholder for now, so we don't yet
        need to add a healpy dependency. It will look something like the
        following code, much simpler than the rectangular case:

        ```python
        import healpy as hp
        flat_indices_input_grid_output_frame = hp.ang2pix(
            nside=spatial_object_output_frame.nside,
            theta=np.rad2deg(obj1_az_el_points_frame2[:, 0]),  # Lon
            phi=np.rad2deg(obj1_az_el_points_frame2[:, 1]),  # Lat
            nest=False,
            lonlat=True,
        )
        ```
        """
        raise NotImplementedError(
            "Index matching for output tiling type Healpix is not yet implemented."
        )

    else:
        raise ValueError(
            "Tiling type of the output frame must be either RECTANGULAR or HEALPIX."
        )

    return flat_indices_input_grid_output_frame


# Define the pointing set classes
class PointingSet(ABC):
    """
    Abstract class to contain pointing set (PSET) data in the context of ENA sky maps.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the pointing set data.
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set.
    """

    @abstractmethod
    def __init__(self, dataset: xr.Dataset, spice_reference_frame: geometry.SpiceFrame):
        """Abstract method to initialize the pointing set object."""
        self.spice_reference_frame = spice_reference_frame
        self.num_points = 0
        self.az_el_points = np.zeros((self.num_points, 2))
        self.data = xr.Dataset()

    def __repr__(self) -> str:
        """
        Return a string representation of the pointing set.

        Returns
        -------
        str
            String representation of the pointing set.
        """
        return (
            f"{self.__class__} PointingSet"
            f"(spice_reference_frame={self.spice_reference_frame})"
        )


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
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set. Default is IMAP_DPS.

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
        spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
    ):
        # Store the reference frame of the pointing set
        self.spice_reference_frame = spice_reference_frame

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
        self.tiling_type = SkyTilingType.RECTANGULAR

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

        # Build the azimuth and elevation grids with an AzElSkyGrid object
        # and check that the 1D axes match the dataset's az and el.
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=self.spacing_deg,
        )

        for dim, constructed_bins in zip(
            ["azimuth", "elevation"],
            [self.sky_grid.az_bin_midpoints, self.sky_grid.el_bin_midpoints],
        ):
            if not np.allclose(
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
            (
                self.sky_grid.az_grid.ravel(order=RAVEL_ORDER),
                self.sky_grid.el_grid.ravel(order=RAVEL_ORDER),
            )
        )
        self.num_points = self.az_el_points.shape[0]

        # Also store the bin edges for the pointing set to allow for "pull" method
        # of index matching (not yet implemented).
        # These are 1D arrays of different lengths and cannot be stacked.
        self.az_bin_edges = self.sky_grid.az_bin_edges
        self.el_bin_edges = self.sky_grid.el_bin_edges

    def __repr__(self) -> str:
        """
        Return a string representation of the UltraPointingSet.

        Returns
        -------
        str
            String representation of the UltraPointingSet.
        """
        return (
            f"UltraPointingSet\n\t(spice_reference_frame="
            f"{self.spice_reference_frame}, epoch={self.epoch}, "
            f"num_points={self.num_points})"
        )


# Define the Map classes
class AbstractSkyMap(ABC):
    """Abstract base class to contain map data in the context of ENA sky maps."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the map.

        Returns
        -------
        str
            String representation of the map.
        """
        return f"{self.__class__} Map)"


class RectangularSkyMap(AbstractSkyMap):
    """
    Map which tiles the sky with a 2D rectangular grid of azimuth/elevation pixels.

    NOTE: Internally, the map is stored as a 1D array of pixels.

    Parameters
    ----------
    spacing_deg : float
        The spacing of the rectangular grid in degrees.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.
    """

    def __init__(
        self,
        spacing_deg: float,
        spice_frame: geometry.SpiceFrame,
    ):
        # Define the core properties of the map:
        self.tiling_type = SkyTilingType.RECTANGULAR  # Type of tiling of the sky
        self.spacing_deg = spacing_deg
        self.spice_reference_frame = spice_frame
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=self.spacing_deg,
        )

        # Solid angles of each pixel in the map grid in units of steradians
        self.solid_angle_grid = spatial_utils.build_solid_angle_map(
            spacing_deg=self.spacing_deg,
        )

        # Unwrap the az, el, solid angle grids to series of points tiling the sky
        az_points = self.sky_grid.az_grid.ravel(order=RAVEL_ORDER)
        el_points = self.sky_grid.el_grid.ravel(order=RAVEL_ORDER)
        self.az_el_points = np.column_stack((az_points, el_points))
        self.solid_angle_points = self.solid_angle_grid.ravel(order=RAVEL_ORDER)
        self.num_points = self.az_el_points.shape[0]

        # Initialize empty data dictionary to store map data
        self.data_dict: dict[str, NDArray] = {}

    def project_pset_values_to_map(
        self,
        pointing_set: PointingSet,
        value_keys: list[tuple[str, IndexMatchMethod]] | None = None,
    ) -> None:
        """
        Project a pointing set's values to the map grid.

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set containing the values to project to the map.
        value_keys : list[tuple[str, IndexMatchMethod]] | None
            The keys of the values to project to the map, and method of index matching.
            Ex.: [("counts", IndexMatchMethod.PUSH), ("flux", IndexMatchMethod.PULL)]
            data_vars named each key must be present, and of the same dimensionality in
            each pointing set which is to be projected to the map.
            Default is None, in which case all data_vars in the pointing set are used,
            with the "push" method of index matching.

        Raises
        ------
        ValueError
            If a value key is not found in the pointing set.
        """
        if value_keys is None:
            value_keys = [
                (key, IndexMatchMethod.PUSH)
                for key in pointing_set.data.data_vars.keys()
            ]

        for value_key, _ in value_keys:
            if value_key not in pointing_set.data.data_vars:
                raise ValueError(f"Value key {value_key} not found in pointing set.")

        # Determine the indices of the sky map grid that correspond to
        # each pixel in the pointing set.
        matched_indices_push = match_coords_to_indices(
            spatial_object_input_frame=pointing_set,
            spatial_object_output_frame=self,
        )

        for value_key, method in value_keys:
            # If multiple spatial axes present
            # (i.e (az, el) for rectangular coordinate PSET),
            # flatten them in the values array to match the raveled indices
            raveled_pset_data = np.reshape(
                np.array(pointing_set.data[value_key]),
                (pointing_set.num_points, -1),
                order=RAVEL_ORDER,
            )

            if value_key not in self.data_dict:
                # Initialize the map data array if it doesn't exist (values start at 0)
                output_shape = (self.num_points, *raveled_pset_data.shape[1:])
                self.data_dict[value_key] = np.zeros(output_shape)

            if method == IndexMatchMethod.PUSH:
                pointing_projected_values = map_utils.bin_single_array_at_indices(
                    value_array=raveled_pset_data,
                    projection_grid_shape=(
                        len(self.sky_grid.az_bin_midpoints),
                        len(self.sky_grid.el_bin_midpoints),
                    ),
                    projection_indices=matched_indices_push,
                )
            else:
                raise NotImplementedError(
                    "The 'pull' method of index matching is not yet implemented."
                )
            self.data_dict[value_key] += pointing_projected_values

    def __repr__(self) -> str:
        """
        Return a string representation of the RectangularSkyMap.

        Returns
        -------
        str
            String representation of the RectangularSkyMap.
        """
        return (
            "RectangularSkyMap\n\t(reference_frame="
            f"{self.spice_reference_frame.name} ({self.spice_reference_frame.value}), "
            f"spacing_deg={self.spacing_deg}, num_points={self.num_points})"
        )


# TODO:
# Add pulling index matching in match_pset_coords_to_indices

# TODO:
# Add ability to push some, pull other indices in project_pset_values_to_map
# dict{"value_key": (array, "push"| "pull")}

# TODO:
# Check units of time which will be read in. Do we need to add j2000ns_to_j2000s?
