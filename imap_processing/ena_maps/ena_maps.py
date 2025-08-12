"""Define classes for handling pointing sets and maps for ENA data."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TypeVar

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps.utils import map_utils, spatial_utils

# The coordinate names can vary between L1C and L2 data (e.g. azimuth vs longitude),
# so we define an enum to handle the coordinate names.
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.time import ttj2000ns_to_et

logger = logging.getLogger(__name__)

# Set the maximum recursion depth for the conversion from Healpix to rectangular SkyMap.
MAX_SUBDIV_RECURSION_DEPTH = 8


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
    transformed pointing set pixel falls.
    This method ensures that all pointing set pixels (and thus all counts) are
    captured in the map, but does not ensure that all pixels in the map receive data.

    **Pull Method**

    The "pull" method takes each pixel in the map grid and transforms its coordinates
    to the frame of the pointing set, then determines into which pixel in the
    pointing set grid the transformed map pixel falls.
    This method ensures that all pixels in the map receive data, but can result in
    some pointing set pixels not being captured in the map, and others being captured
    multiple times.
    """

    PUSH = "Push"
    PULL = "Pull"


def match_coords_to_indices(
    input_object: PointingSet | AbstractSkyMap,
    output_object: PointingSet | AbstractSkyMap,
    event_et: float | None = None,
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

    At present, the allowable inputs are either:
    - A PointingSet object and a SkyMap object, in either order of input/output.
    The event time will be taken from the PointingSet object.
    - Two SkyMap objects, in which case the event time must be specified.

    Parameters
    ----------
    input_object : PointingSet | AbstractSkyMap
        An object containing 1D spatial pixel centers in azimuth and elevation,
        which will be matched to 1D indices of spatial pixels in the output frame.
        Must contain the Spice frame in which the pixel centers are defined.
    output_object : PointingSet | AbstractSkyMap
        The object containing a grid or tessellation of spatial pixels
        into which the input spatial pixel centers will 'land', and be matched to
        corresponding pixel 1D indices in the output frame.
    event_et : float, optional
        Event time at which to transform the input spatial object to the output frame.
        This can be manually specified, e.g., for converting between Maps which do not
        contain an epoch value.
        If specified, must be in SPICE compatible ET.
        The default value is None, in which case the event time of the PointingSet
        object is used.

    Returns
    -------
    flat_indices_input_grid_output_frame : NDArray
        1D array of pixel indices of the output object corresponding to each pixel in
        the input object. The length of the array is equal to the number of pixels in
        the input object, and may contain 0, 1, or multiple occurrences of the same
        output index.

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
    if isinstance(input_object, PointingSet) and isinstance(output_object, PointingSet):
        raise ValueError("Cannot match indices between two PointingSet objects.")

    # If event_et is not specified, use epoch of the PointingSet, if present.
    # The epoch will be in units of terrestrial time (TT) J2000 nanoseconds,
    # which must be converted to ephemeris time (ET) for SPICE.
    if event_et is None:
        if isinstance(input_object, PointingSet):
            event_et = ttj2000ns_to_et(input_object.epoch)
        elif isinstance(output_object, PointingSet):
            event_et = ttj2000ns_to_et(output_object.epoch)
        else:
            raise ValueError(
                "Event time must be specified if both objects are SkyMaps."
            )

    # Az/El pixel center coords of the input object in its own frame
    input_obj_az_el_input_frame = input_object.az_el_points

    # Transform the input pixel centers to the output frame
    input_obj_az_el_output_frame = geometry.frame_transform_az_el(
        et=event_et,
        az_el=input_obj_az_el_input_frame,
        from_frame=input_object.spice_reference_frame,
        to_frame=output_object.spice_reference_frame,
        degrees=True,
    )

    # The way indices are matched depends on the tiling type of the 2nd object
    if output_object.tiling_type is SkyTilingType.RECTANGULAR:
        # To match to a rectangular grid, we need to digitize the transformed az, el
        # pixel centers onto the bin edges of the output frame's grid, then
        # use ravel_multi_index to get the 1D indices of the pixels in the output frame.
        az_indices = (
            np.digitize(
                input_obj_az_el_output_frame[:, 0],
                output_object.sky_grid.az_bin_edges,
            )
            - 1
        )
        el_indices = (
            np.digitize(
                input_obj_az_el_output_frame[:, 1],
                output_object.sky_grid.el_bin_edges,
            )
            - 1
        )
        flat_indices_input_grid_output_frame = np.ravel_multi_index(
            multi_index=(az_indices, el_indices),
            dims=(
                len(output_object.sky_grid.az_bin_midpoints),
                len(output_object.sky_grid.el_bin_midpoints),
            ),
        )

    elif output_object.tiling_type is SkyTilingType.HEALPIX:
        # To match to a Healpix tessellation, we need to use the healpy function ang2pix
        # which directly returns the index on the output frame's Healpix tessellation.
        flat_indices_input_grid_output_frame = hp.ang2pix(
            nside=output_object.nside,
            theta=input_obj_az_el_output_frame[:, 0],  # Lon in degrees
            phi=input_obj_az_el_output_frame[:, 1],  # Lat in degrees
            nest=output_object.nested,
            lonlat=True,
        )
    else:
        raise ValueError(
            "Tiling type of the output frame must be either RECTANGULAR or HEALPIX."
            f"Received: {output_object.tiling_type}"
        )

    return flat_indices_input_grid_output_frame


# Define a TypeVar type to dynamically hint the return type of the base PointingSet
# class classmethod
T = TypeVar("T", bound="PointingSet")


# Define the pointing set classes
class PointingSet(ABC):
    """
    Abstract class to contain pointing set (PSET) data in the context of ENA sky maps.

    Any spatial axes - (azimuth, elevation) for Rectangularly gridded tilings or
    (pixel index) for Healpix - must be stored in the last axis/axes of each data array.

    Parameters
    ----------
    dataset : xr.Dataset | str | Path
        Dataset or path to CDF file containing the pointing set data.
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set.
    """

    # The minimum set of class attributes for any PointingSet to function with
    # a SkyMap using only the PUSH method of projecting are defined here.

    # ======== Attributes that are set in the ABC __init__ method ========
    # The xarray.Dataset containing the data from the PSET CDF
    data: xr.Dataset
    # The spice frame that the az_el_points are expressed in
    spice_reference_frame: geometry.SpiceFrame

    # ======== Attributes required to be set in a subclass ========
    # Azimuth and elevation coordinates of each spatial pixel. The ndarray should
    # have the shape (n, 2) where n is the number of spatial pixels
    az_el_points: np.ndarray
    # Tuple containing the names of each spatial coordinate of the xarray.Dataset
    # stored in the data attribute
    spatial_coords: tuple[str, ...]

    @abstractmethod
    def __init__(
        self,
        dataset: xr.Dataset | str | Path,
        spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
    ):
        """Abstract method to initialize the pointing set object."""
        self.spice_reference_frame = spice_reference_frame

        if isinstance(dataset, (str, Path)):
            dataset = load_cdf(dataset)
            self.data = dataset
        else:
            # If the dataset is already an xarray Dataset,
            # deep copy it to avoid modifying original PSET data
            self.data = dataset.copy(deep=True)

        # A PSET must have a single epoch
        if len(np.unique(self.data["epoch"].values)) > 1:
            raise ValueError("Multiple epochs found in the dataset.")

    @property
    def num_points(self) -> int:
        """
        The number of spatial pixels in the pointing set.

        Returns
        -------
        num_points: int
            The number of spatial pixels in the pointing set.
        """
        return self.az_el_points.shape[0]

    @property
    def epoch(self) -> int:
        """
        The singular epoch value from the xarray.Dataset.

        Returns
        -------
        epoch: int
            The epoch value [J2000 TT ns] of the pointing set.
        """
        return self.data["epoch"].values[0]

    @property
    def unwrapped_dims_dict(self) -> dict[str, tuple[str, ...]]:
        """
        Get dimensions of each variable in the pointing set, with only 1 spatial dim.

        Returns
        -------
        unwrapped_dims_dict : dict[str, tuple[str, ...]]
            Dictionary of variable names and their dimensions, with only 1 spatial dim.
            The generic pixel dimension is always included.
            E.g.: {"counts": ("epoch", "energy", "pixel")} .
        """
        variable_dims = {}
        for var_name in self.data.data_vars:
            pset_dims = self.data[var_name].dims
            non_spatial_dims = tuple(
                dim for dim in pset_dims if dim not in self.spatial_coords
            )

            variable_dims[var_name] = (
                *non_spatial_dims,
                CoordNames.GENERIC_PIXEL.value,
            )
        return variable_dims

    @property
    def non_spatial_coords(self) -> dict[str, xr.DataArray]:
        """
        Get the non-spatial coordinates of the pointing set.

        Returns
        -------
        non_spatial_coords : dict[str, xr.DataArray]
            Dictionary of coordinate names and their data arrays.
            E.g.: {"epoch": [12345,], "energy": [100, 200, 300]} .
        """
        non_spatial_coords = {}
        for coord_name in self.data.coords:
            if coord_name not in self.spatial_coords:
                non_spatial_coords[coord_name] = self.data[coord_name]
        return non_spatial_coords

    def __repr__(self) -> str:
        """
        Return a string representation of the pointing set.

        Returns
        -------
        str
            String representation of the pointing set.
        """
        return (
            f"{self.__class__.__name__} PointingSet"
            f"(spice_reference_frame={self.spice_reference_frame})"
        )


class RectangularPointingSet(PointingSet):
    """
    Pointing set object for rectangularly tiled data. Currently used in testing.

    Parameters
    ----------
    dataset : xr.Dataset | str | Path
        Dataset or path to CDF file containing the pointing set data.
        Currently, the dataset is expected to be tiled in a rectangular grid,
        with data_vars indexed along the coordinates:
            - 'epoch' : time value (1 value per PSET)
            - 'longitude' : (number of longitude/az bins in L1C)
            - 'latitude' : (number of latitude/el bins in L1C)
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set. Default is IMAP_DPS.

    Raises
    ------
    ValueError
        If the longitude/az or latitude/el bin centers don't match the constructed grid.
        Or if the longitude or latitude bin spacing is not uniform.
    ValueError
        If multiple epochs are found in the dataset.
    """

    # In addition to the required attributes defined in the base PointingSet
    # class, the following attributes are required for a RectangularPointingSet
    # to be projected using the PULL method.
    tiling_type: SkyTilingType = SkyTilingType.RECTANGULAR
    sky_grid: spatial_utils.AzElSkyGrid

    def __init__(
        self,
        dataset: xr.Dataset | str | Path,
        spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
    ):
        super().__init__(dataset, spice_reference_frame)

        self.spatial_coords = (
            CoordNames.AZIMUTH_L1C.value,
            CoordNames.ELEVATION_L1C.value,
        )

        # Ensure 1D axes grids are uniformly spaced,
        # then set spacing based on data's azimuth bin spacing.
        az_bin_delta = np.diff(self.data[CoordNames.AZIMUTH_L1C.value])
        el_bin_delta = np.diff(self.data[CoordNames.ELEVATION_L1C.value])
        if not np.allclose(az_bin_delta, az_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Azimuth bin spacing is not uniform.")
        if not np.allclose(el_bin_delta, el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError("Elevation bin spacing is not uniform.")
        if not np.isclose(az_bin_delta[0], el_bin_delta[0], atol=1e-10, rtol=0):
            raise ValueError(
                "Azimuth and elevation bin spacing do not match: "
                f"az {az_bin_delta[0]} != el {el_bin_delta[0]}."
            )
        spacing_deg = az_bin_delta[0]

        # Build the az/azimuth and el/elevation grids with an AzElSkyGrid object
        # and check that the 1D axes match the dataset's az and el.
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=spacing_deg,
        )

        for dim, constructed_bins in zip(
            [CoordNames.AZIMUTH_L1C.value, CoordNames.ELEVATION_L1C.value],
            [self.sky_grid.az_bin_midpoints, self.sky_grid.el_bin_midpoints],
        ):
            if not np.allclose(
                sorted(constructed_bins),
                self.data[dim],
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    f"{dim} bin centers do not match."
                    f"Constructed: {constructed_bins}"
                    f"Dataset: {self.data[dim]}"
                )

        # Unwrap the az, el grids to series of points tiling the sky and combine them
        # into shape (number of points in tiling of the sky, 2) where
        # column 0 (az_el_points[:, 0]) is the azimuth of that point and
        # column 1 (az_el_points[:, 1]) is the elevation of that point.
        self.az_el_points = np.column_stack(
            (
                self.sky_grid.az_grid.ravel(),
                self.sky_grid.el_grid.ravel(),
            )
        )


class HealpixPointingSet(PointingSet, ABC):
    """
    Abstract base class for Healpix pointing sets.

    Defines additional properties and absract properties that are required
    for a PointingSet instance to be used with the match_coords_to_indices
    function.
    """

    tiling_type: SkyTilingType = SkyTilingType.HEALPIX

    @property
    def nside(self) -> int:
        """
        Number of pixels on the side of one of the 12 top-level healpix tiles.

        Returns
        -------
        npix: int
            The number of pixels on the side of one of the 12 ‘top-level’ healpix
            tiles.
        """
        return hp.npix_to_nside(self.num_points)

    @property
    @abstractmethod
    def nested(self) -> bool:
        """Abstract property for getting nested boolean."""
        raise NotImplementedError


class UltraPointingSet(HealpixPointingSet):
    """
    Pointing set object specifically for Healpix-tiled ULTRA data, nominally at Level1C.

    Parameters
    ----------
    dataset : xr.Dataset | str | Path
        Dataset or path to CDF file containing the pointing set data.
        Currently, the dataset is expected to be tiled in a HEALPix tessellation,
        with data_vars indexed along the coordinates:
            - 'epoch' : time value (1 value per PSET, from the mean of the PSET)
            - 'energy' : (number of energy bins in L1C)
            - 'healpix_index' : HEALPix pixel index
        Only the 'healpix_index' coordinate is used in this class for projection.
    spice_reference_frame : geometry.SpiceFrame
        The reference Spice frame of the pointing set. Default is IMAP_DPS.

    Raises
    ------
    ValueError
        If the longitude/az or latitude/el bin centers don't match the constructed grid.
        Or if the longitude or latitude bin spacing is not uniform.
    ValueError
        If multiple epochs are found in the dataset.
    """

    def __init__(
        self,
        dataset: xr.Dataset | str | Path,
        spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.IMAP_DPS,
    ):
        super().__init__(dataset, spice_reference_frame)

        # Set the spatial coordinates and number of points
        self.spatial_coords = (CoordNames.HEALPIX_INDEX.value,)

        # Tracks Per-Pixel Solid Angle in steradians.
        self.solid_angle = hp.nside2pixarea(self.nside, degrees=False)

        # Get the azimuth and elevation coordinates of the healpix pixel centers (deg)
        azimuth_pixel_center, elevation_pixel_center = hp.pix2ang(
            nside=self.nside,
            ipix=np.arange(self.num_points),
            nest=self.nested,
            lonlat=True,
        )

        # Verify that the azimuth and elevation of the healpix pixel centers
        # match the data's azimuth and elevation bin centers.
        # NOTE: They can have different names in the L1C dataset
        # (e.g. "longitude"/"latitude" vs "azimuth"/"elevation").
        for dim, constructed_bins in zip(
            [CoordNames.AZIMUTH_L1C.value, CoordNames.ELEVATION_L1C.value],
            [azimuth_pixel_center, elevation_pixel_center],
        ):
            if not np.allclose(
                self.data[dim],
                constructed_bins,
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    f"{dim} pixel centers do not match the data's {dim} bin centers."
                    f"Constructed: {constructed_bins}"
                    f"Dataset: {self.data[dim]}"
                )

        # The coordinates of the healpix pixel centers are stored as a 2D array
        # of shape (num_points, 2) where column 0 is the lon/az
        # and column 1 is the lat/el.
        self.az_el_points = np.column_stack(
            (azimuth_pixel_center, elevation_pixel_center)
        )

    @property
    def num_points(self) -> int:
        """
        Override the base class property to get the number from the dataset.

        Returns
        -------
        num_points: int
            The number of healpix pixels in the pointing set.
        """
        return self.data[CoordNames.HEALPIX_INDEX.value].size

    @property
    def nested(self) -> bool:
        """
        Whether the healpix tessellation is nested.

        Returns
        -------
        nested: bool
            Whether the healpix tessellation is nested.
        """
        return bool(
            self.data[CoordNames.HEALPIX_INDEX.value].attrs.get("nested", False)
        )

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


class HiPointingSet(PointingSet):
    """
    PointingSet object specific to Hi L1C PSet data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Hi L1C pointing set data loaded in an xarray.DataArray.
    """

    def __init__(self, dataset: xr.Dataset):
        super().__init__(dataset, spice_reference_frame=geometry.SpiceFrame.ECLIPJ2000)

        # Rename some PSET vars to match L2 variables
        self.data = self.data.rename(
            {
                "exposure_times": "exposure_factor",
                "background_rates": "bg_rates",
                "background_rates_uncertainty": "bg_rates_unc",
            }
        )

        # Add obs_date variable to be used in determining a map mean obs_date
        self.data["obs_date"] = xr.full_like(
            self.data["exposure_factor"], self.data["epoch"].values[0]
        )

        self.az_el_points = np.column_stack(
            (
                np.squeeze(self.data["hae_longitude"]),
                np.squeeze(self.data["hae_latitude"]),
            )
        )
        self.spatial_coords = ("spin_angle_bin",)


class LoPointingSet(PointingSet):
    """
    PointingSet object specific to Lo L1C PSet data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lo L1C pointing set data loaded in an xarray.DataArray.
    """

    def __init__(self, dataset: xr.Dataset):
        super().__init__(dataset, spice_reference_frame=geometry.SpiceFrame.IMAP_DPS)
        # TODO: Use spatial_utils.az_el_grid instead of
        #  manually creating the lon/lat values
        inferred_spacing_deg = 360 / dataset.longitude.size
        longitude_bin_centers = np.arange(
            0 + inferred_spacing_deg / 2, 360, inferred_spacing_deg
        )
        latitude_bin_centers = np.arange(
            -2 + inferred_spacing_deg / 2, 2, inferred_spacing_deg
        )

        # Could be wrong about the order here
        longitude_grid, latitude_grid = np.meshgrid(
            longitude_bin_centers,
            latitude_bin_centers,
            indexing="ij",
        )

        longitude = longitude_grid.ravel()
        latitude = latitude_grid.ravel()

        self.az_el_points = np.column_stack((longitude, latitude))
        self.spatial_coords = ("longitude", "latitude")


# Define the Map classes
class AbstractSkyMap(ABC):
    """
    Abstract base class to contain map data in the context of ENA sky maps.

    Data values are stored internally in an xarray Dataset, in the .data_1d attribute.
    where the final (-1) axis is the only spatial dimension.
    If the map is rectangular, this axis is the raveled 2D grid.
    If the map is Healpix, this axis is the 1D array of Healpix pixel indices.

    The data can be also accessed via the to_dataset method, which rewraps the data to
    a 2D grid shape if the map is rectangular and formats the data as an xarray
    Dataset with the correct dims and coords.

    Parameters
    ----------
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.
    """

    # ======== Attributes that are set in the ABC __init__ method ========
    # The spice frame that the az_el_points are expressed in
    spice_frame: geometry.SpiceFrame
    # Lists of variables to project using push and pull methods
    values_to_push_project: list[str]
    values_to_pull_project: list[str]
    # Variables used to track min/max epoch of data that gets projected to map
    min_epoch: int
    max_epoch: int

    # ======== Attributes required to be set in a subclass ========
    # Azimuth and elevation coordinates of each spatial pixel. The ndarray should
    # have the shape (n, 2) where n is the number of spatial pixels
    az_el_points: np.ndarray
    # Type of sky tiling
    tiling_type: SkyTilingType
    # Dictionary of xr.DataArray objects for each non-spatial coordinate in the SkyMap
    non_spatial_coords: dict[str, xr.DataArray | NDArray]
    # Dictionary of xr.DataArray objects for each spatial coordinate in the SkyMap
    spatial_coords: dict[str, xr.DataArray | NDArray]
    # 1D Array of spatial pixel solid angles
    solid_angle_points: np.ndarray
    # Dataset to store map data projected from PointingSet objects
    data_1d: xr.Dataset

    @abstractmethod
    def __init__(self, spice_frame: geometry.SpiceFrame) -> None:
        self.spice_reference_frame = spice_frame

        # Initialize values to be used by the instrument code to push/pull
        self.values_to_push_project = []
        self.values_to_pull_project = []

        # Initialize min and max epoch variables that are used to keep track of
        # PSET epochs that get binned to the map
        self.min_epoch = np.iinfo(np.int64).max
        self.max_epoch = np.iinfo(np.int64).min

    @abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """
        Get the SkyMap data as a formatted xarray Dataset.

        Returns
        -------
        xr.Dataset
            The SkyMap data as a formatted xarray Dataset with dims and coords.
            If the SkyMap is empty, an empty xarray Dataset is returned.
            If the SkyMap is Rectangular, the data is rewrapped to a 2D grid of
            lon/lat (AKA az/el) coordinates.
            If the SkyMap is Healpix, the data is unchanged from the data_1d, but
            the pixel coordinate is renamed to CoordNames.HEALPIX_INDEX.value.
        """
        raise NotImplementedError("AbstractSkyMap.to_dataset() not implemented.")

    @property
    @abstractmethod
    def binning_grid_shape(self) -> tuple[int]:
        """
        Shape of the binning grid.

        Returns
        -------
        binning_grid_shape : tuple[int]
            Shape of the binning grid.
        """
        raise NotImplementedError("binning_grid_shape property method not implemented.")

    @property
    def num_points(self) -> int:
        """
        The number of spatial pixels in the SkyMap.

        Returns
        -------
        num_points: int
            The number of spatial pixels in the pointing set.
        """
        return self.az_el_points.shape[0]

    def project_pset_values_to_map(
        self,
        pointing_set: PointingSet,
        value_keys: list[str] | None = None,
        index_match_method: IndexMatchMethod = IndexMatchMethod.PUSH,
    ) -> None:
        """
        Project a pointing set's values to the map grid.

        Here, the term "project" refers to the process of determining which pixels in
        the map grid correspond to which pixels in the pointing set grid, and then
        binning the values at those indices from the pointing set to the map.

        Parameters
        ----------
        pointing_set : PointingSet
            The pointing set containing the values to project to the map.
        value_keys : list[tuple[str, IndexMatchMethod]] | None
            The keys of the values in the PointingSet to project to the map.
            Ex.: ["counts", "flux"]
            data_vars named each key must be present, and of the same dimensionality in
            each pointing set which is to be projected to the map.
            Default is None, in which case all data_vars in the pointing set are used.
        index_match_method : IndexMatchMethod, optional
            The method of index matching to use for all values.
            Default is IndexMatchMethod.PUSH.

        Raises
        ------
        ValueError
            If a value key is not found in the pointing set.
        """
        if value_keys is None:
            value_keys = list(pointing_set.data.data_vars.keys())
        for value_key in value_keys:
            if value_key not in pointing_set.data.data_vars:
                raise ValueError(f"Value key {value_key} not found in pointing set.")

        if index_match_method is IndexMatchMethod.PUSH:
            # Determine the indices of the sky map grid that correspond to
            # each pixel in the pointing set.
            matched_indices_push = match_coords_to_indices(
                input_object=pointing_set,
                output_object=self,
            )
        elif index_match_method is IndexMatchMethod.PULL:
            # Determine the indices of the pointing set grid that correspond to
            # each pixel in the sky map.
            matched_indices_pull = match_coords_to_indices(
                input_object=self,
                output_object=pointing_set,
            )
        else:
            raise NotImplementedError(
                "Only PUSH and PULL index matching methods are supported."
            )

        for value_key in value_keys:
            pset_values = pointing_set.data[value_key]

            # If multiple spatial axes present
            # (i.e (az, el) for rectangular coordinate PSET),
            # flatten them in the values array to match the raveled indices
            non_spatial_axes_shape = tuple(
                size
                for key, size in pset_values.sizes.items()
                if key not in pointing_set.spatial_coords
            )
            raveled_pset_data = pset_values.data.reshape(
                *non_spatial_axes_shape,
                pointing_set.num_points,
            )

            if value_key not in self.data_1d.data_vars:
                # Initialize the map data array if it doesn't exist (values start at 0)
                output_shape = (*raveled_pset_data.shape[:-1], self.num_points)
                self.data_1d[value_key] = xr.DataArray(
                    np.zeros(output_shape),
                    dims=pointing_set.unwrapped_dims_dict[value_key],
                )

                # Make coordinates for the map data array if they don't exist
                self.data_1d.coords.update(
                    {
                        dim: pointing_set.data[dim]
                        for dim in self.data_1d[value_key].dims
                        if dim not in self.data_1d.coords
                    }
                )

            if index_match_method is IndexMatchMethod.PUSH:
                # Bin the values at the matched indices. There may be multiple
                # pointing set pixels that correspond to the same sky map pixel.
                pointing_projected_values = map_utils.bin_single_array_at_indices(
                    value_array=raveled_pset_data,
                    projection_grid_shape=self.binning_grid_shape,
                    projection_indices=matched_indices_push,
                )
            elif index_match_method is IndexMatchMethod.PULL:
                # We know that there will only be one value per sky map pixel,
                # so we can use the matched indices directly
                pointing_projected_values = raveled_pset_data[..., matched_indices_pull]
            else:
                raise NotImplementedError(
                    "Only PUSH and PULL index matching methods are supported."
                )

            # TODO: we may need to allow for unweighted/weighted means here by
            # dividing pointing_projected_values by some binned weights.
            # For unweighted means, we could use the number of pointing set pixels
            # that correspond to each map pixel as the weights.
            self.data_1d[value_key] += pointing_projected_values

        # TODO: The max epoch needs to include the pset duration. Right now it
        #     is just capturing the start epoch. See issue #1747
        self.min_epoch = min(self.min_epoch, pointing_set.epoch)
        self.max_epoch = max(self.max_epoch, pointing_set.epoch)

    @classmethod
    def from_properties_json(
        cls, json_path: str | Path
    ) -> RectangularSkyMap | HealpixSkyMap:
        """
        Create a SkyMap object from a JSON configuration file.

        Parameters
        ----------
        json_path : str | Path
            Path to the JSON configuration file.

        Returns
        -------
        RectangularSkyMap | HealpixSkyMap
            An instance of a SkyMap object with the specified properties.
        """
        with open(json_path) as f:
            properties = json.load(f)
        return cls.from_properties_dict(properties)

    @classmethod
    def from_properties_dict(
        cls, properties: dict
    ) -> RectangularSkyMap | HealpixSkyMap:
        """
        Create a SkyMap object from a dictionary of properties.

        Parameters
        ----------
        properties : dict
            Dictionary containing the map properties. The required keys are:
            - "spice_reference_frame" : str
                The reference Spice frame of the map as a string. The available
                options are defined in the spice geometry module:
                `imap_processing.geometry.spice.SpiceFrame`. Example: "ECLIPJ2000".
            - "sky_tiling_type" : str
                The type of sky tiling, either "HEALPIX" or "RECTANGULAR".
            - if "HEALPIX":
                - "nside" : int
                    The nside parameter for the Healpix tessellation.
                - "nested" : bool
                    Whether the Healpix tessellation is nested or not.
            - if "RECTANGULAR":
                - "spacing_deg" : float
                    The spacing of the rectangular grid in degrees.
            - "values_to_push_project" : list[str], optional
                The names of the variables to project to the map with the PUSH method.
                NOTE: The projection is done by the instrument code, so this value can
                only be used to inform that code. No values are projected automatically.
            - "values_to_pull_project" : list[str], optional
                The names of the variables to project to the map with the PULL method.
                See the above note for more details.

            See example dictionary in notes section.

        Returns
        -------
        RectangularSkyMap | HealpixSkyMap
            An instance of a SkyMap object with the specified properties.

        Raises
        ------
        ValueError
            If the sky tiling type is not recognized.

        Notes
        -----
        Example dictionary:

        ```python
        properties = {
            "spice_reference_frame": "ECLIPJ2000",
            "sky_tiling_type": "HEALPIX",
            "nside": 32,
            "nested": False,
            "values_to_push_project": ['counts', 'flux'],
            "values_to_pull_project": []
        }
        ```
        """
        sky_tiling_type = SkyTilingType[properties["sky_tiling_type"].upper()]
        spice_reference_frame = geometry.SpiceFrame[properties["spice_reference_frame"]]

        skymap: RectangularSkyMap | HealpixSkyMap  # Mypy gets confused by if/elif types
        if sky_tiling_type is SkyTilingType.HEALPIX:
            skymap = HealpixSkyMap(
                nside=properties["nside"],
                nested=properties["nested"],
                spice_frame=spice_reference_frame,
            )
        elif sky_tiling_type is SkyTilingType.RECTANGULAR:
            skymap = RectangularSkyMap(
                spacing_deg=properties["spacing_deg"],
                spice_frame=spice_reference_frame,
            )
        else:
            raise ValueError(
                f"Unknown sky tiling type: {sky_tiling_type}. "
                f"Must be one of: {SkyTilingType.__members__.keys()}"
            )

        # Store requested variables to push/pull, which will be done by the instrument
        # code which creates and uses the SkyMap object.
        skymap.values_to_push_project = properties.get("values_to_push_project", [])
        skymap.values_to_pull_project = properties.get("values_to_pull_project", [])
        return skymap

    @abstractmethod
    def to_properties_dict(self) -> dict:
        """
        Convert the SkyMap object to a dictionary of properties.

        Returns
        -------
        dict
            Dictionary containing the map properties.
        """
        raise NotImplementedError("to_dict must be implemented in a subclass.")

    def to_properties_json(self, json_path: str | Path) -> None:
        """
        Save the SkyMap object to a JSON configuration file.

        Parameters
        ----------
        json_path : str | Path
            Path to the JSON file where the properties will be saved.
        """
        with open(json_path, "w") as f:
            json.dump(self.to_properties_dict(), f, indent=4)


class RectangularSkyMap(AbstractSkyMap):
    """
    Map which tiles the sky with a 2D rectangular grid of azimuth/elevation pixels.

    Parameters
    ----------
    spacing_deg : float
        The spacing of the rectangular grid in degrees.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.

    Notes
    -----
    Internally, the map is stored as a 1D array of pixels, and all data arrays
    are stored with the final (-1) axis as the only spatial axis, representing the
    pixel index in the 1D array (See Figs 1-2, which demonstrate the 1D pixel index
    corresponding to the 2D grid of coordinates).

    ^  |15,  75|45,  75|75,  75|105,  75|...|255,  75|285,  75|315,  75|345,  75|
    |  |15,  45|45,  45|75,  45|105,  45|...|255,  45|285,  45|315,  45|345,  45|
    |  |15,  15|45,  15|75,  15|105,  15|...|255,  15|285,  15|315,  15|345,  15|
    |  |15, -15|45, -15|75, -15|105, -15|...|255, -15|285, -15|315, -15|345, -15|
    |  |15, -45|45, -45|75, -45|105, -45|...|255, -45|285, -45|315, -45|345, -45|
    |  |15, -75|45, -75|75, -75|105, -75|...|255, -75|285, -75|315, -75|345, -75|
    |
    ---------------------------------------------------------------> Azimuth (degrees)
    Elevation (degrees)

    Fig. 1: Example of a rectangular grid of pixels in azimuth and elevation coordinates
    in degrees, with a spacing of 30 degrees. There will be 12 azimuth bins and 6
    elevation bins in this example, resulting in 72 pixels in the map.

    A multidimensional value (e.g. counts, with energy levels at each pixel)
    will be stored as a 2D array with the first axis as the energy dimension and the
    second axis as the pixel index.

    ^  |5|11|17|23|29|35|41|47|53|59|65|71|
    |  |4|10|16|22|28|34|40|46|52|58|64|70|
    |  |3|9 |15|21|27|33|39|45|51|57|63|69|
    |  |2|8 |14|20|26|32|38|44|50|56|62|68|
    |  |1|7 |13|19|25|31|37|43|49|55|61|67|
    |  |0|6 |12|18|24|30|36|42|48|54|60|66|
    ---------------------------------------> Azimuth
    Elevation

    Fig. 2: The 1D indices of the pixels in Fig. 1.
    Note that the indices are raveled from the 2D grid of (az, el) such that as one
    increases in pixel index, elevation increments first, then azimuth.
    """

    tiling_type = SkyTilingType.RECTANGULAR  # Type of tiling of the sky

    # ======== Attributes unique to RectangularSkyMap ========
    sky_grid: spatial_utils.AzElSkyGrid
    solid_angle_grid: np.ndarray

    def __init__(
        self,
        spacing_deg: float,
        spice_frame: geometry.SpiceFrame,
    ):
        # Define the core properties of the map:
        super().__init__(spice_frame)

        # Angular spacing of the map grid (degrees) defines the number, size of pixels.
        self.sky_grid = spatial_utils.AzElSkyGrid(
            spacing_deg=spacing_deg,
        )

        self.non_spatial_coords = {}
        self.spatial_coords = {
            CoordNames.AZIMUTH_L1C.value: xr.DataArray(
                self.sky_grid.az_bin_midpoints,
                dims=[CoordNames.AZIMUTH_L1C.value],
                attrs={"UNITS": "degrees"},
            ),
            CoordNames.ELEVATION_L1C.value: xr.DataArray(
                self.sky_grid.el_bin_midpoints,
                dims=[CoordNames.ELEVATION_L1C.value],
                attrs={"UNITS": "degrees"},
            ),
        }

        # Unwrap the az, el grids to 1D array of points tiling the sky
        az_points = self.sky_grid.az_grid.ravel()
        el_points = self.sky_grid.el_grid.ravel()

        # Stack so axis 0 is different pixels, and axis 1 is (az, el) of the pixel
        self.az_el_points = np.column_stack((az_points, el_points))

        # Calculate solid angles of each pixel in the map grid in units of steradians
        self.solid_angle_grid = spatial_utils.build_solid_angle_map(
            spacing_deg=self.spacing_deg,
        )
        self.solid_angle_points = self.solid_angle_grid.ravel()

        # Initialize xarray Dataset to store map data projected from pointing sets
        self.data_1d: xr.Dataset = xr.Dataset(
            coords={
                CoordNames.GENERIC_PIXEL.value: np.arange(self.num_points),
            },
        )

    @property
    def spacing_deg(self) -> float:
        """
        Pixel spacing angle in degrees.

        Returns
        -------
        spacing_deg : float
            Spacing angle in degrees.
        """
        return self.sky_grid.spacing_deg

    @property
    def binning_grid_shape(self) -> tuple[int]:
        """
        Shape of the AzElSkyGrid.

        Returns
        -------
        binning_grid_shape : tuple[int]
            Shape of the AzElSkyGrid (num_az_bins, num_el_bins).
        """
        return self.sky_grid.grid_shape

    def to_dataset(self) -> xr.Dataset:
        """
        Get the SkyMap data as a formatted xarray.Dataset.

        Returns
        -------
        xr.Dataset
            The SkyMap data as a formatted xarray Dataset with dims and coords.
            If the SkyMap is empty, an empty xarray Dataset is returned.
            The 1D data used for projecting is rewrapped to a 2D grid of
            lon/lat (AKA az/el) coordinates.
        """
        if len(self.data_1d.data_vars) == 0:
            # If the map is empty, return an empty xarray Dataset,
            # with the unaltered spatial coords of the map
            return xr.Dataset(
                {},
                coords={**self.spatial_coords},
            )
        # Add the solid angle variable to the data_1d Dataset
        self.data_1d["solid_angle"] = xr.DataArray(
            self.solid_angle_points[np.newaxis, :].astype(np.float32),
            name="solid_angle",
            dims=[CoordNames.TIME.value, CoordNames.GENERIC_PIXEL.value],
        )
        # Rewrap each data array in the data_1d to the original 2D grid shape
        rewrapped_data = {}
        for key in self.data_1d.data_vars:
            # drop pixel dim from the end, and add the spatial coords as dims
            rewrapped_dims = [
                dim
                for dim in self.data_1d[key].dims
                if dim != CoordNames.GENERIC_PIXEL.value
            ]
            rewrapped_dims.extend(self.spatial_coords.keys())
            rewrapped_data[key] = xr.DataArray(
                spatial_utils.rewrap_even_spaced_az_el_grid(
                    self.data_1d[key].values,
                    self.binning_grid_shape,
                ),
                dims=rewrapped_dims,
            )
            # Add the output coordinates to the rewrapped data, excluding the pixel
            self.non_spatial_coords.update(
                {
                    coord: self.data_1d[key].coords[coord]
                    for coord in self.data_1d[key].coords
                    if coord != CoordNames.GENERIC_PIXEL.value
                }
            )
        return xr.Dataset(
            rewrapped_data,
            coords={**self.non_spatial_coords, **self.spatial_coords},
        )

    def build_cdf_dataset(
        self,
        instrument: str,
        level: str,
        frame: str,
        descriptor: str,
        sensor: str | None = None,
    ) -> xr.Dataset:
        """
        Format the data into a xarray.Dataset and add required CDF variables.

        Parameters
        ----------
        instrument : str
            Instrument name. "hi", "lo", "ultra".
        level : str
            Product level. "l2" or "l3".
        frame : str
            Map frame. "sf", "hf" or "hk".
        descriptor : str
            Descriptor for filename.
        sensor : str, optional
            Sensor number "45" or "90".

        Returns
        -------
        xarray.Dataset
            - Data is reshaped into expected shapes with coordinates:
            (epoch, energy, longitude, latitude)
            - Label variables are generated for coordinates other than epoch
            - Delta variables are generated and set for spatial coordinates
            - Delta plus/minus variables are generated and set to NaN for energy.
        """
        # TODO: some of this could be moved into a AbstractSkyMap method so
        #    that HealpixSkyMap would benefit from this as well.

        logger.info("Building CDF ready dataset from RectangularSkyMap data.")

        cdf_ds = self.to_dataset()

        # Set the value of the epoch coord
        cdf_ds = cdf_ds.assign_coords(**{CoordNames.TIME.value: [self.min_epoch]})

        # Drop variables dependent on dimensions not in L2 output
        # Need to iterate over CoordNames.__members__ because for an Enum, labels
        # with duplicate values are turned into aliases for the first such label.
        l2_coords = [
            coord_name.value
            for name, coord_name in CoordNames.__members__.items()
            if ("L2" in name)
        ]
        l2_coords.append(CoordNames.TIME.value)
        for map_coord in cdf_ds.dims.keys():
            if map_coord not in l2_coords:
                cdf_ds = cdf_ds.drop_dims(map_coord)

        # Add coordinate label and delta variables (not needed for epoch)
        for coord_name in l2_coords:
            # Label variable not needed for epoch
            if coord_name != CoordNames.TIME.value:
                cdf_ds[f"{coord_name}_label"] = xr.DataArray(
                    cdf_ds[coord_name].values.astype(str),
                    name=f"{coord_name}_label",
                    dims=[coord_name],
                )
            # We can set the correct delta value of the spatial coordinates
            if coord_name == CoordNames.TIME.value:
                cdf_ds[f"{coord_name}_delta"] = xr.DataArray(
                    xr.full_like(cdf_ds[coord_name], self.max_epoch - self.min_epoch),
                    name=f"{coord_name}_delta",
                    dims=[coord_name],
                )
            elif coord_name in self.spatial_coords:
                cdf_ds[f"{coord_name}_delta"] = xr.DataArray(
                    xr.full_like(cdf_ds[coord_name], self.spacing_deg / 2),
                    name=f"{coord_name}_delta",
                    dims=[coord_name],
                )
            # Add energy delta_minus and delta_plus variables
            elif coord_name == CoordNames.ENERGY_L2.value:
                cdf_ds[f"{coord_name}_delta_minus"] = xr.DataArray(
                    xr.full_like(cdf_ds[coord_name], np.nan),
                    name=f"{coord_name}_delta",
                    dims=[coord_name],
                )
                cdf_ds[f"{coord_name}_delta_plus"] = xr.DataArray(
                    xr.full_like(cdf_ds[coord_name], np.nan),
                    name=f"{coord_name}_delta",
                    dims=[coord_name],
                )

        # Object which holds CDF attributes for the map
        cdf_attrs = ImapCdfAttributes()
        cdf_attrs.add_instrument_global_attrs(instrument=instrument)
        cdf_attrs.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
        # Load rectangular map specific variable attributes
        cdf_attrs.add_instrument_variable_attrs(
            instrument="enamaps", level="l2-rectangular"
        )

        # Now set global attributes
        map_attrs = cdf_attrs.get_global_attributes(
            f"imap_{instrument}_{level}_enamap-{frame}"
        )
        map_attrs["Spacing_degrees"] = str(self.spacing_deg)
        for key in ["Data_type", "Logical_source", "Logical_source_description"]:
            map_attrs[key] = map_attrs[key].format(
                descriptor=descriptor,
                sensor=sensor,
            )
            # Always add the following attributes to the map
        map_attrs.update(
            {
                "Sky_tiling_type": self.tiling_type.value,
                "Spice_reference_frame": self.spice_reference_frame.name,
            }
        )
        cdf_ds.attrs.update(map_attrs)

        # Set the variable attributes
        for var in [*cdf_ds.data_vars, *cdf_ds.coords]:
            try:
                # Don't check schema on label or delta variables
                ignore_schema_substrings = ["_label", "_delta"]
                check_schema = (
                    False if any(s in var for s in ignore_schema_substrings) else True
                )
                var_attrs = cdf_attrs.get_variable_attributes(
                    variable_name=var,
                    check_schema=check_schema,
                )
            except KeyError as e:
                raise KeyError(
                    f"Attributes for variable {var} not found in "
                    f"loaded variable attributes."
                ) from e

            cdf_ds[var].attrs.update(var_attrs)

        return cdf_ds

    def to_properties_dict(self) -> dict:
        """
        Convert the RectangularSkyMap object to a dictionary of properties.

        Returns
        -------
        dict
            Dictionary containing the map properties.
        """
        map_properties_dict = {
            "sky_tiling_type": "RECTANGULAR",
            "spice_reference_frame": self.spice_reference_frame.name,
            "spacing_deg": self.spacing_deg,
            "values_to_push_project": self.values_to_push_project,
            "values_to_pull_project": self.values_to_pull_project,
        }
        return map_properties_dict

    def __repr__(self) -> str:
        """
        Return a string representation of the RectangularSkyMap.

        Returns
        -------
        str
            String representation of the RectangularSkyMap.
        """
        return (
            f"{self.__class__.__name__}\n\t(reference_frame="
            f"{self.spice_reference_frame.name} ({self.spice_reference_frame.value}), "
            f"spacing_deg={self.spacing_deg}, num_points={self.num_points})"
        )


class HealpixSkyMap(AbstractSkyMap):
    """
    Map which tiles the sky with a Healpix tessellation of equal-area pixels.

    Parameters
    ----------
    nside : int
        The nside parameter of the Healpix tessellation.
    spice_frame : geometry.SpiceFrame
        The reference Spice frame of the map.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.
    """

    tiling_type = SkyTilingType.HEALPIX

    # ======== Attributes unique to HealpixSkyMap ========
    nside: int
    nested: bool
    approx_resolution: float
    solid_angle: float

    def __init__(
        self, nside: int, spice_frame: geometry.SpiceFrame, nested: bool = False
    ):
        # Define the core properties of the map:
        super().__init__(spice_frame)

        # Tile the sky with a Healpix tessellation. Defined by nside, nested parameters.
        self.nside = nside
        self.nested = nested

        # Calculate the approximate resolution (deg)
        self.approx_resolution = np.rad2deg(hp.nside2resol(nside, arcmin=False))

        # The centers of each pixel in the Healpix tessellation in azimuth (az) and
        # elevation (el) coordinates (degrees) within the map's Spice frame.
        pixel_az, pixel_el = hp.pix2ang(
            nside=nside, ipix=np.arange(hp.nside2npix(nside)), nest=nested, lonlat=True
        )
        # Stack so axis 0 is different pixels, and axis 1 is (az, el) of the pixel
        self.az_el_points = np.column_stack((pixel_az, pixel_el))

        self.spatial_coords = {
            CoordNames.HEALPIX_INDEX.value: xr.DataArray(
                np.arange(self.num_points),
                dims=[CoordNames.HEALPIX_INDEX.value],
            )
        }

        # Tracks Per-Pixel Solid Angle in steradians.
        self.solid_angle = hp.nside2pixarea(nside, degrees=False)

        # Solid angle is equal at all pixels, but define
        # solid_angle_points to be consistent with RectangularSkyMap
        self.solid_angle_points = np.full(self.num_points, self.solid_angle)

        # Initialize xarray Dataset to store map data projected from pointing sets
        self.data_1d: xr.Dataset = xr.Dataset(
            coords={
                CoordNames.GENERIC_PIXEL.value: np.arange(self.num_points),
            }
        )

    @property
    def binning_grid_shape(self) -> tuple[int]:
        """
        Binning_grid_shape for consistency with RectangularSkyMap.

        Returns
        -------
        binning_grid_shape : tuple[int]
            Shape of the HealpixSkyGrid (num_points,).
        """
        return (self.num_points,)

    def to_dataset(self) -> xr.Dataset:
        """
        Get the SkyMap data as a formatted xarray Dataset.

        Returns
        -------
        xr.Dataset
            The SkyMap data as a formatted xarray Dataset with dims and coords.
            If the SkyMap is empty, an empty xarray Dataset is returned.
            Otherwise, he data is unchanged from the data_1d, but
            the pixel coordinate is renamed to CoordNames.HEALPIX_INDEX.value.
        """
        if len(self.data_1d.data_vars) == 0:
            # If the map is empty, return an empty xarray Dataset,
            # with the unaltered spatial coords of the map
            return xr.Dataset(
                {},
                coords={**self.spatial_coords},
            )
        # Add the solid angle variable to the data_1d Dataset
        self.data_1d["solid_angle"] = xr.DataArray(
            self.solid_angle_points[np.newaxis, :].astype(np.float32),
            name="solid_angle",
            dims=[CoordNames.TIME.value, CoordNames.GENERIC_PIXEL.value],
        )
        # return the data_1d as is, but with the pixel coordinate
        # renamed to CoordNames.HEALPIX_INDEX.value
        return self.data_1d.rename(
            {CoordNames.GENERIC_PIXEL.value: CoordNames.HEALPIX_INDEX.value}
        )

    # Define several methods for converting a Healpix map to a Rectangular map:
    def calculate_rect_pixel_value_from_healpix_map_n_subdivisions(
        self,
        rect_pix_center_lon_lat: np.typing.NDArray | tuple[float, float],
        rect_pix_spacing_deg: float,
        value_array: xr.DataArray,
        num_subdivisions: int,
    ) -> np.typing.NDArray:
        """
        Interpolate the value of a rectangular pixel from a healpix map w/ subdivisions.

        This function splits a single rectangular pixel into smaller subpixels
        and calculates the solid angle weighted mean value of
        the healpix map at all of the subpixel centers.

        Parameters
        ----------
        rect_pix_center_lon_lat : np.typing.NDArray | tuple[float, float]
            The center longitude and latitude of the rectangular pixel.
        rect_pix_spacing_deg : float
            The spacing of the rectangular pixel in degrees.
        value_array : xr.DataArray
            The data array containing the healpix map values.
        num_subdivisions : int
            The number of subdivisions to create for the rectangular pixel.
            The more subdivisions, the more accurate the interpolation, but also
            the more computationally expensive it is.

        Returns
        -------
        np.typing.NDArray
            The mean value of the healpix map at the subpixel centers.

            If value_array has a single value at each pixel, the output
            will be a single value, but if there are other dimensions,
            (e.g., if self.data_1d['flux'].sizes =
            {"epoch": 1, "energy": 24, "pixel": 16200}),
            the output will be an array with the same dims except the pixel dimension
            (e.g., (1, 24)).
        """
        # Assumes that you already checked the pixel doesn't fall entirely in an HP pix
        # TODO: Ask Nick if we need to add this here to mimic his code.
        # It shouldn't really be necessary, as the next function
        # get_pixel_value_recursive_subdivs will finish at 1 subdivision

        # Ensure input contains lon in the first column and lat in the second column
        rect_pix_center_lon_lat = np.array(rect_pix_center_lon_lat).reshape(-1, 2)

        # Calculate the number of subdivisions and the spacing of the subpixels
        # Then calculate the subpixel centers
        n_subpix_side = 2**num_subdivisions
        subpix_spacing = rect_pix_spacing_deg / n_subpix_side
        left_edge_lon = rect_pix_center_lon_lat[:, 0] - rect_pix_spacing_deg / 2
        bottom_edge_lat = rect_pix_center_lon_lat[:, 1] - rect_pix_spacing_deg / 2

        rect_subpix_lon_ctrs = (
            left_edge_lon
            + subpix_spacing * np.arange(n_subpix_side)
            + subpix_spacing / 2
        )
        rect_subpix_lat_ctrs = (
            bottom_edge_lat
            + subpix_spacing * np.arange(n_subpix_side)
            + subpix_spacing / 2
        )

        # We must weight by solid angle, which is not exactly equal for all subpixels
        # Calculate the solid angle of the full rectangular pixel (sterad)
        full_rect_pixel_solid_angle = np.deg2rad(rect_pix_spacing_deg) * (
            np.sin(np.deg2rad(bottom_edge_lat + rect_pix_spacing_deg))
            - np.sin(np.deg2rad(bottom_edge_lat))
        )

        # Calculate solid angle of each subpix from the rect_subpix_lat_ctrs (sterad)
        all_edges_lat = bottom_edge_lat + np.arange(n_subpix_side + 1) * subpix_spacing
        sine_all_edges_lat = np.sin(np.deg2rad(all_edges_lat))
        rect_subpix_solid_angle_by_lat = np.diff(sine_all_edges_lat) * np.deg2rad(
            subpix_spacing
        )
        rect_subpix_solid_angle_by_lat = np.repeat(
            rect_subpix_solid_angle_by_lat[np.newaxis, :], n_subpix_side, axis=0
        ).reshape(-1)

        rect_subpix_ctrs = (
            np.array(
                np.meshgrid(rect_subpix_lon_ctrs, rect_subpix_lat_ctrs, indexing="ij")
            )
            .reshape(2, -1)
            .T
        )

        # Get the healpix pixel indices at the rectangular subpixel centers
        hp_pix_at_rect_subpix_ctrs = hp.ang2pix(
            nside=self.nside,
            nest=self.nested,
            theta=rect_subpix_ctrs[:, 0],
            phi=rect_subpix_ctrs[:, 1],
            lonlat=True,
        )
        # Get the healpix values at the rectangular subpixel centers
        hp_vals_at_rect_pix_ctrs = value_array.values[..., hp_pix_at_rect_subpix_ctrs]

        # Weighted mean (weighted by solid angle) of these values over the pixel axis,
        # which is the last axis of this array
        weighted_hp_vals_at_rect_pix_ctrs = (
            hp_vals_at_rect_pix_ctrs * rect_subpix_solid_angle_by_lat
        )
        mean_pixel_value = (
            weighted_hp_vals_at_rect_pix_ctrs.sum(axis=-1) / full_rect_pixel_solid_angle
        )
        # Log the mean pixel value and the number of subdivisions for debugging
        logger.debug(
            f"    Mean pixel value at Number of subdivisions: {num_subdivisions}: "
            f"array of shape {mean_pixel_value.shape}: {mean_pixel_value}"
        )
        return mean_pixel_value

    def get_rect_pixel_value_recursive_subdivs(
        self,
        rect_pix_center_lon_lat: np.typing.NDArray | tuple[float, float],
        rect_pix_spacing_deg: float,
        value_array: xr.DataArray,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-12,
        max_subdivision_depth: int = MAX_SUBDIV_RECURSION_DEPTH,
    ) -> tuple[np.typing.NDArray, int]:
        """
        Recursively subdivide a rectangular pixel to get a mean value within tolerances.

        Takes a rectangular pixel, and recursively breaks it up into
        smaller and smaller subpixels, then calculates the solid-angle weighted mean
        of the healpix map's value at this pixel, until the difference
        between the mean values of two consecutive subdivisions is within the
        specified tolerances at all values in that pixel (e.g., all energy bins).
        The function returns the mean value at the final level
        of subdivision and the depth of recursion.

        Parameters
        ----------
        rect_pix_center_lon_lat : np.typing.NDArray | tuple[float, float]
            The center longitude and latitude of the rectangular pixel.
        rect_pix_spacing_deg : float
            The spacing of the rectangular pixel in degrees.
        value_array : xr.DataArray
            The data array containing the healpix map values to interpolate from.
        rtol : float, optional
            The relative tolerance for convergence, by default 1e-3.
        atol : float, optional
            The absolute tolerance for convergence, by default 1e-12.
        max_subdivision_depth : int, optional
            The maximum depth of recursion for subdivision,
            by default MAX_SUBDIV_RECURSION_DEPTH.
            Computation grows exponentially with depth, but only where the value
            has a significant gradient between adjacent healpix pixels.
            If the value is smooth, the recursion depth will be low.

        Returns
        -------
        tuple[list[float], int]
            The mean value at the final level of subdivision and the depth of recursion.
        """
        # Recursively subdivide a pixel and calculate its mean value until either the
        # difference between consecutive levels is within the specified tolerances
        # or the maximum recursion depth is reached
        depth = 0
        previous_mean_pixel_value: NDArray = np.full((1,), np.nan)
        while depth < max_subdivision_depth:
            mean_pixel_value = (
                self.calculate_rect_pixel_value_from_healpix_map_n_subdivisions(
                    rect_pix_center_lon_lat=rect_pix_center_lon_lat,
                    rect_pix_spacing_deg=rect_pix_spacing_deg,
                    value_array=value_array,
                    num_subdivisions=depth,
                )
            )

            # Determine if tolerance is met,
            # (skip on the 0th iteration, as there's no delta to compare).

            # Note:  Using allclose() means that, while we calculate the mean pixel
            # value(s) over  all the subdivided subpixels, this mean may have multiple
            # values (e.g., different energy bins), so we check if
            # all of them are within the tolerances. This can result in
            # over-subdividing some energy bins, where there is little spatial gradient,
            # but where another energy bin has a large gradient.
            if depth > 0:
                if np.allclose(
                    mean_pixel_value,
                    previous_mean_pixel_value,
                    rtol=rtol,
                    atol=atol,
                ):
                    break
            depth += 1
            previous_mean_pixel_value = mean_pixel_value

        logger.debug(
            f"Pixel at ({rect_pix_center_lon_lat} deg size={rect_pix_spacing_deg} deg,)"
            f" converged to {mean_pixel_value.mean()} in {depth} subdivisions."
            f" Previous mean was {previous_mean_pixel_value.mean()}."
        )
        # Only keep the last (best) mean pixel value
        return mean_pixel_value, depth

    def to_rectangular_skymap(
        self,
        rect_spacing_deg: float,
        value_keys: list[str],
        max_subdivision_depth: int = MAX_SUBDIV_RECURSION_DEPTH,
    ) -> tuple[RectangularSkyMap, dict[str, np.typing.NDArray]]:
        """
        Interpolate a healpix map to a rectangular map using recursive subdivision.

        Parameters
        ----------
        rect_spacing_deg : float
            The spacing of the rectangular map in degrees.
        value_keys : list[str]
            The names of the values to interpolate from the healpix map.
            Each must be independently interpolated because the subdivision depth
            depends on the gradient of the value between adjacent healpix pixels.
        max_subdivision_depth : int, optional
            The maximum depth of recursion for subdivision,
            by default MAX_SUBDIV_RECURSION_DEPTH.

        Returns
        -------
        tuple[RectangularSkyMap, dict[str, np.typing.NDArray]]
            A RectangularSkyMap containing the interpolated values, and a dictionary of
            each value and its corresponding subdivision depth by pixel.
        """
        # Begin by defining the rectangular map we want to create, which must be
        # in the same spice reference frame as the healpix map
        rect_map = RectangularSkyMap(
            spacing_deg=rect_spacing_deg,
            spice_frame=self.spice_reference_frame,
        )

        # Depending on the maximum recursion depth, the number of pixels in the
        # RectangularSkyMap, and the number of value keys, and especially on the
        # gradients of the values, the number of operations can be very large, so
        # log key information about the expected number of operations.
        approx_max_operations = (
            (4**max_subdivision_depth) * self.num_points * len(value_keys)
        )
        logger.info(
            f"Converting from a HealpixSkyMap(nside={self.nside}) to a "
            f"RectangularSkyMap(spacing_deg={rect_spacing_deg}) with recursive "
            "subdivision.\n The maximum recursion depth is "
            f"{max_subdivision_depth}, yielding a maximum number of healpix calls"
            f" of {approx_max_operations:.3e}."
        )

        # Dict to hold the subdivision depth by pixel for each value key
        subdiv_depth_dict = {}
        for value_key in value_keys:
            # For each of the values, calculate each pixel's value with
            # recursive subdivision. Unfortunately, this must be done independently
            # for each value key.

            # Yields a list of tuple (mean_value, depth) for each pixel in the map
            healpix_values_array = self.data_1d[value_key]
            best_value_and_recursion_depth_by_pixel = [
                self.get_rect_pixel_value_recursive_subdivs(
                    rect_pix_center_lon_lat=lon_lat,
                    rect_pix_spacing_deg=rect_map.spacing_deg,
                    value_array=healpix_values_array,
                    max_subdivision_depth=max_subdivision_depth,
                )
                for lon_lat in rect_map.az_el_points
            ]

            # Separate the best value and the recursion depth for each pixel
            # into two lists, then convert both to numpy arrays
            # and move the pixel dim to the last dim of values
            interpolated_data_by_rect_pixel, subdiv_depth_of_value_by_pixel = zip(
                *best_value_and_recursion_depth_by_pixel
            )
            interpolated_data_by_rect_pixel = np.moveaxis(
                np.array(interpolated_data_by_rect_pixel), 0, -1
            )
            subdiv_depth_of_value_by_pixel = np.array(subdiv_depth_of_value_by_pixel)

            # This can introduce an extra dim as the last dim of the array
            # to values with only one dimension
            if len(healpix_values_array.dims) == 1:
                interpolated_data_by_rect_pixel = np.squeeze(
                    interpolated_data_by_rect_pixel,
                )

            # Store the best value(s) of each pixel in the rectangular map with the
            # leading coordinates of the healpix map, and the pixel coordinate last
            rect_map.data_1d[value_key] = xr.DataArray(
                data=interpolated_data_by_rect_pixel,
                dims=(*healpix_values_array.dims[:-1], CoordNames.GENERIC_PIXEL.value),
            )

            # Update the coordinates of the rectangular map with any new coordinates
            # from the healpix map except the pixel coord,
            # which will be different in the rectangular map.
            for coord in healpix_values_array.coords:
                if coord not in (
                    CoordNames.GENERIC_PIXEL.value,
                    CoordNames.HEALPIX_INDEX.value,
                ):
                    rect_map.data_1d.coords[coord] = healpix_values_array.coords[coord]

            # Add the subdivision depth by pixel of this value_key to the dictionary
            # This may be necessary for uncertainty estimation
            subdiv_depth_dict[value_key] = subdiv_depth_of_value_by_pixel
            logger.info(
                f"Summary of subdivision depth for {value_key}:\n"
                "Mean +/- std number of subdivisions for the "
                f"{rect_map.num_points} pixels of {value_key} is:\n"
                f"    {np.mean(subdiv_depth_of_value_by_pixel):.6f}."
                f" +/- {np.std(subdiv_depth_of_value_by_pixel):.6f}.\n"
                "Min / Max number of subdivisions: \n"
                f"    {np.min(subdiv_depth_of_value_by_pixel):.6f} / "
                f"{np.max(subdiv_depth_of_value_by_pixel):.6f}.\n"
                f"The maximum allowed depth is {max_subdivision_depth}."
            )

        return rect_map, subdiv_depth_dict

    def to_properties_dict(self) -> dict:
        """
        Convert the HealpixSkyMap object to a dictionary of properties.

        Returns
        -------
        dict
            Dictionary containing the map properties.
        """
        map_properties_dict = {
            "sky_tiling_type": "HEALPIX",
            "spice_reference_frame": self.spice_reference_frame.name,
            "nside": self.nside,
            "nested": self.nested,
            "values_to_push_project": self.values_to_push_project,
            "values_to_pull_project": self.values_to_pull_project,
        }
        return map_properties_dict

    def __repr__(self) -> str:
        """
        Return a string representation of the HealpixSkyMap.

        Returns
        -------
        str
            String representation of the HealpixSkyMap.
        """
        return (
            f"{self.__class__.__name__}\n\t(reference_frame="
            f"{self.spice_reference_frame.name} ({self.spice_reference_frame.value}), "
            f"nside={self.nside}, num_points={self.num_points})"
        )
