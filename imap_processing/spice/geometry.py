"""
Functions for computing geometry, many of which use SPICEYPY.

Paradigms for developing this module:

* Use @ensure_spice decorator on functions that directly wrap spiceypy functions
* Vectorize everything at the lowest level possible (e.g. the decorated spiceypy
  wrapper function)
* Always return numpy arrays for vectorized calls.
"""

import typing
from enum import IntEnum
from typing import Union

import numpy as np
import numpy.typing as npt
import spiceypy
from numpy.typing import NDArray

from imap_processing.spice.kernels import ensure_spice


class SpiceBody(IntEnum):
    """Enum containing SPICE IDs for bodies that we use."""

    # A subset of IMAP Specific bodies as defined in imap_wkcp.tf
    IMAP = -43
    IMAP_SPACECRAFT = -43000
    # IMAP Pointing Frame (Despun) as defined in imap_science_0001.tf
    IMAP_DPS = -43901
    # Standard NAIF bodies
    SOLAR_SYSTEM_BARYCENTER = spiceypy.bodn2c("SOLAR_SYSTEM_BARYCENTER")
    SUN = spiceypy.bodn2c("SUN")
    EARTH = spiceypy.bodn2c("EARTH")


class SpiceFrame(IntEnum):
    """Enum containing SPICE IDs for reference frames, defined in imap_wkcp.tf."""

    # Standard SPICE Frames
    J2000 = spiceypy.irfnum("J2000")
    ECLIPJ2000 = spiceypy.irfnum("ECLIPJ2000")
    ITRF93 = 13000
    # IMAP Pointing Frame (Despun) as defined in imap_science_0001.tf
    IMAP_DPS = -43901
    # IMAP specific as defined in imap_wkcp.tf
    IMAP_SPACECRAFT = -43000
    IMAP_LO_BASE = -43100
    IMAP_LO_STAR_SENSOR = -43103
    IMAP_LO = -43105
    IMAP_HI_45 = -43150
    IMAP_HI_90 = -43160
    IMAP_ULTRA_45 = -43200
    IMAP_ULTRA_90 = -43210
    IMAP_MAG = -43250
    IMAP_SWE = -43300
    IMAP_SWAPI = -43350
    IMAP_CODICE = -43400
    IMAP_HIT = -43500
    IMAP_IDEX = -43700
    IMAP_GLOWS = -43750


BORESIGHT_LOOKUP = {
    SpiceFrame.IMAP_LO_BASE: np.array([0, -1, 0]),
    SpiceFrame.IMAP_HI_45: np.array([0, 1, 0]),
    SpiceFrame.IMAP_HI_90: np.array([0, 1, 0]),
    SpiceFrame.IMAP_ULTRA_45: np.array([0, 0, 1]),
    SpiceFrame.IMAP_ULTRA_90: np.array([0, 0, 1]),
    SpiceFrame.IMAP_MAG: np.array([0, 0, 1]),
    SpiceFrame.IMAP_SWE: np.array([-1, 0, 0]),
    SpiceFrame.IMAP_SWAPI: np.array([0, 1, 0]),
    SpiceFrame.IMAP_CODICE: np.array([0, 0, 1]),
    SpiceFrame.IMAP_HIT: np.array([0, 1, 0]),
    SpiceFrame.IMAP_IDEX: np.array([0, 1, 0]),
    SpiceFrame.IMAP_GLOWS: np.array([0, 0, -1]),
}


@typing.no_type_check
@ensure_spice
def imap_state(
    et: Union[np.ndarray, float],
    ref_frame: SpiceFrame = SpiceFrame.ECLIPJ2000,
    abcorr: str = "NONE",
    observer: SpiceBody = SpiceBody.SUN,
) -> np.ndarray:
    """
    Get the state (position and velocity) of the IMAP spacecraft.

    By default, the state is returned in the ECLIPJ2000 frame as observed by the Sun.

    Parameters
    ----------
    et : np.ndarray or float
        Epoch time(s) [J2000 seconds] to get the IMAP state for.
    ref_frame : SpiceFrame (Optional)
        Reference frame which the IMAP state is expressed in. Default is
        SpiceFrame.ECLIPJ2000.
    abcorr : str (Optional)
        Aberration correction flag. Default is "NONE".
    observer : SpiceBody (Optional)
        Observing body. Default is SpiceBody.SUN.

    Returns
    -------
    state : np.ndarray
     The Cartesian state vector representing the position and velocity of the
     IMAP spacecraft.
    """
    state, _ = spiceypy.spkezr(
        SpiceBody.IMAP.name, et, ref_frame.name, abcorr, observer.name
    )
    return np.asarray(state)


def get_spacecraft_to_instrument_spin_phase_offset(instrument: SpiceFrame) -> float:
    """
    Get the spin phase offset from the spacecraft to the instrument.

    For now, the offset is a fixed lookup based on `Table 1: Nominal Instrument
    to S/C CS Transformations` in document `7516-0011_drw.pdf`. These fixed
    values will need to be updated based on calibration data or retrieved using
    SPICE and the latest IMAP frame kernel.

    Parameters
    ----------
    instrument : SpiceFrame
        Instrument to get the spin phase offset for.

    Returns
    -------
    spacecraft_to_instrument_spin_phase_offset : float
        The spin phase offset from the spacecraft to the instrument.
    """
    # TODO: Implement retrieval from SPICE?
    offset_lookup = {
        SpiceFrame.IMAP_LO_BASE: 330 / 360,
        SpiceFrame.IMAP_HI_45: 255 / 360,
        SpiceFrame.IMAP_HI_90: 285 / 360,
        SpiceFrame.IMAP_ULTRA_45: 33 / 360,
        SpiceFrame.IMAP_ULTRA_90: 210 / 360,
        SpiceFrame.IMAP_SWAPI: 168 / 360,
        SpiceFrame.IMAP_IDEX: 90 / 360,
        SpiceFrame.IMAP_CODICE: 136 / 360,
        SpiceFrame.IMAP_HIT: 30 / 360,
        SpiceFrame.IMAP_SWE: 153 / 360,
        SpiceFrame.IMAP_GLOWS: 127 / 360,
        SpiceFrame.IMAP_MAG: 0 / 360,
    }
    return offset_lookup[instrument]


def frame_transform(
    et: Union[float, npt.NDArray],
    position: npt.NDArray,
    from_frame: SpiceFrame,
    to_frame: SpiceFrame,
) -> npt.NDArray:
    """
    Transform an <x, y, z> vector between reference frames (rotation only).

    This function is a vectorized equivalent to performing the following SPICE
    calls for each input time and position vector to perform the transform.
    The matrix multiplication step is done using `numpy.matmul` rather than
    `spiceypy.mxv`.
    >>> rotation_matrix = spiceypy.pxform(from_frame, to_frame, et)
    ... result = spiceypy.mxv(rotation_matrix, position)

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) corresponding to position(s).
    position : np.ndarray
        <x, y, z> vector or array of vectors in reference frame `from_frame`.
        There are several possible shapes for the input position and et:
        1. A single position vector may be provided for multiple `et` query times
        2. A single `et` may be provided for multiple position vectors,
        3. The same number of `et` and position vectors may be provided.
        But it is not allowed to have n position vectors and m `et`, where n != m.
    from_frame : SpiceFrame
        Reference frame of input vector(s).
    to_frame : SpiceFrame
        Reference frame of output vector(s).

    Returns
    -------
    result : np.ndarray
        3d Cartesian position vector(s) in reference frame `to_frame`.
    """
    # If from_frame and to_frame are the same, no rotation needed
    if from_frame == to_frame:
        return position

    if position.ndim == 1:
        if not len(position) == 3:
            raise ValueError(
                "Position vectors with one dimension must have 3 elements."
            )
    elif position.ndim == 2:
        if not position.shape[1] == 3:
            raise ValueError(
                f"Invalid position shape: {position.shape}. "
                f"Each input position vector must have 3 elements."
            )
        if not len(position) == np.asarray(et).size:
            if np.asarray(et).size != 1:
                raise ValueError(
                    "Mismatch in number of position vectors and "
                    "Ephemeris times provided."
                    f"Position has {len(position)} elements and et has "
                    f"{np.asarray(et).size} elements."
                )

    # rotate will have shape = (3, 3) or (n, 3, 3)
    # position will have shape = (3,) or (n, 3)
    rotate = get_rotation_matrix(et, from_frame, to_frame)
    # adding a dimension to position results in the following input and output
    # shapes from matrix multiplication
    # Single et/position:      (3, 3),(3, 1) -> (3, 1)
    # Multiple et single pos:  (n, 3, 3),(3, 1) -> (n, 3, 1)
    # Multiple et/positions :  (n, 3, 3),(n, 3, 1) -> (n, 3, 1)
    result = np.squeeze(rotate @ position[..., np.newaxis])

    return result


def frame_transform_az_el(
    et: Union[float, npt.NDArray],
    az_el: npt.NDArray,
    from_frame: SpiceFrame,
    to_frame: SpiceFrame,
    degrees: bool = True,
) -> npt.NDArray:
    """
    Transform azimuth and elevation coordinates between reference frames.

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) corresponding to position(s).
    az_el :  np.ndarray
        <azimuth, elevation> vector or array of vectors in reference frame `from_frame`.
        There are several possible shapes for the input az_el and et:
        1. A single az_el vector may be provided for multiple `et` query times
        2. A single `et` may be provided for multiple az_el vectors,
        3. The same number of `et` and az_el vectors may be provided.
        It is not allowed to have n az_el vectors and m `et`, where n != m.
    from_frame : SpiceFrame
        Reference frame of input coordinates.
    to_frame : SpiceFrame
        Reference frame of output coordinates.
    degrees : bool
        If True, azimuth and elevation input and output will be in degrees.

    Returns
    -------
    to_frame_az_el : np.ndarray
        Azimuth/elevation coordinates in reference frame `to_frame`. This
        output coordinate vector will have shape (2,) if a single `az_el` position
        vector and single `et` time are input. Otherwise, it will have shape (n, 2)
        where n is the number of input position vector or ephemeris times. The last
        axis of the output vector contains azimuth in the 0th position and elevation
        in the 1st position.
    """
    # Convert input az/el to Cartesian vectors
    spherical_coords_in = np.array(
        [np.ones_like(az_el[..., 0]), az_el[..., 0], az_el[..., 1]]
    ).T
    from_frame_cartesian = spherical_to_cartesian(spherical_coords_in)
    # Transform to to_frame
    to_frame_cartesian = frame_transform(et, from_frame_cartesian, from_frame, to_frame)
    # Convert to spherical and extract azimuth/elevation
    to_frame_az_el = cartesian_to_spherical(to_frame_cartesian, degrees=degrees)
    return to_frame_az_el[..., 1:3]


@typing.no_type_check
@ensure_spice
def get_rotation_matrix(
    et: Union[float, npt.NDArray],
    from_frame: SpiceFrame,
    to_frame: SpiceFrame,
) -> npt.NDArray:
    """
    Get the rotation matrix/matrices that can be used to transform between frames.

    This is a vectorized wrapper around `spiceypy.pxform`
    "Return the matrix that transforms position vectors from one specified frame
    to another at a specified epoch."
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pxform_c.html

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) for which to get the rotation matrices.
    from_frame : SpiceFrame
        Reference frame to transform from.
    to_frame : SpiceFrame
        Reference frame to transform to.

    Returns
    -------
    rotation : np.ndarray
        If `et` is a float, the returned rotation matrix is of shape `(3, 3)`. If
        `et` is a np.ndarray, the returned rotation matrix is of shape `(n, 3, 3)`
        where `n` matches the number of elements in et.
    """
    vec_pxform = np.vectorize(
        spiceypy.pxform,
        excluded=["fromstr", "tostr"],
        signature="(),(),()->(3,3)",
        otypes=[np.float64],
    )
    return vec_pxform(from_frame.name, to_frame.name, et)


def instrument_pointing(
    et: Union[float, npt.NDArray],
    instrument: SpiceFrame,
    to_frame: SpiceFrame,
    cartesian: bool = False,
) -> npt.NDArray:
    """
    Compute the instrument pointing at the specified times.

    By default, the coordinates returned are (Longitude, Latitude) coordinates in
    the reference frame `to_frame`. Cartesian coordinates can be returned if
    desired by setting `cartesian=True`.

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) to at which to compute instrument pointing.
    instrument : SpiceFrame
        Instrument reference frame to compute the pointing for.
    to_frame : SpiceFrame
        Reference frame in which the pointing is to be expressed.
    cartesian : bool
        If set to True, the pointing is returned in Cartesian coordinates.
        Defaults to False.

    Returns
    -------
    pointing : np.ndarray
        The instrument pointing at the specified times.
    """
    pointing = frame_transform(et, BORESIGHT_LOOKUP[instrument], instrument, to_frame)
    if cartesian:
        return pointing
    if isinstance(et, typing.Collection):
        return np.rad2deg([spiceypy.reclat(vec)[1:] for vec in pointing])
    return np.rad2deg(spiceypy.reclat(pointing)[1:])


def basis_vectors(
    et: Union[float, npt.NDArray],
    from_frame: SpiceFrame,
    to_frame: SpiceFrame,
) -> npt.NDArray:
    """
    Get the basis vectors of the `from_frame` expressed in the `to_frame`.

    The rotation matrix defining a frame transform are the basis vectors of
    the `from_frame` expressed in the `to_frame`. This function just transposes
    each rotation matrix retrieved from the `get_rotation_matrix` function so
    that basis vectors can be indexed 0 for x, 1 for y, and 2 for z.

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) for which to get the rotation matrices.
    from_frame : SpiceFrame
        Reference frame to transform from.
    to_frame : SpiceFrame
        Reference frame to transform to.

    Returns
    -------
    basis_vectors : np.ndarray
        If `et` is a float, the returned basis vector matrix is of shape `(3, 3)`. If
        `et` is a np.ndarray, the returned basis vector matrix is of shape `(n, 3, 3)`
        where `n` matches the number of elements in et and basis vectors are the rows
        of the 3 by 3 matrices.

    Examples
    --------
    >>> from imap_processing.spice.geometry import basis_vectors
    ... from imap_processing.spice.time import ttj2000ns_to_et
    ... et = ttj2000ns_to_et(dataset.epoch.values)
    ... basis_vectors = basis_vectors(
    ...     et, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.ECLIPJ2000
    ... )
    ... spacecraft_x = basis_vectors[:, 0]
    ... spacecraft_y = basis_vectors[:, 1]
    ... spacecraft_z = basis_vectors[:, 2]
    """
    return np.moveaxis(get_rotation_matrix(et, from_frame, to_frame), -1, -2)


def cartesian_to_spherical(
    v: NDArray,
    degrees: bool = True,
) -> NDArray:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    v : np.ndarray
        A NumPy array with shape (n, 3) where each
        row represents a vector
        with x, y, z-components.
    degrees : bool
        If True, the azimuth and elevation angles are returned in degrees.
        Defaults to True.

    Returns
    -------
    spherical_coords : np.ndarray
        A NumPy array with shape (n, 3), where each row contains
        the spherical coordinates (r, azimuth, elevation):

        - r : Distance of the point from the origin.
        - azimuth : angle in the xy-plane
          In degrees if degrees parameter is True (by default):
          output range=[0, 360) degrees,
          otherwise in radians if degrees parameter is False:
          output range=[0, 2*pi) radians.
        - elevation : angle from the xy-plane
          In degrees if degrees parameter is True (by default):
          output range=[-90, 90) degrees,
          otherwise in radians if degrees parameter is False:
          output range=[-pi/2, pi/2) radians.
    """
    # Magnitude of the velocity vector
    magnitude_v = np.linalg.norm(v, axis=-1, keepdims=True)

    vhat = v / magnitude_v

    # Elevation angle (angle from the xy-plane, range: [-pi/2, pi/2])
    el = np.arcsin(vhat[..., 2])

    # Azimuth angle (angle in the xy-plane, range: [0, 2*pi])
    az = np.arctan2(vhat[..., 1], vhat[..., 0])

    # Ensure azimuth is from 0 to 2PI
    az = az % (2 * np.pi)

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    spherical_coords = np.stack((np.squeeze(magnitude_v), az, el), axis=-1)

    return spherical_coords


def spherical_to_cartesian(spherical_coords: NDArray) -> NDArray:
    """
    Convert spherical coordinates (angles in degrees) to Cartesian coordinates.

    Parameters
    ----------
    spherical_coords : np.ndarray
        A NumPy array with shape (n, 3), where each row contains
        the spherical coordinates (r, azimuth, elevation):

        - r : Distance of the point from the origin.
        - azimuth : angle in the xy-plane in degrees. Range is [0, 360) degrees.
        - elevation : angle from the xy-plane in degrees. Range is [-90, 90) degrees.

    Returns
    -------
    cartesian_coords : np.ndarray
        Cartesian coordinates.
    """
    r = spherical_coords[..., 0]
    azimuth = spherical_coords[..., 1]
    elevation = spherical_coords[..., 2]

    # Convert to radians for numpy trigonometric operations
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    cartesian_coords = np.stack((x, y, z), axis=-1)

    return cartesian_coords


def cartesian_to_latitudinal(coords: NDArray, degrees: bool = True) -> NDArray:
    """
    Convert cartesian coordinates to latitudinal coordinates in radians.

    This is a vectorized wrapper around `spiceypy.reclat`
    "Convert from rectangular coordinates to latitudinal coordinates."
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/reclat_c.html

    Parameters
    ----------
    coords : np.ndarray
        Either shape (n, 3) or (3) where the last dimension represents a vector
        with x, y, z-components.
    degrees : bool
        If True, the longitude and latitude coords are returned in degrees.
        Defaults to True.

    Returns
    -------
    np.ndarray
        A NumPy array with shape (n, 3) or (3), where the last dimension contains
        the latitudinal coordinates (radius, longitude, latitude).
    """
    # If coords is 1d, add another dimension
    while coords.ndim < 2:
        coords = np.expand_dims(coords, axis=0)
    latitudinal_coords = np.array([spiceypy.reclat(vec) for vec in coords])

    if degrees:
        latitudinal_coords[..., 1:] = np.degrees(latitudinal_coords[..., 1:])
    # Return array of latitudinal and remove the first dimension if it is 1.
    return np.squeeze(latitudinal_coords)


def solar_longitude(
    et: Union[np.ndarray, float],
    degrees: bool = True,
) -> Union[float, npt.NDArray]:
    """
    Compute the solar longitude of the Imap Spacecraft.

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) to at which to compute solar longitude.
    degrees : bool
        If True, the longitude is returned in degrees.
        Defaults to True.

    Returns
    -------
    float or np.ndarray
        The solar longitude at the specified times.
    """
    # Get position of IMAP in ecliptic frame
    imap_pos = imap_state(et, observer=SpiceBody.SUN)[..., 0:3]
    lat_coords = cartesian_to_latitudinal(imap_pos, degrees=degrees)[..., 1]

    return float(lat_coords) if lat_coords.size == 1 else lat_coords
