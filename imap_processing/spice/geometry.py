"""
Functions for computing geometry, many of which use SPICE.

Paradigms for developing this module:

* Use @ensure_spice decorator on functions that directly wrap spiceypy functions
* Vectorize everything at the lowest level possible (e.g. the decorated spiceypy
  wrapper function)
* Always return numpy arrays for vectorized calls.
"""

import os
import typing
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import spiceypy as spice
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
    SOLAR_SYSTEM_BARYCENTER = spice.bodn2c("SOLAR_SYSTEM_BARYCENTER")
    SUN = spice.bodn2c("SUN")
    EARTH = spice.bodn2c("EARTH")


class SpiceFrame(IntEnum):
    """Enum containing SPICE IDs for reference frames, defined in imap_wkcp.tf."""

    # Standard SPICE Frames
    J2000 = spice.irfnum("J2000")
    ECLIPJ2000 = spice.irfnum("ECLIPJ2000")
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


# TODO: Update boresight for in-situ instruments
# TODO: Confirm ENA boresight vectors
BORESIGHT_LOOKUP = {
    SpiceFrame.IMAP_LO: np.array([0, -1, 0]),
    SpiceFrame.IMAP_HI_45: np.array([0, 1, 0]),
    SpiceFrame.IMAP_HI_90: np.array([0, 1, 0]),
    SpiceFrame.IMAP_ULTRA_45: np.array([0, 0, 1]),
    SpiceFrame.IMAP_ULTRA_90: np.array([0, 0, 1]),
    SpiceFrame.IMAP_MAG: np.array([0, 0, 1]),
    SpiceFrame.IMAP_SWE: np.array([0, 0, 1]),
    SpiceFrame.IMAP_SWAPI: np.array([0, 0, 1]),
    SpiceFrame.IMAP_CODICE: np.array([0, 0, 1]),
    SpiceFrame.IMAP_HIT: np.array([0, 0, 1]),
    SpiceFrame.IMAP_IDEX: np.array([0, 0, 1]),
    SpiceFrame.IMAP_GLOWS: np.array([0, 0, 1]),
}


@typing.no_type_check
@ensure_spice
def imap_state(
    et: Union[np.ndarray, float],
    ref_frame: SpiceFrame = SpiceFrame.ECLIPJ2000,
    observer: SpiceBody = SpiceBody.SUN,
) -> np.ndarray:
    """
    Get the state (position and velocity) of the IMAP spacecraft.

    By default, the state is returned in the ECLIPJ2000 frame as observed by the Sun.

    Parameters
    ----------
    et : np.ndarray or float
        Epoch time(s) [J2000 seconds] to get the IMAP state for.
    ref_frame : SpiceFrame
        Reference frame which the IMAP state is expressed in. Default is
        SpiceFrame.ECLIPJ2000.
    observer : SpiceBody
        Observing body. Default is SpiceBody.SUN.

    Returns
    -------
    state : np.ndarray
     The Cartesian state vector representing the position and velocity of the
     IMAP spacecraft.
    """
    state, _ = spice.spkezr(
        SpiceBody.IMAP.name, et, ref_frame.name, "NONE", observer.name
    )
    return np.asarray(state)


def get_spin_data() -> pd.DataFrame:
    """
    Read spin file using environment variable and return spin data.

    SPIN_DATA_FILEPATH environment variable would be a fixed value.
    It could be s3 filepath that can be used to download the data
    through API or it could be path EFS or Batch volume mount path.

    Spin data should contain the following fields:
        * spin_number
        * spin_start_sec
        * spin_start_subsec
        * spin_period_sec
        * spin_period_valid
        * spin_phase_valid
        * spin_period_source
        * thruster_firing

    Returns
    -------
    spin_data : pandas.DataFrame
        Spin data.
    """
    spin_data_filepath = os.getenv("SPIN_DATA_FILEPATH")
    if spin_data_filepath is not None:
        path_to_spin_file = Path(spin_data_filepath)
    else:
        # Handle the case where the environment variable is not set
        raise ValueError("SPIN_DATA_FILEPATH environment variable is not set.")

    spin_df = pd.read_csv(path_to_spin_file, comment="#")
    # Combine spin_start_sec and spin_start_subsec to get the spin start
    # time in seconds. The spin start subseconds are in milliseconds.
    spin_df["spin_start_time"] = (
        spin_df["spin_start_sec"] + spin_df["spin_start_subsec"] / 1e3
    )

    return spin_df


def get_spacecraft_spin_phase(
    query_met_times: Union[float, npt.NDArray],
) -> Union[float, npt.NDArray]:
    """
    Get the spacecraft spin phase for the input query times.

    Formula to calculate spin phase:
        spin_phase = (query_met_times - spin_start_time) / spin_period_sec

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = get_spin_data()

    # Ensure query_met_times is an array
    query_met_times = np.asarray(query_met_times)
    is_scalar = query_met_times.ndim == 0
    if is_scalar:
        # Force scalar to array because np.asarray() will not
        # convert scalar to array
        query_met_times = np.atleast_1d(query_met_times)
    # Empty array check
    if query_met_times.size == 0:
        return query_met_times

    # Create an empty array to store spin phase results
    spin_phases = np.zeros_like(query_met_times)

    # Find all spin time that are less or equal to query_met_times.
    # To do that, use side right, a[i-1] <= v < a[i], in the searchsorted.
    # Eg.
    # >>> df['a']
    # array([0, 15, 30, 45, 60])
    # >>> np.searchsorted(df['a'], [0, 13, 15, 32, 70], side='right')
    # array([1, 1, 2, 3, 5])
    last_spin_indices = (
        np.searchsorted(spin_df["spin_start_time"], query_met_times, side="right") - 1
    )
    # Make sure input times are within the bounds of spin data
    spin_df_start_time = spin_df["spin_start_time"].values[0]
    spin_df_end_time = (
        spin_df["spin_start_time"].values[-1] + spin_df["spin_period_sec"].values[-1]
    )
    input_start_time = query_met_times.min()
    input_end_time = query_met_times.max()
    if input_start_time < spin_df_start_time or input_end_time >= spin_df_end_time:
        raise ValueError(
            f"Query times, {query_met_times} are outside of the spin data range, "
            f"{spin_df_start_time, spin_df_end_time}."
        )

    # Calculate spin phase
    spin_phases = (
        query_met_times - spin_df["spin_start_time"].values[last_spin_indices]
    ) / spin_df["spin_period_sec"].values[last_spin_indices]

    # Check for invalid spin phase using below checks:
    # 1. Check that the spin phase is in valid range, [0, 1).
    # 2. Check invalid spin phase using spin_phase_valid,
    #   spin_period_valid columns.
    invalid_spin_phase_range = (spin_phases < 0) | (spin_phases >= 1)

    invalid_spins = (spin_df["spin_phase_valid"].values[last_spin_indices] == 0) | (
        spin_df["spin_period_valid"].values[last_spin_indices] == 0
    )
    bad_spin_phases = invalid_spin_phase_range | invalid_spins
    spin_phases[bad_spin_phases] = np.nan

    if is_scalar:
        return spin_phases[0]
    return spin_phases


def get_instrument_spin_phase(
    query_met_times: Union[float, npt.NDArray],
    instrument: SpiceFrame,
) -> Union[float, npt.NDArray]:
    """
    Get the instrument spin phase for the input query times.

    Formula to calculate spin phase:
        instrument_spin_phase = (spacecraft_spin_phase + instrument_spin_offset) % 1

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    instrument : SpiceFrame
        Instrument frame to calculate spin phase for.

    Returns
    -------
    spin_phase : float or np.ndarray
        Instrument spin phase for the input query times. Spin phase is a
        floating point number in the range [0, 1) corresponding to the
        spin angle / 360.
    """
    spacecraft_spin_phase = get_spacecraft_spin_phase(query_met_times)
    instrument_spin_phase_offset = get_spacecraft_to_instrument_spin_phase_offset(
        instrument
    )
    return (spacecraft_spin_phase + instrument_spin_phase_offset) % 1


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
        SpiceFrame.IMAP_LO: 330 / 360,
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
    `spice.mxv`.
    >>> rotation_matrix = spice.pxform(from_frame, to_frame, et)
    ... result = spice.mxv(rotation_matrix, position)

    Parameters
    ----------
    et : float or np.ndarray
        Ephemeris time(s) corresponding to position(s).
    position : np.ndarray
        <x, y, z> vector or array of vectors in reference frame `from_frame`.
        A single position vector may be provided for multiple `et` query times
        but only a single position vector can be provided for a single `et`.
    from_frame : SpiceFrame
        Reference frame of input vector(s).
    to_frame : SpiceFrame
        Reference frame of output vector(s).

    Returns
    -------
    result : np.ndarray
        3d Cartesian position vector(s) in reference frame `to_frame`.
    """
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
            raise ValueError(
                "Mismatch in number of position vectors and Ephemeris times provided."
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
        spice.pxform,
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

    By default, the coordinates returned are Latitude/Longitude coordinates in
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
        return np.rad2deg([spice.reclat(vec)[1:] for vec in pointing])
    return np.rad2deg(spice.reclat(pointing)[1:])


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
    ... from imap_processing.spice.time import j2000ns_to_j2000s
    ... et = j2000ns_to_j2000s(dataset.epoch.values)
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
) -> NDArray:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    v : np.ndarray
        A NumPy array with shape (n, 3) where each
        row represents a vector
        with x, y, z-components.

    Returns
    -------
    spherical_coords : np.ndarray
        A NumPy array with shape (n, 3), where each row contains
        the spherical coordinates (r, azimuth, elevation):

        - r : Distance of the point from the origin.
        - azimuth : angle in the xy-plane in radians [0, 2*pi].
        - elevation : angle from the z-axis in radians [-pi/2, pi/2].
    """
    # Magnitude of the velocity vector
    magnitude_v = np.linalg.norm(v, axis=-1, keepdims=True)

    vhat = v / magnitude_v

    # Elevation angle (angle from the z-axis, range: [-pi/2, pi/2])
    el = np.arcsin(vhat[..., 2])

    # Azimuth angle (angle in the xy-plane, range: [0, 2*pi])
    az = np.arctan2(vhat[..., 1], vhat[..., 0])

    # Ensure azimuth is from 0 to 2PI
    az = az % (2 * np.pi)
    spherical_coords = np.stack(
        (np.squeeze(magnitude_v), np.degrees(az), np.degrees(el)), axis=-1
    )

    return spherical_coords


def spherical_to_cartesian(spherical_coords: NDArray) -> NDArray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    spherical_coords : np.ndarray
        A NumPy array with shape (n, 3), where each row contains
        the spherical coordinates (r, azimuth, elevation):

        - r : Distance of the point from the origin.
        - azimuth : angle in the xy-plane in radians [0, 2*pi].
        - elevation : angle from the z-axis in radians [-pi/2, pi/2].

    Returns
    -------
    cartesian_coords : np.ndarray
        Cartesian coordinates.
    """
    r = spherical_coords[..., 0]
    azimuth = spherical_coords[..., 1]
    elevation = spherical_coords[..., 2]

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    cartesian_coords = np.stack((x, y, z), axis=-1)

    return cartesian_coords
