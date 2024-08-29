"""Functions for furnishing and tracking SPICE kernels."""

import functools
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray
from spiceypy.utils.exceptions import SpiceyError

logger = logging.getLogger(__name__)


def ensure_spice(
    f_py: Optional[Callable] = None, time_kernels_only: bool = False
) -> Callable:
    """
    Decorator/wrapper that automatically furnishes SPICE kernels.

    Parameters
    ----------
    f_py : Callable
        The function requiring SPICE that we are going to wrap if being used
        explicitly, otherwise None, in which case ensure_spice is being used,
        not as a function wrapper (see l2a_processing.py) but as a true
        decorator without an explicit function argument.
    time_kernels_only : bool
        Specify that we only need to furnish time kernels (if SPICE_METAKERNEL
        is set, we still just furnish that metakernel and assume the time
        kernels are included.

    Returns
    -------
    Callable
        Decorated function, with spice error handling.

    Notes
    -----
    Before trying to understand this piece of code, read this:
    https://stackoverflow.com/questions/5929107/decorators-with-parameters/60832711#60832711

    **Control flow overview:**
    1. Try simply calling the wrapped function naively.
        * SUCCESS? Great! We're done.
        * SpiceyError? Go to step 2.

    2. Furnish metakernel at SPICE_METAKERNEL
        * SUCCESS? Great, return the original function again (so it can be
          re-run).
        * KeyError? Seems like SPICE_METAKERNEL isn't set, no problem. Go to
          step 3.

    3. Did we get the parameter time_kernels_only=True?
    --> YES? We only need LSK and SCLK kernels to run this function. Go fetch
        those and furnish and return the original function (so it can be re-run).
    --> NO? Dang. This is sort of the end of the line. Re-raise the error
        generated from the failed spiceypy function call but add a better
        message to it.

    Examples
    --------
    There are three ways to use this object

    1. A decorator with no arguments
        >>> @ensure_spice
        ... def my_spicey_func(a, b):
        ...     pass

    2. A decorator with parameters. This is useful
    if we only need the latest SCLK and LSK kernels for the function involved.
        >>> @ensure_spice(time_kernels_only=True)
        ... def my_spicey_time_func(a, b):
        ...     pass

    3. An explicit wrapper function, providing a dynamically set value for
    parameters, e.g. time_kernels_only
        >>> wrapped = ensure_spice(spicey_func, time_kernels_only=True)
        ... result = wrapped(*args, **kwargs)
    """
    if f_py and not callable(f_py):
        raise ValueError(
            f"Received a non-callable object {f_py} as the f_py argument to"
            f"ensure_spice.  f_py must be a callable object."
        )

    def _decorator(func: Callable[..., Callable]) -> Callable:
        """
        Decorate or wrap input function depending on how ensure_spice is used.

        Parameters
        ----------
        func : Callable
            The function to be decorated/wrapped.

        Returns
        -------
        Callable
            If used as a function wrapper, the decorated function is returned.
        """

        @functools.wraps(func)
        def wrapper_ensure_spice(*args: Any, **kwargs: Any) -> Any:
            """
            Wrap the function that ensure_spice is used on.

            Parameters
            ----------
            *args : list
                The positional arguments passed to the decorated function.
            **kwargs
                The keyword arguments passed to the decorated function.

            Returns
            -------
            Object
                Output from wrapped function.
            """
            try:
                # Step 1.
                return func(
                    *args, **kwargs
                )  # Naive first try. Maybe SPICE is already furnished.
            except SpiceyError as spicey_err:
                try:
                    # Step 2.
                    metakernel_path = os.environ["SPICE_METAKERNEL"]
                    spice.furnsh(metakernel_path)
                except KeyError:
                    # TODO: An additional step that was used on EMUS was to get
                    #  a custom metakernel from the SDC API based on an input
                    #  time range.
                    if time_kernels_only:
                        # Step 3.
                        # TODO: Decide if this is useful for IMAP. Possible
                        #  implementation could include downloading
                        #  the most recent leapsecond kernel from NAIF (see:
                        #  https://lasp.colorado.edu/nucleus/projects/LIBSDC/repos/libera_utils/browse/libera_utils/spice_utils.py
                        #  for LIBERA implementation of downloading and caching
                        #  kernels) and finding the most recent IMAP clock
                        #  kernel in EFS.
                        raise NotImplementedError from spicey_err
                    else:
                        raise SpiceyError(
                            "When calling a function requiring SPICE, we failed"
                            "to load a metakernel. SPICE_METAKERNEL is not set,"
                            "and time_kernels_only is not set to True"
                        ) from spicey_err
                return func(*args, **kwargs)

        return wrapper_ensure_spice

    # Note: This return was originally implemented as a ternary operator, but
    # this caused mypy to fail due to this bug:
    # https://github.com/python/mypy/issues/4134
    if callable(f_py):
        return _decorator(f_py)
    else:
        return _decorator


@contextmanager
def spice_ck_file(pointing_frame_path: str) -> Generator[int, None, None]:
    """
    Context manager for handling SPICE CK files.

    Parameters
    ----------
    pointing_frame_path : str
        Path to the CK file.

    Yields
    ------
    handle : int
        Handle to the opened CK file.
    """
    handle = spice.ckopn(pointing_frame_path, "CK", 0)
    try:
        yield handle
    finally:
        spice.ckcls(handle)


@ensure_spice
def create_pointing_frame(pointing_frame_path: Optional[Path] = None) -> Path:
    """
    Create the pointing frame.

    Parameters
    ----------
    pointing_frame_path : Path
        Directory of where pointing frame will be saved.

    Returns
    -------
    pointing_frame_path : Path
        Path to pointing frame.

    References
    ----------
    https://numpydoc.readthedocs.io/en/latest/format.html#references
    """
    ck_kernel, _, _, _ = spice.kdata(0, "ck")

    # Get timerange for the pointing frame kernel.
    et_start, et_end, et_times = _get_et_times(ck_kernel)
    # Create a rotation matrix
    rotation_matrix = _create_rotation_matrix(et_times)

    # Convert the rotation matrix to a quaternion.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
    q_avg = spice.m2q(rotation_matrix)

    # TODO: come up with naming convention.
    if pointing_frame_path is None:
        pointing_frame_path = Path(ck_kernel).parent / "imap_dps.bc"

    # Open a new CK file, returning the handle of the opened file.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckopn
    with spice_ck_file(str(pointing_frame_path)) as handle:
        # Get the SCLK ID.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
        id_imap_sclk = spice.gipool("CK_-43000_SCLK", 0, 1)
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
        # Convert start and end times to SCLK.
        sclk_begtim = spice.sce2c(int(id_imap_sclk), et_start)
        sclk_endtim = spice.sce2c(int(id_imap_sclk), et_end)

        # Get the pointing frame ID.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
        id_imap_dps = spice.gipool("FRAME_IMAP_DPS", 0, 1)
        # TODO: Figure out how to write new pointings to same CK kernel.
        # Create the pointing frame kernel.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckw02
        spice.ckw02(
            # Handle of an open CK file.
            handle,
            # Start time of the segment.
            sclk_begtim,
            # End time of the segment.
            sclk_endtim,
            # Pointing frame ID.
            int(id_imap_dps),
            # Reference frame.
            "ECLIPJ2000",  # Reference frame
            # Identifier.
            "IMAP_DPS",
            # Number of pointing intervals.
            1,
            # Start times of individual pointing records within segment.
            # Since there is only a single record this is equal to sclk_begtim.
            np.array([sclk_begtim]),
            # End times of individual pointing records within segment.
            # Since there is only a single record this is equal to sclk_endtim.
            np.array([sclk_endtim]),  # Single stop time
            # Average quaternion.
            q_avg,
            # 0.0 Angular rotation terms.
            np.array([0.0, 0.0, 0.0]),
            # Rates (seconds per tick) at which the quaternion and
            # angular velocity change.
            np.array([1.0]),
        )

    return pointing_frame_path


@ensure_spice
def _get_et_times(ck_kernel: str) -> tuple[float, float, np.ndarray]:
    """
    Get times for pointing start and stop.

    Parameters
    ----------
    ck_kernel : str
        Path of ck_kernel used to create the pointing frame.

    Returns
    -------
    et_start : float
        Pointing start time.
    et_end : float
        Pointing end time.
    et_times : numpy.ndarray
        Array of times between et_start and et_end.
    """
    # Get the spacecraft ID.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
    id_imap_spacecraft = spice.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)

    # TODO: Queried pointing start and stop times here.
    # TODO removing the @ensure_spice decorator when using the repointing table.

    # Get the coverage window
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckcov
    cover = spice.ckcov(ck_kernel, int(id_imap_spacecraft), True, "SEGMENT", 0, "TDB")
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.wnfetd
    et_start, et_end = spice.wnfetd(cover, 0)
    # 1 spin/15 seconds; 10 quaternions / spin
    num_samples = (et_end - et_start) / 15 * 10
    et_times = np.linspace(et_start, et_end, int(num_samples))

    return et_start, et_end, et_times


@ensure_spice
def _average_quaternions(et_times: np.ndarray) -> NDArray:
    """
    Average the quaternions.

    Parameters
    ----------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.

    Returns
    -------
    q_avg : np.ndarray
        Average quaternion.
    """
    aggregate = np.zeros((4, 4))
    for tdb in et_times:
        # we use a quick and dirty method here for grabbing the quaternions
        # from the attitude kernel.  Depending on how well the kernel input
        # data is built and sampled, there may or may not be aliasing with this
        # approach.  If it turns out that we need to pull the quaternions
        # directly from the CK there are several routines that exist to do this
        # but it's not straight forward.  We'll revisit this if needed.

        # Rotation matrix from IMAP spacecraft frame to ECLIPJ2000.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.pxform
        body_rots = spice.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", tdb)
        # Convert rotation matrix to quaternion.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
        body_quat = spice.m2q(body_rots)

        # Standardize the quaternion so that they may be compared.
        body_quat = body_quat * np.sign(body_quat[0])
        # Aggregate quaternions into a single matrix.
        aggregate += np.outer(body_quat, body_quat)

    # Reference: "On Averaging Rotations".
    # Link: https://link.springer.com/content/pdf/10.1023/A:1011129215388.pdf
    aggregate /= len(et_times)

    # Compute eigen values and vectors of the matrix A
    # Eigenvalues tell you how much "influence" each
    # direction (eigenvector) has.
    # The largest eigenvalue corresponds to the direction
    # that has the most influence.
    # The eigenvector corresponding to the largest
    # eigenvalue points in the direction that has the most
    # combined rotation influence.
    eigvals, eigvecs = np.linalg.eig(aggregate)
    # q0: The scalar part of the quaternion.
    # q1, q2, q3: The vector part of the quaternion.
    q_avg = eigvecs[:, np.argmax(eigvals)]

    return q_avg


def _create_rotation_matrix(et_times: np.ndarray) -> NDArray:
    """
    Create a rotation matrix.

    Parameters
    ----------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix.
    """
    # Averaged quaternions.
    q_avg = _average_quaternions(et_times)

    # Converts the averaged quaternion (q_avg) into a rotation matrix
    # and get inertial z axis.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.q2m
    z_avg = spice.q2m(list(q_avg))[:, 2]
    # y_avg is perpendicular to both z_avg and the standard Z-axis.
    y_avg = np.cross(z_avg, [0, 0, 1])
    # x_avg is perpendicular to y_avg and z_avg.
    x_avg = np.cross(y_avg, z_avg)

    # Construct the rotation matrix from x_avg, y_avg, z_avg
    rotation_matrix = np.asarray([x_avg, y_avg, z_avg])

    return rotation_matrix
