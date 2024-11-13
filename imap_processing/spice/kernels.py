"""Functions that generate, furnish, and retrieve metadata from SPICE kernels."""

import functools
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union, overload

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray
from spiceypy.utils.exceptions import SpiceyError

from imap_processing import imap_module_directory

logger = logging.getLogger(__name__)


# Declarations to help with typing. Taken from mypy documentation on
# decorator-factories:
# https://mypy.readthedocs.io/en/stable/generics.html#decorator-factories
# Bare decorator usage
@overload
def ensure_spice(
    __func: Callable[..., Any],
) -> Callable[..., Any]: ...  # numpydoc ignore=GL08
# Decorator with arguments
@overload
def ensure_spice(
    *, time_kernels_only: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...  # numpydoc ignore=GL08
# Implementation
def ensure_spice(
    __func: Optional[Callable[..., Any]] = None, *, time_kernels_only: bool = False
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """
    Decorator/wrapper that automatically furnishes SPICE kernels.

    Parameters
    ----------
    __func : Callable
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
        * YES? We only need LSK and SCLK kernels to run this function. Go fetch
          those and furnish and return the original function (so it can be re-run).
        * NO? Dang. This is sort of the end of the line. Re-raise the error
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
        ... result = wrapped(args, kwargs)
    """

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
                    if os.getenv("SPICE_METAKERNEL"):
                        metakernel_path = os.getenv("SPICE_METAKERNEL")
                        spice.furnsh(metakernel_path)
                    else:
                        furnish_time_kernel()
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
                            "When calling a function requiring SPICE, we failed "
                            "to load a metakernel. SPICE_METAKERNEL is not set,"
                            "and time_kernels_only is not set to True"
                        ) from spicey_err
                return func(*args, **kwargs)

        return wrapper_ensure_spice

    # Note: This return was originally implemented as a ternary operator, but
    # this caused mypy to fail due to this bug:
    # https://github.com/python/mypy/issues/4134
    if callable(__func):
        return _decorator(__func)
    else:
        return _decorator


@contextmanager
def open_spice_ck_file(pointing_frame_path: Path) -> Generator[int, None, None]:
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
    # TODO: We will need to figure out if ck kernel changes
    # and how that will affect appending to the pointing
    # frame kernel.
    if pointing_frame_path.exists():
        handle = spice.dafopw(str(pointing_frame_path))
    else:
        handle = spice.ckopn(str(pointing_frame_path), "CK", 0)
    try:
        yield handle
    finally:
        spice.ckcls(handle)


@ensure_spice
def create_pointing_frame(pointing_frame_path: Path, ck_path: Path) -> None:
    """
    Create the pointing frame.

    Parameters
    ----------
    pointing_frame_path : pathlib.Path
        Location of pointing frame kernel.
    ck_path : pathlib.Path
        Location of the CK kernel.

    Notes
    -----
    Kernels required to be furnished:
    "imap_science_0001.tf",
    "imap_sclk_0000.tsc",
    "imap_sim_ck_2hr_2secsampling_with_nutation.bc" or
    "sim_1yr_imap_attitude.bc",
    "imap_wkcp.tf",
    "naif0012.tls"

    Assumptions:
    - The MOC has removed timeframe in which nutation/procession are present.
    TODO: We may come back and have a check for this.
    - We will continue to append to the pointing frame kernel.
    TODO: Figure out how we want to handle the file size becoming too large.
    - For now we can only furnish a single ck kernel.
    TODO: This will not be the case once we add the ability to query the .csv.

    References
    ----------
    https://numpydoc.readthedocs.io/en/latest/format.html#references
    """
    # Get IDs.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
    id_imap_dps = spice.gipool("FRAME_IMAP_DPS", 0, 1)
    id_imap_sclk = spice.gipool("CK_-43000_SCLK", 0, 1)

    # Verify that only ck_path kernel is loaded.
    count = spice.ktotal("ck")
    loaded_ck_kernel, _, _, _ = spice.kdata(count - 1, "ck")

    if count != 1 or str(ck_path) != loaded_ck_kernel:
        raise ValueError(f"Error: Expected CK kernel {ck_path}")

    # If the pointing frame kernel already exists, find the last time.
    if pointing_frame_path.exists():
        # Get the last time in the pointing frame kernel.
        pointing_cover = spice.ckcov(
            str(pointing_frame_path), int(id_imap_dps), True, "SEGMENT", 0, "TDB"
        )
        num_segments = spice.wncard(pointing_cover)
        _, et_end_pointing_frame = spice.wnfetd(pointing_cover, num_segments - 1)
    else:
        et_end_pointing_frame = None

    # TODO: Query for .csv file to get the pointing start and end times.
    # TODO: Remove next four lines once query is added.
    id_imap_spacecraft = spice.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)
    ck_cover = spice.ckcov(
        str(ck_path), int(id_imap_spacecraft), True, "INTERVAL", 0, "TDB"
    )
    num_intervals = spice.wncard(ck_cover)

    with open_spice_ck_file(pointing_frame_path) as handle:
        # TODO: this will change to the number of pointings.
        for i in range(num_intervals):
            # Get the coverage window
            # TODO: this will change to pointing start and end time.
            et_start, et_end = spice.wnfetd(ck_cover, i)
            et_times = _get_et_times(et_start, et_end)

            # TODO: remove after query is added.
            if (
                et_end_pointing_frame is not None
                and et_times[0] < et_end_pointing_frame
            ):
                break

            # Create a rotation matrix
            rotation_matrix = _create_rotation_matrix(et_times)

            # Convert the rotation matrix to a quaternion.
            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
            q_avg = spice.m2q(rotation_matrix)

            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
            # Convert start and end times to SCLK.
            sclk_begtim = spice.sce2c(int(id_imap_sclk), et_times[0])
            sclk_endtim = spice.sce2c(int(id_imap_sclk), et_times[-1])

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


def _get_et_times(et_start: float, et_end: float) -> NDArray[np.float64]:
    """
    Get times for pointing start and stop.

    Parameters
    ----------
    et_start : float
        Pointing start time.
    et_end : float
        Pointing end time.

    Returns
    -------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.
    """
    # TODO: Queried pointing start and stop times here.
    # TODO removing the @ensure_spice decorator when using the repointing table.

    # 1 spin/15 seconds; 10 quaternions / spin.
    num_samples = (et_end - et_start) / 15 * 10
    # There were rounding errors when using spice.pxform so np.ceil and np.floor
    # were used to ensure the start and end times were included in the array.
    et_times = np.linspace(
        np.ceil(et_start * 1e6) / 1e6, np.floor(et_end * 1e6) / 1e6, int(num_samples)
    )

    return et_times


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


def furnish_time_kernel() -> None:
    """Furnish the time kernels."""
    spice_test_data_path = imap_module_directory / "tests/spice/test_data"

    # TODO: we need to load these kernels from EFS volumen that is
    # mounted to batch volume and extend this to generate metakernell
    # which is TBD.
    spice.furnsh(str(spice_test_data_path / "imap_sclk_0000.tsc"))
    spice.furnsh(str(spice_test_data_path / "naif0012.tls"))
