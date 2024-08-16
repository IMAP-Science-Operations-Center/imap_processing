"""Functions for furnishing and tracking SPICE kernels."""

import functools
import logging
import os
from typing import Any, Callable, Optional

import spiceypy as spice
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
