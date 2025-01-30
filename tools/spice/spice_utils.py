"""Various utility functions to support the use of SPICE kernels."""

import logging
import os
from typing import Optional

import spiceypy

logger = logging.getLogger(__name__)


def list_files_with_extensions(
    directory: str, extensions: Optional[list[str]] = None
) -> list[str]:
    """
    List all files in a given directory that have the specified extensions.

    Parameters
    ----------
    directory : str
        The directory to search in.
    extensions : list[str], (optional)
        A list of file extensions to filter the files.

    Returns
    -------
    matching_files : list[str]
        A list of file paths in the specified directory
        that match the given extensions.
    """
    # Default set of extensions
    if extensions is None:
        extensions = [".bpc", ".bsp", ".ti", ".tf", ".tls", ".tsc"]
    else:
        extensions = [ext.lower() for ext in extensions]

    matching_files = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in extensions):
            matching_files.append(os.path.join(directory, file))
        else:
            logger.debug(f"Skipping file {file}.")

    return matching_files


def list_loaded_kernels(extensions: Optional[list[str]] = None) -> list:
    """
    List furnished spice kernels, optionally filtered by specific extensions.

    Parameters
    ----------
    extensions : list of str or None, optional
        Extensions to filter the kernels. If None, list all kernels.

    Returns
    -------
    result : list
        A list of kernel filenames.
    """
    count = spiceypy.ktotal("ALL")
    result = []

    for i in range(count):
        file, _, _, _ = spiceypy.kdata(i, "ALL")
        # Append the file if no specific extensions are provided
        if extensions is None:
            result.append(file)
        # Check if the file ends with any of the specified extensions
        elif any(file.endswith(ext) for ext in extensions):
            result.append(file)

    return result


def list_all_constants() -> dict:
    """
    List all constants in the Spice constant pool.

    Returns
    -------
    result : dict
        Dictionary of kernel constants.
    """
    # retrieve names of kernel variables using below inputs per
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/gdpool_c.html
    # name = '*', means name of the variable whose value is to be returned.
    # start = 0, Which component to start retrieving for `name'.
    # room = 1000, The largest number of values to return.
    # n = 81, Number of values returned for `name'.
    kernel_vars = spiceypy.gnpool("*", 0, 1000, 81)

    result = {}
    for kernel_var in sorted(kernel_vars):
        # retrieve data about a kernel variable
        n, kernel_type = spiceypy.dtpool(kernel_var)
        # numerical data type
        if kernel_type == "N":
            values = spiceypy.gdpool(kernel_var, 0, n)
            result[kernel_var] = values
        # character data type
        elif kernel_type == "C":
            values = spiceypy.gcpool(kernel_var, 0, n, 81)
            result[kernel_var] = values
    return result
