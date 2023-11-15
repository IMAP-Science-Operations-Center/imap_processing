import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class SpiceKernelManager:
    """Manages the loading and handling of SPICE kernels.

    Parameters
    ----------
    kernel_path : Path
        The local directory path where the SPICE kernel files are stored.
    extensions : list of str, optional
        A list of file extensions to be considered as valid SPICE kernel files.
    """

    def __init__(self, kernel_path: Path, extensions: Optional[list[str]] = None):
        self.kernel_path = kernel_path
        if extensions is None:
            self.extensions = [".bpc", ".bsp", ".ti", ".tf", ".tls", ".tsc"]
        else:
            self.extensions = extensions

    def furnsh(self):
        """Load a kernel file into the SPICE system."""
        self.clear()
        files_loaded = []

        for filename in os.listdir(self.kernel_path):
            if filename.endswith(tuple(self.extensions)):
                filepath = os.path.join(self.kernel_path, filename)
                spice.furnsh(filepath)
                files_loaded.append(filename)
            else:
                logger.info(f"Invalid kernel_type extension: {filename}.")

        return files_loaded

    def clear(self):
        """Clear the SPICE kernel pool."""
        logger.info("Clearing SPICE kernel pool.")
        spice.kclear()


def ls_kernels(extensions=None) -> list:
    """List furnished spice kernels, optionally filtered by specific extensions.

    Parameters
    ----------
    extensions: list of str, optional
        Extensions to filter the kernels. If None, list all kernels.

    Returns
    -------
    : list
        A list of kernel filenames.
    """
    count = spice.ktotal("ALL")
    result = []

    for i in range(count):
        file, _, _, _ = spice.kdata(i, "ALL")
        # Append the file if no specific extensions are provided
        if extensions is None:
            result.append(file)
        # Check if the file ends with any of the specified extensions
        elif any(file.endswith(ext) for ext in extensions):
            result.append(file)

    return result


def ls_spice_constants() -> dict:
    """List all constants in the Spice constant pool.

    Returns
    -------
    : dict
        Dictionary of kernel constants
    """
    try:
        # retrieve names of kernel variables
        # * means matches all variables
        # 0 : starting from first match
        # 1000 : num variable names retrieved in call
        # 81 : max len of variable name
        kervars = spice.gnpool("*", 0, 1000, 81)
    except NotFoundError:  # Happens if there are no constants in the pool
        return {}

    result = {}
    for kervar in sorted(kervars):
        # retrieve data about a kernel variable
        n, kernel_type = spice.dtpool(kervar)  # pylint: disable=W0632
        # numerical data type
        if kernel_type == "N":
            values = spice.gdpool(kervar, 0, n)
            result[kervar] = values
        # character data type
        elif kernel_type == "C":
            values = spice.gcpool(kervar, 0, n, 81)
            result[kervar] = values
    return result


def ls_attitude_coverage(custom_pattern=None) -> tuple:
    """Process attitude kernels to extract and convert dates to UTC format.

    Parameters
    ----------
    custom_pattern : str, optional
        A custom regular expression pattern to match file names.

    Returns
    -------
    : tuple
        Tuple giving the most recent start and end time in ET.
    """
    att_kernels = ls_kernels([".ah.bc", ".ah.a"])
    filtered_files = []

    for filedir in att_kernels:
        basename = os.path.basename(filedir)
        _, extension = os.path.splitext(basename)

        # Historical attitude kernels only
        if custom_pattern is None:
            pattern = (r'imap_(\d{4})_(\d{3})_(\d{4})_(\d{3})_' +
                       rf'(\d{{2}})(\.ah\{extension})')
        else:
            pattern = custom_pattern

        matching_pattern = re.match(pattern, basename)
        if matching_pattern:
            parts = matching_pattern.groups()
            start_date_utc = datetime(int(parts[0]), 1, 1) + timedelta(
                int(parts[1]) - 1
            )
            end_date_utc = datetime(int(parts[2]), 1, 1) + timedelta(int(parts[3]) - 1)
            filtered_files.append((start_date_utc, end_date_utc))
        else:
            raise ValueError(f"Invalid pattern: {pattern}.")

    if not filtered_files:
        return None

    max_tuple = max(filtered_files, key=lambda x: x[1])
    return max_tuple
