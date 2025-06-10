"""Interstellar Mapping and Acceleration Probe (IMAP) data processing package.

This package contains the IMAP data processing software. The package is
organized into submodules for each instrument. Each instrument submodule
contains code for each processing level.

There are utilities to read and write IMAP data files in
the CDF file format, and to interact with the SPICE toolkit.
"""

# When imap_processing is installed using pip, we need to be able to find the
# packet definitions directory path.
#
# This directory is used by the imap_processing package to find the packet definitions.
from pathlib import Path

from imap_processing._version import __version__, __version_tuple__  # noqa: F401

# Eg. imap_module_directory = /usr/local/lib/python3.11/site-packages/imap_processing
imap_module_directory = Path(__file__).parent

PROCESSING_LEVELS = {
    "codice": ["l1a", "l1b", "l2"],
    "glows": ["l1a", "l1b", "l2"],
    "hi": ["l1a", "l1b", "l1c", "l2"],
    "hit": ["l1a", "l1b", "l2"],
    "idex": ["l1a", "l1b", "l2a", "l2b", "l2c"],
    "lo": ["l1a", "l1b", "l1c", "l2"],
    "mag": ["l1a", "l1b", "l1c", "l1d", "l2"],
    "spacecraft": ["l1a", "spice"],
    "swapi": ["l1a", "l1b", "l1", "l2", "l3a", "l3b"],
    "swe": ["l1a", "l1b", "l2"],
    "ultra": ["l1a", "l1b", "l1c", "l2"],
}
