"""Interstellar Mapping and Acceleration Probe (IMAP) data processing package.

This package contains the IMAP data processing software. The package is
organized into submodules for each instrument. Each instrument submodule
contains code for each processing level.

There are utilities to read and write IMAP data files in
the CDF file format, and to interact with the SPICE toolkit.
"""
__version__ = "0.2.0"

# When imap_processing is installed using pip, we need to be able to find the
# packet definitions directory path.
#
# This directory is used by the imap_processing package to find the packet definitions.
from pathlib import Path

# Eg. imap_module_directory = /usr/local/lib/python3.11/site-packages/imap_processing
imap_module_directory = Path(__file__).parent

INSTRUMENTS = [
    "codice",
    "glows",
    "hi",
    "hit",
    "idex",
    "lo",
    "mag",
    "swapi",
    "swe",
    "ultra",
]

PROCESSING_LEVELS = {
    "codice": ["l0", "l1a", "l1b", "l2"],
    "glows": ["l0", "l1a", "l1b", "l2"],
    "hi": ["l0", "l1a", "l1b", "l1c", "l2"],
    "hit": ["l0", "l1a", "l1b", "l2"],
    "idex": ["l0", "l1a", "l1b", "l1c", "l2"],
    "lo": ["l0", "l1a", "l1b", "l1c", "l2"],
    "mag": ["l0", "l1a", "l1b", "l1c", "l2pre", "l2"],
    "swapi": ["l0", "l1", "l2"],
    "swe": ["l0", "l1a", "l1b", "l2"],
    "ultra": ["l0", "l1a", "l1b", "l1c", "l1d", "l2"],
}
