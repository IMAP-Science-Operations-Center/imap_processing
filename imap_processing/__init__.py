__version__ = "0.1.0"

# When imap_processing is installed using pip, we need to be able to find the
# packet definitions directory path.
#
# This directory is used by the imap_processing package to find the packet definitions.
from pathlib import Path

# Eg. imap_module_directory = /usr/local/lib/python3.11/site-packages/imap_processing
imap_module_directory = Path(__file__).parent

instruments = [
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

processing_levels = {
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
