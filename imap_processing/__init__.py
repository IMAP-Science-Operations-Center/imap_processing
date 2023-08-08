__version__ = "0.1.0"

# When imap_processing is installed using pip, we need to be able to find the
# packet definitions directory path.
#
# This directory is used by the imap_processing package to find the packet definitions.
from pathlib import Path

# Eg. imap_module_directory = /usr/local/lib/python3.11/site-packages/imap_processing
imap_module_directory = Path(__file__).parent

# Relative to imap_module_directory, set path of packet definitions directory.
packet_definition_directory = f"{imap_module_directory}/packet_definitions/"
