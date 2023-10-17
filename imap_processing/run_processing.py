"""Serves as a command line utility to run the processing for a specific
instrument and data level.

Use
---
    python run_processing.py <instrument> <data_level>
"""

import argparse

from imap_processing import instruments, processing_levels


def _parse_args():
    """Parse the command line arguments.

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values
    """
    description = (
        "This command line program invokes the processing pipeline "
        "for a specific instrument and data level. Example usage: "
        '"python run_processing swe l1a".'
    )

    instrument_help = (
        "The instrument to process. Acceptable values are: " '"swe", and "codice".'
    )
    level_help = (
        "The data level to process. Acceptable values are: "
        '"l0", "l1a", "l1b", "l1c", and "l2".'
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("instrument", type=str, help=instrument_help)
    parser.add_argument("level", type=str, help=level_help)

    args = parser.parse_args()

    return args


def _validate_args(args):
    """Ensure that the provided arguments are valid before kicking off the
    processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values
    """
    assert (
        args.instrument in instruments
    ), f"{args.instrument} is not a supported instrument"
    assert (
        args.level in processing_levels[args.instrument]
    ), f"{args.level} is not a supported data level"


class ProcessInstrument:
    """A class containing methods to process each instrument.

    Attributes
    ----------
    level : str
        The data level to process (e.g. ``l1a``)
    """

    def __init__(self, level):
        self.level = level

    def process_codice(self):
        """Perform CoDICE specific processing."""
        print(f"Processing CoDICE {self.level}")

    def process_glows(self):
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {self.level}")

    def process_hi(self):
        """Perform IMAP-Hi specific processing."""
        print(f"Processing IMAP-Hi {self.level}")

    def process_hit(self):
        """Perform HIT specific processing."""
        print(f"Processing HIT {self.level}")

    def process_idex(self):
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {self.level}")

    def process_lo(self):
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {self.level}")

    def process_mag(self):
        """Perform MAG specific processing."""
        print(f"Processing MAG {self.level}")

    def process_swapi(self):
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {self.level}")

    def process_swe(self):
        """Perform SWE specific processing."""
        print(f"Processing swe {self.level}")

    def process_ultra(self):
        """Perform IMAP-Ultra specific processing."""
        print(f"Processing IMAP-Ultra {self.level}")


if __name__ == "__main__":
    args = _parse_args()
    _validate_args(args)

    # Determine which function to invoke
    process = ProcessInstrument(args.level)
    method = getattr(process, f"process_{args.instrument}")
    method()
