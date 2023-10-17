#!/usr/bin/env python3
"""Serves as a command line utility to run the processing for a specific
instrument and data level.

Use
---
    python run_processing.py <instrument> <data_level>
"""

import argparse
import sys
from abc import ABC, abstractmethod

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

    instrument_help = f"The instrument to process. Acceptable values are: {instruments}"
    level_help = (
        f"The data level to process. Acceptable values are: {processing_levels}"
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--level", type=str, required=True, help=level_help)

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
    if args.instrument not in instruments:
        raise ValueError(
            f"{args.instrument} is not in the supported instrument list: {instruments}"
        )
    if args.level not in processing_levels[args.instrument]:
        raise ValueError(
            f"{args.level} is not a supported data level for the {args.instrument} instrument, valid levels are: {processing_levels[args.instrument]}"
        )


class ProcessInstrument(ABC):
    """An abstract base class containing a method to process an instrument.

    Attributes
    ----------
    level : str
        The data level to process (e.g. ``l1a``)
    """

    @abstractmethod
    def process(level):
        """Perform instrument specific processing."""
        raise NotImplementedError


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def process(level):
        """Perform CoDICE specific processing."""
        print(f"Processing CoDICE {level}")


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def process(level):
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {level}")


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def process(level):
        """Perform IMAP-Hi specific processing."""
        print(f"Processing IMAP-Hi {level}")


class Hit(ProcessInstrument):
    """Process HIT."""

    def process(level):
        """Perform HIT specific processing."""
        print(f"Processing HIT {level}")


class Idex(ProcessInstrument):
    """Process IDEX."""

    def process(level):
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {level}")


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def process(level):
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {level}")


class Mag(ProcessInstrument):
    """Process MAG."""

    def process(level):
        """Perform MAG specific processing."""
        print(f"Processing MAG {level}")


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def process(level):
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {level}")


class Swe(ProcessInstrument):
    """Process SWE."""

    def process(level):
        """Perform SWE specific processing."""
        print(f"Processing SWE {level}")


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def process(level):
        """Perform IMAP-Ultra specific processing."""
        print(f"Processing IMAP-Ultra {level}")


if __name__ == "__main__":
    args = _parse_args()
    _validate_args(args)

    # Determine which function to invoke
    cls = getattr(sys.modules[__name__], args.instrument.capitalize())
    cls.process(args.level)
