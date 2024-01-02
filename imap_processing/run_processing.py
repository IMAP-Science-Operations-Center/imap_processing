#!/usr/bin/env python3

"""Run the processing for a specific instrument & data level.

This module serves as a command line utility to invoke the processing for
a user-supplied instrument and data level.

Use
---
    python run_processing.py <instrument> <data_level>
"""

import argparse
import sys
from abc import ABC, abstractmethod
import spiceypy as spice

from imap_processing import instruments, processing_levels
from tools.spice.spice_utils import list_loaded_kernels

# Constants for symlink
LATEST_ATTITUDE_KERNEL_PATH = "/mnt/spice/latest_attitude_kernel.ah.a"
LATEST_EPHEMERIS_KERNEL_PATH = "/mnt/spice/latest_ephemeris_kernel.bsp"


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
    """Ensure that the  arguments are valid before kicking off the processing.

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
            f"{args.level} is not a supported data level for the {args.instrument}"
            f" instrument, valid levels are: {processing_levels[args.instrument]}"
        )


class ProcessInstrument(ABC):
    """An abstract base class containing a method to process an instrument.

    Attributes
    ----------
    level : str
        The data level to process (e.g. ``l1a``)
    """

    def __init__(self, level):
        self.level = level

    @abstractmethod
    def process(self):
        """Perform instrument specific processing."""
        raise NotImplementedError

    @staticmethod
    def get_latest_data():
        """Get the latest ephemeris and attitude data."""
        with spice.KernelPool(LATEST_ATTITUDE_KERNEL_PATH):
            attitude = list_loaded_kernels()
        with spice.KernelPool(LATEST_EPHEMERIS_KERNEL_PATH):
            ephemeris = list_loaded_kernels()
        return attitude, ephemeris


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def process(self):
        """Perform CoDICE specific processing."""
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        print(f"Processing CoDICE {self.level} using {attitude} and {ephemeris}")


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {self.level} using {self.level} using {attitude} and {ephemeris}")


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform IMAP-Hi specific processing."""
        print(f"Processing IMAP-Hi {self.level} using {attitude} and {ephemeris}")


class Hit(ProcessInstrument):
    """Process HIT."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform HIT specific processing."""
        print(f"Processing HIT {self.level} using {attitude} and {ephemeris}")


class Idex(ProcessInstrument):
    """Process IDEX."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {self.level} using {attitude} and {ephemeris}")


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {self.level} using {attitude} and {ephemeris}")


class Mag(ProcessInstrument):
    """Process MAG."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform MAG specific processing."""
        print(f"Processing MAG {self.level} using {attitude} and {ephemeris}")


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {self.level} using {attitude} and {ephemeris}")


class Swe(ProcessInstrument):
    """Process SWE."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform SWE specific processing."""
        print(f"Processing SWE {self.level} using {attitude} and {ephemeris}")


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def process(self):
        attitude, ephemeris = ProcessInstrument.get_latest_data()
        """Perform IMAP-Ultra specific processing."""
        print(f"Processing IMAP-Ultra {self.level} using {attitude} and {ephemeris}")


def main():
    args = _parse_args()
    _validate_args(args)
    cls = getattr(sys.modules[__name__], args.instrument.capitalize())
    instrument = cls(args.level)
    instrument.process()
