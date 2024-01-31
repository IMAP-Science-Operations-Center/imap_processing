#!/usr/bin/env python3

"""Run the processing for a specific instrument & data level.

This module serves as a command line utility to invoke the processing for
a user-supplied instrument and data level.

Use
---
    imap_cli --instrument <instrument> --level <data_level>
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import imap_data_access
from cdflib.xarray import cdf_to_xarray

import imap_processing
from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b


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
        '"python run_processing --instrument swe --level l1a".'
    )
    data_dir_help = (
        "Directory to use for reading and writing IMAP data. "
        "The default is an 'imap-data/' folder in the "
        "current working directory. This can also be "
        "set using the IMAP_DATA_DIR environment variable."
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_processing.INSTRUMENTS}"
    )
    level_help = (
        "The data level to process. Acceptable values are: "
        f"{imap_processing.PROCESSING_LEVELS}"
    )
    depdency_help = (
        "Dependency information in str format."
        "Example: '[{instrument: swe, level: l0, version: v00-01}]'"
    )

    parser = argparse.ArgumentParser(prog="imap_cli", description=description)
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--level", type=str, required=True, help=level_help)
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Full path to the file in the S3 bucket.",
    )
    parser.add_argument(
        "--dependency",
        type=str,
        required=True,
        help=depdency_help,
    )
    parser.add_argument("--data-dir", type=str, required=False, help=data_dir_help)
    args = parser.parse_args()

    return args


def _validate_args(args):
    """Ensure that the  arguments are valid before kicking off the processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values
    """
    if args.instrument not in imap_processing.INSTRUMENTS:
        raise ValueError(
            f"{args.instrument} is not in the supported instrument list: "
            f"{imap_processing.INSTRUMENTS}"
        )
    if args.level not in imap_processing.PROCESSING_LEVELS[args.instrument]:
        raise ValueError(
            f"{args.level} is not a supported data level for the {args.instrument}"
            " instrument, valid levels are: "
            f"{imap_processing.PROCESSING_LEVELS[args.instrument]}"
        )
    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory {args.data_dir} does not exist")
        # Set the data directory to the user-supplied value
        imap_processing.config["DATA_DIR"] = data_path


class ProcessInstrument(ABC):
    """An abstract base class containing a method to process an instrument.

    Attributes
    ----------
    level : str
        The data level to process (e.g. ``l1a``)
    """

    def __init__(self, level, file_path):
        self.level = level
        self.file_path = file_path

    @abstractmethod
    def process(self):
        """Perform instrument specific processing."""
        raise NotImplementedError


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def process(self):
        """Perform CoDICE specific processing."""
        print(f"Processing CoDICE {self.level}")


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def process(self):
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {self.level}")


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def process(self):
        """Perform IMAP-Hi specific processing."""
        print(f"Processing IMAP-Hi {self.level}")


class Hit(ProcessInstrument):
    """Process HIT."""

    def process(self):
        """Perform HIT specific processing."""
        print(f"Processing HIT {self.level}")


class Idex(ProcessInstrument):
    """Process IDEX."""

    def process(self):
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {self.level}")


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def process(self):
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {self.level}")


class Mag(ProcessInstrument):
    """Process MAG."""

    def process(self):
        """Perform MAG specific processing."""
        print(f"Processing MAG {self.level}")


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def process(self):
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {self.level}")


class Swe(ProcessInstrument):
    """Process SWE."""

    def process(self):
        """Perform SWE specific processing."""
        # self.file_path example:
        # imap/swe/l1a/2023/09/imap_swe_l1a_sci_20230927_20230927_v01-00.cdf
        print(f"Processing SWE {self.level}")
        if self.level == "l1a":
            # create download path
            download_path = self.file_path.replace("l1a", "l0").replace("cdf", "pkts")
            print(f"download_path: {download_path}")
            output_path = imap_data_access.download(download_path)
            processed_data = swe_l1a(output_path)
            for data in processed_data:
                # write data to cdf
                cdf_file_path = write_cdf(
                    data=data["data"], descriptor=data["descriptor"]
                )
                imap_data_access.upload(cdf_file_path)
                print(f"uploaded {cdf_file_path}")
        elif self.level == "l1b":
            # create download path
            download_path = self.file_path.replace("l1b", "l1a")
            print(f"download_path: {download_path}")
            output_path = imap_data_access.download(download_path)
            print(f"downloaded folder {output_path}")
            # read CDF file
            l1a_dataset = cdf_to_xarray(output_path)
            processed_data = swe_l1b(l1a_dataset)
            descriptor = os.path.basename(self.file_path).split("_")[3]
            processed_file_path = write_cdf(data=processed_data, descriptor=descriptor)
            print(f"processed file path: {processed_file_path}")
            imap_data_access.upload(processed_file_path)
            print(f"finished uploading - {processed_file_path}")
        else:
            print("No code to process this level")


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def process(self):
        """Perform IMAP-Ultra specific processing."""
        print(f"Processing IMAP-Ultra {self.level}")


def main():
    """Run the processing for a specific instrument & data level.

    Set up the command line arguments, parse them, and then invoke the
    appropriate instrument processing function.
    """
    # NOTE: This is to allow the cli script to be installed and reference
    #       this function for an entrypoint.
    args = _parse_args()

    _validate_args(args)

    cls = getattr(sys.modules[__name__], args.instrument.capitalize())
    instrument = cls(args.level, args.file_path)
    instrument.process()


if __name__ == "__main__":
    main()
