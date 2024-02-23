#!/usr/bin/env python3
# ruff: noqa: PLR0913
"""Run the processing for a specific instrument & data level.

This module serves as a command line utility to invoke the processing for
a user-supplied instrument and data level.

Use
---
    imap_cli --instrument <instrument> --level <data_level>
"""

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from json import loads
from pathlib import Path
from urllib.error import HTTPError

import imap_data_access
from cdflib.xarray import cdf_to_xarray

import imap_processing

# TODO: change how we import things and also folder
# structure may?
# From this:
#   from imap_processing.cdf.utils import write_cdf
# To this:
#   from imap_processing import cdf
# In code:
#   call cdf.utils.write_cdf
from imap_processing.cdf.utils import write_cdf
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b

logger = logging.getLogger(__name__)


def _parse_args():
    """Parse the command line arguments.

    The expected input format is:
    --instrument "mag"
    --data-level "l1a"
    --start-date "20231212"
    --end-date "20231212"
    --version "v00-01"
    --dependency "[
        {
            'instrument': 'mag',
            'data_level': 'l0',
            'descriptor': 'sci',
            'version': 'v00-01',
            'start_date': '20231212',
            'end_date': '20231212'
        }]"
    --upload-to-sdc

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values
    """
    description = (
        "This command line program invokes the processing pipeline "
        "for a specific instrument and data level. Example usage: "
        '"imap_cli --instrument "mag" '
        '--data-level "l1a"'
        ' --start-date "20231212"'
        '--end-date "20231212"'
        '--version "v00-01"'
        '--dependency "['
        '   {"instrument": "mag",'
        '   "data_level": "l0"',
        '   "descriptor": "sci"',
        '   "version": "v00-01"',
        '   "start_date": "20231212"',
        '   "end_date": "20231212"',
        '}]" --upload-to-sdc"',
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_processing.INSTRUMENTS}"
    )
    level_help = (
        "The data level to process. Acceptable values are: "
        f"{imap_processing.PROCESSING_LEVELS}"
    )
    dependency_help = (
        "Dependency information in str format."
        "Example: '[{'instrument': 'mag',"
        "'data_level': 'l0',"
        "'descriptor': 'sci',"
        "'version': 'v00-01',"
        "'start_date': '20231212',"
        "'end_date': '20231212'}]"
    )

    parser = argparse.ArgumentParser(prog="imap_cli", description=description)
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--data-level", type=str, required=True, help=level_help)

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start time for the output data. Format: YYYYMMDD",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=False,
        help="End time for the output data. If not provided, start_time will be used "
        "for end_time. Format: YYYYMMDD",
    )
    # TODO: Will need to add some way of including pointing numbers

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version of the data. Format: vxx-xx",
    )
    parser.add_argument(
        "--dependency",
        type=str,
        required=True,
        help=dependency_help,
    )

    parser.add_argument(
        "--upload-to-sdc",
        action="store_true",
        required=False,
        help="Upload completed output files to the IMAP SDC.",
    )
    args = parser.parse_args()

    return args


def _validate_args(args):
    """Ensure that the  arguments are valid before kicking off the processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values
    """
    if args.instrument not in imap_data_access.VALID_INSTRUMENTS:
        raise ValueError(
            f"{args.instrument} is not in the supported instrument list: "
            f"{imap_processing.INSTRUMENTS}"
        )
    if args.data_level not in imap_processing.PROCESSING_LEVELS[args.instrument]:
        raise ValueError(
            f"{args.data_level} is not a supported data level for the {args.instrument}"
            " instrument, valid levels are: "
            f"{imap_processing.PROCESSING_LEVELS[args.instrument]}"
        )


class ProcessInstrument(ABC):
    """An abstract base class containing a method to process an instrument.

    Attributes
    ----------
    data_level : str
        The data level to process (e.g. ``l1a``)
    dependency_str : str
        A string representation of the dependencies for the instrument in the
        format: "[{
            'instrument': 'mag',
            'data_level': 'l0',
            'descriptor': 'sci',
            'version': 'v00-01',
            'start_date': '20231212',
            'end_date': '20231212'
        }]"
    start_date : str
        The start date for the output data. Format: YYYYMMDD
    end_date : str
        The end date for the output data. Format: YYYYMMDD
    version : str
        The version of the data. Format: vxx-xx
    upload_to_sdc : bool
        A flag indicating whether to upload the output file to the SDC.
    """

    def __init__(
        self,
        data_level: str,
        dependency_str: str,
        start_date: str,
        end_date: str,
        version: str,
        upload_to_sdc: bool,
    ) -> None:
        self.data_level = data_level

        # Convert string into a dictionary
        self.dependencies = loads(dependency_str.replace("'", '"'))

        self.start_date = start_date
        self.end_date = end_date
        if not end_date:
            self.end_date = start_date
            print(f"Setting end time to start time: {start_date}")

        self.version = version
        self.upload_to_sdc = upload_to_sdc

    def download_dependencies(self):
        """Download the dependencies for the instrument.

        Returns
        -------
        file_list: list[str]
            A list of file paths to the downloaded dependencies.
        """
        file_list = []
        for dependency in self.dependencies:
            try:
                # TODO: Validate dep dict
                # TODO: determine what dependency information is optional
                # TODO: Add in timestamps and descriptor to query
                return_query = imap_data_access.query(
                    instrument=dependency["instrument"],
                    data_level=dependency["data_level"],
                    version=dependency["version"],
                )
            except HTTPError as e:
                raise ValueError(f"Unable to download files from {dependency}") from e

            if not return_query:
                raise FileNotFoundError(
                    f"File not found for required dependency "
                    f"{dependency} while attempting to create file."
                    f"This should never occur "
                    f"in normal processing."
                )

            file_list.append(imap_data_access.download(return_query[0]["file_path"]))
        return file_list

    @abstractmethod
    def process(self):
        """Perform instrument specific processing."""
        raise NotImplementedError


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def process(self):
        """Perform CoDICE specific processing."""
        print(f"Processing CoDICE {self.data_level}")


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def process(self):
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {self.data_level}")


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def process(self):
        """Perform IMAP-Hi specific processing."""
        print(f"Processing IMAP-Hi {self.data_level}")


class Hit(ProcessInstrument):
    """Process HIT."""

    def process(self):
        """Perform HIT specific processing."""
        print(f"Processing HIT {self.data_level}")


class Idex(ProcessInstrument):
    """Process IDEX."""

    def process(self):
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {self.data_level}")


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def process(self):
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {self.data_level}")


class Mag(ProcessInstrument):
    """Process MAG."""

    def process(self):
        """Perform MAG specific processing."""
        print(f"Processing MAG {self.data_level}")
        file_paths = self.download_dependencies()

        if self.data_level == "l1a":
            # File path is expected output file path
            if len(file_paths) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1A:"
                    f"{file_paths}. Expected only one dependency."
                )
            filename_norm = imap_data_access.ScienceFilePath.generate_from_inputs(
                "mag", "l1a", "raw-norm", self.start_date, self.end_date, self.version
            ).construct_path()
            filename_burst = imap_data_access.ScienceFilePath.generate_from_inputs(
                "mag", "l1a", "raw-burst", self.start_date, self.end_date, self.version
            ).construct_path()
            mag_l1a(file_paths[0], filename_norm, filename_burst)

            if self.upload_to_sdc:
                # TODO: figure out data_dir, because now this fails.
                #  Should switch to using IMAP_DATA_DIR env var.
                if filename_norm.exists():
                    logging.info(f"Uploading file: {filename_norm}")
                    imap_data_access.upload(filename_norm)
                if filename_burst.exists():
                    logging.info(f"Uploading file: {filename_burst}")
                    imap_data_access.upload(filename_burst)


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def process(self):
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {self.data_level}")


class Swe(ProcessInstrument):
    """Process SWE."""

    def process(self):
        """Perform SWE specific processing."""
        # self.file_path example:
        # imap/swe/l1a/2023/09/imap_swe_l1a_sci_20230927_20230927_v01-00.cdf
        dependencies = self.download_dependencies()
        print(f"Processing SWE {self.data_level}")

        # TODO: currently assumes just the first path returned is the one to use

        if self.data_level == "l1a":
            processed_data = swe_l1a(Path(dependencies[0]))
            for data in processed_data:
                # write data to cdf
                file = imap_data_access.ScienceFilePath.generate_from_inputs(
                    "swe",
                    "l1a",
                    data["descriptor"],
                    self.start_date,
                    self.end_date,
                    self.version,
                )

                cdf_file_path = write_cdf(
                    data=data["data"], filepath=file.construct_path()
                )

                print(f"processed file path: {cdf_file_path}")

                if self.upload_to_sdc:
                    imap_data_access.upload(cdf_file_path)
                    print(f"Uploading file: {cdf_file_path}")

        elif self.data_level == "l1b":
            # read CDF file
            l1a_dataset = cdf_to_xarray(dependencies[0])
            processed_data = swe_l1b(l1a_dataset)
            # TODO: Update this descriptor
            descriptor = "test"
            file = imap_data_access.ScienceFilePath.generate_from_inputs(
                "swe", "l1b", descriptor, self.start_date, self.end_date, self.version
            )

            cdf_file_path = write_cdf(
                data=processed_data, filepath=file.construct_path()
            )
            print(f"processed file path: {cdf_file_path}")
            if self.upload_to_sdc:
                imap_data_access.upload(cdf_file_path)
                print(f"Uploading file: {cdf_file_path}")

        else:
            print("No code to process this level")


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def process(self):
        """Perform IMAP-Ultra specific processing."""
        print(f"Processing IMAP-Ultra {self.data_level}")


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
    instrument = cls(
        args.data_level,
        args.dependency,
        args.start_date,
        args.end_date,
        args.version,
        args.upload_to_sdc,
    )
    instrument.process()


if __name__ == "__main__":
    main()
