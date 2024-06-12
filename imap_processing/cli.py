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
from typing import final
from urllib.error import HTTPError

import imap_data_access

import imap_processing
from imap_processing.cdf.utils import load_cdf, write_cdf

# TODO: change how we import things and also folder
# structure may?
# From this:
#   from imap_processing.cdf.utils import write_cdf
# To this:
#   from imap_processing import cdf
# In code:
#   call cdf.utils.write_cdf
from imap_processing.codice import codice_l1a
from imap_processing.glows.l1a.glows_l1a import glows_l1a
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.hi.l1a import hi_l1a
from imap_processing.hit.l1a.hit_l1a import hit_l1a
from imap_processing.idex.idex_packet_parser import PacketParser
from imap_processing.lo.l1a import lo_l1a
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b import ultra_l1b
from imap_processing.ultra.l1c import ultra_l1c

logger = logging.getLogger(__name__)


def _parse_args():
    """Parse the command line arguments.

    The expected input format is:
    --instrument "mag"
    --data-level "l1a"
    --start-date "20231212"
    --version "v001"
    --dependency "[
        {
            'instrument': 'mag',
            'data_level': 'l0',
            'descriptor': 'sci',
            'version': 'v001',
            'start_date': '20231212'
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
        '--version "v001"'
        '--dependency "['
        '   {"instrument": "mag",'
        '   "data_level": "l0"',
        '   "descriptor": "sci"',
        '   "version": "v001"',
        '   "start_date": "20231212"',
        '}]" --upload-to-sdc"',
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_data_access.VALID_INSTRUMENTS}"
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
        "'version': 'v001',"
        "'start_date': '20231212'}]"
    )

    parser = argparse.ArgumentParser(prog="imap_cli", description=description)
    # TODO: Add version here and change our current "version" to "data-version"?
    # parser.add_argument(
    #     "--version",
    #     action="version",
    #     version=f"%(prog)s {imap_processing.__version__}",
    # )
    # Logging level
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Add verbose output",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
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
        help="Version of the data. Format: vXXX",
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
            f"{imap_data_access.VALID_INSTRUMENTS}"
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
            'start_date': '20231212'
        }]"
    start_date : str
        The start date for the output data. Format: YYYYMMDD
    end_date : str
        The end date for the output data. Format: YYYYMMDD
    version : str
        The version of the data. Format: vXXX
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
                    descriptor=dependency["descriptor"],
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

    def upload_products(self, products: list[str]):
        """
        Upload data products to the IMAP SDC.

        Attributes
        ----------
        products : list[str]
        A list of file paths to upload to the SDC.
        """
        if self.upload_to_sdc:
            if len(products) == 0:
                logger.info("No files to upload.")
            for filename in products:
                logger.info(f"Uploading file: {filename}")
                imap_data_access.upload(filename)

    @final
    def process(self):
        """
        Run the processing workflow and cannot be overridden by subclasses.

        Each IMAP processing step consists of three steps:
        1. Pre-processing actions such as downloading dependencies for processing.
        2. Do the data processing. The result of this step will usually be a list
        of new products (files).
        3. Post-processing actions such as uploading files to the IMAP SDC.
        """
        logger.info(f"IMAP Processing Version: {imap_processing.__version__}")
        logger.info(f"Processing {self.__class__.__name__} level {self.data_level}")
        logger.info("Beginning preprocessing (download dependencies)")
        dependencies = self.pre_processing()
        logger.info("Beginning actual processing")
        products = self.do_processing(dependencies)
        logger.info("Beginning postprocessing (uploading data products)")
        self.post_processing(products)
        logger.info("Processing complete")

    def pre_processing(self):
        """
        Complete pre-processing.

        For this baseclass, pre-processing consists of downloading dependencies
        for processing. Child classes can override this method to customize the
        pre-processing actions.

        Returns
        -------
        List of dependencies downloaded from the IMAP SDC.
        """
        return self.download_dependencies()

    @abstractmethod
    def do_processing(self, dependencies: list):
        """
        Abstract method that processes the IMAP processing steps.

        All child classes must implement this method. Input and outputs are
        typically lists of file paths but are free to any list.

        Attributes
        ----------
        dependencies : list
            List of dependencies to process

        Returns
        -------
        list
            List of products produced
        """
        raise NotImplementedError

    def post_processing(self, products: list[str]):
        """
        Complete post-processing.

        Default post-processing consists of uploading newly generated products
        to the IMAP SDC. Child classes can override this method to customize the
        post-processing actions.

        Attributes
        ----------
        products : list[str]
            A list of file paths (products) produced by do_processing method.
        """
        self.upload_products(products)


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def do_processing(self, dependencies):
        """Perform CoDICE specific processing."""
        print(f"Processing CoDICE {self.data_level}")

        if self.data_level == "l1a":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for CoDICE L1a:"
                    f"{dependencies}. Expected only one dependency."
                )
            # process data
            dataset = codice_l1a.process_codice_l1a(dependencies[0])
            cdf_file_path = dataset.attrs["cdf_filename"]
            return [cdf_file_path]


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def do_processing(self, dependencies):
        """Perform GLOWS specific processing."""
        print(f"Processing GLOWS {self.data_level}")
        products = []
        if self.data_level == "l1a":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1A:"
                    f"{dependencies}. Expected only one input dependency."
                )
            datasets = glows_l1a(dependencies[0], self.version)
            products = [write_cdf(dataset) for dataset in datasets]

        if self.data_level == "l1b":
            if len(dependencies) < 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1B:"
                    f"{dependencies}. Expected at least one input dependency."
                )
            dataset = glows_l1b(dependencies[0], self.version)
            products = write_cdf(dataset)

        return products


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def do_processing(self, dependencies: list):
        """
        Perform IMAP-Hi specific processing.

        Attributes
        ----------
        dependencies: list
        List of dependencies to process
        """
        print(f"Processing IMAP-Hi {self.data_level}")

        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for Hi L1A:"
                    f"{dependencies}. Expected only one dependency."
                )
            datasets = hi_l1a.hi_l1a(dependencies[0])
            products = [write_cdf(dataset) for dataset in datasets]
            return products


class Hit(ProcessInstrument):
    """Process HIT."""

    def do_processing(self, dependencies):
        """Perform HIT specific processing."""
        print(f"Processing HIT {self.data_level}")

        if self.data_level == "l1a":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1A:"
                    f"{dependencies}. Expected only one dependency."
                )
            # process data and write all processed data to CDF files
            products = hit_l1a(dependencies[0])
            return products


class Idex(ProcessInstrument):
    """Process IDEX."""

    def do_processing(self, dependencies):
        """Perform IDEX specific processing."""
        print(f"Processing IDEX {self.data_level}")

        if self.data_level == "l1":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1:"
                    f"{dependencies}. Expected only one dependency."
                )
            # read CDF file
            processed_data = PacketParser(dependencies[0]).data
            cdf_file_path = write_cdf(processed_data)
            print(f"processed file path: {cdf_file_path}")
            return [cdf_file_path]


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def do_processing(self, dependencies):
        """Perform IMAP-Lo specific processing."""
        print(f"Processing IMAP-Lo {self.data_level}")

        if self.data_level == "l1a":
            # L1A packet / products are 1 to 1. Should only have
            # one dependency file
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IMAP-Lo L1A:"
                    f"{dependencies}. Expected only one dependency."
                )
            output_files = lo_l1a.lo_l1a(dependencies[0])
            return [output_files]


class Mag(ProcessInstrument):
    """Process MAG."""

    def do_processing(self, file_paths):
        """Perform MAG specific processing."""
        print(f"Processing MAG {self.data_level}")

        if self.data_level == "l1a":
            # File path is expected output file path
            if len(file_paths) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1A:"
                    f"{file_paths}. Expected only one dependency."
                )
            output_files = mag_l1a(file_paths[0], data_version=self.version)
            return output_files


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def do_processing(self, dependencies):
        """Perform SWAPI specific processing."""
        print(f"Processing SWAPI {self.data_level}")

        if self.data_level == "l1":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1:"
                    f"{dependencies}. Expected only one dependency."
                )
            # process data
            processed_data = swapi_l1(dependencies[0])
            # Write all processed data to CDF files
            products = [write_cdf(dataset) for dataset in processed_data]
            return products


class Swe(ProcessInstrument):
    """Process SWE."""

    def do_processing(self, dependencies):
        """Perform SWE specific processing."""
        print(f"Processing SWE {self.data_level}")

        if self.data_level == "l1a":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1A:"
                    f"{dependencies}. Expected only one dependency."
                )
            processed_data = swe_l1a(Path(dependencies[0]), data_version=self.version)
            # Right now, we only process science data. Therefore,
            # we expect only one dataset to be returned.
            cdf_file_path = write_cdf(processed_data)
            print(f"processed file path: {cdf_file_path}")
            return [cdf_file_path]

        elif self.data_level == "l1b":
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1B:"
                    f"{dependencies}. Expected only one dependency."
                )
            # read CDF file
            l1a_dataset = load_cdf(dependencies[0])
            processed_data = swe_l1b(l1a_dataset)
            cdf_file_path = write_cdf(processed_data)
            print(f"processed file path: {cdf_file_path}")
            return [cdf_file_path]
        else:
            print("Did not recognize data level. No processing done.")


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def do_processing(self, dependencies: list):
        """
        Perform IMAP-Ultra specific processing.

        Attributes
        ----------
        dependencies: list
        List of dependencies to process
        """
        print(f"Processing IMAP-Ultra {self.data_level}")

        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependencies) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for ULTRA L1A:"
                    f"{dependencies}. Expected only one dependency."
                )

            datasets = ultra_l1a.ultra_l1a(dependencies[0])
            products = [write_cdf(dataset) for dataset in datasets]
            return products
        elif self.data_level == "l1b":
            data_dict = {}
            for dependency in dependencies:
                dataset = load_cdf(dependency)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1b.ultra_l1b(data_dict)
            products = [write_cdf(dataset) for dataset in datasets]
            return products
        elif self.data_level == "l1c":
            data_dict = {}
            for dependency in dependencies:
                dataset = load_cdf(dependency)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1c.ultra_l1c(data_dict)
            products = [write_cdf(dataset) for dataset in datasets]
            return products


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
