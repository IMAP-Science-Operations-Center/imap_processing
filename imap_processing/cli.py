#!/usr/bin/env python3
"""
Run the processing for a specific instrument & data level.

This module serves as a command line utility to invoke the processing for
a user-supplied instrument and data level.

Examples
--------
    imap_cli --instrument <instrument> --level <data_level>
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

import imap_data_access
import xarray as xr
from imap_data_access.processing_input import (
    ProcessingInputCollection,
)

import imap_processing
from imap_processing._version import __version__, __version_tuple__  # noqa: F401
from imap_processing.cdf.utils import load_cdf, write_cdf

# TODO: change how we import things and also folder
# structure may?
# From this:
#   from imap_processing.cdf.utils import write_cdf
# To this:
#   from imap_processing import cdf
# In code:
#   call cdf.utils.write_cdf
from imap_processing.codice import codice_l1a, codice_l1b
from imap_processing.glows.l1a.glows_l1a import glows_l1a
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l2.glows_l2 import glows_l2
from imap_processing.hi.l1a import hi_l1a
from imap_processing.hi.l1b import hi_l1b
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hit.l1a.hit_l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import hit_l1b
from imap_processing.hit.l2.hit_l2 import hit_l2
from imap_processing.idex.idex_l1a import PacketParser
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.lo.l1a import lo_l1a
from imap_processing.lo.l1b import lo_l1b
from imap_processing.lo.l1c import lo_l1c
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1b.mag_l1b import mag_l1b
from imap_processing.mag.l1c.mag_l1c import mag_l1c
from imap_processing.mag.l2.mag_l2 import mag_l2
from imap_processing.spacecraft import quaternions
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import swapi_l2
from imap_processing.swapi.swapi_utils import read_swapi_lut_table
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b import ultra_l1b
from imap_processing.ultra.l1c import ultra_l1c

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    The expected input format is:
    --instrument "mag"
    --data-level "l1a"
    --descriptor "all"
    --start-date "20231212"
    --version "v001"
    --dependency '[
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-cal_20250101_v001.cdf",
                    "imap_mag_l1b-cal_20250103_20250104_v002.cdf"
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_idex_l2_sci_20240312_v000.cdf",
                    "imap_idex_l2_sci_20240312_v001.cdf"
                ]
            }
        ]'
    --upload-to-sdc

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    description = (
        "This command line program invokes the processing pipeline "
        "for a specific instrument and data level. Example usage: "
        '"imap_cli --instrument "mag" '
        '--data-level "l1a" '
        '--descriptor "all" '
        ' --start-date "20231212" '
        '--repointing "repoint12345" '
        '--version "v001" '
        '--dependency "['
        "    {"
        '        "type": "ancillary",'
        '        "files": ['
        '            "imap_mag_l1b-cal_20250101_v001.cdf",'
        '            "imap_mag_l1b-cal_20250103_20250104_v002.cdf"'
        "        ]"
        "    },"
        "    {"
        '        "type": "science",'
        '        "files": ['
        '            "imap_idex_l2_sci_20240312_v000.cdf",'
        '            "imap_idex_l2_sci_20240312_v001.cdf"'
        "        ]"
        "    }"
        "]"
        ' --upload-to-sdc"'
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_data_access.VALID_INSTRUMENTS}"
    )
    level_help = (
        "The data level to process. Acceptable values are: "
        f"{imap_processing.PROCESSING_LEVELS}"
    )
    descriptor_help = (
        "The descriptor of the product to process. This could be 'all' or a specific "
        "descriptor like 'sci-1min'. Default is 'all'."
    )
    dependency_help = (
        "Dependency information in str format."
        "Example:"
        "'["
        "    {"
        '        "type": "ancillary",'
        '        "files": ['
        '            "imap_mag_l1b-cal_20250101_v001.cdf",'
        '            "imap_mag_l1b-cal_20250103_20250104_v002.cdf"'
        "        ]"
        "    },"
        "    {"
        '        "type": "science",'
        '        "files": ['
        '            "imap_idex_l2_sci_20240312_v000.cdf",'
        '            "imap_idex_l2_sci_20240312_v001.cdf"'
        "        ]"
        "    }"
        "]'"
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
    # TODO: unused for now, but needed for batch job handling
    # pass through of status in AWS
    parser.add_argument(
        "--descriptor", type=str, required=False, help=descriptor_help, default="all"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=False,
        help="Start time for the output data. Format: YYYYMMDD",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=False,
        help="DEPRECATED: Do not use this."
        "End time for the output data. If not provided, start_time will be used "
        "for end_time. Format: YYYYMMDD",
    )
    parser.add_argument(
        "--repointing",
        type=str,
        required=False,
        help="Repointing time for output data. Replaces start_time if both are "
        "provided. Format: repoint#####",
    )

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


def _validate_args(args: argparse.Namespace) -> None:
    """
    Ensure that the  arguments are valid before kicking off the processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
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
    if args.start_date is None and args.repointing is None:
        raise ValueError(
            "Either start_date or repointing must be provided. "
            "Run 'imap_cli -h' for more information."
        )

    if (
        args.start_date is not None
        and not imap_data_access.ScienceFilePath.is_valid_date(args.start_date)
    ):
        raise ValueError(f"{args.start_date} is not a valid date, use format YYYYMMDD.")

    if (
        args.repointing is not None
        and not imap_data_access.ScienceFilePath.is_valid_repointing(args.repointing)
    ):
        raise ValueError(
            f"{args.repointing} is not a valid repointing, use format repoint#####."
        )

    if getattr(args, "end_date", None) is not None:
        logger.warning(
            "The end_date argument is deprecated and will be ignored. Do not use."
        )


class ProcessInstrument(ABC):
    """
    An abstract base class containing a method to process an instrument.

    Parameters
    ----------
    data_level : str
        The data level to process (e.g. ``l1a``).
    data_descriptor : str
        The descriptor of the data to process (e.g. ``sci``).
    dependency_str : str
        A string representation of the dependencies for the instrument in the
        format:
        '[
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-cal_20250101_v001.cdf",
                    "imap_mag_l1b-cal_20250103_20250104_v002.cdf"
                ]
            },
            {
                "type": "ancillary",
                "files": [
                    "imap_mag_l1b-lut_20250101_v001.cdf",
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_mag_l1a_norm-magi_20240312_v000.cdf",
                    "imap_mag_l1a_norm-magi_20240312_v001.cdf"
                ]
            },
            {
                "type": "science",
                "files": [
                    "imap_idex_l2_sci_20240312_v000.cdf",
                    "imap_idex_l2_sci_20240312_v001.cdf"
                ]
            }
        ]'
        This is what ProcessingInputCollection.serialize() outputs.
    start_date : str
        The start date for the output data in YYYYMMDD format.
    repointing : str
        The repointing for the output data in the format 'repoint#####'.
    version : str
        The version of the data in vXXX format.
    upload_to_sdc : bool
        A flag indicating whether to upload the output file to the SDC.
    """

    def __init__(
        self,
        data_level: str,
        data_descriptor: str,
        dependency_str: str,
        start_date: str | None,
        repointing: str | None,
        version: str,
        upload_to_sdc: bool,
    ) -> None:
        self.data_level = data_level
        self.descriptor = data_descriptor

        self.dependency_str = dependency_str

        self.start_date = start_date
        self.repointing = repointing

        self.version = version
        self.upload_to_sdc = upload_to_sdc

    def upload_products(self, products: list[Path]) -> None:
        """
        Upload data products to the IMAP SDC.

        Parameters
        ----------
        products : list[Path]
            A list of file paths to upload to the SDC.
        """
        if self.upload_to_sdc:
            if len(products) == 0:
                logger.info("No files to upload.")
            for filename in products:
                logger.info(f"Uploading file: {filename}")
                imap_data_access.upload(filename)

    @final
    def process(self) -> None:
        """
        Run the processing workflow and cannot be overridden by subclasses.

        Each IMAP processing step consists of three steps:
        1. Pre-processing actions such as downloading dependencies for processing.
        2. Do the data processing. The result of this step will usually be a list
        of new products (files).
        3. Post-processing actions such as uploading files to the IMAP SDC.
        """
        logger.info(f"IMAP Processing Version: {imap_processing._version.__version__}")
        logger.info(f"Processing {self.__class__.__name__} level {self.data_level}")
        logger.info("Beginning preprocessing (download dependencies)")
        dependencies = self.pre_processing()
        logger.info("Beginning actual processing")
        products = self.do_processing(dependencies)
        logger.info("Beginning postprocessing (uploading data products)")
        self.post_processing(products, dependencies)
        logger.info("Processing complete")

    def pre_processing(self) -> ProcessingInputCollection:
        """
        Complete pre-processing.

        For this baseclass, pre-processing consists of downloading dependencies
        for processing. Child classes can override this method to customize the
        pre-processing actions.

        Returns
        -------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.
        """
        dependencies = ProcessingInputCollection()
        dependencies.deserialize(self.dependency_str)
        dependencies.download_all_files()
        return dependencies

    @abstractmethod
    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Abstract method that processes the IMAP processing steps.

        All child classes must implement this method. Input is
        object containing dependencies and output is
        list of xr.Dataset containing processed data(s).

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        list[xr.Dataset]
            List of products produced.
        """
        raise NotImplementedError

    def post_processing(
        self, datasets: list[xr.Dataset], dependencies: ProcessingInputCollection
    ) -> None:
        """
        Complete post-processing.

        Default post-processing consists of writing the datasets to local storage
        and then uploading those newly generated products to the IMAP SDC.
        Child classes can override this method to customize the
        post-processing actions.

        The values from start_date and/or repointing are used to generate the output
        file name if supplied. All other filename fields are derived from the
        dataset attributes.

        Parameters
        ----------
        datasets : list[xarray.Dataset]
            A list of datasets (products) produced by do_processing method.
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.
        """
        if len(datasets) == 0:
            logger.info("No products to write to CDF file.")
            return

        logger.info("Writing products to local storage")

        logger.info("Dataset version: %s", self.version)
        # Parent files used to create these datasets
        # https://spdf.gsfc.nasa.gov/istp_guide/gattributes.html.
        parent_files = [p.name for p in dependencies.get_file_paths()]
        logger.info("Parent files: %s", parent_files)

        # Format version to vXXX if not already in that format. Eg.
        # If version is passed in as 1 or 001, it will be converted to v001.
        r = re.compile(r"v\d{3}")
        if not isinstance(self.version, str) or r.match(self.version) is None:
            self.version = f"v{int(self.version):03d}"  # vXXX

        # Start date is either the start date or the repointing.
        # if it is the repointing, default to using the first epoch in the file as
        # start_date.
        # If it is start_date, skip repointing in the output filename.

        products = []
        for ds in datasets:
            ds.attrs["Data_version"] = self.version
            ds.attrs["Repointing"] = self.repointing
            ds.attrs["Start_date"] = self.start_date
            ds.attrs["Parents"] = parent_files
            products.append(write_cdf(ds))

        self.upload_products(products)


class Codice(ProcessInstrument):
    """Process CoDICE."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform CoDICE specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
         dataset : xr.Dataset
            Xr.Dataset of cdf file paths.
        """
        print(f"Processing CoDICE {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for CoDICE L1a:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data
            science_files = dependencies.get_file_paths(source="codice")
            datasets = codice_l1a.process_codice_l1a(science_files[0])

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for CoDICE L1b:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data
            science_files = dependencies.get_file_paths(source="codice")
            dependency = load_cdf(science_files[0])
            datasets = [codice_l1b.process_codice_l1b(dependency)]

        return datasets


class Glows(ProcessInstrument):
    """Process GLOWS."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform GLOWS specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.dataset of products.
        """
        print(f"Processing GLOWS {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1A:"
                    f"{dependency_list}. Expected only one input dependency."
                )
            science_files = dependencies.get_file_paths(source="glows")
            datasets = glows_l1a(science_files[0])

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L1B:"
                    f"{dependency_list}. Expected at least one input dependency."
                )
            science_files = dependencies.get_file_paths(source="glows")
            input_dataset = load_cdf(science_files[0])
            datasets = [glows_l1b(input_dataset)]

        if self.data_level == "l2":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for GLOWS L2:"
                    f"{dependency_list}. Expected only one input dependency."
                )
            science_files = dependencies.get_file_paths(source="glows")
            input_dataset = load_cdf(science_files[0])
            datasets = glows_l2(input_dataset)

        return datasets


class Hi(ProcessInstrument):
    """Process IMAP-Hi."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Hi specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing IMAP-Hi {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for Hi L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            science_files = dependencies.get_file_paths(source="hi")
            datasets = hi_l1a.hi_l1a(science_files[0])
        elif self.data_level == "l1b":
            l0_files = dependencies.get_file_paths(source="hi", descriptor="raw")
            if l0_files:
                datasets = hi_l1b.hi_l1b(l0_files[0])
            else:
                l1a_files = dependencies.get_file_paths(source="hi")
                datasets = hi_l1b.hi_l1b(load_cdf(l1a_files[0]))
        elif self.data_level == "l1c":
            # TODO: Add PSET calibration product config file dependency and remove
            #    below injected dependency
            hi_dependencies = dependencies.get_file_paths(source="hi")
            hi_dependencies.append(
                Path(__file__).parent
                / "tests/hi/test_data/l1"
                / "imap_his_pset-calibration-prod-config_20240101_v001.csv"
            )
            hi_dependencies[0] = load_cdf(hi_dependencies[0])
            datasets = hi_l1c.hi_l1c(hi_dependencies)
        else:
            raise NotImplementedError(
                f"Hi processing not implemented for level {self.data_level}"
            )
        return datasets


class Hit(ProcessInstrument):
    """Process HIT."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform HIT specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of datasets.
        """
        print(f"Processing HIT {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data to L1A products
            science_files = dependencies.get_file_paths(source="hit")
            datasets = hit_l1a(science_files[0])

        elif self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )
            data_dict = {}
            # TODO: Check this and update with new features as needed.
            l0_files = dependencies.get_file_paths(source="hit", descriptor="raw")
            l1a_files = dependencies.get_file_paths(source="hit")
            if len(l0_files) > 0:
                # Add path to CCSDS file to process housekeeping
                data_dict["imap_hit_l0_raw"] = l0_files[0]
            else:
                # Add L1A dataset to process science data
                l1a_dataset = load_cdf(l1a_files[0])
                data_dict[l1a_dataset.attrs["Logical_source"]] = l1a_dataset
            # process data to L1B products
            datasets = hit_l1b(data_dict)
        elif self.data_level == "l2":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L2:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # Add L1B dataset to process science data
            science_files = dependencies.get_file_paths(source="hit")
            l1b_dataset = load_cdf(science_files[0])
            # process data to L2 products
            datasets = hit_l2(l1b_dataset)

        return datasets


class Idex(ProcessInstrument):
    """Process IDEX."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IDEX specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of cdf file paths.
        """
        print(f"Processing IDEX {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # get l0 file
            science_files = dependencies.get_file_paths(source="idex")
            datasets = PacketParser(science_files[0]).data
        elif self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1B:"
                    f"{dependency_list}. Expected only one science dependency."
                )
            # get CDF file
            science_files = dependencies.get_file_paths(source="idex")
            # process data
            dependency = load_cdf(science_files[0])
            datasets = [idex_l1b(dependency)]
        return datasets


class Lo(ProcessInstrument):
    """Process IMAP-Lo."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Lo specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of output files.
        """
        print(f"Processing IMAP-Lo {self.data_level}")
        datasets: list[xr.Dataset] = []
        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # L1A packet / products are 1 to 1. Should only have
            # one dependency file
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IMAP-Lo L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            science_files = dependencies.get_file_paths(source="lo")
            datasets = lo_l1a.lo_l1a(science_files[0])

        elif self.data_level == "l1b":
            data_dict = {}
            # TODO: Check this and update with new features as needed.
            for input_type in dependencies.processing_input:
                science_files = dependencies.get_file_paths(
                    source="lo", descriptor=input_type.descriptor
                )
                dataset = load_cdf(science_files[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = lo_l1b.lo_l1b(data_dict)

        elif self.data_level == "l1c":
            data_dict = {}
            for input_type in dependencies.processing_input:
                science_files = dependencies.get_file_paths(
                    source="lo", descriptor=input_type.descriptor
                )
                dataset = load_cdf(science_files[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            # TODO: This is returning the wrong type
            datasets = lo_l1c.lo_l1c(data_dict)

        return datasets


class Mag(ProcessInstrument):
    """Process MAG."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform MAG specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of output files.
        """
        print(f"Processing MAG {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        science_files = dependencies.get_file_paths(source="mag")
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # TODO: Update this type

            datasets = mag_l1a(science_files[0])

        if self.data_level == "l1b":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )
            input_data = load_cdf(science_files[0])
            datasets = [mag_l1b(input_data)]

        if self.data_level == "l1c":
            input_data = [load_cdf(dep) for dep in science_files]
            # Input datasets can be in any order, and are validated within mag_l1c
            if len(input_data) == 1:
                datasets = [mag_l1c(input_data[0])]
            elif len(input_data) == 2:
                datasets = [mag_l1c(input_data[0], input_data[1])]
            else:
                raise ValueError(
                    f"Invalid dependencies found for MAG L1C:"
                    f"{dependencies}. Expected one or two dependencies."
                )

        if self.data_level == "l2":
            # TODO: Overwrite dependencies with versions from offsets file
            # TODO: Ensure that parent_files attribute works with that
            input_data = load_cdf(science_files[0])
            # TODO: use ancillary from input
            calibration_dataset = load_cdf(
                Path(__file__).parent
                / "tests"
                / "mag"
                / "validation"
                / "calibration"
                / "imap_mag_l2-calibration-matrices_20251017_v004.cdf"
            )

            offset_dataset = load_cdf(
                Path(__file__).parent
                / "tests"
                / "mag"
                / "validation"
                / "calibration"
                / "imap_mag_l2-offsets-norm_20251017_20251017_v001.cdf"
            )
            # TODO: Test data missing
            datasets = mag_l2(calibration_dataset, offset_dataset, input_data)

        return datasets


class Spacecraft(ProcessInstrument):
    """Process Spacecraft data."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform Spacecraft specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing Spacecraft {self.data_level}")

        if self.data_level != "l1a":
            raise NotImplementedError(
                f"Spacecraft processing not implemented for level {self.data_level}"
            )

        # File path is expected output file path
        input_files = dependencies.get_file_paths(source="spacecraft")
        if len(input_files) > 1:
            raise ValueError(
                f"Unexpected dependencies found for Spacecraft L1A: "
                f"{input_files}. Expected only one dependency."
            )
        datasets = list(quaternions.process_quaternions(input_files[0]))
        return datasets


class Swapi(ProcessInstrument):
    """Process SWAPI."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform SWAPI specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing SWAPI {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1":
            # For science, we expect l0 raw file and L1 housekeeping file
            if self.descriptor == "sci" and len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 science:"
                    f"{dependency_list}. Expected only two dependencies."
                )
            # For housekeeping, we expect only L0 raw file
            if self.descriptor == "hk" and len(dependency_list) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 housekeeping:"
                    f"{dependency_list}. Expected only one dependency."
                )

            dependent_files = []
            l0_files = dependencies.get_file_paths(descriptor="raw")
            # TODO: handle multiples files as needed in the future
            dependent_files.append(l0_files[0])

            if self.descriptor == "sci":
                # TODO: handle multiples files as needed in the future
                hk_files = dependencies.get_file_paths(descriptor="hk")
                dependent_files.append(hk_files[0])

            # process science or housekeeping data
            datasets = swapi_l1(dependent_files)
        elif self.data_level == "l2":
            if len(dependency_list) != 3:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L2:"
                    f"{dependency_list}. Expected 3 dependencies."
                )
            # process data
            science_files = dependencies.get_file_paths(
                source="swapi", descriptor="sci"
            )
            esa_table_files = dependencies.get_file_paths(
                source="swapi", descriptor="esa-unit-conversion"
            )
            lut_notes_files = dependencies.get_file_paths(
                source="swapi", descriptor="lut-notes"
            )
            esa_table_df = read_swapi_lut_table(esa_table_files[0])
            lut_notes_df = read_swapi_lut_table(lut_notes_files[0])
            l1_dataset = load_cdf(science_files[0])
            datasets = [swapi_l2(l1_dataset, esa_table_df, lut_notes_df)]

        return datasets


class Swe(ProcessInstrument):
    """Process SWE."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform SWE specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        dataset : xr.Dataset
            Path to cdf file.
        """
        print(f"Processing SWE {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            science_files = dependencies.get_file_paths(source="swe")
            datasets = swe_l1a(str(science_files[0]))
            # Right now, we only process science data. Therefore,
            # we expect only one dataset to be returned.

        elif self.data_level == "l1b":
            if len(dependency_list) != 4:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1B:"
                    f"{dependency_list}. Expected exactly four dependencies."
                )

            science_files = dependencies.get_file_paths("swe", "sci")
            if len(science_files) > 1:
                raise ValueError(
                    "Multiple science files processing is not supported for SWE L1B."
                )

            datasets = swe_l1b(dependencies)
        else:
            print("Did not recognize data level. No processing done.")

        return datasets


class Ultra(ProcessInstrument):
    """Process IMAP-Ultra."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset]:
        """
        Perform IMAP-Ultra specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : xr.Dataset
            Xr.Dataset of products.
        """
        print(f"Processing IMAP-Ultra {self.data_level}")
        datasets: list[xr.Dataset] = []

        dependency_list = dependencies.processing_input
        if self.data_level == "l1a":
            # File path is expected output file path
            if len(dependency_list) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for ULTRA L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            science_files = dependencies.get_file_paths(source="ultra")
            datasets = ultra_l1a.ultra_l1a(science_files[0])

        elif self.data_level == "l1b":
            data_dict = {}
            for dep in dependency_list:
                dataset = load_cdf(dep.imap_file_paths[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1b.ultra_l1b(data_dict)

        elif self.data_level == "l1c":
            data_dict = {}
            for dep in dependency_list:
                dataset = load_cdf(dep.imap_file_paths[0])
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = ultra_l1c.ultra_l1c(data_dict)

        return datasets


def main() -> None:
    """
    Run the processing for a specific instrument & data level.

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
        args.descriptor,
        args.dependency,
        args.start_date,
        args.repointing,
        args.version,
        args.upload_to_sdc,
    )
    instrument.process()


if __name__ == "__main__":
    main()
