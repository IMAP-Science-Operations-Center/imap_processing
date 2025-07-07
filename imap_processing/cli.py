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
import numpy as np
import spiceypy
import xarray as xr
from imap_data_access import ScienceFilePath
from imap_data_access.io import download
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ProcessingInputType,
    RepointInput,
    SPICESource,
    SpinInput,
)

import imap_processing
from imap_processing._version import __version__, __version_tuple__  # noqa: F401
from imap_processing.ancillary.ancillary_dataset_combiner import MagAncillaryCombiner
from imap_processing.cdf.utils import load_cdf, write_cdf

# TODO: change how we import things and also folder
# structure may?
# From this:
#   from imap_processing.cdf.utils import write_cdf
# To this:
#   from imap_processing import cdf
# In code:
#   call cdf.utils.write_cdf
from imap_processing.codice import codice_l1a, codice_l1b, codice_l2
from imap_processing.glows.l1a.glows_l1a import glows_l1a
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l2.glows_l2 import glows_l2
from imap_processing.hi import hi_l1a, hi_l1b, hi_l1c, hi_l2
from imap_processing.hit.l1a.hit_l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import hit_l1b
from imap_processing.hit.l2.hit_l2 import hit_l2
from imap_processing.idex.idex_l1a import PacketParser
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import idex_l2a
from imap_processing.idex.idex_l2b import idex_l2b
from imap_processing.idex.idex_l2c import idex_l2c
from imap_processing.lo.l1a import lo_l1a
from imap_processing.lo.l1b import lo_l1b
from imap_processing.lo.l1c import lo_l1c
from imap_processing.lo.l2 import lo_l2
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1b.mag_l1b import mag_l1b
from imap_processing.mag.l1c.mag_l1c import mag_l1c
from imap_processing.mag.l2.mag_l2 import mag_l2
from imap_processing.spacecraft import quaternions
from imap_processing.spice import pointing_frame, repoint, spin
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import swapi_l2
from imap_processing.swapi.swapi_utils import read_swapi_lut_table
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l2.swe_l2 import swe_l2
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b import ultra_l1b
from imap_processing.ultra.l1c import ultra_l1c
from imap_processing.ultra.l2 import ultra_l2

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
        "    A path to a JSON file containing this same information may also be"
        "passed in. If dependency is a string ending in '.json', it will be interpreted"
        " as such a file path."
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

    # If the dependency argument was passed in as a json file, read it into a string
    if args.dependency.endswith(".json"):
        logger.info(
            f"Interpreting dependency argument as a JSON file: {args.dependency}"
        )
        dependency_filepath = download(args.dependency)
        with open(dependency_filepath) as f:
            args.dependency = f.read()

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

    class ImapFileExistsError(Exception):
        """Indicates a failure because the files already exist."""

        pass

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
            # Validate that the files don't already exist
            for filename in products:
                file_path = ScienceFilePath(filename)
                existing_file = imap_data_access.query(
                    instrument=file_path.instrument,
                    data_level=file_path.data_level,
                    descriptor=file_path.descriptor,
                    start_date=file_path.start_date,
                    end_date=file_path.start_date,
                    repointing=file_path.repointing,
                    version=file_path.version,
                    extension="cdf",
                    table="science",
                )
                if existing_file:
                    raise ProcessInstrument.ImapFileExistsError(
                        f"File {filename} already exists in the IMAP SDC. "
                        "No files were uploaded."
                        f"Generated files: {products}."
                    )

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
        4. Final cleanup actions.
        """
        logger.info(f"IMAP Processing Version: {imap_processing._version.__version__}")
        logger.info(f"Processing {self.__class__.__name__} level {self.data_level}")
        logger.info("Beginning preprocessing (download dependencies)")
        dependencies = self.pre_processing()
        logger.info("Beginning actual processing")
        products = self.do_processing(dependencies)
        logger.info("Beginning postprocessing (uploading data products)")
        self.post_processing(products, dependencies)
        self.cleanup()
        logger.info("Processing complete")

    def pre_processing(self) -> ProcessingInputCollection:
        """
        Complete pre-processing.

        For this baseclass, pre-processing consists of downloading dependencies
        for processing and furnishing any spice kernels in the input
        dependencies. Child classes can override this method to customize the
        pre-processing actions.

        Returns
        -------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.
        """
        dependencies = ProcessingInputCollection()
        dependencies.deserialize(self.dependency_str)
        dependencies.download_all_files()

        # Furnish spice kernels
        kernel_paths = dependencies.get_file_paths(data_type=SPICESource.SPICE.value)
        logger.info(f"Furnishing kernels: {[k.name for k in kernel_paths]}")
        spiceypy.furnsh([str(kernel_path.resolve()) for kernel_path in kernel_paths])

        # Set spin table paths in mutable module attributes
        spin.set_global_spin_table_paths(
            dependencies.get_file_paths(data_type=SpinInput.data_type)
        )

        # Set repoint table path in mutable module attribute
        repoint.set_global_repoint_table_paths(
            dependencies.get_file_paths(data_type=RepointInput.data_type)
        )

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
        self,
        processed_data: list[xr.Dataset | Path],
        dependencies: ProcessingInputCollection,
    ) -> None:
        """
        Complete post-processing.

        Default post-processing consists of the following:
        For each xarray.Dataset:
            1. Set `Data_version` global attribute.
            2. Set `Repointing` global attribute for appropriate products.
            3. Set `Start_date` global attribute.
            4. Set `Parents` global attribute.
            5. Write the xarray.Dataset to a local CDF file.
        The resulting paths to CDF files as well as any Path included in the
        `processed_data` input are then uploaded to the IMAP SDC.

        Child classes can override this method to customize the
        post-processing actions.

        The values from start_date and/or repointing are used to generate the output
        file name if supplied. All other filename fields are derived from the
        dataset attributes.

        Parameters
        ----------
        processed_data : list[xarray.Dataset | Path]
            A list of datasets (products) and paths produced by the do_processing
            method.
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.
        """
        if len(processed_data) == 0:
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
        for ds in processed_data:
            if isinstance(ds, xr.Dataset):
                ds.attrs["Data_version"] = self.version[1:]  # Strip 'v' from version
                if self.repointing is not None:
                    ds.attrs["Repointing"] = self.repointing
                ds.attrs["Start_date"] = self.start_date
                ds.attrs["Parents"] = parent_files
                products.append(write_cdf(ds))
            else:
                # A path to a product that was already written out
                products.append(ds)

        self.upload_products(products)

    @final
    def cleanup(self) -> None:
        """Cleanup from processing."""
        logger.info("Clearing furnished SPICE kernels")
        spiceypy.kclear()


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

        if self.data_level == "l1a":
            science_files = dependencies.get_file_paths(source="codice")
            if len(science_files) != 1:
                raise ValueError(
                    f"CoDICE L1A requires exactly one input science file, received: "
                    f"{science_files}."
                )
            # process data
            datasets = codice_l1a.process_codice_l1a(science_files[0])

        if self.data_level == "l1b":
            science_files = dependencies.get_file_paths(source="codice")
            if len(science_files) != 1:
                raise ValueError(
                    f"CoDICE L1B requires exactly one input science file, received: "
                    f"{science_files}."
                )
            # process data
            datasets = [codice_l1b.process_codice_l1b(science_files[0])]

        if self.data_level == "l2":
            science_files = dependencies.get_file_paths(source="codice")
            if len(science_files) != 1:
                raise ValueError(
                    f"CoDICE L2 requires exactly one input science file, received: "
                    f"{science_files}."
                )
            datasets = [codice_l2.process_codice_l2(science_files[0])]

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

        if self.data_level == "l1a":
            science_files = dependencies.get_file_paths(source="glows")
            if len(science_files) != 1:
                raise ValueError(
                    f"GLOWS L1A requires exactly one input science file, received: "
                    f"{science_files}."
                )
            datasets = glows_l1a(science_files[0])

        if self.data_level == "l1b":
            science_files = dependencies.get_file_paths(source="glows")
            if len(science_files) != 1:
                raise ValueError(
                    f"GLOWS L1A requires exactly one input science file, received: "
                    f"{science_files}."
                )
            input_dataset = load_cdf(science_files[0])
            datasets = [glows_l1b(input_dataset)]

        if self.data_level == "l2":
            science_files = dependencies.get_file_paths(source="glows")
            if len(science_files) != 1:
                raise ValueError(
                    f"GLOWS L1A requires exactly one input science file, received: "
                    f"{science_files}."
                )
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

        if self.data_level == "l1a":
            science_files = dependencies.get_file_paths(source="hi")
            if len(science_files) != 1:
                raise ValueError(
                    f"Unexpected science_files found for Hi L1A:"
                    f"{science_files}. Expected only one dependency."
                )
            datasets = hi_l1a.hi_l1a(science_files[0])
        elif self.data_level == "l1b":
            l0_files = dependencies.get_file_paths(source="hi", descriptor="raw")
            if l0_files:
                datasets = hi_l1b.hi_l1b(l0_files[0])
            else:
                l1a_files = dependencies.get_file_paths(source="hi", data_type="l1a")
                datasets = hi_l1b.hi_l1b(load_cdf(l1a_files[0]))
        elif self.data_level == "l1c":
            science_paths = dependencies.get_file_paths(source="hi", data_type="l1b")
            if len(science_paths) != 1:
                raise ValueError(
                    f"Expected only one science dependency. Got {science_paths}"
                )
            anc_paths = dependencies.get_file_paths(data_type="ancillary")
            if len(anc_paths) != 1:
                raise ValueError(
                    f"Expected only one ancillary dependency. Got {anc_paths}"
                )
            datasets = hi_l1c.hi_l1c(load_cdf(science_paths[0]), anc_paths[0])
        elif self.data_level == "l2":
            science_paths = dependencies.get_file_paths(source="hi", data_type="l1c")
            # TODO get ancillary paths
            geometric_factors_path = ""
            esa_energies_path = ""
            datasets = hi_l2.hi_l2(
                science_paths,
                geometric_factors_path,
                esa_energies_path,
                self.descriptor,
            )
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
            # 1 science files and 2 spice files
            if len(dependency_list) > 3:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # process data to L1A products
            science_files = dependencies.get_file_paths(source="hit", descriptor="raw")
            datasets = hit_l1a(science_files[0])

        elif self.data_level == "l1b":
            data_dict = {}
            # TODO: Sean removed the file number error handling to work with the
            #  new SPICE dependencies for SIT-4. Need to review and make changes
            #  if needed.
            l0_files = dependencies.get_file_paths(source="hit", descriptor="raw")
            l1a_files = dependencies.get_file_paths(source="hit", data_type="l1a")
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
            if len(dependency_list) != 5:
                raise ValueError(
                    f"Unexpected dependencies found for HIT L2:"
                    f"{dependency_list}. Expected only one dependency."
                )
            # Add L1B dataset to process science data
            science_files = dependencies.get_file_paths(
                source="hit", descriptor="-rates"
            )
            ancillary_files = dependencies.get_file_paths(
                source="hit", descriptor="-dt"
            )
            if len(science_files) > 1:
                raise ValueError(
                    "Multiple science files processing is not supported for HIT L2."
                )
            if len(ancillary_files) != 4:
                raise ValueError(
                    "Unexpected ancillary files found for HIT L2:"
                    f"{ancillary_files}. Expected 4 ancillary files."
                )
            l1b_dataset = load_cdf(science_files[0])
            # process data to L2 products
            datasets = hit_l2(l1b_dataset, ancillary_files)

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
            if len(dependency_list) > 2:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1A:"
                    f"{dependency_list}. Expected only two dependency."
                )
            # get l0 file
            science_files = dependencies.get_file_paths(source="idex")
            datasets = PacketParser(science_files[0]).data
        elif self.data_level == "l1b":
            if len(dependency_list) != 3:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L1B:"
                    f"{dependency_list}. Expected only three dependencies."
                )
            # get CDF file
            science_files = dependencies.get_file_paths(source="idex")
            # process data
            dependency = load_cdf(science_files[0])
            datasets = [idex_l1b(dependency)]
        elif self.data_level == "l2a":
            if len(dependency_list) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L2A:"
                    f"{dependency_list}. Expected only one dependency."
                )
            science_files = dependencies.get_file_paths(source="idex")
            dependency = load_cdf(science_files[0])
            datasets = [idex_l2a(dependency)]
        elif self.data_level == "l2b":
            if len(dependency_list) < 3 or len(dependency_list) > 4:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L2B:"
                    f"{dependency_list}. Expected three or four dependencies."
                )
            sci_files = dependencies.get_file_paths(
                source="idex", descriptor="sci-1week"
            )
            sci_dependencies = [load_cdf(f) for f in sci_files]
            hk_files = dependencies.get_file_paths(source="idex", descriptor="evt")
            # Remove duplicate housekeeping files
            hk_dependencies = [load_cdf(dep) for dep in list(set(hk_files))]
            datasets = [idex_l2b(sci_dependencies, hk_dependencies)]
        elif self.data_level == "l2c":
            if len(dependency_list) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for IDEX L2C:"
                    f"{dependency_list}. Expected only one dependency."
                )
            sci_files = dependencies.get_file_paths(source="idex", descriptor="sci-1mo")
            dependency = load_cdf(sci_files[0])
            datasets = idex_l2c(dependency)
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
        if self.data_level == "l1a":
            # L1A packet / products are 1 to 1. Should only have
            # one dependency file
            science_files = dependencies.get_file_paths(source="lo", data_type="l0")
            if len(science_files) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for IMAP-Lo L1A:"
                    f"{science_files}. Expected only one dependency."
                )
            datasets = lo_l1a.lo_l1a(science_files[0])

        elif self.data_level == "l1b":
            data_dict = {}
            science_files = dependencies.get_file_paths(source="lo", data_type="l1a")
            logger.info(f"Science files for L1B: {science_files}")
            for file in science_files:
                dataset = load_cdf(file)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = lo_l1b.lo_l1b(data_dict)

        elif self.data_level == "l1c":
            data_dict = {}
            anc_dependencies: list = dependencies.get_file_paths(
                source="lo", descriptor="goodtimes"
            )
            science_files = dependencies.get_file_paths(source="lo", descriptor="de")
            for file in science_files:
                dataset = load_cdf(file)
                data_dict[dataset.attrs["Logical_source"]] = dataset
            datasets = lo_l1c.lo_l1c(data_dict, anc_dependencies)

        elif self.data_level == "l2":
            data_dict = {}
            # TODO: Add ancillary descriptors when maps using them are
            #  implemented.
            anc_dependencies = dependencies.get_file_paths(
                source="lo",
            )
            science_files = dependencies.get_file_paths(source="lo", descriptor="pset")
            psets = []
            for file in science_files:
                psets.append(load_cdf(file))
            data_dict[psets[0].attrs["Logical_source"]] = psets
            datasets = lo_l2.lo_l2(data_dict, anc_dependencies)
        return datasets


class Mag(ProcessInstrument):
    """Process MAG."""

    def do_processing(  # noqa: PLR0912
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

        # If the calibration files have no end date on them, we need to designate
        # one. This ensures we have 3 days past the processing day in the
        # calibration file.

        if self.start_date is not None:
            current_day = np.datetime64(
                f"{self.start_date[:4]}-{self.start_date[4:6]}-{self.start_date[6:]}"
            )
            day_buffer = current_day + np.timedelta64(3, "D")
        else:
            raise ValueError("Start date is not set for MAG L2 processing.")

        if self.data_level == "l1a":
            science_files = dependencies.get_file_paths(source="mag", data_type="l0")
            # File path is expected output file path
            if len(science_files) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1A:"
                    f"{dependency_list}. Expected only one dependency."
                )

            datasets = mag_l1a(science_files[0])

        if self.data_level == "l1b":
            science_files = dependencies.get_file_paths(source="mag", data_type="l1a")
            if len(science_files) != 1:
                raise ValueError(
                    f"Unexpected dependencies found for MAG L1B:"
                    f"{dependency_list}. Expected only one dependency."
                )

            calibration = dependencies.get_processing_inputs(
                descriptor="l1b-calibration"
            )
            if len(calibration) != 1:
                raise ValueError(
                    f"Missing required dependencies for MAG L1B. Need at least one"
                    f"l1b-calibration-matrices file. Received: {dependencies}"
                )

            combined_calibration = MagAncillaryCombiner(calibration[0], day_buffer)

            input_data = load_cdf(science_files[0])
            datasets = [
                mag_l1b(input_data, current_day, combined_calibration.combined_dataset)
            ]

        if self.data_level == "l1c":
            science_files = dependencies.get_file_paths(source="mag", data_type="l1b")
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
            science_files = dependencies.get_file_paths(source="mag", data_type="l1b")
            science_files.extend(
                dependencies.get_file_paths(source="mag", data_type="l1c")
            )
            # TODO: Overwrite dependencies with versions from offsets file
            # TODO: Ensure that parent_files attribute works with that
            input_data = load_cdf(science_files[0])

            descriptor_no_frame = str.split(self.descriptor, "-")[0]

            # We expect either a norm or a burst input descriptor.
            offsets_desc = f"l2-{descriptor_no_frame}-offsets"
            offsets = dependencies.get_processing_inputs(descriptor=offsets_desc)

            calibration = dependencies.get_processing_inputs(
                descriptor="l2-calibration"
            )

            if (
                len(offsets) != 1
                or len(offsets[0].filename_list) != 1
                or len(calibration) != 1
            ):
                anc_dependencies = dependencies.get_processing_inputs(
                    input_type=ProcessingInputType.ANCILLARY_FILE
                )
                raise ValueError(
                    f"Unexpected dependencies found in MAG L2."
                    f"Expected exactly one offsets dependency input file "
                    f"and at least one calibration file."
                    f"All ancillary dependencies: "
                    f"{anc_dependencies}"
                )

            combined_calibration = MagAncillaryCombiner(calibration[0], day_buffer)
            offset_dataset = load_cdf(offsets[0].imap_file_paths[0].construct_path())
            # TODO: get input data from offsets file
            # TODO: Test data missing
            datasets = mag_l2(
                combined_calibration.combined_dataset,
                offset_dataset,
                input_data,
                current_day,
                mode=DataMode(descriptor_no_frame.upper()),
            )

        for ds in datasets:
            if "raw" not in ds.attrs["Logical_source"] and not np.all(
                ds["epoch"].values[1:] > ds["epoch"].values[:-1]
            ):
                raise ValueError(
                    "Timestamps for output file are not monotonically increasing."
                )
        return datasets


class Spacecraft(ProcessInstrument):
    """Process Spacecraft data."""

    def do_processing(
        self, dependencies: ProcessingInputCollection
    ) -> list[xr.Dataset | Path]:
        """
        Perform Spacecraft specific processing.

        Parameters
        ----------
        dependencies : ProcessingInputCollection
            Object containing dependencies to process.

        Returns
        -------
        datasets : list[xarray.Dataset | Path]
            The list of processed products.
        """
        print(f"Processing Spacecraft {self.data_level}")

        if self.data_level == "l1a":
            # File path is expected output file path
            input_files = dependencies.get_file_paths(source="spacecraft")
            if len(input_files) > 1:
                raise ValueError(
                    f"Unexpected dependencies found for Spacecraft L1A: "
                    f"{input_files}. Expected only one dependency."
                )
            datasets = list(quaternions.process_quaternions(input_files[0]))
            return datasets
        elif self.data_level == "spice":
            spice_inputs = dependencies.get_file_paths(
                data_type=SPICESource.SPICE.value
            )
            ah_paths = [path for path in spice_inputs if ".ah" in path.suffixes]
            if len(ah_paths) != 1:
                raise ValueError(
                    f"Unexpected spice dependencies found for Spacecraft "
                    f"pointing_kernel: {ah_paths}. Expected exactly one "
                    f"attitude history file."
                )
            pointing_kernel_paths = pointing_frame.generate_pointing_attitude_kernel(
                ah_paths[0]
            )
            return pointing_kernel_paths
        else:
            raise NotImplementedError(
                f"Spacecraft processing not implemented for level {self.data_level}"
            )


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
            if self.descriptor == "sci" and len(dependency_list) != 3:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 science:"
                    f"{dependency_list}. Expected only three dependencies,"
                    "HK, L0 and time kernels."
                )
            # For housekeeping, we expect only L0 raw file
            if self.descriptor == "hk" and len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWAPI L1 housekeeping:"
                    f"{dependency_list}. Expected only two dependenccies,"
                    "L0 and time kernels."
                )

            # process science or housekeeping data
            datasets = swapi_l1(dependencies)
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
            if len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1A:"
                    f"{dependency_list}. Expected only two dependencies."
                    "L0 data and time kernels."
                )
            science_files = dependencies.get_file_paths(source="swe")
            datasets = swe_l1a(str(science_files[0]))
            # Right now, we only process science data. Therefore,
            # we expect only one dataset to be returned.

        elif self.data_level == "l1b" and self.descriptor == "sci":
            if len(dependency_list) != 5:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1B:"
                    f"{dependency_list}. Expected exactly five dependencies."
                    "L1A science, in-fligth cal, esa LUT, EU conversion and "
                    "time kernels."
                )

            science_files = dependencies.get_file_paths("swe", "sci")
            if len(science_files) > 1:
                raise ValueError(
                    "Multiple science files processing is not supported for SWE L1B."
                )

            datasets = swe_l1b(dependencies)
        elif self.data_level == "l1b" and self.descriptor == "hk":
            if len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L1B HK:"
                    f"{dependency_list}. Expected exactly two dependencies."
                    "L0 data and time kernels."
                )
            # process data
            datasets = swe_l1b(dependencies)
        elif self.data_level == "l2":
            if len(dependency_list) != 2:
                raise ValueError(
                    f"Unexpected dependencies found for SWE L2:"
                    f"{dependency_list}. Expected exactly two dependencies."
                    "L1B science and spin data."
                )
            # process data
            science_files = dependencies.get_file_paths(source="swe", descriptor="sci")
            if len(science_files) > 1:
                raise ValueError(
                    "Multiple science files processing is not supported for SWE L2."
                )
            l1b_datasets = load_cdf(science_files[0])
            datasets = [swe_l2(l1b_datasets)]
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

        if self.data_level == "l1a":
            science_files = dependencies.get_file_paths(source="ultra")
            if len(science_files) != 1:
                raise ValueError(
                    f"Unexpected science_files found for ULTRA L1A:"
                    f"{science_files}. Expected only one dependency."
                )
            datasets = ultra_l1a.ultra_l1a(science_files[0])
        elif self.data_level == "l1b":
            science_files = dependencies.get_file_paths(source="ultra", data_type="l1a")
            l1a_dict = {
                dataset.attrs["Logical_source"]: dataset
                for dataset in [load_cdf(sci_file) for sci_file in science_files]
            }
            science_files = dependencies.get_file_paths(source="ultra", data_type="l1b")
            l1b_dict = {
                dataset.attrs["Logical_source"]: dataset
                for dataset in [load_cdf(sci_file) for sci_file in science_files]
            }
            combined = {**l1a_dict, **l1b_dict}
            anc_paths = dependencies.get_file_paths(data_type="ancillary")
            ancillary_files = {}
            for path in anc_paths:
                ancillary_files[path.stem.split("_")[2]] = path
            datasets = ultra_l1b.ultra_l1b(combined, ancillary_files)
        elif self.data_level == "l1c":
            science_files = dependencies.get_file_paths(source="ultra", data_type="l1a")
            l1a_dict = {
                dataset.attrs["Logical_source"]: dataset
                for dataset in [load_cdf(sci_file) for sci_file in science_files]
            }
            science_files = dependencies.get_file_paths(source="ultra", data_type="l1b")
            l1b_dict = {
                dataset.attrs["Logical_source"]: dataset
                for dataset in [load_cdf(sci_file) for sci_file in science_files]
            }
            combined = {**l1a_dict, **l1b_dict}
            anc_paths = dependencies.get_file_paths(data_type="ancillary")
            ancillary_files = {}
            for path in anc_paths:
                ancillary_files[path.stem.split("_")[2]] = path
            spice_paths = dependencies.get_file_paths(data_type="spice")
            if spice_paths:
                has_spice = True
            else:
                has_spice = False
            datasets = ultra_l1c.ultra_l1c(combined, ancillary_files, has_spice)
        elif self.data_level == "l2":
            all_pset_filepaths = dependencies.get_file_paths(
                source="ultra", descriptor="pset"
            )
            # There can be many PSET files, so avoid reading them all in.
            # The filename stem (logical_file_id) contains
            # all the information needed in the key.
            data_dict = {
                pset_filepath.stem: pset_filepath
                for pset_filepath in all_pset_filepaths
            }
            datasets = ultra_l2.ultra_l2(
                data_dict,
                descriptor=self.descriptor,
            )

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
