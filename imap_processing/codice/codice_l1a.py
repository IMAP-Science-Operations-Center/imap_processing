"""
Perform CoDICE l1a processing.

This module processes CoDICE L0 files and creates L1a data products.

Notes
-----
    from imap_processing.codice.codice_l1a import process_codice_l1a
    processed_datasets = process_codice_l1a(path_to_l0_file)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CODICEAPID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Determine what should go in event data CDF and how it should be
#       structured.


class CoDICEL1aPipeline:
    """
    Contains methods for processing L0 data and creating L1a data products.

    Parameters
    ----------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration.
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet.
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet.
    view_id : int
        Provides information about how data was collapsed and/or compressed.

    Methods
    -------
    decompress_data(science_values)
        Perform decompression on the data.
    define_coordinates()
        Create ``xr.DataArrays`` for the coords needed in the final dataset.
    define_data_variables()
        Define and add the appropriate data variables to the dataset.
    define_dimensions()
        Define the dimensions of the data arrays for the final dataset.
    define_support_variables()
        Define and add 'support' CDF data variables to the dataset.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_energy_table()
        Retrieve the ESA sweep values.
    reshape_data()
        Reshape the data arrays based on the data product being made.
    set_data_product_config()
        Set the various settings for defining the data products.
    """

    def __init__(self, table_id: int, plan_id: int, plan_step: int, view_id: int):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def decompress_data(self, science_values: list[str]) -> None:
        """
        Perform decompression on the data.

        The science data within the packet is a compressed byte string of
        values. Apply the appropriate decompression algorithm to get an array
        of decompressed values.

        Parameters
        ----------
        science_values : list[str]
            A list of byte strings representing the science values of the data
            for each packet.
        """
        # The compression algorithm depends on the instrument and view ID
        if self.config["instrument"] == "lo":
            compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]
        elif self.config["instrument"] == "hi":
            compression_algorithm = constants.HI_COMPRESSION_ID_LOOKUP[self.view_id]

        self.raw_data = []
        for packet_data, byte_count in zip(
            science_values, self.dataset.byte_count.data
        ):
            # Convert from numpy array to byte object
            values = ast.literal_eval(str(packet_data))

            # Only use the values up to the byte count. Bytes after this are
            # used as padding and are not needed
            values = values[:byte_count]

            decompressed_values = decompress(values, compression_algorithm)
            self.raw_data.append(decompressed_values)

    def define_coordinates(self) -> None:
        """
        Create ``xr.DataArrays`` for the coords needed in the final dataset.

        The coordinates for the dataset depend on the data product being made.
        """
        self.coords = {}

        for name in self.config["coords"]:
            if name == "epoch":
                values = self.dataset.epoch.data
            elif name == "inst_az":
                values = np.arange(self.config["num_positions"])
            elif name == "spin_sector":
                values = np.arange(self.config["num_spin_sectors"])
            elif name == "esa_step":
                values = np.arange(self.config["num_energy_steps"])
            else:
                # TODO: Need to implement other types of coords
                continue

            coord = xr.DataArray(
                values,
                name=name,
                dims=[name],
                attrs=self.cdf_attrs.get_variable_attributes(name),
            )

            self.coords[name] = coord

    def define_data_variables(self) -> xr.Dataset:
        """
        Define and add the appropriate data variables to the dataset.

        The data variables included in the dataset depend on the data product
        being made. The method returns the ``xarray.Dataset`` object that can
        then be written to a CDF file.

        Returns
        -------
        processed_dataset : xarray.Dataset
            The 'final' ``xarray`` dataset.
        """
        # Create the main dataset to hold all the variables
        dataset = xr.Dataset(
            coords=self.coords,
            attrs=self.cdf_attrs.get_global_attributes(self.config["dataset_name"]),
        )

        # Stack the data so that it is easier to reshape and iterate over
        all_data = np.stack(self.data)

        # The dimension of all data is (epoch, num_counters, num_positions,
        # num_spin_sectors, num_energy_steps) (or may be slightly different
        # depending on the data product). In any case, iterate over the
        # num_counters dimension to isolate the data for each counter so
        # that it can be placed in a CDF data variable.
        for counter, variable_name in zip(
            range(all_data.shape[1]), self.config["variable_names"]
        ):
            counter_data = all_data[:, counter, :, :, :]

            # Get the CDF attributes
            descriptor = self.config["dataset_name"].split("imap_codice_l1a_")[-1]
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            attrs = self.cdf_attrs.get_variable_attributes(cdf_attrs_key)

            # Create the CDF data variable
            dataset[variable_name] = xr.DataArray(
                counter_data,
                name=variable_name,
                dims=self.config["dims"],
                attrs=attrs,
            )

        # Add support data variables based on data product
        dataset = self.define_support_variables(dataset)

        return dataset

    def define_support_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Define and add 'support' CDF data variables to the dataset.

        These variables include instrument metadata, energies, times, etc. that
        help further define the L1a CDF data product. The variables included
        depend on the data product being made.

        Parameters
        ----------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product.

        Returns
        -------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product, with added support variables.
        """
        for variable_name in self.config["support_variables"]:
            if variable_name == "energy_table":
                variable_data = self.get_energy_table()
                dims = ["esa_step"]
                attrs = self.cdf_attrs.get_variable_attributes("esa_step")

            elif variable_name == "acquisition_time_per_step":
                variable_data = self.get_acquisition_times()
                dims = ["esa_step"]
                attrs = self.cdf_attrs.get_variable_attributes(
                    "acquisition_time_per_step"
                )

            else:
                # TODO: Need to implement methods to gather and set other
                #       support attributes
                continue

            # Add variable to the dataset
            dataset[variable_name] = xr.DataArray(
                variable_data,
                dims=dims,
                attrs=attrs,
            )

        return dataset

    def get_acquisition_times(self) -> list[float]:
        """
        Retrieve the acquisition times via the Lo stepping table.

        Get the acquisition times from the data file based on the values of
        ``plan_id`` and ``plan_step``

        The Lo stepping table defines how many voltage steps and which steps are
        used during each spacecraft spin. A full cycle takes 16 spins. The table
        provides the timing for a given energy step, and most importantly
        provides the "acquisition time", which is the acquisition time, in
        milliseconds, for the energy step.

        Returns
        -------
        acquisition_times : list[float]
            The list of acquisition times from the Lo stepping table.
        """
        # Read in the Lo stepping data table
        lo_stepping_data_file = Path(
            f"{imap_module_directory}/codice/data/lo_stepping_values.csv"
        )
        lo_stepping_data = pd.read_csv(lo_stepping_data_file)

        # Determine which Lo stepping table is needed
        lo_stepping_table_id = constants.LO_STEPPING_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        # Get the appropriate values
        lo_stepping_values = lo_stepping_data[
            lo_stepping_data["table_num"] == lo_stepping_table_id
        ]

        # Create a list for the acquisition times
        acquisition_times = []

        # Only need the energy columns from the table
        energy_steps = lo_stepping_values[
            ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
        ].astype(str)  # convert to string to avoid confusion with table index value

        # For each energy step (0-127), scan the energy columns and find the row
        # number, which corresponds to a specific acquisition time, then append
        # it to the list
        for step_number in range(128):
            row_number = np.argmax(energy_steps == str(step_number), axis=1).argmax()
            acquisition_times.append(lo_stepping_values.acq_time[row_number])

        return acquisition_times

    def get_energy_table(self) -> NDArray[float]:
        """
        Retrieve the ESA sweep values.

        Get the ElectroStatic Analyzer (ESA) sweep values from the data file
        based on the values of ``plan_id`` and ``plan_step``

        CoDICE-Lo measures ions between ~0.5 and 80 keV/q that enter the
        aperture and are selected and focused according to their E/q into the
        Time of Flight (TOF) assembly.  The E/q sweeping steps up to the max
        voltage for the next stepping cycle when solar wind count rate exceed a
        predefined threshold rate.

        The ESA sweep table defines the voltage steps that are used to cover the
        full energy per charge range.

        Returns
        -------
        energy_table : NDArray[float]
            The list of ESA sweep values (i.e. voltage steps).
        """
        # Read in the ESA sweep data table
        esa_sweep_data_file = Path(
            f"{imap_module_directory}/codice/data/esa_sweep_values.csv"
        )
        sweep_data = pd.read_csv(esa_sweep_data_file)

        # Determine which ESA sweep table is needed
        sweep_table_id = constants.ESA_SWEEP_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        # Get the appropriate values
        sweep_table = sweep_data[sweep_data["table_idx"] == sweep_table_id]
        energy_table: NDArray[float] = sweep_table["esa_v"].values

        return energy_table

    def reshape_data(self) -> None:
        """
        Reshape the data arrays based on the data product being made.

        These data need to be divided up by species or priorities (or
        what I am calling "counters" as a general term), and re-arranged into
        3D arrays representing dimensions such as spin sectors, positions, and
        energies (depending on the data product).
        """
        self.data = []

        # For CoDICE-lo, data are a 3D arrays with a shape representing
        # [<num_positions>,<num_spin_sectors>,<num_energy_steps>]
        if self.config["instrument"] == "lo":
            for packet_data in self.raw_data:
                if packet_data:
                    reshaped_packet_data = np.array(
                        packet_data, dtype=np.uint32
                    ).reshape(
                        (
                            self.config["num_counters"],
                            self.config["num_positions"],
                            self.config["num_spin_sectors"],
                            self.config["num_energy_steps"],
                        )
                    )
                    self.data.append(reshaped_packet_data)
                else:
                    self.data.append(None)

        # For CoDICE-hi, data are a 3D array with a shape representing
        # [<num_energy_steps>,<num_positions>,<num_spin_sectors>]
        elif self.config["instrument"] == "hi":
            for packet_data in self.raw_data:
                if packet_data:
                    reshaped_packet_data = np.array(
                        packet_data, dtype=np.uint32
                    ).reshape(
                        (
                            self.config["num_counters"],
                            self.config["num_energy_steps"],
                            self.config["num_positions"],
                            self.config["num_spin_sectors"],
                        )
                    )
                    self.data.append(reshaped_packet_data)
                else:
                    self.data.append(None)

        # No longer need to keep the raw data around
        del self.raw_data

    def set_data_product_config(
        self, apid: int, dataset: xr.Dataset, data_version: str
    ) -> None:
        """
        Set the various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        dataset : xarray.Dataset
            The dataset for the APID of interest.
        data_version : str
            Version of the data product being created.
        """
        # Set the packet dataset so that it can be easily called from various
        # methods
        self.dataset = dataset

        # Set various configurations of the data product
        self.config: dict[str, Any] = constants.DATA_PRODUCT_CONFIGURATIONS.get(apid)  # type: ignore

        # Gather and set the CDF attributes
        self.cdf_attrs = ImapCdfAttributes()
        self.cdf_attrs.add_instrument_global_attrs("codice")
        self.cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
        self.cdf_attrs.add_global_attribute("Data_version", data_version)


def create_event_dataset(
    apid: int, packet: xr.Dataset, data_version: str
) -> xr.Dataset:
    """
    Create dataset for event data.

    Parameters
    ----------
    apid : int
        The APID of the packet.
    packet : xarray.Dataset
        The packet to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the event data.
    """
    if apid == CODICEAPID.COD_LO_PHA:
        dataset_name = "imap_codice_l1a_lo-pha"
    elif apid == CODICEAPID.COD_HI_PHA:
        dataset_name = "imap_codice_l1a_hi-pha"

    # Extract the data
    # event_data = packet.event_data.data (Currently turned off, see TODO)

    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    # Define coordinates
    epoch = xr.DataArray(
        packet.epoch,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    # Create the dataset to hold the data variables
    dataset = xr.Dataset(
        coords={
            "epoch": epoch,
        },
        attrs=cdf_attrs.get_global_attributes(dataset_name),
    )

    return dataset


def create_hskp_dataset(
    packet: xr.Dataset,
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packet : xarray.Dataset
        The packet to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the metadata.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    epoch = xr.DataArray(
        packet.epoch,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"),
    )

    for variable in packet:
        attrs = cdf_attrs.get_variable_attributes(variable)

        dataset[variable] = xr.DataArray(
            packet[variable].data, dims=["epoch"], attrs=attrs
        )

    return dataset


def get_params(dataset: xr.Dataset) -> tuple[int, int, int, int]:
    """
    Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset for the APID of interest. We expect each packet in the
        dataset to have the same values for the four main parameters, so the
        first index of the dataset can be used to determine them.

    Returns
    -------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration.
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet.
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet.
    view_id : int
        Provides information about how data was collapsed and/or compressed.
    """
    table_id = int(dataset.table_id.data[0])
    plan_id = int(dataset.plan_id.data[0])
    plan_step = int(dataset.plan_step.data[0])
    view_id = int(dataset.view_id.data[0])

    return table_id, plan_id, plan_step, view_id


def log_dataset_info(datasets: dict[int, xr.Dataset]) -> None:
    """
    Log info about the input data to help with tracking and/or debugging.

    Parameters
    ----------
    datasets : dict[int, xarray.Dataset]
        Mapping from apid to ``xarray`` dataset, one dataset per apid.
    """
    launch_time = np.datetime64("2010-01-01T00:01:06.184", "ns")
    logger.info("\nThis input file contains the following APIDs:\n")
    for apid in datasets:
        num_packets = len(datasets[apid].epoch.data)
        time_deltas = [np.timedelta64(item, "ns") for item in datasets[apid].epoch.data]
        times = [launch_time + delta for delta in time_deltas]
        start = np.datetime_as_string(times[0])
        end = np.datetime_as_string(times[-1])
        logger.info(
            f"{CODICEAPID(apid).name}: {num_packets} packets spanning {start} to {end}"
        )


def process_codice_l1a(file_path: Path, data_version: str) -> list[xr.Dataset]:
    """
    Will process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the CoDICE L0 file to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_datasets : list[xarray.Dataset]
        A list of the ``xarray`` datasets containing the science data and
        supporting metadata.
    """
    # Decom the packets, group data by APID, and sort by time
    datasets = decom_packets(file_path)

    # Log some information about the contents of the data
    log_dataset_info(datasets)

    # Placeholder to hold the final, processed datasets
    processed_datasets = []

    # Process each APID separately
    for apid in datasets:
        dataset = datasets[apid]
        logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

        # Housekeeping data
        if apid == CODICEAPID.COD_NHK:
            processed_dataset = create_hskp_dataset(dataset, data_version)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Event data
        elif apid in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            processed_dataset = create_event_dataset(apid, dataset, data_version)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Everything else
        elif apid in constants.APIDS_FOR_SCIENCE_PROCESSING:
            # Extract the data
            science_values = [packet.data for packet in dataset.data]

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(dataset)

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.set_data_product_config(apid, dataset, data_version)
            pipeline.decompress_data(science_values)
            pipeline.reshape_data()
            pipeline.define_coordinates()
            processed_dataset = pipeline.define_data_variables()

            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # TODO: Still need to implement I-ALiRT and hi-priorities data products
        elif apid in [
            CODICEAPID.COD_HI_INST_COUNTS_PRIORITIES,
            CODICEAPID.COD_HI_IAL,
            CODICEAPID.COD_LO_IAL,
        ]:
            logger.info("\tStill need to properly implement")
            processed_dataset = None

        # For APIDs that don't require processing
        else:
            logger.info(f"\t{apid} does not require processing")
            continue

        processed_datasets.append(processed_dataset)

    return processed_datasets
