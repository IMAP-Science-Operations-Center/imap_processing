"""
Perform CoDICE l1a processing.

This module processes decommutated CoDICE packets and creates L1a data products.

Notes
-----
    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1a import process_codice_l1a
    packets = decom_packets(packet_file)
    dataset = process_codice_l1a(packets)
"""

from __future__ import annotations

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

# TODO: Add support for decomming multiple APIDs from a single file
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

    def decompress_data(self, science_values: str) -> None:
        """
        Perform decompression on the data.

        The science data within the packet is a compressed, binary string of
        values. Apply the appropriate decompression algorithm to get an array
        of decompressed values.

        Parameters
        ----------
        science_values : str
            A string of binary data representing the science values of the data.
        """
        compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]

        # Decompress the binary string into a list of integers
        self.data = decompress(science_values, compression_algorithm)

    def define_coordinates(self) -> None:
        """
        Create ``xr.DataArrays`` for the coords needed in the final dataset.

        The coordinates for the dataset depend on the data product being made.
        """
        self.coords = {}

        for name in self.config["coords"]:
            if name == "epoch":
                values = self.packet_dataset.epoch
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
        dataset : xarray.Dataset
            The 'final' ``xarray`` dataset.
        """
        # Create the main dataset to hold all the variables
        dataset = xr.Dataset(
            coords=self.coords,
            attrs=self.cdf_attrs.get_global_attributes(self.config["dataset_name"]),
        )

        # Create a data variable for each counter
        for variable_data, variable_name in zip(
            self.data, self.config["variable_names"]
        ):
            # Reshape to 4 dimensions to allow for epoch dimension
            reshaped_variable_data = np.expand_dims(variable_data, axis=0)

            # Get the CDF attributes
            descriptor = self.config["dataset_name"].split("imap_codice_l1a_")[-1]
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            attrs = self.cdf_attrs.get_variable_attributes(cdf_attrs_key)

            # Create the CDF data variable
            dataset[variable_name] = xr.DataArray(
                reshaped_variable_data,
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
        # For CoDICE-lo, data are a 3D arrays with a shape representing
        # [<num_positions>,<num_spin_sectors>,<num_energy_steps>]
        if self.config["instrument"] == "lo":
            self.data = np.array(self.data, dtype=np.uint32).reshape(
                (
                    self.config["num_counters"],
                    self.config["num_positions"],
                    self.config["num_spin_sectors"],
                    self.config["num_energy_steps"],
                )
            )

        # For CoDICE-hi, data are a 3D array with a shape representing
        # [<num_energy_steps>,<num_positions>,<num_spin_sectors>]
        elif self.config["instrument"] == "hi":
            self.data = np.array(self.data, dtype=np.uint32).reshape(
                (
                    self.config["num_counters"],
                    self.config["num_energy_steps"],
                    self.config["num_positions"],
                    self.config["num_spin_sectors"],
                )
            )

    def set_data_product_config(
        self, apid: int, packet: xr.Dataset, data_version: str
    ) -> None:
        """
        Set the various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        packet : xarray.Dataset
            A packet for the APID of interest.
        data_version : str
            Version of the data product being created.
        """
        # Set the packet dataset so that it can be easily called from various
        # methods
        self.packet_dataset = packet

        # Set various configurations of the data product
        self.config: dict[str, Any] = constants.DATA_PRODUCT_CONFIGURATIONS.get(
            apid
        )  # mypy: allow-untyped-defs

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


def get_params(packet: xr.Dataset) -> tuple[int, int, int, int]:
    """
    Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    Parameters
    ----------
    packet : xarray.Dataset
        A packet for the APID of interest.

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
    table_id = int(packet.table_id.data)
    plan_id = int(packet.plan_id.data)
    plan_step = int(packet.plan_step.data)
    view_id = int(packet.view_id.data)

    return table_id, plan_id, plan_step, view_id


def process_codice_l1a(file_path: Path, data_version: str) -> xr.Dataset:
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
    dataset : xarray.Dataset
        The ``xarray`` dataset containing the science data and supporting metadata.
    """
    # Decom the packets, group data by APID, and sort by time
    datasets = decom_packets(file_path)

    for apid in datasets:
        packet_dataset = datasets[apid]
        logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

        if apid == CODICEAPID.COD_NHK:
            dataset = create_hskp_dataset(packet_dataset, data_version)

        elif apid in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            dataset = create_event_dataset(apid, packet_dataset, data_version)

        elif apid in constants.APIDS_FOR_SCIENCE_PROCESSING:
            # Extract the data
            science_values = packet_dataset.data.data[0]
            # TODO: improve this as needed
            science_values = "".join(f"{byte:08b}" for byte in science_values)

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(packet_dataset)

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.set_data_product_config(apid, packet_dataset, data_version)
            pipeline.decompress_data(science_values)
            pipeline.reshape_data()
            pipeline.define_coordinates()
            dataset = pipeline.define_data_variables()

    logger.info(f"\nFinal data product:\n{dataset}\n")

    return dataset
