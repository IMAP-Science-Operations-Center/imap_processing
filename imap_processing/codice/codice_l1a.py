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

import numpy as np
import pandas as pd
import space_packet_parser
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import IMAP_EPOCH, met_to_j2000ns
from imap_processing.codice import constants
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.utils import CODICEAPID, create_hskp_dataset
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Decom data arrays need to be decompressed
# TODO: Add metadata attrs to science dataset?
# TODO: In decommutation, how to have a variable length data and then a checksum
#       after it?


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
    create_science_dataset()
        Create an ``xarray`` dataset for the unpacked science data.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_data_products()
        Retrieve the lo data products.
    get_esa_sweep_values()
        Retrieve the ESA sweep values.
    unpack_science_data()
        Make 4D L1a data product from the decompressed science data.
    """

    def __init__(self, table_id: int, plan_id: int, plan_step: int, view_id: int):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def create_science_dataset(self, met: np.int64, data_version: str) -> xr.Dataset:
        """
        Create an ``xarray`` dataset for the unpacked science data.

        The dataset can then be written to a CDF file.

        Parameters
        ----------
        met : numpy.int64
            The mission elapsed time of the packet, used to determine epoch data.
        data_version : str
            Version of the data product being created.

        Returns
        -------
        dataset : xarray.Dataset
            The ``xarray`` dataset containing the science data and supporting metadata.
        """
        # Set the CDF attrs
        cdf_attrs = ImapCdfAttributes()
        cdf_attrs.add_instrument_global_attrs("codice")
        cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
        cdf_attrs.add_global_attribute("Data_version", data_version)

        # Define coordinates
        epoch = xr.DataArray(
            met_to_j2000ns(met),  # TODO: Fix after SIT-3 (see note below)
            name="epoch",
            dims=["epoch"],
            attrs=cdf_attrs.get_variable_attributes("epoch_attrs"),
        )
        energy_steps = xr.DataArray(
            np.arange(self.num_energy_steps),
            name="energy",
            dims=["energy"],
            attrs=cdf_attrs.get_variable_attributes("energy_attrs"),
        )

        # Define labels
        energy_label = xr.DataArray(
            energy_steps.values.astype(str),
            name="energy_label",
            dims=["energy_label"],
            attrs=cdf_attrs.get_variable_attributes("energy_label"),
        )

        # Create the dataset to hold the data variables
        dataset = xr.Dataset(
            coords={
                "epoch": epoch,
                "energy": energy_steps,
                "energy_label": energy_label,
            },
            attrs=cdf_attrs.get_global_attributes(self.dataset_name),
        )

        # Create a data variable for each counter
        for variable_data, variable_name in zip(self.data, self.variable_names):
            # TODO: Currently, cdflib doesn't properly write/read CDF files that
            #       have a single epoch value. To get around this for now, use
            #       two epoch values and reshape accordingly. Revisit this after
            #       SIT-3.
            variable_data_arr = np.array(list(variable_data) * 2, dtype=int).reshape(
                2, self.num_energy_steps
            )
            cdf_attrs_key = (
                f"{self.dataset_name.split('imap_codice_l1a_')[-1]}-{variable_name}"
            )
            dataset[variable_name] = xr.DataArray(
                variable_data_arr,
                name=variable_name,
                dims=["epoch", "energy"],
                attrs=cdf_attrs.get_variable_attributes(cdf_attrs_key),
            )

        # Add ESA Sweep Values and acquisition times (lo only)
        if "_lo_" in self.dataset_name:
            dataset["esa_sweep_values"] = xr.DataArray(
                self.esa_sweep_values,
                dims=["energy"],
                attrs=cdf_attrs.get_variable_attributes("esa_sweep_attrs"),
            )
            dataset["acquisition_times"] = xr.DataArray(
                self.acquisition_times,
                dims=["energy"],
                attrs=cdf_attrs.get_variable_attributes("acquisition_times_attrs"),
            )

        return dataset

    def get_acquisition_times(self) -> None:
        """
        Retrieve the acquisition times via the Lo stepping table.

        Get the acquisition times from the data file based on the values of
        ``plan_id`` and ``plan_step``

        The Lo stepping table defines how many voltage steps and which steps are
        used during each spacecraft spin. A full cycle takes 16 spins. The table
        provides the timing for a given energy step, and most importantly
        provides the "acquisition time", which is the acquisition time, in
        milliseconds, for the energy step.
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
        self.acquisition_times = []

        # Only need the energy columns from the table
        energy_steps = lo_stepping_values[
            ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
        ].astype(str)  # convert to string to avoid confusion with table index value

        # For each energy step (0-127), scan the energy columns and find the row
        # number, which corresponds to a specific acquisition time, then append
        # it to the list
        for step_number in range(128):
            row_number = np.argmax(energy_steps == str(step_number), axis=1).argmax()
            self.acquisition_times.append(lo_stepping_values.acq_time[row_number])

    def get_data_products(self, apid: int) -> None:
        """
        Retrieve various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        """
        config = constants.DATA_PRODUCT_CONFIGURATIONS.get(apid)  # type: ignore[call-overload]
        # TODO Change, No overload variant of "get" of
        #  "dict" matches argument type "str".
        self.num_counters = config["num_counters"]
        self.num_energy_steps = config["num_energy_steps"]
        self.variable_names = config["variable_names"]
        self.dataset_name = config["dataset_name"]

    def get_esa_sweep_values(self) -> None:
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
        self.esa_sweep_values = sweep_table["esa_v"].values

    def unpack_science_data(self, science_values: str) -> None:
        """
        Unpack the science data from the packet.

        For LO SW Species Counts data, the science data within the packet is a
        blob of compressed values of length 2048 bits (16 species * 128 energy
        levels). These data need to be divided up by species so that each
        species can have their own data variable in the L1A CDF file.

        Parameters
        ----------
        science_values : str
            A string of binary data representing the science values of the data.
        """
        self.compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]
        self.collapse_table_id = constants.LO_COLLAPSE_TABLE_ID_LOOKUP[self.view_id]

        # TODO: Turn this back on after SIT-3
        # For SIT-3, just create appropriate length data arrays of all ones
        # Divide up the data by the number of priorities or species
        # science_values = packets[0].data["DATA"].raw_value
        # num_bits = len(science_values)
        # chunk_size = len(science_values) // self.num_counters
        # self.data = [
        #     science_values[i : i + chunk_size] for i in range(0, num_bits, chunk_size)
        # ]
        self.data = [["1"] * 128] * self.num_counters


def get_params(packet: space_packet_parser.parser.Packet) -> tuple[int, int, int, int]:
    """
    Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
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
    table_id = packet.data["TABLE_ID"].raw_value
    plan_id = packet.data["PLAN_ID"].raw_value
    plan_step = packet.data["PLAN_STEP"].raw_value
    view_id = packet.data["VIEW_ID"].raw_value

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
    # TODO: Once simulated data for codice-hi is acquired, there shouldn't be a
    # need to split the processing based on the file_path, so this function can
    # be simplified.

    apids_for_lo_science_processing = [
        CODICEAPID.COD_LO_INST_COUNTS_AGGREGATED,
        CODICEAPID.COD_LO_INST_COUNTS_SINGLES,
        CODICEAPID.COD_LO_SW_ANGULAR_COUNTS,
        CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS,
        CODICEAPID.COD_LO_SW_PRIORITY_COUNTS,
        CODICEAPID.COD_LO_NSW_PRIORITY_COUNTS,
        CODICEAPID.COD_LO_SW_SPECIES_COUNTS,
        CODICEAPID.COD_LO_NSW_SPECIES_COUNTS,
    ]

    # TODO: Temporary workaround in order to create hi data products in absence
    #       of simulated data
    if file_path.name.startswith(("imap_codice_l0_lo", "imap_codice_l0_hskp")):
        # Decom the packets, group data by APID, and sort by time
        packets = decom_packets(file_path)
        grouped_data = group_by_apid(packets)

        for apid in grouped_data:
            logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

            if apid == CODICEAPID.COD_NHK:
                packets = grouped_data[apid]
                sorted_packets = sort_by_time(packets, "SHCOARSE")
                dataset = create_hskp_dataset(sorted_packets, data_version)

            elif apid in apids_for_lo_science_processing:
                # Sort the packets by time
                packets = sort_by_time(grouped_data[apid], "SHCOARSE")

                # Determine the start time of the packet
                met = packets[0].data["ACQ_START_SECONDS"].raw_value
                met = [met, met + 1]  # TODO: Remove after SIT-3
                # Extract the data
                science_values = packets[0].data["DATA"].raw_value

                # Get the four "main" parameters for processing
                table_id, plan_id, plan_step, view_id = get_params(packets[0])

                # Run the pipeline to create a dataset for the product
                pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
                pipeline.get_esa_sweep_values()
                pipeline.get_acquisition_times()
                pipeline.get_data_products(apid)
                pipeline.unpack_science_data(science_values)
                dataset = pipeline.create_science_dataset(met, data_version)

    # TODO: Temporary workaround in order to create hi data products in absence
    #       of simulated data. This is essentially the same process as is for
    #       lo, but don't try to decom any packets, just define the data
    #       outright.
    elif file_path.name.startswith("imap_codice_l0_hi"):
        if file_path.name.startswith("imap_codice_l0_hi-counters-aggregated"):
            apid = CODICEAPID.COD_HI_INST_COUNTS_AGGREGATED
            table_id, plan_id, plan_step, view_id = (1, 0, 0, 3)
        elif file_path.name.startswith("imap_codice_l0_hi-counters-singles"):
            apid = CODICEAPID.COD_HI_INST_COUNTS_SINGLES
            table_id, plan_id, plan_step, view_id = (1, 0, 0, 4)
        elif file_path.name.startswith("imap_codice_l0_hi-omni"):
            apid = CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS
            table_id, plan_id, plan_step, view_id = (1, 0, 0, 5)
        elif file_path.name.startswith("imap_codice_l0_hi-sectored"):
            apid = CODICEAPID.COD_HI_SECT_SPECIES_COUNTS
            table_id, plan_id, plan_step, view_id = (1, 0, 0, 6)

        met0 = (np.datetime64("2024-04-29T00:00") - IMAP_EPOCH).astype("timedelta64[s]")
        met0 = met0.astype(np.int64)
        met = [met0, met0 + 1]  # Using this to match the other data products
        science_values = ""  # Currently don't have simulated data for this

        pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
        pipeline.get_data_products(apid)
        pipeline.unpack_science_data(science_values)
        dataset = pipeline.create_science_dataset(met, data_version)

    # Write dataset to CDF
    logger.info(f"\nFinal data product:\n{dataset}\n")

    return dataset
