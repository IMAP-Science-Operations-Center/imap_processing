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

import collections
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import space_packet_parser
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.codice import constants
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CODICEAPID, add_metadata_to_array
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: In decommutation, how to have a variable length data and then a checksum
#       after it? (Might be fixed with new XTCE script updates)
# TODO: Add support for decomming multiple APIDs from a single file
# TODO: Add these as variables in CDF: SPIN_PERIOD, ST_BIAS_GAIN_MODE,
#       SW_BIAS_GAIN_MODE, RGFO_HALF_SPIN, NSO_HALF_SPIN, DATA_QUALITY
# TODO: Use new packet_file_to_dataset() function to simplify things
# TODO: Determine what should go in event data CDF and how it should be
#       structured.
# TODO: Make sure CDF attributes match expected nomenclature


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
    configure_data_products()
        Set the various settings for defining the data products.
    create_science_dataset()
        Create an ``xarray`` dataset for the unpacked science data.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_esa_sweep_values()
        Retrieve the ESA sweep values.
    unpack_science_data()
        Decompress, unpack, and restructure science data arrays.
    """

    def __init__(self, table_id: int, plan_id: int, plan_step: int, view_id: int):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def configure_data_products(self, apid: int) -> None:
        """
        Set the various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        """
        config = constants.DATA_PRODUCT_CONFIGURATIONS.get(apid)  # type: ignore[call-overload]
        self.num_counters = config["num_counters"]
        self.num_energy_steps = config["num_energy_steps"]
        self.num_spin_sectors = config["num_spin_sectors"]
        self.num_positions = config["num_positions"]
        self.variable_names = config["variable_names"]
        self.dataset_name = config["dataset_name"]
        self.instrument = config["instrument"]

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
            [met_to_j2000ns(met)],
            name="epoch",
            dims=["epoch"],
            attrs=cdf_attrs.get_variable_attributes("epoch"),
        )
        inst_az = xr.DataArray(
            np.arange(self.num_positions),
            name="inst_az",
            dims=["inst_az"],
            attrs=cdf_attrs.get_variable_attributes("inst_az_attrs"),
        )
        spin_sector = xr.DataArray(
            np.arange(self.num_spin_sectors),
            name="spin_sector",
            dims=["spin_sector"],
            attrs=cdf_attrs.get_variable_attributes("spin_sector_attrs"),
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
                "inst_az": inst_az,
                "spin_sector": spin_sector,
                "energy": energy_steps,
                "energy_label": energy_label,
            },
            attrs=cdf_attrs.get_global_attributes(self.dataset_name),
        )

        # Create a data variable for each counter
        for variable_data, variable_name in zip(self.data, self.variable_names):
            # Data arrays are structured depending on the instrument
            if self.instrument == "lo":
                variable_data_arr = np.array(variable_data).reshape(
                    1, self.num_positions, self.num_spin_sectors, self.num_energy_steps
                )
                dims = ["epoch", "inst_az", "spin_sector", "energy"]
            elif self.instrument == "hi":
                variable_data_arr = np.array(variable_data).reshape(
                    1, self.num_energy_steps, self.num_positions, self.num_spin_sectors
                )
                dims = ["epoch", "energy", "inst_az", "spin_sector"]

            # Get the CDF attributes
            cdf_attrs_key = (
                f"{self.dataset_name.split('imap_codice_l1a_')[-1]}-{variable_name}"
            )
            attrs = cdf_attrs.get_variable_attributes(cdf_attrs_key)

            # Create the CDF data variable
            dataset[variable_name] = xr.DataArray(
                variable_data_arr,
                name=variable_name,
                dims=dims,
                attrs=attrs,
            )

        # Add ESA Sweep Values and acquisition times (lo only)
        if self.instrument == "lo":
            self.get_esa_sweep_values()
            self.get_acquisition_times()
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
        Decompress, unpack, and restructure science data arrays.

        The science data within the packet is a compressed, binary string of
        values. These data need to be divided up by species or priorities (or
        what I am calling "counters" as a general term), and re-arranged into
        3D arrays representing spin sectors, positions, and energies (the order
        of which depends on the instrument).

        Parameters
        ----------
        science_values : str
            A string of binary data representing the science values of the data.
        """
        self.compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]

        # Decompress the binary string into a list of integers
        science_values_decompressed = decompress(
            science_values, self.compression_algorithm
        )

        # Re-arrange the counter data
        # For CoDICE-lo, data are a 3D arrays with a shape representing
        # [<num_positions>,<num_spin_sectors>,<num_energy_steps>]
        if self.instrument == "lo":
            self.data = np.array(science_values_decompressed, dtype=np.uint32).reshape(
                self.num_counters,
                self.num_positions,
                self.num_spin_sectors,
                self.num_energy_steps,
            )

        # For CoDICE-hi, data are a 3D array with a shape representing
        # [<num_energy_steps>,<num_positions>,<num_spin_sectors>]
        elif self.instrument == "hi":
            self.data = np.array(science_values_decompressed, dtype=np.uint32).reshape(
                self.num_counters,
                self.num_energy_steps,
                self.num_positions,
                self.num_spin_sectors,
            )


def create_event_dataset(
    met: list[int], event_data: str, dataset_name: str, data_version: str
) -> xr.Dataset:
    """
    Create dataset for event data.

    Parameters
    ----------
    met : list[int]
        The Mission Elapsed Time of the data.
    event_data : str
        A string of binary numbers representing the event data.
    dataset_name : str
        The name for the dataset.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the event data.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    # Define coordinates
    epoch = xr.DataArray(
        met_to_j2000ns(met),  # TODO: Fix after SIT-3 (see note below)
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
    packets: list[space_packet_parser.parser.Packet],
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        The list of packets to process.
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

    metadata_arrays: dict = collections.defaultdict(list)

    for packet in packets:
        add_metadata_to_array(packet, metadata_arrays)

    epoch = xr.DataArray(
        met_to_j2000ns(
            metadata_arrays["SHCOARSE"],
            reference_epoch=np.datetime64("2010-01-01T00:01:06.184", "ns"),
        ),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch"),
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"),
    )

    # TODO: Change 'TBD' catdesc and fieldname
    #       Once housekeeping packet definition file is re-generated with
    #       updated version of space_packet_parser, can get fieldname and
    #       catdesc info via:
    #           for key, value in (packet.header | packet.data).items():
    #               fieldname = value.short_description
    #              catdesc = value.long_description
    #       I am holding off making this change until I acquire updated
    #       housekeeping packets/validation data that match the latest telemetry
    #       definitions
    for key, value in metadata_arrays.items():
        attrs = cdf_attrs.get_variable_attributes("codice_support_attrs")
        attrs["CATDESC"] = "TBD"
        attrs["DEPEND_0"] = "epoch"
        attrs["FIELDNAM"] = "TBD"
        attrs["LABLAXIS"] = key

        dataset[key] = xr.DataArray(value, dims=["epoch"], attrs=attrs)

    return dataset


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
    # Decom the packets, group data by APID, and sort by time
    packets = decom_packets(file_path)
    grouped_data = group_by_apid(packets)

    for apid in grouped_data:
        logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

        if apid == CODICEAPID.COD_NHK:
            packets = grouped_data[apid]
            sorted_packets = sort_by_time(packets, "SHCOARSE")
            dataset = create_hskp_dataset(sorted_packets, data_version)

        elif apid in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            if apid == CODICEAPID.COD_LO_PHA:
                dataset_name = "imap_codice_l1a_lo_pha"
            elif apid == CODICEAPID.COD_HI_PHA:
                dataset_name = "imap_codice_l1a_hi_pha"

            # Sort the packets by time
            packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            # Determine the start time of the packet
            met = packets[0].data["ACQ_START_SECONDS"].raw_value
            met = [met, met + 1]  # TODO: Remove after cdflib fix

            # Extract the data
            event_data = packets[0].data["EVENT_DATA"].raw_value

            # Create the dataset
            dataset = create_event_dataset(met, event_data, dataset_name, data_version)

        elif apid in constants.APIDS_FOR_SCIENCE_PROCESSING:
            # Sort the packets by time
            packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            # Determine the start time of the packet
            met = packets[0].data["ACQ_START_SECONDS"].raw_value

            # Extract the data
            science_values = packets[0].data["DATA"].raw_value

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(packets[0])

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.configure_data_products(apid)
            pipeline.unpack_science_data(science_values)
            dataset = pipeline.create_science_dataset(met, data_version)

    logger.info(f"\nFinal data product:\n{dataset}\n")

    return dataset
