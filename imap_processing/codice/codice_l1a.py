"""Perform CoDICE l1a processing.

This module processes decommutated CoDICE packets and creates L1a data products.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1a import codice_l1a
    packets = decom_packets(packet_file)
    dataset = process_codice_l1a(packets)
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.codice import cdf_attrs
from imap_processing.codice.constants import (
    ESA_SWEEP_TABLE_ID_LOOKUP,
    LO_COLLAPSE_TABLE_ID_LOOKUP,
    LO_COMPRESSION_ID_LOOKUP,
    LO_NSW_SPECIES_NAMES,
    LO_STEPPING_TABLE_ID_LOOKUP,
    LO_SW_PRIORITY_NAMES,
    LO_SW_SPECIES_NAMES,
)
from imap_processing.codice.utils import CODICEAPID, create_hskp_dataset
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CoDICEL1aPipeline:
    """Contains methods for processing L0 data and creating L1a data products.

    Attributes
    ----------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet
    view_id : int
        Provides information about how data was collapsed and/or compressed

    Methods
    -------
    create_science_dataset()
        Create an ``xarray`` dataset for the unpacked science data.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_esa_sweep_values()
        Retrieve the ESA sweep values.
    get_lo_data_products()
        Retrieve the lo data products.
    unpack_science_data()
        Make 4D L1a data product from the decompressed science data.
    """

    def __init__(self, table_id: int, plan_id: int, plan_step: int, view_id: int):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def create_science_dataset(self, packets: list) -> xr.Dataset:
        """Create an ``xarray`` dataset for the unpacked science data.

        The dataset can then be written to a CDF file.

        Parameters
        ----------
        packet : list[space_packet_parser.parser.Packet]
            List of packets for the APID of interest

        Returns
        -------
        xr.Dataset
            ``xarray`` dataset containing the science data and supporting metadata

        # TODO: Pull out common code and put in codice.utils alongside
        # create_hskp_dataset()
        """
        epoch = xr.DataArray(
            [calc_start_time(packets[0].data["SHCOARSE"].raw_value)],
            name="epoch",
            dims=["epoch"],
            attrs=ConstantCoordinates.EPOCH,
        )

        energy_steps = xr.DataArray(
            np.arange(self.num_energy_steps),
            name="energy",
            dims=["energy"],
            attrs=cdf_attrs.energy_attrs.output(),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch, "energy": energy_steps},
            attrs=self.cdf_attrs.output(),
        )

        # Create a data variable for each species
        for variable_data, name in zip(self.data, self.variable_names):
            varname, fieldname = name
            variable_data_arr = [int(item) for item in variable_data]
            variable_data_arr = np.array(variable_data_arr).reshape(
                -1, self.num_energy_steps
            )

            dataset[varname] = xr.DataArray(
                variable_data_arr,
                name=varname,
                dims=["epoch", "energy"],
                attrs=dataclasses.replace(
                    cdf_attrs.counts_attrs, fieldname=fieldname
                ).output(),
            )

        # TODO: Add in the ESA sweep values and acquisition times? (Confirm with Joey)

        return dataset

    def get_acquisition_times(self):
        """Retrieve the acquisition times via the Lo stepping table.

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
        lo_stepping_table_id = LO_STEPPING_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        # Get the appropriate values
        # TODO: update lo_stepping_values.csv with updated data
        lo_stepping_values = lo_stepping_data[
            lo_stepping_data["table_num"] == lo_stepping_table_id
        ]

        # Get the acquisition times
        self.acquisition_times = lo_stepping_values.acq_time

        # TODO: Expand acquisition times list so that each energy step has an
        # associated time

    def get_esa_sweep_values(self):
        """Retrieve the ESA sweep values.

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
        sweep_table_id = ESA_SWEEP_TABLE_ID_LOOKUP[(self.plan_id, self.plan_step)]

        # Get the appropriate values
        sweep_table = sweep_data[sweep_data["table_idx"] == sweep_table_id]
        self.esa_sweep_values = sweep_table["esa_v"].values

    def get_lo_data_products(self, apid: int):
        """Retrieve the lo data products table.

        Parameters
        ----------
        apid : int
            The APID of interest.
        """
        if apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
            self.num_counters = 16
            self.num_energy_steps = 128
            self.variable_names = LO_SW_SPECIES_NAMES
            self.cdf_attrs = cdf_attrs.l1a_lo_sw_species_counts_attrs
        elif apid == CODICEAPID.COD_LO_NSW_SPECIES_COUNTS:
            self.num_counters = 8
            self.num_energy_steps = 112
            self.variable_names = LO_NSW_SPECIES_NAMES
            self.cdf_attrs = cdf_attrs.l1a_lo_nsw_species_counts_attrs
        elif apid == CODICEAPID.COD_LO_SW_PRIORITY_COUNTS:
            self.num_counters = 5
            self.num_energy_steps = 211
            self.variable_names = LO_SW_PRIORITY_NAMES
            self.cdf_attrs = cdf_attrs.l1a_lo_sw_priority_counts_attrs

    def unpack_science_data(self, packets: list):
        """Unpack the science data from the packet.

        For LO SW Species Counts data, the science data within the packet is a
        blob of compressed values of length 2048 bits (16 species * 128 energy
        levels). These data need to be divided up by species so that each
        species can have their own data variable in the L1A CDF file.

        Parameters
        ----------
        packet : list[space_packet_parser.parser.Packet]
            List of packets for the APID of interest

        TODO: Check to see if we expect to have multiple packets?
        """
        self.compression_algorithm = LO_COMPRESSION_ID_LOOKUP[self.view_id]
        self.collapse_table_id = LO_COLLAPSE_TABLE_ID_LOOKUP[self.view_id]

        science_values = packets[0].data["DATA"].raw_value

        # Divide up the data by the number of priorities or species
        num_bytes = len(science_values)
        chunk_size = len(science_values) // self.num_counters
        self.data = [
            science_values[i : i + chunk_size] for i in range(0, num_bytes, chunk_size)
        ]


def get_params(packet) -> tuple[int, int, int, int]:
    """Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        A packet for the APID of interest

    Returns
    -------
    table_id : int
        A unique ID assigned to a specific table configuration. This field is
        used to link the overall acquisition and processing settings to a
        specific table configuration
    plan_id : int
        The plan table that was in use.  In conjunction with ``plan_step``,
        describes which counters are included in the data packet
    plan_step : int
        Plan step that was active when the data was acquired and processed. In
        conjunction with ``plan_id``, describes which counters are included
        in the data packet
    view_id : int
        Provides information about how data was collapsed and/or compressed
    """
    table_id = packet.data["TABLE_ID"].raw_value
    plan_id = packet.data["PLAN_ID"].raw_value
    plan_step = packet.data["PLAN_STEP"].raw_value
    view_id = packet.data["VIEW_ID"].raw_value

    return table_id, plan_id, plan_step, view_id


def process_codice_l1a(packets) -> xr.Dataset:
    """Process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        Decom data list that contains all APIDs

    Returns
    -------
    dataset : xarray.Dataset
        ``xarray`` dataset containing the science data and supporting metadata
    """
    apids_for_lo_science_processing = [
        CODICEAPID.COD_LO_SW_SPECIES_COUNTS,
        CODICEAPID.COD_LO_NSW_SPECIES_COUNTS,
        CODICEAPID.COD_LO_SW_PRIORITY_COUNTS,
    ]

    # Group data by APID and sort by time
    grouped_data = group_by_apid(packets)

    for apid in grouped_data.keys():
        logger.info(f"\nProcessing {CODICEAPID(apid).name} packet")

        if apid == CODICEAPID.COD_NHK:
            packets = grouped_data[apid]
            sorted_packets = sort_by_time(packets, "SHCOARSE")
            dataset = create_hskp_dataset(packets=sorted_packets)

        elif apid in apids_for_lo_science_processing:
            packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(packets[0])

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.get_esa_sweep_values()
            pipeline.get_acquisition_times()
            pipeline.get_lo_data_products(apid)
            pipeline.unpack_science_data(packets)
            dataset = pipeline.create_science_dataset(packets)

        elif apid == CODICEAPID.COD_LO_PHA:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_NSW_PRIORITY_COUNTS:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_SW_ANGULAR_COUNTS:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_PHA:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS:
            logger.info(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_SECT_SPECIES_COUNTS:
            logger.info(f"{apid} is currently not supported")
            continue

    # Write dataset to CDF
    logger.info(f"\nFinal data product:\n{dataset}\n")
    dataset.attrs["cdf_filename"] = write_cdf(dataset)
    logger.info(f"\tCreated CDF file: {dataset.cdf_filename}")

    return dataset
