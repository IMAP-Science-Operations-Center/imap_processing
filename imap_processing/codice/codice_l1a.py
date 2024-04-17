"""Perform CoDICE l1a processing.

This module processes decommutated CoDICE packets and creates L1a data products.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1a import codice_l1a
    packets = decom_packets(packet_file, xtce_document)
    cdf_filename = codice_l1a(packets)
"""

# TODO: Change print statements to logging
# TODO: Add type hints

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
    LO_STEPPING_TABLE_ID_LOOKUP,
    LO_SW_SPECIES_FIELDNAMES,
    LO_SW_SPECIES_VARNAMES,
)
from imap_processing.codice.utils import CODICEAPID, create_hskp_dataset
from imap_processing.utils import group_by_apid, sort_by_time

logger = logging.getLogger(__name__)


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

    def create_science_dataset(self) -> xr.Dataset:
        """Create an ``xarray`` dataset for the unpacked science data.

        The dataset can then be written to a CDF file.

        Returns
        -------
        xr.Dataset
            xarray dataset containing the science data and supporting metadata

        # TODO: Pull out common code and put in codice.utils alongside
        # create_hskp_dataset()
        """
        epoch = xr.DataArray(
            [calc_start_time(self.packets[0].data["SHCOARSE"].raw_value)],
            name="epoch",
            dims=["epoch"],
            attrs=ConstantCoordinates.EPOCH,
        )

        energy = xr.DataArray(
            np.arange(128),
            name="energy",
            dims=["energy"],
            attrs=cdf_attrs.energy_attrs.output(),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch, "energy": energy},
            attrs=cdf_attrs.l1a_lo_sw_species_attrs.output(),
        )

        # Create a data variable for each species
        for species_data, varname, fieldname in zip(
            self.data, LO_SW_SPECIES_VARNAMES, LO_SW_SPECIES_FIELDNAMES
        ):
            species_data_arr = [int(item) for item in species_data]
            species_data_arr = np.array(species_data_arr).reshape(-1, 128)
            dataset[varname] = xr.DataArray(
                species_data_arr,
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

    def get_lo_data_products(self):
        """Retrieve the lo data products table."""
        # TODO: implement this

        pass

    def unpack_science_data(self):
        """Unpack the science data from the packet.

        For LO SW Species Counts data, the science data within the packet is a
        blob of compressed values of length 2048 bits (16 species * 128 energy
        levels). These data need to be divided up by species so that each
        species can have their own data variable in the L1A CDF file.

        TODO: Make this method more generalized for other APIDs
        TODO: Check to see if we expect to have multiple packets?
        """
        print("Unpacking science data")

        self.compression_algorithm = LO_COMPRESSION_ID_LOOKUP[self.view_id]
        self.collapse_table_id = LO_COLLAPSE_TABLE_ID_LOOKUP[self.view_id]

        science_values = self.packets[0].data["DATA"].raw_value

        # Divide up the data by the number of species
        num_bytes = len(science_values)
        num_species = 16
        chunk_size = len(science_values) // num_species
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


def process_codice_l1a(packets) -> str:
    """Process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        Decom data list that contains all APIDs

    Returns
    -------
    cdf_filename : str
        The path to the CDF file that was created
    """
    # Group data by APID and sort by time
    print("Grouping the data by APID")
    grouped_data = group_by_apid(packets)

    for apid in grouped_data.keys():
        if apid == CODICEAPID.COD_NHK:
            print("Processing COD_NHK packet")
            packets = grouped_data[apid]
            sorted_packets = sort_by_time(packets, "SHCOARSE")
            dataset = create_hskp_dataset(packets=sorted_packets)

        elif apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
            print("Processing COD_LO_SW_SPECIES_COUNTS packet")

            packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(packets[0])

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.packets = packets
            pipeline.get_esa_sweep_values()
            pipeline.get_acquisition_times()
            pipeline.get_lo_data_products()
            pipeline.unpack_science_data()
            dataset = pipeline.create_science_dataset()

        elif apid == CODICEAPID.COD_LO_PHA:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_PRIORITY_COUNTS:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_NSW_SPECIES_COUNTS:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_SW_ANGULAR_COUNTS:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_PHA:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS:
            print(f"{apid} is currently not supported")
            continue

        elif apid == CODICEAPID.COD_HI_SECT_SPECIES_COUNTS:
            print(f"{apid} is currently not supported")
            continue

    # Write dataset to CDF
    print(f"\nFinal data product:\n{dataset}\n")
    cdf_filename = write_cdf(dataset)
    print(f"Created CDF file: {cdf_filename}")
    return cdf_filename
