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

import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory, launch_time
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.codice import cdf_attrs
from imap_processing.codice.constants import (
    ESA_SWEEP_TABLE_ID_LOOKUP,
    LO_COLLAPSE_TABLE_ID_LOOKUP,
    LO_COMPRESSION_ID_LOOKUP,
    LO_STEPPING_TABLE_ID_LOOKUP,
    LO_SW_SPECIES_NAMES,
)
from imap_processing.codice.decompress import decompress
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
    use_simulated_data : bool
        When ``True``, simulated science data is generated and used in the
        processing pipeline. This is useful for development and testing in the
        absence of actual CoDICE testing data., and is intended to be removed
        once simulated science data is no longer needed.

    Methods
    -------
    _generate_simulated_data(length_in_bits)
        Return a list of random bytes to provide simulated science data.
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

    def __init__(self, table_id, plan_id, plan_step, view_id):
        """Initialize a ``CoDICEL1aPipeline`` class instance."""
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id
        self.use_simulated_data = True

    def _generate_simulated_data(self, length_in_bits):
        """Return a list of random bytes to provide simulated science data.

        This method is used as a temporary workaround to simulate science data
        in the absence of real data to test with.

        Parameters
        ----------
        length_in_bits : int
            The number of bits used to generate the list of bytes. For example,
            a ``length_in_bits`` of 80 yields a list of 10 bytes.

        Returns
        -------
        random_bytes : bytes
            A list of random bytes to be used as simulated science data
        """
        logger.info("\tUsing simulated data")
        # Generate string of random bits of proper length
        bit_string = bin(random.getrandbits(length_in_bits))[2:]
        bit_string = bit_string.zfill(length_in_bits)
        print(f"\tLength of data in bits: {len(bit_string)}")

        # Convert list of random bits into byte-size list of integers
        random_bytes_str = [bit_string[i : i + 8] for i in range(0, len(bit_string), 8)]
        random_bytes_int = [int(item, 2) for item in random_bytes_str]
        random_bytes = bytes(random_bytes_int)
        print(f"\tLength of data in bytes: {len(random_bytes)}")

        return random_bytes

    def create_science_dataset(self):
        """Create an ``xarray`` dataset for the unpacked science data.

        The dataset can then be written to a CDF file.

        Returns
        -------
        xarray.Dataset
            xarray dataset containing the science data and supporting metadata

        # TODO: Pull out common code and put in codice.utils alongside
        # create_hskp_dataset()
        """
        # Define the attributes specific to species data
        species_attrs = dataclasses.replace(
            cdf_attrs.l1a_science_attrs,
            depend_0="species",
            depend_1=None,
            depend_2=None,
            catdesc="TBD",
            fieldname="TBD",
            label_axis="TBD",
        ).output()

        epoch_time = xr.DataArray(
            [
                launch_time for i in range(len(self.data))
            ],  # Temporary workaround to get epoch in correct dimensions
            name="epoch",
            dims=["epoch"],
            attrs=ConstantCoordinates.EPOCH,
        )

        species_array = xr.DataArray(
            name="species",
            data=np.zeros(36864),
            dims=("species"),
            attrs=species_attrs,
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "species": species_array},
            attrs=cdf_attrs.codice_l1a_global_attrs.output(),
        )

        # Create a data variable for each species
        for species_data, species_name in zip(self.data, LO_SW_SPECIES_NAMES):
            data = xr.DataArray(
                name=species_name,
                data=list(species_data),
                dims=("species"),
                attrs=species_attrs,
            )
            dataset[species_name] = data

        dataset.attrs["Logical_source"] = "imap_codice_l1a_lo-sw-species-counts"

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
        """Make 4D L1a data product from the decompressed science data.

        Take the decompressed science data and reorganize the bytes to
        create a four-dimensional data product

        TODO: Describe the data product in more detail in docstring
        """
        print("Unpacking science data")

        self.compression_algorithm = LO_COMPRESSION_ID_LOOKUP[self.view_id]
        self.collapse_table_id = LO_COLLAPSE_TABLE_ID_LOOKUP[self.view_id]

        # Generate simulated science data
        # TODO: Take hard coded bit length value out and use variable instead
        if self.use_simulated_data:
            science_values = self._generate_simulated_data(
                4718592
            )  # 16 species * 128 energies * 24 positions * 12 spin angles * 8 bits
        else:
            # Decompress the science data
            compressed_values = None
            print(
                f"Decompressing science data using {self.compression_algorithm.name} algorithm"  # noqa
            )
            science_values = [
                decompress(compressed_value, self.compression_algorithm)
                for compressed_value in compressed_values
            ]

        # TODO: Confirm with joey that I am unpacking this correctly
        # Chunk of the data by the number of species
        num_bytes = len(science_values)
        num_species = 16
        chunk_size = len(science_values) // num_species
        self.data = [
            science_values[i : i + chunk_size] for i in range(0, num_bytes, chunk_size)
        ]


def get_params(apid):
    """Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

    This function is intended to serve as a temporary workaround until proper
    testing data are acquired. These values should be able to be derived in the
    test data. Once proper testing data are acquired, this function will grab
    the values from the CCSDS packet.

    Parameters
    ----------
    apid : IntEnum
        The APID of interest

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
    # table_id will likely always be 1
    table_id = 1

    if apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
        plan_id = 1
        plan_step = 1
        view_id = 5

    # Default values
    else:
        plan_id = 1
        plan_step = 1
        view_id = 1

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

            # These are currently commented out because the test data does not
            # have SHCOARSE
            # TODO: Turn these "on" once proper testing data is acquired
            # packets = grouped_data[apid]
            # sorted_packets = sort_by_time(packets, "SHCOARSE")

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(apid)

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
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
