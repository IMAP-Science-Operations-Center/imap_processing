"""Perform CoDICE l1a processing.

This module processes decommutated CoDICE packets and creates L1a data products.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1a import codice_l1a
    packets = decom_packets(packet_file, xtce_document)
    cdf_filename = codice_l1a(packets)
"""

import collections
import logging
import random
from pathlib import Path

import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.codice.cdf_attrs import codice_l1a_global_attrs
from imap_processing.codice.constants import ESA_SWEEP_TABLE, LO_STEPPING_TABLE
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    CoDICECompression,
    create_dataset,
)
from imap_processing.utils import group_by_apid, sort_by_time


class CoDICEL1a:
    """Contains methods for processing L0 data and creating L1a data products.

    Attributes
    ----------
    TODO: Fill in documentation here

    Methods
    -------
    TODO: Fill in documentation here
    """

    def __init__(self, table_id, plan_id, plan_step, view_id):
        self.table_id = table_id
        self.plan_id = plan_id
        self.plan_step = plan_step
        self.view_id = view_id

    def _get_random_bits(self, length):
        """Return a list of random bits of given length.

        Parameters
        ----------
        length : int
            The number of bits to generate
        """
        return [random.randomint(0, 1) for _ in range(length)]

    def decompress_science_data(self):
        """Decompress the compressed science values.

        The science values are decompressed based on the compression algorithm
        that was used in the creation of the data packet.
        """
        # The compression algorithm can be derived from the packet data
        # For now, simulate which algorithm to use
        compression_algorithm = CoDICECompression.LOSSY_A

        # Get the science values
        compressed_values = self._get_random_bits(1179648)
        self.science_values = decompress(compressed_values, compression_algorithm)

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
        lo_stepping_table_id = LO_STEPPING_TABLE[(self.plan_id, self.plan_step)]

        # Get the appropriate values
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
        sweep_table_id = ESA_SWEEP_TABLE[(self.plan_id, self.plan_step)]

        # Get the appropriate values
        self.esa_sweep_values = sweep_data[sweep_data["table_idx"] == sweep_table_id]

        # TODO: Only select the esa_v values from the dataframe?

    def make_cdf_data(self):
        """Create the ``xarray`` datasets needed for the L1a CDF file.

        Returns
        -------
        dataset : xr.Dataset
            The dataset used in the L1a CDF file
        """
        metadata_arrays = collections.defaultdict(list)

        epoch_time = xr.DataArray(
            metadata_arrays["SHCOARSE"],
            name="Epoch",
            dims=["Epoch"],
            attrs=ConstantCoordinates.EPOCH,
        )

        energy = xr.Dataset(
            self.esa_sweep_values, name="Energy", dims=["Energy"], attrs="TBD"
        )

        times = xr.Dataset(
            self.acquisition_times, name="Times", dims=["Times"], attrs="TBD"
        )

        dataset = xr.Dataset(
            coords={
                "Epoch": epoch_time,
                "Energy": energy,
                "Times": times,
            },
            attrs=codice_l1a_global_attrs.output(),
        )

        dataset["SCIENCE_DATA"] = self.science_values

        return dataset

    def unpack_science_data(self):
        """Make 4D L1a data product from the decompressed science data.

        Take the decompressed science data and reorganize the bytes to
        create a four-dimensional data product

        TODO: Describe the data product in more detail
        """
        # TODO: Implement this
        pass


def get_params(packets):
    """Return the four 'main' parameters used for l1a processing.

    The combination of these parameters largely determines what steps/values
    are used to create CoDICE L1a data products and what steps are needed in
    the pipeline algorithm.

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
    # These can be derived from the packet, but for now just simulate the values
    table_id = 1
    plan_id = 1
    plan_step = 1
    view_id = 1

    return table_id, plan_id, plan_step, view_id


def process_codice_l1a(packets, cdf_directory: str) -> str:
    """Process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        Decom data list that contains all APIDs
    cdf_directory : str
        The directory in which to write the output CDF file.

    Returns
    -------
    cdf_filename : str
        The path to the CDF file that was created
    """
    # Group data by APID and sort by time
    grouped_data = group_by_apid(packets)
    grouped_data = {1154: []}

    for apid in grouped_data.keys():
        if apid == CODICEAPID.COD_NHK:
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")
            data = create_dataset(packets=sorted_packets)

        elif apid == CODICEAPID.COD_LO_INSTRUMENT_COUNTERS:
            packets = grouped_data[apid]
            table_id, plan_id, plan_step, view_id = get_params(packets)

            pipeline = CoDICEL1a(table_id, plan_id, plan_step, view_id)
            pipeline.get_esa_sweep_values()
            pipeline.get_acquisition_times()
            pipeline.decompress_science_data()
            pipeline.unpack_science_data()
            data = pipeline.make_cdf_data()

        else:
            logging.debug(f"{apid} is currently not supported")

    # Write data to CDF
    cdf_filename = write_cdf(
        data,
        mode="",
        description="hk",
        directory=cdf_directory,
    )

    return cdf_filename


if __name__ == "__main__":
    process_codice_l1a([], "")
