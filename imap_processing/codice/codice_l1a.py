"""
Perform CoDICE l1a processing.

This module processes CoDICE L0 files and creates L1a data products.

Notes
-----
    from imap_processing.codice.codice_l1a import process_codice_l1a
    processed_datasets = process_codice_l1a(path_to_l0_file)
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
from imap_processing.codice.utils import CODICEAPID, CoDICECompression
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    define_energy_bins()
        Define/add variables to the dataset that correspond to the energy bins.
    define_support_variables()
        Define and add 'support' CDF data variables to the dataset.
    get_acquisition_times()
        Retrieve the acquisition times via the Lo stepping table.
    get_energy_table()
        Retrieve the ESA sweep values.
    get_hi_energy_table_data(species)
        Retrieve energy table data for CoDICE-Hi products
    reshape_binned_data(dataset)
        Reshape data arrays for binned datasets.
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

    def decompress_data(self, science_values: list[NDArray[str]] | list[str]) -> None:
        """
        Perform decompression on the data.

        The science data within the packet is a compressed byte string of
        values. Apply the appropriate decompression algorithm to get an array
        of decompressed values.

        Parameters
        ----------
        science_values : list[NDArray[str]] | list[str]
            A list of byte strings (or bit strings, in the case of I-ALiRT)
            representing the science values of the data for each packet.
        """
        # The compression algorithm depends on the instrument and view ID
        if self.config["instrument"] == "lo":
            compression_algorithm = constants.LO_COMPRESSION_ID_LOOKUP[self.view_id]
        elif self.config["instrument"] == "hi":
            compression_algorithm = constants.HI_COMPRESSION_ID_LOOKUP[self.view_id]

        self.raw_data = []

        # I-ALiRT data already has byte count cut-off applied, so treat
        # it slightly differently
        if "ialirt" in self.config["dataset_name"]:
            for packet_data in science_values:
                # Convert from bit string to byte object
                values = int(packet_data, 2).to_bytes(
                    len(packet_data) // 8, byteorder="big"
                )
                decompressed_values = decompress(values, compression_algorithm)
                self.raw_data.append(decompressed_values)

        else:
            for packet_data, byte_count in zip(
                science_values, self.dataset.byte_count.data
            ):
                # Convert from numpy array to byte object
                values = packet_data[()]

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

        coord_names = [
            *self.config["output_dims"].keys(),
            *[key + "_label" for key in self.config["output_dims"].keys()],
        ]

        # Define epoch coordinates
        epochs, epoch_delta_minus, epoch_delta_plus = calculate_epoch_values(
            self.dataset.acq_start_seconds, self.dataset.acq_start_subseconds
        )
        for name, var in [
            ("epoch", epochs),
            ("epoch_delta_minus", epoch_delta_minus),
            ("epoch_delta_plus", epoch_delta_plus),
        ]:
            coord = xr.DataArray(
                var,
                name=name,
                dims=[name],
                attrs=self.cdf_attrs.get_variable_attributes(name, check_schema=False),
            )
            self.coords[name] = coord

        # Define the values for the coordinates
        for name in coord_names:
            if name in [
                "esa_step",
                "inst_az",
                "spin_sector",
                "spin_sector_pairs",
                "spin_sector_index",
                "ssd_index",
            ]:
                values = np.arange(self.config["output_dims"][name])
                dims = [name]
            elif name == "spin_sector_pairs_label":
                values = np.array(
                    [
                        "0-30 deg",
                        "30-60 deg",
                        "60-90 deg",
                        "90-120 deg",
                        "120-150 deg",
                        "150-180 deg",
                    ]
                )
                dims = [name]
            elif name in [
                "spin_sector_label",
                "esa_step_label",
                "inst_az_label",
                "spin_sector_index_label",
                "ssd_index_label",
            ]:
                key = name.removesuffix("_label")
                values = np.arange(self.config["output_dims"][key]).astype(str)
                dims = [key]

            coord = xr.DataArray(
                values,
                name=name,
                dims=dims,
                attrs=self.cdf_attrs.get_variable_attributes(name, check_schema=False),
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

        # The dimension of all_data is something like (epoch, num_counters,
        # num_energy_steps, num_positions, num_spin_sectors) (or may be slightly
        # different depending on the data product). In any case, iterate over
        # the num_counters dimension to isolate the data for each counter so
        # each counter's data can be placed in a separate CDF data variable.
        for counter, variable_name in zip(
            range(all_data.shape[1]), self.config["variable_names"]
        ):
            # Extract the counter data
            counter_data = all_data[:, counter, ...]

            # Get the CDF attributes
            descriptor = self.config["dataset_name"].split("imap_codice_l1a_")[-1]
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            attrs = self.cdf_attrs.get_variable_attributes(cdf_attrs_key)

            # For most products, the final CDF dimensions always has "epoch" as
            # the first dimension followed by the dimensions for the specific
            # data product
            dims = ["epoch", *list(self.config["output_dims"].keys())]

            # However, CoDICE-Hi products use specific energy bins for the
            # energy dimension
            # TODO: This bit of code may no longer be needed once I can figure
            #       out how to run hi-sectored product through the
            #       create_binned_dataset function
            if self.config["dataset_name"] == "imap_codice_l1a_hi-sectored":
                dims = [
                    f"energy_{variable_name}" if item == "esa_step" else item
                    for item in dims
                ]

            # Create the CDF data variable
            dataset[variable_name] = xr.DataArray(
                counter_data,
                name=variable_name,
                dims=dims,
                attrs=attrs,
            )

        # Add support data variables based on data product
        dataset = self.define_support_variables(dataset)

        # For CoDICE-Hi products, since energy dimension was replaced, we no
        # longer need the "esa_step" coordinate
        # TODO: This bit of code may no longer be needed once I can figure
        #       out how to run hi-sectored product through the
        #       create_binned_dataset function
        if self.config["dataset_name"] == "imap_codice_l1a_hi-sectored":
            for species in self.config["energy_table"]:
                dataset = self.define_energy_bins(dataset, species)
            dataset = dataset.drop_vars(["esa_step", "esa_step_label"])

        return dataset

    def define_energy_bins(self, dataset: xr.Dataset, species: str) -> xr.Dataset:
        """
        Define/add variables to the dataset that correspond to the energy bins.

        For hi-omni and hi-sectored data products specifically, the L1a data
        product contains the energy bin centers and deltas. This method
        handles adding these bins as CDF variables and their attributes.

        Parameters
        ----------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product.
        species : str
            The species for which to add the energy bins (e.g. "he3").

        Returns
        -------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product, with added energy variables.
        """
        energy_bin_name = f"energy_{species}"
        centers, deltas = self.get_hi_energy_table_data(
            energy_bin_name.split("energy_")[-1]
        )

        # Add bin centers and deltas to the dataset
        dataset[energy_bin_name] = xr.DataArray(
            centers,
            dims=[energy_bin_name],
            attrs=self.cdf_attrs.get_variable_attributes(
                f"{self.config['dataset_name'].split('_')[-1]}-{energy_bin_name}",
                check_schema=False,
            ),
        )
        dataset[f"{energy_bin_name}_delta"] = xr.DataArray(
            deltas,
            dims=[f"{energy_bin_name}_delta"],
            attrs=self.cdf_attrs.get_variable_attributes(
                f"{self.config['dataset_name'].split('_')[-1]}-{energy_bin_name}_delta",
                check_schema=False,
            ),
        )

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
        # These variables can be gathered from the packet data
        packet_data_variables = [
            "rgfo_half_spin",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
        ]

        for variable_name in self.config["support_variables"]:
            # These variables require reading in external tables
            if variable_name == "energy_table":
                variable_data = self.get_energy_table()
                dims = ["esa_step"]
                attrs = self.cdf_attrs.get_variable_attributes(
                    "energy_table", check_schema=False
                )

            elif variable_name == "acquisition_time_per_step":
                variable_data = self.get_acquisition_times()
                dims = ["esa_step"]
                attrs = self.cdf_attrs.get_variable_attributes(
                    "acquisition_time_per_step", check_schema=False
                )

            # These variables can be gathered straight from the packet data
            elif variable_name in packet_data_variables:
                variable_data = self.dataset[variable_name].data
                dims = ["epoch"]
                attrs = self.cdf_attrs.get_variable_attributes(variable_name)

            # Data quality is named differently in packet data and needs to be
            # treated slightly differently
            elif variable_name == "data_quality":
                if "hi-omni" in self.config["dataset_name"]:
                    continue
                variable_data = self.dataset.suspect.data
                dims = ["epoch"]
                attrs = self.cdf_attrs.get_variable_attributes("data_quality")

            # Spin period requires the application of a conversion factor
            # See Table B.5 in the algorithm document
            elif variable_name == "spin_period":
                if "hi-omni" in self.config["dataset_name"]:
                    continue
                variable_data = (
                    self.dataset.spin_period.data * constants.SPIN_PERIOD_CONVERSION
                ).astype(np.float32)
                dims = ["epoch"]
                attrs = self.cdf_attrs.get_variable_attributes("spin_period")

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

        Get the acquisition times from the lookup table based on the values of
        ``plan_id`` and ``plan_step``

        The Lo stepping table defines how many voltage steps and which steps are
        used during each spacecraft spin. A full cycle takes 16 spins. The table
        provides the "acquisition time", which is the acquisition time, in
        milliseconds, for the energy step.

        Returns
        -------
        acquisition_times : list[float]
            The list of acquisition times from the Lo stepping table.
        """
        # Determine which Lo stepping table is needed
        lo_stepping_table_id = constants.LO_STEPPING_TABLE_ID_LOOKUP[
            (self.plan_id, self.plan_step)
        ]

        acquisition_times: list[float] = constants.ACQUISITION_TIMES[
            lo_stepping_table_id
        ]

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

    def get_hi_energy_table_data(
        self, species: str
    ) -> tuple[NDArray[float], NDArray[float]]:
        """
        Retrieve energy table data for CoDICE-Hi products.

        This includes the centers and deltas of the energy bins for a given
        species. These data eventually get included in the CoDICE-Hi CDF data
        products.

        Parameters
        ----------
        species : str
            The species of interest, which determines which lookup table to
            use (e.g. ``h``).

        Returns
        -------
        centers : NDArray[float]
            An array whose values represent the centers of the energy bins.
        deltas : NDArray[float]
            An array whose values represent the deltas of the energy bins.
        """
        data_product = self.config["dataset_name"].split("-")[-1].upper()
        energy_table = getattr(constants, f"{data_product}_ENERGY_TABLE")[species]

        # Find the centers and deltas of the energy bins
        centers = np.array(
            [
                (energy_table[i] + energy_table[i + 1]) / 2
                for i in range(len(energy_table) - 1)
            ]
        )
        deltas = energy_table[1:] - centers

        return centers, deltas

    def reshape_binned_data(self, dataset: xr.Dataset) -> dict[str, list]:
        """
        Reshape data arrays for binned datasets.

        Binned datasets get reshaped based on the number of species and their
        corresponding number of energy bins. Additionally, the number of spins
        during data acquisition are collapsed/summed which also needs to be taken
        into account when reshaping into the correct dimensions.

        Parameters
        ----------
        dataset : xarray.Dataset
            ``xarray`` dataset for the data product.

        Returns
        -------
        data : dict[str, list]
            Data arrays for each species.
        """
        # This will hold all of the data per-species and support variables,
        # ready to be put in a CDF file
        data: dict[str, list] = {}
        for species in self.config["energy_table"]:
            data[species] = []
            data["epoch"] = []
            data["spin_period"] = []
            data["data_quality"] = []

        # Get the number of spins per species
        num_spins = self.config["num_spins"]

        # Iterate through each epoch's data and pull out the data for each
        # species
        stacked_data = np.array(self.raw_data, dtype=np.uint32)
        for i, epoch in enumerate(stacked_data):
            current_epoch = dataset.epoch.data[i]
            position = 0
            for species in self.config["energy_table"]:
                # Subtracting one here since the table includes endpoints
                num_bins = len(self.config["energy_table"][species]) - 1
                species_data = (
                    epoch[position : position + num_bins * self.config["num_spins"]]
                    .reshape(num_bins, num_spins)
                    .T
                )

                # Now pull out the data for each spin within the species data
                for spin_data in species_data:
                    data[species].append(spin_data)

                    # We only need one set of support variables in the CDF,
                    # so just iterate using one species for these
                    if species == "h":
                        # For each spin, we add <spin_period>*<num_spins> to the
                        # epoch value
                        spin_period = (
                            dataset.spin_period.data[i]
                            * constants.SPIN_PERIOD_CONVERSION
                        )
                        epoch_value = current_epoch + np.int64(
                            (spin_period * num_spins) * 1e9  # Convert from s to ns
                        )
                        data["epoch"].append(epoch_value)
                        current_epoch = epoch_value

                        # Other support variables
                        data["spin_period"].append(spin_period)
                        data["data_quality"].append(dataset.suspect.data[i])

                position += num_bins * num_spins

        return data

    def reshape_data(self) -> None:
        """
        Reshape the data arrays based on the data product being made.

        These data need to be divided up by species or priorities (or
        what I am calling "counters" as a general term), and re-arranged into
        4D arrays representing dimensions such as time, spin sectors, positions,
        and energies (depending on the data product).

        However, the existence and order of these dimensions can vary depending
        on the specific data product, so we define this in the "input_dims"
        and "output_dims" values configuration dictionary; the "input_dims"
        defines how the dimensions are written into the packet data, while
        "output_dims" defines how the dimensions should be written to the final
        CDF product.
        """
        # This will contain the reshaped data for all counters
        self.data = []

        # First reshape the data based on how it is written to the data array of
        # the packet data. The number of counters is the first dimension / axis,
        # with the exception of lo-counters-aggregated which is treated slightly
        # differently
        if self.config["dataset_name"] != "imap_codice_l1a_lo-counters-aggregated":
            reshape_dims = (
                self.config["num_counters"],
                *self.config["input_dims"].values(),
            )
        else:
            reshape_dims = (
                *self.config["input_dims"].values(),
                self.config["num_counters"],
            )

        # Then, transpose the data based on how the dimensions should be written
        # to the CDF file. Since this is specific to each data product, we need
        # to determine this dynamically based on the "output_dims" config.
        # Again, lo-counters-aggregated is treated slightly differently
        input_keys = ["num_counters", *self.config["input_dims"].keys()]
        output_keys = ["num_counters", *self.config["output_dims"].keys()]
        if self.config["dataset_name"] != "imap_codice_l1a_lo-counters-aggregated":
            transpose_axes = [input_keys.index(dim) for dim in output_keys]
        else:
            transpose_axes = [1, 2, 0]  # [esa_step, spin_sector_pairs, num_counters]

        for packet_data in self.raw_data:
            reshaped_packet_data = np.array(packet_data, dtype=np.uint32).reshape(
                reshape_dims
            )
            reshaped_cdf_data = np.transpose(reshaped_packet_data, axes=transpose_axes)

            self.data.append(reshaped_cdf_data)

        # No longer need to keep the raw data around
        del self.raw_data

    def set_data_product_config(self, apid: int, dataset: xr.Dataset) -> None:
        """
        Set the various settings for defining the data products.

        Parameters
        ----------
        apid : int
            The APID of interest.
        dataset : xarray.Dataset
            The dataset for the APID of interest.
        """
        # Set the packet dataset so that it can be easily called from various
        # methods
        self.dataset = dataset

        # Set various configurations of the data product
        self.config: dict[str, Any] = constants.DATA_PRODUCT_CONFIGURATIONS[apid]

        # Gather and set the CDF attributes
        self.cdf_attrs = ImapCdfAttributes()
        self.cdf_attrs.add_instrument_global_attrs("codice")
        self.cdf_attrs.add_instrument_variable_attrs("codice", "l1a")


def calculate_epoch_values(
    acq_start_seconds: xr.DataArray, acq_start_subseconds: xr.DataArray
) -> tuple[NDArray[int], NDArray[int], NDArray[int]]:
    """
    Calculate and return the values to be used for `epoch`.

    On CoDICE, the epoch values are derived from the `acq_start_seconds` and
    `acq_start_subseconds` fields in the packet.

    Note that the `acq_start_subseconds` field needs to be converted from
    microseconds to seconds.

    Parameters
    ----------
    acq_start_seconds : xarray.DataArray
        The acquisition times to calculate the epoch values from.
    acq_start_subseconds : xarray.DataArray
        The subseconds portion of the acquisition times.

    Returns
    -------
    epoch : NDArray[int]
        List of centered epoch values.
    epoch_delta_minus: NDArray[int]
        List of values that represent the length of time from acquisition
        start to the center of the acquisition time bin.
    epoch_delta_plus: NDArray[int]
        List of values that represent the length of time from the center of
        the acquisition time bin to the end of acquisition.
    """
    # First calculate an epoch value based on the acquisition start
    acq_start = met_to_ttj2000ns(acq_start_seconds + acq_start_subseconds / 1e6)

    # Apply correction to center the epoch bin
    epoch = (acq_start[:-1] + acq_start[1:]) // 2
    epoch_delta_minus = epoch - acq_start[:-1]
    epoch_delta_plus = acq_start[1:] - epoch

    # Since the centers and deltas are determined by averaging sequential bins,
    # the last elements must be calculated differently. For this, we just use
    # the last acquisition start and the previous deltas
    epoch = np.concatenate([epoch, [acq_start[-1]]])
    epoch_delta_minus = np.concatenate([epoch_delta_minus, [epoch_delta_minus[-1]]])
    epoch_delta_plus = np.concatenate([epoch_delta_plus, [epoch_delta_plus[-1]]])

    return epoch, epoch_delta_minus, epoch_delta_plus


def group_ialirt_data(
    packets: xr.Dataset, data_field_range: range, prefix: str
) -> list[bytearray]:
    """
    Group together the individual I-ALiRT data fields.

    Parameters
    ----------
    packets : xarray.Dataset
        The dataset containing the I-ALiRT data packets.
    data_field_range : range
        The range of the individual data fields (15 or lo, 6 for hi).
    prefix : str
        The prefix used to index the data (i.e. ``cod_lo`` or ``cod_hi``).

    Returns
    -------
    grouped_data : list[bytearray]
        The list of grouped I-ALiRT data.
    """
    current_data_stream = bytearray()
    grouped_data = []

    # Workaround to get this function working for both I-ALiRT spacecraft
    # data and CoDICE-specific I-ALiRT test data from Joey
    # TODO: Once CoDICE I-ALiRT processing is more established, we can probably
    #       do away with processing the test data from Joey and just use the
    #       I-ALiRT data that is constructed closer to what we expect in-flight.
    if hasattr(packets, "acquisition_time"):
        time_key = "acquisition_time"
        counter_key = "counter"
        data_key = "data"
    else:
        time_key = f"{prefix}_acq"
        counter_key = f"{prefix}_counter"
        data_key = f"{prefix}_data"

    # When a counter value of 255 is encountered, this signifies the
    # end of the data stream
    for packet_num in range(0, len(packets[time_key].data)):
        counter = packets[counter_key].data[packet_num]
        if counter != 255:
            for field in data_field_range:
                current_data_stream.extend(
                    bytearray([packets[f"{data_key}_{field:02}"].data[packet_num]])
                )
        else:
            # At this point, if there are data, the data stream is ready
            # to be processed like an SW Species product (for lo) or an
            # Omni Species product (for hi)
            if len(current_data_stream) > 0:
                grouped_data.append(current_data_stream)
            current_data_stream = bytearray()

    return grouped_data


def create_binned_dataset(
    apid: int, dataset: xr.Dataset, science_values: list[str]
) -> xr.Dataset:
    """
    Create dataset for data that is binned by energy.

    This applies to the ``hi-omni`` and ``hi-sectored`` datasets. In addition to
    data for species (e.g. ``h``, ``c``, ``o``, etc.), we add CDF variables
    for their respective energy bin centers and deltas (e.g. ``energy_h``,
    ``energy_h_delta``, etc.)

    Parameters
    ----------
    apid : int
        The APID of the packet.
    dataset : xarray.Dataset
        The packets to process.
    science_values : list[str]
        The values of the "data" field of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the final processed dataset.
    """
    # TODO: hi-sectored data product should be processed similar to hi-omni,
    #       so I should be able to use this method.

    # Get the four "main" parameters for processing
    table_id, plan_id, plan_step, view_id = get_params(dataset)

    # Run some of the pipeline methods to set configs and decompress
    # the data
    pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
    pipeline.set_data_product_config(apid, dataset)
    pipeline.decompress_data(science_values)

    data = pipeline.reshape_binned_data(dataset)

    # Create the main dataset to hold all the variables
    coord = xr.DataArray(
        np.array(data["epoch"], dtype=np.uint64),
        name="epoch",
        dims=["epoch"],
        attrs=pipeline.cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    # TODO: Figure out how to calculate epoch centers and deltas and store them
    #       in variables here
    dataset = xr.Dataset(
        coords={"epoch": coord},
        attrs=pipeline.cdf_attrs.get_global_attributes(pipeline.config["dataset_name"]),
    )

    # Add the data variables
    descriptor = pipeline.config["dataset_name"].removeprefix("imap_codice_l1a_")
    for species in pipeline.config["energy_table"]:
        # Add the species data to the dataset
        values = np.array(data[species], dtype=np.uint32)
        attrs = pipeline.cdf_attrs.get_variable_attributes(f"{descriptor}-{species}")
        dims = ["epoch", f"energy_{species}"]
        dataset[species] = xr.DataArray(
            values,
            name=species,
            dims=dims,
            attrs=attrs,
        )

        # Add the energy bins to the dataset
        dataset = pipeline.define_energy_bins(dataset, species)

    # Add support variables to the dataset
    dataset["spin_period"] = xr.DataArray(
        np.array(data["spin_period"]),
        name="spin_period",
        dims=["epoch"],
        attrs=pipeline.cdf_attrs.get_variable_attributes("spin_period"),
    )
    dataset["data_quality"] = xr.DataArray(
        np.array(data["data_quality"]),
        name="data_quality",
        dims=["epoch"],
        attrs=pipeline.cdf_attrs.get_variable_attributes("data_quality"),
    )

    return dataset


def create_direct_event_dataset(apid: int, packets: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for direct event data.

    For direct event data, the raw data from the spacecraft is organized first
    by epoch, then by priority, then by events. For example, for a CoDICE-Lo
    dataset with 10 epochs, we expect the length of the `event_data` field to
    be (10 epochs * 8 priorities) = 80 items, with each item being a compressed
    byte object representing a variable number of events (up to 10000 events).
    Each compressed byte object is comprised of several fields with specific
    bit lengths/positions, described by the constants.[LO|HI]_DE_BIT_STRUCTURE
    dictionary. Padding is added to any fields that have less than 10000 events.

    In order to process these data, we must take the decommed raw data, group
    the packets appropriately based on their `seq_flgs`, decompress the data,
    then arrange the data into CDF data variables for each priority and bit
    field. For example, P2_SpinAngle represents the spin angles for the 2nd
    priority data.

    Parameters
    ----------
    apid : int
        The APID of the packet.
    packets : xarray.Dataset
        The packets to process.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the direct event data.
    """
    # Group and decompress the data
    grouped_data = group_data(packets)
    decompressed_data = [
        decompress(group, CoDICECompression.LOSSLESS) for group in grouped_data
    ]

    # Reshape the packet data into CDF-ready variables
    data = reshape_de_data(packets, decompressed_data, apid)

    # Gather the CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    # Determine the epochs to use in the dataset, which are the epochs whenever
    # there is a start of a segment and the priority is 0
    epoch_indices = np.where(
        ((packets.seq_flgs.data == 3) | (packets.seq_flgs.data == 1))
        & (packets.priority.data == 0)
    )[0]
    acq_start_seconds = packets.acq_start_seconds[epoch_indices]
    acq_start_subseconds = packets.acq_start_subseconds[epoch_indices]

    # Calculate epoch variables
    epochs, epochs_delta_minus, epochs_delta_plus = calculate_epoch_values(
        acq_start_seconds, acq_start_subseconds
    )

    # Define coordinates
    epoch = xr.DataArray(
        epochs,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    epoch_delta_minus = xr.DataArray(
        epochs_delta_minus,
        name="epoch_delta_minus",
        dims=["epoch_delta_minus"],
        attrs=cdf_attrs.get_variable_attributes(
            "epoch_delta_minus", check_schema=False
        ),
    )
    epoch_delta_plus = xr.DataArray(
        epochs_delta_plus,
        name="epoch_delta_plus",
        dims=["epoch_delta_plus"],
        attrs=cdf_attrs.get_variable_attributes("epoch_delta_plus", check_schema=False),
    )
    event_num = xr.DataArray(
        np.arange(10000),
        name="event_num",
        dims=["event_num"],
        attrs=cdf_attrs.get_variable_attributes("event_num", check_schema=False),
    )
    event_num_label = xr.DataArray(
        np.arange(10000).astype(str),
        name="event_num_label",
        dims=["event_num"],
        attrs=cdf_attrs.get_variable_attributes("event_num_label", check_schema=False),
    )

    # Create the dataset to hold the data variables
    if apid == CODICEAPID.COD_LO_PHA:
        attrs = cdf_attrs.get_global_attributes("imap_codice_l1a_lo-pha")
    elif apid == CODICEAPID.COD_HI_PHA:
        attrs = cdf_attrs.get_global_attributes("imap_codice_l1a_hi-pha")
    dataset = xr.Dataset(
        coords={
            "epoch": epoch,
            "epoch_delta_minus": epoch_delta_minus,
            "epoch_delta_plus": epoch_delta_plus,
            "event_num": event_num,
            "event_num_label": event_num_label,
        },
        attrs=attrs,
    )

    # Create the CDF data variables for each Priority and Field
    for i in range(constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["num_priorities"]):
        for field in constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["cdf_fields"]:
            variable_name = f"P{i}_{field}"
            attrs = cdf_attrs.get_variable_attributes(variable_name)
            if field in ["NumEvents", "DataQuality"]:
                dims = ["epoch"]
            else:
                dims = ["epoch", "event_num"]
            dataset[variable_name] = xr.DataArray(
                np.array(data[variable_name]),
                name=variable_name,
                dims=dims,
                attrs=attrs,
            )

    return dataset


def create_hskp_dataset(packet: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packet : xarray.Dataset
        The packet to process.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the metadata.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    epoch = xr.DataArray(
        packet.epoch,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"),
    )

    # These variables don't need to carry over from L0 to L1a
    exclude_variables = [
        "spare_1",
        "spare_2",
        "spare_3",
        "spare_4",
        "spare_5",
        "spare_6",
        "spare_62",
        "spare_68",
        "chksum",
    ]

    for variable in packet:
        if variable in exclude_variables:
            continue

        # The housekeeping spin_period variable has different values than
        # the spin_value attribute in other datasets, so it gets special
        # treatment
        if variable == "spin_period":
            attrs = cdf_attrs.get_variable_attributes("spin_period_hskp")
        else:
            attrs = cdf_attrs.get_variable_attributes(variable)

        dataset[variable] = xr.DataArray(
            packet[variable].data, dims=["epoch"], attrs=attrs
        )

    return dataset


def create_ialirt_dataset(apid: int, packets: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for lo- and hi-ialirt data.

    I-ALiRT data are packed identically to regular science data
    (``lo-sw-species`` for CoDICE-lo, and ``hi-omni`` for CoDICE-hi), except
    for some slight differences in the metadata that are transmitted.
    Additionally, data are transmitted in separate, individual single-byte
    fields (there are 15 of these for CoDICE-lo and 6 for CoDICE-hi).

    This function will process these I-ALiRT data while using some of the same
    code used for processing the ``lo-sw-species`` and ``hi-omni`` L1a data
    products.

    Parameters
    ----------
    apid : int
        The APID of the packet.
    packets : xarray.Dataset
        The packets to process.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the direct event data.

    References
    ----------
    See section 9.4 of the CoDICE algorithm document for further details.
    """
    # I-ALiRT packet data gets split up into multiple data fields,
    # specific to lo- and hi-
    # See sections 10.4.1 and 10.4.2 in the algorithm document
    if apid == CODICEAPID.COD_LO_IAL:
        data_field_range = range(0, 15)
        prefix = "cod_lo"
    elif apid == CODICEAPID.COD_HI_IAL:
        data_field_range = range(0, 5)
        prefix = "cod_hi"

    # Group together packets of I-ALiRT data to form complete data sets
    grouped_data = group_ialirt_data(packets, data_field_range, prefix)

    if grouped_data:
        # Process each group to get the science data and corresponding metadata
        science_values, metadata_values = process_ialirt_data_streams(grouped_data)

        # How data are processed is different for lo-iarlirt and hi-ialirt
        if apid == CODICEAPID.COD_HI_IAL:
            # Set some necessary values and process as a binned dataset similar to
            # a hi-omni data product
            metadata_for_processing = [
                "table_id",
                "plan_id",
                "plan_step",
                "view_id",
                "spin_period",
                "suspect",
            ]
            for var in metadata_for_processing:
                packets[var] = metadata_values[var.upper()]
            dataset = create_binned_dataset(apid, packets, science_values)

        elif apid == CODICEAPID.COD_LO_IAL:
            # Create a nominal instance of the pipeline and process similar to a
            # lo-sw-species data product
            pipeline = CoDICEL1aPipeline(
                metadata_values["TABLE_ID"][0],
                metadata_values["PLAN_ID"][0],
                metadata_values["PLAN_STEP"][0],
                metadata_values["VIEW_ID"][0],
            )
            pipeline.set_data_product_config(apid, packets)
            pipeline.decompress_data(science_values)
            pipeline.reshape_data()

            # The calculate_epoch_values method needs acq_start_seconds and
            # acq_start_subseconds attributes on the dataset
            pipeline.dataset["acq_start_seconds"] = (
                "_",
                metadata_values["ACQ_START_SECONDS"],
            )
            pipeline.dataset["acq_start_subseconds"] = (
                "_",
                metadata_values["ACQ_START_SUBSECONDS"],
            )

            pipeline.define_coordinates()

            # The dataset also needs the metadata that will be carried through
            # to the final data product
            for field in [
                "spin_period",
                "suspect",
                "st_bias_gain_mode",
                "sw_bias_gain_mode",
                "rgfo_half_spin",
                "nso_half_spin",
            ]:
                pipeline.dataset[field] = ("_", metadata_values[field.upper()])

            dataset = pipeline.define_data_variables()

        return dataset

    else:
        logger.warning("No I-ALiRT data found")
        return None


def get_de_metadata(packets: xr.Dataset, segment: int) -> bytes:
    """
    Gather and return packet metadata (From packet_version through byte_count).

    Extract the metadata in the segmented direct event packet, which is then
    used to construct the full data of the group of segments.

    Parameters
    ----------
    packets : xarray.Dataset
        The segmented direct event packet data.
    segment : int
        The index of the segment of interest.

    Returns
    -------
    metadata : bytes
        The compressed metadata for the segmented packet.
    """
    # String together the metadata fields and convert the data to a bytes obj
    metadata_str = ""
    for field, num_bits in constants.DE_METADATA_FIELDS.items():
        metadata_str += f"{packets[field].data[segment]:0{num_bits}b}"
    metadata_chunks = [metadata_str[i : i + 8] for i in range(0, len(metadata_str), 8)]
    metadata_ints = [int(item, 2) for item in metadata_chunks]
    metadata = bytes(metadata_ints)

    return metadata


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


def group_data(packets: xr.Dataset) -> list[bytes]:
    """
    Organize continuation packets into appropriate groups.

    Some packets are continuation packets, as in, they are packets that are
    part of a group of packets. These packets are marked by the `seq_flgs` field
    in the CCSDS header of the packet. For CoDICE, the values are defined as
    follows:

    3 = Packet is not part of a group
    1 = Packet is the first packet of the group
    0 = Packet is in the middle of the group
    2 = Packet is the last packet of the group

    For packets that are part of a group, the byte count associated with the
    first packet of the group signifies the byte count for the entire group.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets to group.

    Returns
    -------
    grouped_data : list[bytes]
        The packet data, converted to bytes and grouped appropriately.
    """
    grouped_data = []  # Holds the properly grouped data to be decompressed
    current_group = bytearray()  # Temporary storage for current group
    group_byte_count = None  # Temporary storage for current group byte count

    for segment in range(len(packets.event_data.data)):
        packet_data = packets.event_data.data[segment]
        group_code = packets.seq_flgs.data[segment]
        byte_count = packets.byte_count.data[segment]

        # If the group code is 3, this means the data is not part of a group
        # and can be decompressed as-is
        if group_code == 3:
            grouped_data.append(packet_data[:byte_count])

        # If the group code is 1, this means the data is the first data in a
        # group. Also, set the byte count for the group
        elif group_code == 1:
            group_byte_count = byte_count
            current_group += packet_data

        # If the group code is 0, this means the data is part of the middle of
        # the group.
        elif group_code == 0:
            current_group += get_de_metadata(packets, segment)
            current_group += packet_data

        # If the group code is 2, this means the data is the last data in the
        # group
        elif group_code == 2:
            current_group += get_de_metadata(packets, segment)
            current_group += packet_data

            # The grouped data is now ready to be decompressed
            values_to_decompress = current_group[:group_byte_count]
            grouped_data.append(values_to_decompress)

            # Reset the current group
            current_group = bytearray()
            group_byte_count = None

    return grouped_data


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
    for apid, ds in datasets.items():
        num_packets = len(ds.epoch.data)
        time_deltas = [np.timedelta64(item, "ns") for item in ds.epoch.data]
        times = [launch_time + delta for delta in time_deltas]
        start = np.datetime_as_string(times[0])
        end = np.datetime_as_string(times[-1])
        logger.info(
            f"{CODICEAPID(apid).name}: {num_packets} packets spanning {start} to {end}"
        )


def process_ialirt_data_streams(
    grouped_data: list[bytearray],
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Process each I-ALiRT science data stream to extract individual data fields.

    Each data stream is converted to binary so that each metadata and science
    data field and their values can be separated out. These fields and values
    eventually will be stored in CDF data/support variables.

    Parameters
    ----------
    grouped_data : list[bytearray]
        A list of grouped I-ALiRT data.

    Returns
    -------
    science_values : list[str]
        The science values / data array portion of the I-ALiRT data in the form
        of a binary string.
    metadata_values : dict[str, list[int]]
        The extracted metadata fields and their values.
    """
    # Initialize placeholders for the processed data
    science_values = []
    metadata_values: dict[str, list[int]] = {}
    for field in constants.IAL_BIT_STRUCTURE:
        metadata_values[field] = []

    # Process each complete data stream
    for data_stream in grouped_data:
        try:
            # Convert the data to binary
            bit_string = "".join(f"{byte:08b}" for byte in data_stream)

            # Separate the data into its individual fields
            bit_position = 0
            for field in constants.IAL_BIT_STRUCTURE:
                # Convert from binary to integer
                value = int(
                    bit_string[
                        bit_position : bit_position + constants.IAL_BIT_STRUCTURE[field]
                    ],
                    2,
                )

                # If we encounter an SHCOARSE of 0, the packet is bad
                if field == "SHCOARSE" and value == 0:
                    raise ValueError("Bad packet encountered")

                metadata_values[field].append(value)
                bit_position += constants.IAL_BIT_STRUCTURE[field]
                if field == "BYTE_COUNT":
                    byte_count = value * 8  # Convert from bytes to number of bits

            # The rest is the data field, up to the byte count
            data_field = bit_string[bit_position : bit_position + byte_count]
            science_values.append(data_field)
        except ValueError:
            pass

    return science_values, metadata_values


def reshape_de_data(
    packets: xr.Dataset, decompressed_data: list[list[int]], apid: int
) -> dict[str, np.ndarray]:
    """
    Reshape the decompressed direct event data into CDF-ready arrays.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets, needed to determine priority order
        and data quality.
    decompressed_data : list[list[int]]
        The decompressed data to reshape, in the format <epoch>[<priority>[<event>]].
    apid : int
        The APID of the packet, used primarily to determine if the data are from
        CoDICE-Lo or CoDICE-Hi.

    Returns
    -------
    data : dict[str, numpy.ndarray]
        The reshaped, CDF-ready arrays. The keys of the dictionary represent the
        CDF variable names, and the values represent the data.
    """
    # Dictionary to hold all the (soon to be restructured) direct event data
    data: dict[str, np.ndarray] = {}

    # Extract some useful variables
    num_priorities = constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["num_priorities"]
    bit_structure = constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]["bit_structure"]

    # Determine the number of epochs to help with data array initialization
    # There is one epoch per set of priorities
    num_epochs = len(decompressed_data) // num_priorities

    # Get num_events, data quality, and priorities data for beginning of segments
    segment_starts = np.where(
        (packets.seq_flgs.data == 3) | (packets.seq_flgs.data == 1)
    )[0]
    num_events_arr = packets.num_events.data[segment_starts]
    data_quality_arr = packets.suspect.data[segment_starts]
    priorities_arr = packets.priority.data[segment_starts]

    # Initialize data arrays for each priority and field to store the data
    # We also need arrays to hold number of events and data quality
    for priority_num in range(num_priorities):
        for field in bit_structure:
            if field not in ["Priority", "Spare"]:
                data[f"P{priority_num}_{field}"] = np.full(
                    (num_epochs, 10000),
                    bit_structure[field]["fillval"],
                    dtype=bit_structure[field]["dtype"],
                )
        data[f"P{priority_num}_NumEvents"] = np.full(num_epochs, 65535, dtype=np.uint16)
        data[f"P{priority_num}_DataQuality"] = np.full(num_epochs, 255, dtype=np.uint8)

    # decompressed_data is one large list of values of length
    # (<number of epochs> * <number of priorities>)
    # Chunk the data into each epoch
    for epoch_index in range(num_epochs):
        # Determine the starting and ending indices of the epoch
        epoch_start = epoch_index * num_priorities
        epoch_end = epoch_start + num_priorities

        # Extract the data for the epoch
        epoch_data = decompressed_data[epoch_start:epoch_end]

        # The order of the priorities and data quality flags are unique to each
        # epoch and can be gathered from the packet data
        priority_order = priorities_arr[epoch_start:epoch_end]
        data_quality = data_quality_arr[epoch_start:epoch_end]

        # For each epoch/priority combo, iterate over each event
        for i, priority_num in enumerate(priority_order):
            priority_data = epoch_data[i]

            # Number of events and data quality can be determined at this stage
            num_events = num_events_arr[epoch_start:epoch_end][i]
            data[f"P{priority_num}_NumEvents"][epoch_index] = num_events
            data[f"P{priority_num}_DataQuality"][epoch_index] = data_quality[i]

            # Iterate over each event
            for event_index in range(num_events):
                event_start = event_index * 8  # The 8 is for 8 bytes
                event_end = event_start + 8
                event = priority_data[event_start:event_end]

                # Separate out each individual field from the bit string
                # The fields are packed into the bit string in reverse order, so
                # we need to back them out in reverse order
                bit_string = (
                    f"{int.from_bytes(event, byteorder='big'):0{len(event) * 8}b}"
                )

                bit_position = 0
                for field_name, field_components in reversed(bit_structure.items()):
                    # We don't need to carry Priority and Spare fields through
                    if field_name in ["Priority", "Spare"]:
                        bit_position += field_components["bit_length"]
                        continue

                    # Convert from binary to integer
                    value = int(
                        bit_string[
                            bit_position : bit_position + field_components["bit_length"]
                        ],
                        2,
                    )

                    # Set the value into the data array
                    data[f"P{priority_num}_{field_name}"][epoch_index, event_index] = (
                        value
                    )
                    bit_position += field_components["bit_length"]

    return data


def process_codice_l1a(file_path: Path) -> list[xr.Dataset]:
    """
    Will process CoDICE l0 data to create l1a data products.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the CoDICE L0 file to process.

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
            processed_dataset = create_hskp_dataset(dataset)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Event data
        elif apid in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            processed_dataset = create_direct_event_dataset(apid, dataset)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # I-ALiRT data
        elif apid in [CODICEAPID.COD_LO_IAL, CODICEAPID.COD_HI_IAL]:
            processed_dataset = create_ialirt_dataset(apid, dataset)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # hi-omni data
        elif apid == CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS:
            science_values = [packet.data for packet in dataset.data]
            processed_dataset = create_binned_dataset(apid, dataset, science_values)
            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # Everything else
        elif apid in constants.APIDS_FOR_SCIENCE_PROCESSING:
            # Extract the data
            science_values = [packet.data for packet in dataset.data]

            # Get the four "main" parameters for processing
            table_id, plan_id, plan_step, view_id = get_params(dataset)

            # Run the pipeline to create a dataset for the product
            pipeline = CoDICEL1aPipeline(table_id, plan_id, plan_step, view_id)
            pipeline.set_data_product_config(apid, dataset)
            pipeline.decompress_data(science_values)
            pipeline.reshape_data()
            pipeline.define_coordinates()
            processed_dataset = pipeline.define_data_variables()

            logger.info(f"\nFinal data product:\n{processed_dataset}\n")

        # For APIDs that don't require processing
        else:
            logger.info(f"\t{apid} does not require processing")
            continue

        processed_datasets.append(processed_dataset)

    return processed_datasets
