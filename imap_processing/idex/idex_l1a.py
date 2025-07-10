"""
Perform IDEX L1a Processing.

This module processes decommutated IDEX packets and creates L1a data products.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1a_data.write_l1a_cdf()
"""

import logging
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import space_packet_parser
import xarray as xr
from xarray import Dataset

from imap_processing.idex.decode import rice_decode
from imap_processing.idex.idex_constants import IDEXAPID
from imap_processing.idex.idex_l0 import decom_packets
from imap_processing.idex.idex_utils import get_idex_attrs
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.utils import convert_to_binary_string

logger = logging.getLogger(__name__)


class Scitype(IntEnum):
    """Define parameters for IDEX Science Type."""

    FIRST_PACKET = 1
    TOF_HIGH = 2
    TOF_LOW = 4
    TOF_MID = 8
    TARGET_LOW = 16
    TARGET_HIGH = 32
    ION_GRID = 64


class PacketParser:
    """
    IDEX L1a packet parsing class.

    Encapsulates the decom work needed to decom a daily file of IDEX L0 data
    received from the POC. The class is instantiated with a reference to a L0
    file as it exists on the local file system.

    Parameters
    ----------
    packet_file : str
        The path and filename to the L0 file to read.
    """

    def __init__(self, packet_file: Union[str, Path]) -> None:
        """
        Read a L0 pkts file and perform all of the decom work.

        Parameters
        ----------
        packet_file : pathlib.Path | str
          The path and filename to the L0 file to read.

        Notes
        -----
            Currently assumes one L0 file will generate exactly one L1a file.
        """
        self.data = []
        self.idex_attrs = get_idex_attrs("l1a")
        epoch_attrs = self.idex_attrs.get_variable_attributes(
            "epoch", check_schema=False
        )

        science_packets, raw_datset_by_apid, derived_datasets_by_apid = decom_packets(
            packet_file
        )

        if science_packets:
            logger.info("Processing IDEX L1A Science data.")
            self.data.append(self._create_science_dataset(science_packets))

        datasets_by_level = {"l1a": raw_datset_by_apid, "l1b": derived_datasets_by_apid}
        for level, dataset in datasets_by_level.items():
            if IDEXAPID.IDEX_EVT in dataset:
                logger.info(f"Processing IDEX {level} Event Message data")
                data = dataset[IDEXAPID.IDEX_EVT]
                data.attrs = self.idex_attrs.get_global_attributes(
                    f"imap_idex_{level}_evt"
                )
                data["epoch"] = calculate_idex_epoch_time(
                    data["shcoarse"], data["shfine"]
                )
                data["epoch"].attrs = epoch_attrs
                self.data.append(data)

            if IDEXAPID.IDEX_CATLST in dataset:
                logger.info(f"Processing IDEX {level} Catalog List Summary data.")
                data = dataset[IDEXAPID.IDEX_CATLST]
                data.attrs = self.idex_attrs.get_global_attributes(
                    f"imap_idex_{level}_catlst"
                )
                data["epoch"] = calculate_idex_epoch_time(
                    data["shcoarse"], data["shfine"]
                )
                data["epoch"].attrs = epoch_attrs
                self.data.append(data)

        logger.info("IDEX L1A data processing completed.")

    def _create_science_dataset(self, science_decom_packet_list: list) -> xr.Dataset:
        """
        Process IDEX science packets into an xarray Dataset.

        Parameters
        ----------
        science_decom_packet_list : list
            List of decommutated science packets.

        Returns
        -------
        xarray.Dataset
            Dataset containing processed dust events.
        """
        dust_events = {}
        for packet in science_decom_packet_list:
            if "IDX__SCI0TYPE" in packet:
                scitype = packet["IDX__SCI0TYPE"]
                event_number = packet["IDX__SCI0EVTNUM"]
                if scitype == Scitype.FIRST_PACKET:
                    # Initial packet for new dust event
                    # Further packets will fill in data
                    dust_events[event_number] = RawDustEvent(packet)
                elif event_number not in dust_events:
                    raise KeyError(
                        f"Have not receive header information from event number\
                            {event_number}.  Packets are possibly out of order!"
                    )
                else:
                    # Populate the IDEXRawDustEvent with 1's and 0's
                    dust_events[event_number]._populate_bit_strings(packet)
            else:
                logger.warning(f"Unhandled packet received: {packet}")

        processed_dust_impact_list = [
            dust_event.process() for dust_event in dust_events.values()
        ]
        processed_dust_impact_list = [
            x for x in processed_dust_impact_list if x is not None
        ]
        data = xr.concat(processed_dust_impact_list, dim="epoch")
        data.attrs = self.idex_attrs.get_global_attributes("imap_idex_l1a_sci")
        data = data.sortby("epoch")
        # Add high and low sample rate coords
        data["time_low_sample_rate_index"] = xr.DataArray(
            np.arange(len(data["time_low_sample_rate"][0])),
            name="time_low_sample_rate_index",
            dims=["time_low_sample_rate_index"],
            attrs=self.idex_attrs.get_variable_attributes(
                "time_low_sample_rate_index", check_schema=False
            ),
        )

        data["time_high_sample_rate_index"] = xr.DataArray(
            np.arange(len(data["time_high_sample_rate"][0])),
            name="time_high_sample_rate_index",
            dims=["time_high_sample_rate_index"],
            attrs=self.idex_attrs.get_variable_attributes(
                "time_high_sample_rate_index", check_schema=False
            ),
        )
        # NOTE: LABL_PTR_1 should be CDF_CHAR.
        data["time_low_sample_rate_label"] = xr.DataArray(
            data.time_low_sample_rate_index.values.astype(str),
            name="time_low_sample_rate_label",
            dims=["time_low_sample_rate_index"],
            attrs=self.idex_attrs.get_variable_attributes(
                "time_low_sample_rate_label", check_schema=False
            ),
        )

        data["time_high_sample_rate_label"] = xr.DataArray(
            data.time_high_sample_rate_index.values.astype(str),
            name="time_high_sample_rate_label",
            dims=["time_high_sample_rate_index"],
            attrs=self.idex_attrs.get_variable_attributes(
                "time_high_sample_rate_label", check_schema=False
            ),
        )

        return data


def _read_waveform_bits(waveform_raw: str, high_sample: bool = True) -> list[int]:
    """
    Convert the raw waveform binary string to ints.

    Parse a binary string representing a waveform.
    If the data is a high sample waveform:
        - Data arrives in 32-bit chunks, divided up into:
            * 2 bits of padding
            * 3x10 bits of integer data.
        - The very last four numbers are usually bad, so remove those.
    If the data is a low sample waveform:
        - Data arrives in 32-bit chunks, divided up into:
            * 8 bits of padding
            * 2x12 bits of integer data.

    Parameters
    ----------
    waveform_raw : str
        The binary string representing the waveform.
    high_sample : bool
        If true, parse the waveform according to the high sample pattern,
        otherwise use the low sample pattern.

    Returns
    -------
    ints : list[int]
        List of the waveform.
    """
    ints: list[int] = []
    if high_sample:
        for i in range(0, len(waveform_raw), 32):
            # 32-bit chunks, divided up into 2, 10, 10, 10
            # skip first two bits
            ints += [
                int(waveform_raw[i + 2 : i + 12], 2),
                int(waveform_raw[i + 12 : i + 22], 2),
                int(waveform_raw[i + 22 : i + 32], 2),
            ]
        ints = ints[:-4]  # Remove last 4 numbers
    else:
        for i in range(0, len(waveform_raw), 32):
            # 32-bit chunks, divided up into 8, 12, 12
            # skip first eight bits
            ints += [
                int(waveform_raw[i + 8 : i + 20], 2),
                int(waveform_raw[i + 20 : i + 32], 2),
            ]
    return ints


def calculate_idex_epoch_time(
    shcoarse_time: Union[float, np.ndarray], shfine_time: Union[float, np.ndarray]
) -> npt.NDArray[np.int64]:
    """
    Calculate the epoch time from the FPGA header time variables.

    We are given the MET seconds, we need to convert it to nanoseconds in j2000. IDEX
    epoch is calculated with shcoarse and shfine time values. The shcoarse time counts
    the number of whole seconds elapsed since the epoch (Jan 1st 2010), while shfine
    time counts the number of additional 20-microsecond intervals beyond the whole
    seconds. Together, these time measurements establish when a dust event took place.

    Parameters
    ----------
    shcoarse_time : float, numpy.ndarray
        The coarse time value from the FPGA header. Number of seconds since epoch.
    shfine_time : float, numpy.ndarray
        The fine time value from the FPGA header. Number of 20 microsecond "ticks" since
         the last second.

    Returns
    -------
    numpy.ndarray[numpy.int64]
        The mission elapsed time converted to nanoseconds since the J2000 epoch
        in the terrestrial time (TT) timescale.
    """
    # Get met time in seconds including shfine (number of 20 microsecond ticks)
    met = shcoarse_time + shfine_time * 20e-6
    return met_to_ttj2000ns(met)


class RawDustEvent:
    """
    Encapsulate IDEX Raw Dust Event.

    Encapsulates the work needed to convert a single dust event into a processed
    ``xarray`` ``dateset`` object.

    Parameters
    ----------
    header_packet : space_packet_parser.packets.CCSDSPacket
        The FPGA metadata event header.

    Attributes
    ----------
    HIGH_SAMPLE_RATE: float
        The high sample rate in microseconds per sample.
    LOW_SAMPLE_RATE: float
        The low sample rate in microseconds per sample.
    NUMBER_SAMPLES_PER_LOW_SAMPLE_BLOCK: int
        The number of samples in a "block" of low sample data.
    NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK: int
        The number of samples in a "block" of high sample data.
    MAX_HIGH_BLOCKS: int
        The maximum number of "blocks" for high sample data.
    MAX_LOW_BLOCKS: int
        The maximum number of "blocks" for low sample data.

    Methods
    -------
    _append_raw_data(scitype, bits)
        Append data to the appropriate bit string.
    _set_sample_trigger_times(packet)
        Calculate the actual sample trigger time.
    _parse_high_sample_waveform(waveform_raw)
        Will process the high sample waveform.
    _parse_low_sample_waveform(waveform_raw)
        Will process the low sample waveform.
    _calc_low_sample_resolution(num_samples)
        Calculate the resolution of the low samples.
    _calc_high_sample_resolution(num_samples)
        Calculate the resolution of high samples.
    _populate_bit_strings(packet)
        Parse IDEX data packets to populate bit strings.
    process()
        Will process the raw data into a xarray.Dataset.
    """

    # Constants
    HIGH_SAMPLE_RATE = 1 / 260  # microseconds per sample
    LOW_SAMPLE_RATE = 1 / 4.0625  # microseconds per sample

    NUMBER_SAMPLES_PER_LOW_SAMPLE_BLOCK = (
        8  # The number of samples in a "block" of low sample data
    )
    NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK = (
        512  # The number of samples in a "block" of high sample data
    )
    # Maximum amount of data
    MAX_HIGH_BLOCKS = 16
    MAX_LOW_BLOCKS = 64

    def __init__(self, header_packet: space_packet_parser.packets.CCSDSPacket) -> None:
        """
        Initialize a raw dust event, with an FPGA Header Packet from IDEX.

        The values we care about are:

        self.impact_time - When the impact occurred.
        self.low_sample_trigger_time - When the low sample stuff actually triggered.
        self.high_sample_trigger_time - When the high sample stuff actually triggered.

        Parameters
        ----------
        header_packet : space_packet_parser.packets.CCSDSPacket
            The FPGA metadata event header.
        """
        # Calculate the impact time in seconds since epoch
        self.impact_time = 0
        self.impact_time = calculate_idex_epoch_time(
            header_packet["SHCOARSE"], header_packet["SHFINE"]
        )
        self.event_number = header_packet["IDX__SCI0EVTNUM"]

        # The actual trigger time for the low and high sample rate in
        # microseconds since the impact time
        self.low_sample_trigger_time = 0
        self.high_sample_trigger_time = 0
        self._set_sample_trigger_times(header_packet)

        # Iterate through every telemetry item not in the header and pull out the values
        self.telemetry_items = {
            key.lower(): val
            for key, val in header_packet.items()
            if key not in header_packet.header.keys()
        }

        logger.debug(
            f"telemetry_items:\n{self.telemetry_items}"
        )  # Log values here in case of error

        # Initialize the binary data received from future packets
        self.TOF_High_bits = ""
        self.TOF_Mid_bits = ""
        self.TOF_Low_bits = ""
        self.Target_Low_bits = ""
        self.Target_High_bits = ""
        self.Ion_Grid_bits = ""

        self.compressed = self.telemetry_items["idx__sci0comp"]
        self.cdf_attrs = get_idex_attrs("l1a")

    def _append_raw_data(self, scitype: Scitype, bits: str) -> None:
        """
        Append data to the appropriate bit string.

        This function determines which variable to append the bits to, given a
        specific scitype.

        Parameters
        ----------
        scitype : Scitype
            The science type of the data.
        bits : str
            The binary data to append.
        """
        if scitype == Scitype.TOF_HIGH:
            self.TOF_High_bits += bits
        elif scitype == Scitype.TOF_LOW:
            self.TOF_Low_bits += bits
        elif scitype == Scitype.TOF_MID:
            self.TOF_Mid_bits += bits
        elif scitype == Scitype.TARGET_LOW:
            self.Target_Low_bits += bits
        elif scitype == Scitype.TARGET_HIGH:
            self.Target_High_bits += bits
        elif scitype == Scitype.ION_GRID:
            self.Ion_Grid_bits += bits
        else:
            logger.warning("Unknown science type received: [%s]", scitype)

    def _set_sample_trigger_times(
        self, packet: space_packet_parser.packets.CCSDSPacket
    ) -> None:
        """
        Calculate the actual sample trigger time.

        Determines how many samples of data are included before the dust impact
        triggered the instrument.

        Parameters
        ----------
        packet : space_packet_parser.packets.CCSDSPacket
            The IDEX FPGA header packet info.

        Notes
        -----
        A "sample" is one single data point.

        A "block" is ~1.969 microseconds of data collection (8/4.0625). The only
        time that a block of data matters is in this function.

        Because the low sample data are taken every 1/4.0625 microseconds, there
        are 8 samples in one block of data.

        Because the high sample data are taken every 1/260 microseconds, there
        are 512 samples in one block of High Sample data.

        The header has information about the number of blocks before triggering,
        rather than the number of samples before triggering.
        """
        # Retrieve the number of samples for high gain delay

        # packet['IDX__TXHDRSAMPDELAY'] is a 32-bit value, with the last 10 bits
        # representing the high gain sample delay and the first 2 bits used for padding.
        # To extract the high gain bits, the bitwise right shift (>> 20) moves the bits
        # 20 positions to the right, and the mask (0b1111111111) keeps only the least
        # significant 10 bits.
        # TODO use the delay corresponding to the trigger
        high_gain_delay = (packet["IDX__TXHDRSAMPDELAY"] >> 22) & 0b1111111111
        n_blocks = packet["IDX__TXHDRBLOCKS"]

        # Retrieve number of low/high sample pre-trigger blocks

        # packet['IDX__TXHDRBLOCKS'] is a 32-bit value:
        # Bits 21-26 represent the number of low sampling pre-trigger blocks.
        #   We can extract this by shifting right by 6 bits and applying a mask to keep
        #   the last 6 bits.
        # Bits 13-16 represent the number of high sampling pre-trigger blocks.
        #   We can extract this by shifting right by 16 bits and applying a mask to keep
        #   the last 4 bits.
        num_low_sample_pretrigger_blocks = (n_blocks >> 6) & 0b111111
        num_high_sample_pretrigger_blocks = (n_blocks >> 16) & 0b1111
        # Calculate the low and high sample trigger times based on the high gain delay
        # and the number of high sample/low sample pretrigger blocks
        self.low_sample_trigger_time = (
            self.LOW_SAMPLE_RATE
            * (num_low_sample_pretrigger_blocks + 1)
            * self.NUMBER_SAMPLES_PER_LOW_SAMPLE_BLOCK
        )
        self.high_sample_trigger_time = (
            self.HIGH_SAMPLE_RATE
            * (num_high_sample_pretrigger_blocks + 1)
            * self.NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK
            - self.HIGH_SAMPLE_RATE * high_gain_delay
        )

    def _parse_high_sample_waveform(self, waveform_raw: str) -> list[int]:
        """
        Will process the high sample waveform.

        Parse a binary string representing a high sample waveform.
        If the data has been compressed, decompress using the RICE Golomb algorithm.

        Parameters
        ----------
        waveform_raw : str
            The binary string representing the high sample waveform.

        Returns
        -------
        ints : list[int]
            List of the high sample waveform.
        """
        samples = self.MAX_HIGH_BLOCKS * self.NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK
        ints: list[int] = []
        if self.compressed.raw_value == 1:
            ints.extend(rice_decode(waveform_raw, nbit10=True, sample_count=samples))
            ints = ints[:-3]
        else:
            ints.extend(_read_waveform_bits(waveform_raw, high_sample=True))
        return ints

    def _parse_low_sample_waveform(self, waveform_raw: str) -> list[int]:
        """
        Will process the low sample waveform.

        Parse a binary string representing a low sample waveform
        If the data has been compressed, decompress using the RICE Golomb algorithm.

        Parameters
        ----------
        waveform_raw : str
            The binary string representing the low sample waveform.

        Returns
        -------
        ints : list[int]
            List of processed low sample waveform.
        """
        samples = self.MAX_LOW_BLOCKS * self.NUMBER_SAMPLES_PER_LOW_SAMPLE_BLOCK
        ints: list[int] = []
        if self.compressed.raw_value == 1:
            ints.extend(rice_decode(waveform_raw, nbit10=False, sample_count=samples))
        else:
            ints.extend(_read_waveform_bits(waveform_raw, high_sample=False))
        return ints

    def _calc_low_sample_resolution(self, num_samples: int) -> npt.NDArray:
        """
        Calculate the resolution of the low samples.

        Calculates the low sample time array based on the number
        of samples of data taken.

        Multiply a linear array by the sample rate.
        Subtract the calculated trigger time.

        Parameters
        ----------
        num_samples : int
            The number of samples.

        Returns
        -------
        time_low_sample_rate_data : numpy.ndarray
            Low time sample data array.
        """
        time_low_sample_rate_init = np.linspace(0, num_samples, num_samples)
        time_low_sample_rate_data = (
            self.LOW_SAMPLE_RATE * time_low_sample_rate_init
            - self.low_sample_trigger_time
        )
        return time_low_sample_rate_data

    def _calc_high_sample_resolution(self, num_samples: int) -> npt.NDArray:
        """
        Calculate the resolution of high samples.

        Calculates the high sample time array based on the number
        of samples of data taken.

        Multiply a linear array by the sample rate.
        Subtract the calculated trigger time.

        Parameters
        ----------
        num_samples : int
            The number of samples.

        Returns
        -------
        time_high_sample_rate_data : numpy.ndarray
            High sample time data array.
        """
        time_high_sample_rate_init = np.linspace(0, num_samples, num_samples)
        time_high_sample_rate_data = (
            self.HIGH_SAMPLE_RATE * time_high_sample_rate_init
            - self.high_sample_trigger_time
        )
        return time_high_sample_rate_data

    def _populate_bit_strings(
        self, packet: space_packet_parser.packets.CCSDSPacket
    ) -> None:
        """
        Parse IDEX data packets to populate bit strings.

        Parameters
        ----------
        packet : space_packet_parser.packets.CCSDSPacket
            A single science data packet for one of the 6.
            IDEX observables.
        """
        scitype = packet["IDX__SCI0TYPE"]
        raw_science_bits = convert_to_binary_string(packet["IDX__SCI0RAW"])
        self._append_raw_data(scitype, raw_science_bits)

    def process(self) -> Dataset | None:
        """
        Will process the raw data into a ``xarray.Dataset``.

        To be called after all packets for the IDEX event have been parsed.
        Parses the binary data into numpy integer arrays, and combines them into
        a ``xarray.Dataset`` object.

        Returns
        -------
        dataset : xarray.Dataset, None
            A Dataset object containing the data from a single impact.
        """
        # Create an object for CDF attrs
        idex_attrs = self.cdf_attrs

        # Gather the huge amount of metadata info
        trigger_vars = {}
        for var, value in self.telemetry_items.items():
            trigger_vars[var] = xr.DataArray(
                name=var,
                data=[value],
                dims=("epoch"),
                attrs=idex_attrs.get_variable_attributes(var),
            )

        data_vars = {
            # Process the 6 primary data variables
            "TOF_High": xr.DataArray(
                name="TOF_High",
                data=[self._parse_high_sample_waveform(self.TOF_High_bits)],
                dims=("epoch", "time_high_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("tof_high_attrs"),
            ),
            "TOF_Low": xr.DataArray(
                name="TOF_Low",
                data=[self._parse_high_sample_waveform(self.TOF_Low_bits)],
                dims=("epoch", "time_high_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("tof_low_attrs"),
            ),
            "TOF_Mid": xr.DataArray(
                name="TOF_Mid",
                data=[self._parse_high_sample_waveform(self.TOF_Mid_bits)],
                dims=("epoch", "time_high_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("tof_mid_attrs"),
            ),
            "Target_High": xr.DataArray(
                name="Target_High",
                data=[self._parse_low_sample_waveform(self.Target_High_bits)],
                dims=("epoch", "time_low_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("target_high_attrs"),
            ),
            "Target_Low": xr.DataArray(
                name="Target_Low",
                data=[self._parse_low_sample_waveform(self.Target_Low_bits)],
                dims=("epoch", "time_low_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("target_low_attrs"),
            ),
            "Ion_Grid": xr.DataArray(
                name="Ion_Grid",
                data=[self._parse_low_sample_waveform(self.Ion_Grid_bits)],
                dims=("epoch", "time_low_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes("ion_grid_attrs"),
            ),
        }
        # Determine coordinate variables
        coords = {
            "epoch": xr.DataArray(
                name="epoch",
                data=[self.impact_time],
                dims=("epoch"),
                attrs=idex_attrs.get_variable_attributes("epoch", check_schema=False),
            ),
        }
        sampling_rates = {
            "time_low_sample_rate": xr.DataArray(
                name="time_low_sample_rate",
                data=[
                    self._calc_low_sample_resolution(len(data_vars["Target_Low"][0]))
                ],
                dims=("epoch", "time_low_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes(
                    "low_sample_rate_attrs", check_schema=False
                ),
            ),
            "time_high_sample_rate": xr.DataArray(
                name="time_high_sample_rate",
                data=[self._calc_high_sample_resolution(len(data_vars["TOF_Low"][0]))],
                dims=("epoch", "time_high_sample_rate_index"),
                attrs=idex_attrs.get_variable_attributes(
                    "high_sample_rate_attrs", check_schema=False
                ),
            ),
        }
        expected_shapes = {
            f"{name}_index": array.shape[1] for name, array in sampling_rates.items()
        }
        if any(
            var.shape[1] != expected_shapes[var.dims[1]] for var in data_vars.values()
        ):
            # The IDEX team requests that a warning be logged for incomplete events
            # (dropped packets) in the data, while still allowing the CDF to be created
            # with the remainder of the complete events.
            logger.warning(
                "Missing packet for event number %s. Skipping event..",
                self.event_number,
            )
            return None

        # Combine to return a dataset object
        dataset = xr.Dataset(
            data_vars=data_vars | trigger_vars | sampling_rates,
            coords=coords,
        )
        return dataset
