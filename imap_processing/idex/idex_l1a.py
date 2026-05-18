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

import json
import logging
import os
from collections import defaultdict
from enum import IntEnum
from os import path
from pathlib import Path

import numpy as np
import numpy.typing as npt
import space_packet_parser
import xarray as xr
from xarray import Dataset

from imap_processing import imap_module_directory
from imap_processing.idex.decode import rice_decode
from imap_processing.idex.evt_msg_decode_utils import render_event_template
from imap_processing.idex.idex_constants import IDEXAPID
from imap_processing.idex.idex_l0 import decom_packets
from imap_processing.idex.idex_utils import get_10_day_window_end_date, get_idex_attrs
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    str_yyyymmdd_to_ttj2000ns,
)
from imap_processing.utils import convert_to_binary_string

logger = logging.getLogger(__name__)


def idex_l1a(
    packet_files: list[Path], window_start_date: str
) -> list[xr.Dataset | None]:
    """
    Process a list of IDEX L0 packet files into a list of xarray Datasets.

    Parameters
    ----------
    packet_files : list[pathlib.Path]
        List of paths to IDEX L0 packet files to process. These l0 files should all
        contain data that belongs in the same 10-day window specified by start_date.
    window_start_date : str
        The start date of the 10-day window in YYYYMMDD format. Used to filter the
        data for the 10-day window.

    Returns
    -------
    list[xarray.Dataset|None]
        A list of xarray Datasets containing the processed IDEX L1a data products. If
        There is no Data found for the 10-day window, None is returned.
    """
    # Sort packet files so latest version comes last
    # This ensures when we drop duplicate events (if any), the latest file's data is
    # kept.
    sorted_packet_files = sorted(packet_files)
    idex_products = []
    # decom each idex l0 file and gather the data for each product type
    # (science, event message, catlst) into separate lists
    data_dicts = defaultdict(list)
    for packet_file in sorted_packet_files:
        data = PacketParser(packet_file).data
        for product, dataset in data.items():
            data_dicts[product].append(dataset)
    # Get the end date of the data window. This will be used to filter events.
    window_end_date = get_10_day_window_end_date(window_start_date)
    # Convert from strings to ttj2000ns for easier epoch comparison
    window_start_date_ns = str_yyyymmdd_to_ttj2000ns(window_start_date)
    window_end_date_ns = str_yyyymmdd_to_ttj2000ns(window_end_date)
    # combine the data for each product type into a single dataset.
    # filter each dataset for epochs that are within the 10-day window range.
    for product, datasets in data_dicts.items():
        concat_ds = (
            xr.concat(
                datasets,
                dim="epoch",
                # Keep non-epoch support variables (e.g. label/index vectors) from a
                # single dataset instead of broadcasting them across epochs.
                data_vars="minimal",
                coords="minimal",
                compat="override",
                # Drop duplicate epochs, keeping the last one (which will be from the
                # latest file due to sorting above)
            )
            .sortby("epoch")
            .drop_duplicates("epoch", keep="last")
        )
        mask = (concat_ds["epoch"] >= window_start_date_ns) & (
            concat_ds["epoch"] < window_end_date_ns
        )
        filtered_ds = concat_ds.isel(epoch=mask)
        if len(filtered_ds.epoch) == 0:
            logger.warning(
                f"No data found for dates {window_start_date_ns} - {window_end_date_ns}"
                f" for {product} in packet files: "
                f"{[path.basename(f) for f in packet_files]}"
            )
            continue
        idex_products.append(filtered_ds)

    return idex_products


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

    def __init__(self, packet_file: str | Path) -> None:
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
        self.data = {}
        self.idex_attrs = get_idex_attrs("l1a")
        epoch_attrs = self.idex_attrs.get_variable_attributes(
            "epoch", check_schema=False
        )

        science_packets, raw_datset_by_apid, derived_datasets_by_apid = decom_packets(
            packet_file
        )
        filename = os.path.basename(packet_file)
        logger.info(f"Processing IDEX L1A Packet {filename}")
        if science_packets:
            logger.info("Processing IDEX L1A Science data.")
            self.data["l1a_sci-10days"] = self._create_science_dataset(science_packets)
        datasets_by_level = {"l1a": raw_datset_by_apid, "l1b": derived_datasets_by_apid}
        for level, dataset in datasets_by_level.items():
            # Only produce l1a products for event messages. L1b will be processed in a
            # another job.
            if IDEXAPID.IDEX_EVT in dataset and level == "l1a":
                logger.info("Processing IDEX L1A Event Message data")
                data = dataset[IDEXAPID.IDEX_EVT]
                processed_data = self._create_evt_msg_data(data)
                processed_data["epoch"].attrs = epoch_attrs
                self.data["l1a_msg-10days"] = processed_data

            if IDEXAPID.IDEX_CATLST in dataset:
                logger.info(f"Processing IDEX {level} CATLST data")
                data = dataset[IDEXAPID.IDEX_CATLST]
                data.attrs = self.idex_attrs.get_global_attributes(
                    f"imap_idex_{level}_catlst-10days"
                )
                data["epoch"] = calculate_idex_event_time(
                    data["shcoarse"].data, data["shfine"].data
                )
                data["epoch"].attrs = epoch_attrs
                self.data[f"{level}_catlst-10days"] = data

        logger.info("IDEX L1A data processing completed.")

    def _create_evt_msg_data(self, data: xr.Dataset) -> xr.Dataset:
        """
        Process IDEX message data into a more usable format.

        Parameters
        ----------
        data : xarray.Dataset
            The raw message data to process.

        Returns
        -------
        xarray.Dataset
            The processed message data.
        """
        # Convert the time to epoch time in nanoseconds since J2000 in the TT timescale
        # elsec_evtpkt is the coarse time in seconds, elssec_evtpkt is the fine time in
        # 20-microsecond intervals. We need to combine these to get the actual event
        # time.
        epoch = calculate_idex_event_time(
            data["elsec_evtpkt"].data, data["elssec_evtpkt"].data
        )
        # initialize dataset with time variables
        l1a_msg_ds = xr.Dataset(
            data_vars={
                "epoch": xr.DataArray(epoch, name="epoch", dims=["epoch"]),
                "elsec_evtpkt": xr.DataArray(
                    data["elsec_evtpkt"].data,
                    dims=["epoch"],
                    attrs=self.idex_attrs.get_variable_attributes("elsec_evtpkt"),
                ),
                "elssec_evtpkt": xr.DataArray(
                    data["elssec_evtpkt"].data,
                    dims=["epoch"],
                    attrs=self.idex_attrs.get_variable_attributes("elssec_evtpkt"),
                ),
            },
            attrs=self.idex_attrs.get_global_attributes("imap_idex_l1a_msg-10days"),
        )
        # Load the event decoding dictionaries
        with open(
            f"{imap_module_directory}/idex/idex_evt_msg_parsing_dictionaries.json"
        ) as f:
            msg_dicts = json.load(f)

        # restore integer keys since JSON stringifies them
        msg_json_data = {
            dict_name: {int(k): v for k, v in pairs.items()}
            for dict_name, pairs in msg_dicts.items()
        }
        # Get the event message templates and log entry name dictionaries
        # These are used to decode the raw event messages into human-readable formats
        # during rendering.
        event_description_templates = msg_json_data.get("eventMsgDictionary", {})
        log_entry_names = msg_json_data.get("logEntryIdDictionary", {})

        # Get the event id - this will tell us what event happened.
        # The following parameter values will tell us additional details about the event
        # For example the event may be a science state change and the parameters will
        # tell us what state it changed to (e.g. on or off).
        event_ids = data["elid_evtpkt"].data
        # Stack the parameter bytes into a single array of shape (num_events, 4) for
        # easier access during rendering.
        params_bytes = np.stack(
            [
                data["el1par_evtpkt"].data,
                data["el2par_evtpkt"].data,
                data["el3par_evtpkt"].data,
                data["el4par_evtpkt"].data,
            ],
            axis=-1,
        )

        # initialize an empty list for messages
        messages = []
        for idx in range(len(event_ids)):
            # Look up the string format using the event_id.
            event_id = event_ids[idx]
            current_desc_template = event_description_templates.get(event_id)
            current_param_bytes = params_bytes[idx].tolist()
            event_name = log_entry_names.get(event_id, f"EVENT_0x{event_id:02X}")
            # Render the event message using the template if available.
            if current_desc_template:
                try:
                    message = render_event_template(
                        current_desc_template, current_param_bytes, msg_json_data
                    )
                except Exception as exc:
                    message = (
                        f"{event_name} [template_render_error={exc}] "
                        f"params="
                        f"({', '.join(f'0x{x:02X}' for x in current_param_bytes)})"
                    )
            else:
                # If no template exists for an event ID, fall back to a message
                # that still preserves the event name and raw parameter bytes.
                phex = ", ".join(f"0x{x:02X}" for x in current_param_bytes)
                message = f"{event_name} ({phex})"

            messages.append(message)

        l1a_msg_ds["messages"] = xr.DataArray(
            messages,
            name="messages",
            dims=["epoch"],
            attrs=self.idex_attrs.get_variable_attributes(
                "messages", check_schema=False
            ),
        )
        l1a_msg_ds.attrs = self.idex_attrs.get_global_attributes(
            "imap_idex_l1a_msg-10days"
        )
        return l1a_msg_ds

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


def calculate_idex_event_time(
    coarse_time_sec: np.ndarray,
    fine_time_subs: np.ndarray,
) -> npt.NDArray[np.int64]:
    """
    Calculate the epoch time from the FPGA header time variables.

    Coarse_time_sec counts the number of whole seconds elapsed since the epoch
    (Jan 1st 2010), while fine_time_subs counts the number of additional 20-microsecond
    intervals beyond the whole seconds. Together, these time measurements establish
    when a dust event took place.

    Parameters
    ----------
    coarse_time_sec : numpy.ndarray
        The coarse event time (seconds).
    fine_time_subs : numpy.ndarray
        The fine event time in 20-microsecond intervals.

    Returns
    -------
    numpy.ndarray[numpy.int64]
        The mission elapsed time converted to nanoseconds since the J2000 epoch
        in the terrestrial time (TT) timescale.
    """
    # Calculate the fine event time in seconds
    fine_event_time = fine_time_subs * 20e-6
    return met_to_ttj2000ns(coarse_time_sec + fine_event_time)


class RawDustEvent:
    """
    Encapsulate IDEX Raw Dust Event.

    Encapsulates the work needed to convert a single dust event into a processed
    ``xarray`` ``dateset`` object.

    Parameters
    ----------
    header_packet : space_packet_parser.SpacePacket
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

    def __init__(self, header_packet: space_packet_parser.SpacePacket) -> None:
        """
        Initialize a raw dust event, with an FPGA Header Packet from IDEX.

        The values we care about are:

        self.impact_time - When the impact occurred.
        self.low_sample_trigger_time - When the low sample stuff actually triggered.
        self.high_sample_trigger_time - When the high sample stuff actually triggered.

        Parameters
        ----------
        header_packet : space_packet_parser.SpacePacket
            The FPGA metadata event header.
        """
        # Calculate the impact time in seconds since epoch
        self.impact_time = 0
        # The elapsed seconds are stored as a 32-bit unsigned integer that is split
        # across two 16-bit words for packetization. As a result, idx__txhdrtimesec1
        # represents multiples of 2^16 seconds, while idx_txhdrtimesec2 represents the
        # remaining seconds within that range. This necessitates bit shifting the upper
        # word by 16 bits when reconstructing the full seconds counter.
        self.impact_time = calculate_idex_event_time(
            (header_packet["IDX__TXHDRTIMESEC1"] << 16)
            + header_packet["IDX__TXHDRTIMESEC2"],
            header_packet["IDX__TXHDRTIMESUBS"],
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
            for i, (key, val) in enumerate(header_packet.items())
            if i > 6  # Skip first 7 header items
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
        self, packet: space_packet_parser.SpacePacket
    ) -> None:
        """
        Calculate the actual sample trigger time.

        Determines how many samples of data are included before the dust impact
        triggered the instrument.

        Parameters
        ----------
        packet : space_packet_parser.SpacePacket
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

        # packet['IDX__TXHDRSAMPDELAY'] is a 32-bit value:
        # bits0-9: high-gain delay,
        # bits10-19: mid-gain delay,
        # bits20-29: low-gain delay.
        # bits30-31 are padding/reserved.
        # Each delay is extracted by right-shifting to align the field,
        # then masking with #0b1111111111 (10 bits).

        n_blocks = packet["IDX__TXHDRBLOCKS"]
        trigger_item = packet["IDX__TXHDRTRIGID"]

        tof_delay = packet["IDX__TXHDRSAMPDELAY"]  # last two bits are padding

        # mask to extract 10-bit values
        tof_mask = 0b1111111111

        # Determine the delay based on the trigger id.
        hg_delay = tof_delay & tof_mask  # first 10 bits (0-9)
        mg_delay = (tof_delay >> 10) & tof_mask  # next 10 bits (10-19)
        lg_delay = (tof_delay >> 20) & tof_mask  # next 10 bits (20-29)

        u10 = trigger_item & 0x3FF
        if (u10 >> 0) & 1:
            delay = hg_delay
        elif (u10 >> 1) & 1:
            delay = lg_delay
        elif (u10 >> 2) & 1:
            delay = mg_delay
        else:
            delay = hg_delay

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
        self.high_sample_trigger_time = self.HIGH_SAMPLE_RATE * (
            num_high_sample_pretrigger_blocks + 1
        ) * self.NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK - self.HIGH_SAMPLE_RATE * (
            delay - 1
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
        time_low_sample_rate_init: np.ndarray = np.arange(num_samples, dtype=np.float64)
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
        time_high_sample_rate_init: np.ndarray = np.arange(
            num_samples, dtype=np.float64
        )
        time_high_sample_rate_data = (
            self.HIGH_SAMPLE_RATE * time_high_sample_rate_init
            - self.high_sample_trigger_time
        )
        return time_high_sample_rate_data

    def _populate_bit_strings(self, packet: space_packet_parser.SpacePacket) -> None:
        """
        Parse IDEX data packets to populate bit strings.

        Parameters
        ----------
        packet : space_packet_parser.SpacePacket
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
            # SCI0AID is not updated properly. To this end, TXHDRFSWAIDCOPY must be
            # used as the proper AID.
            if var == "idx__sci0aid":
                continue
            # rename idx__txhdrfswaidcopy to aid for better readability in the final
            # dataset
            var_name = "aid" if var == "idx__txhdrfswaidcopy" else var
            trigger_vars[var_name] = xr.DataArray(
                name=var_name,
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
