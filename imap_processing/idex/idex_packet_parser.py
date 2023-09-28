import logging

import bitstring
import numpy as np
import xarray as xr
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory

TWENTY_MICROSECONDS = 20 * (10 ** (-6))

SCITYPE_MAPPING_TO_NAMES = {
    2: "TOF_High",
    4: "TOF_Low",
    8: "TOF_Mid",
    16: "Target_Low",
    32: "Target_High",
    64: "Ion_Grid",
}


class PacketParser:
    """IDEX packet parsing class.

    This class encapsulates the decom work needed to decom a daily file of IDEX data
    received from the POC.  The class is instantiated with a reference to a L0 file as
    it exists on the local file system.

    Attributes
    ----------
        data (xarray.Dataset): An object containing all of the relevant data in the file

    TODO
    ----
        * Add method to generate l1a CDF
        * Add method to generate quicklook

    Examples
    --------
        >>> # Print out the data in a L0 file
        >>> from imap_processing.idex.idex_packet_parser import PacketParser
        >>> l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
        >>> l0_data = PacketParser(l0_file)
        >>> print(l0_data.data)

    """

    def __init__(self, packet_file: str):
        """Read a l0 pkts file and perform all of the decom work.

        Parameters
        ----------
            packet_file (str):  The path and filename to the L0 file to read

        Notes
        -----
            Currently assumes one L0 file will generate exactly one l1a file
        """
        xtce_filename = "idex_packet_definition.xml"
        xtce_file = f"{imap_module_directory}/idex/packet_definitions/{xtce_filename}"
        packet_definition = xtcedef.XtcePacketDefinition(xtce_document=xtce_file)
        packet_parser = parser.PacketParser(packet_definition)

        binary_data = bitstring.ConstBitStream(filename=packet_file)
        packet_generator = packet_parser.generator(binary_data)

        dust_events = {}

        for packet in packet_generator:
            if "IDX__SCI0TYPE" in packet.data:
                scitype = packet.data["IDX__SCI0TYPE"].raw_value
                event_number = packet.data["IDX__SCI0EVTNUM"].derived_value
                if scitype == 1:
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
                    dust_events[event_number].parse_packet(packet)
            else:
                logging.warning(f"Unhandled packet received: {packet}")

        processed_dust_impact_list = [
            dust_event.process() for dust_event in dust_events.values()
        ]

        self.data = xr.concat(processed_dust_impact_list, dim="Epoch")


class RawDustEvent:
    """Store data for each raw dust event."""

    # Constants
    HIGH_SAMPLE_RATE = 1 / 260  # nanoseconds per sample
    LOW_SAMPLE_RATE = 1 / 4.0625  # nanoseconds per sample
    FOUR_BIT_MASK = 0b1111
    SIX_BIT_MASK = 0b111111
    TEN_BIT_MASK = 0b1111111111

    def __init__(self, header_packet):
        """Initialize a raw dust event, with an FPGA Header Packet from IDEX.

        The values we care about are:

        self.impact_time - When the impact occured
        self.low_sample_trigger_time - When the low sample stuff actually triggered
        self.high_sample_trigger_time - When the high sample stuff actually triggered

        Parameters
        ----------
            header_packet:  The FPGA metadata event header

        """
        # Calculate the impact time in seconds since Epoch
        self.impact_time = self._calc_impact_time(header_packet)

        (
            self.low_sample_trigger_time,
            self.high_sample_trigger_time,
        ) = self._calc_sample_trigger_times(header_packet)

        # Initialize the binary data received from future packets
        self.TOF_High_bits = ""
        self.TOF_Mid_bits = ""
        self.TOF_Low_bits = ""
        self.Target_Low_bits = ""
        self.Target_High_bits = ""
        self.Ion_Grid_bits = ""

        # Log the rest of the header
        self._log_packet_info(header_packet)

    def _calc_impact_time(self, packet):
        """Calculate the number of seconds since Jan 1 2012."""
        # Number of seconds here
        seconds_since_epoch = packet.data["SHCOARSE"].derived_value
        # Number of 20 microsecond "ticks" since the last second
        num_of_20_microseconds = packet.data["SHFINE"].derived_value

        return seconds_since_epoch + TWENTY_MICROSECONDS * num_of_20_microseconds

    def _calc_sample_trigger_times(self, packet):
        """Calculate the high sample and low sample trigger times."""
        # This is a 32 bit number, consisting of:
        # 2 bits padding
        # 10 bits for low gain delay
        # 10 bits for mid gain delay
        # 10 bits for high gain delay
        time_of_flight_sample_delay_field = (
            packet.data["IDX__TXHDRSAMPDELAY"].raw_value >> 2
        )
        # Retrieve high gain delay from above number
        high_gain_delay = (time_of_flight_sample_delay_field >> 20) & self.TEN_BIT_MASK

        # Retrieve number of low/high sample pretrigger blocks
        num_low_sample_pretrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 6
        ) & self.SIX_BIT_MASK
        num_high_sample_pretrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 16
        ) & self.FOUR_BIT_MASK

        # Calculate the low and high sample trigger times based on the high gain delay
        # and the number of high sample/low sample pretrigger blocks
        low_sample_trigger_time = (
            8 * self.LOW_SAMPLE_RATE * (num_low_sample_pretrigger_blocks + 1)
            - self.HIGH_SAMPLE_RATE * high_gain_delay
        )
        high_sample_trigger_time = (
            512 * self.HIGH_SAMPLE_RATE * (num_high_sample_pretrigger_blocks + 1)
        )

        return low_sample_trigger_time, high_sample_trigger_time

    def _log_packet_info(self, packet):
        """Packet logging routine.

        This function exists solely to log the parameters in the L0
        FPGA header packet for new dust events, nothing here should affect the data.
        """
        event_number = packet.data["IDX__SCI0EVTNUM"].derived_value
        logging.debug(f"^*****Event header {event_number}******^")
        logging.debug(
            f"Timestamp = {self.impact_time} seconds since epoch \
              (Midnight January 1st, 2012)"
        )
        # Extract the number of blocks, pre and post trigger
        # Extract the first six bits
        low_sample_posttrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value
        ) & self.SIX_BIT_MASK
        low_sample_pretrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 6
        ) & self.SIX_BIT_MASK
        high_sample_posttrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 12
        ) & self.FOUR_BIT_MASK
        high_sample_pretrigger_blocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 16
        ) & self.FOUR_BIT_MASK
        logging.debug(
            "High Sample pre trig sampling blocks: "
            + str(high_sample_pretrigger_blocks)
        )
        logging.debug(
            "Low Sample pre trig sampling blocks: " + str(low_sample_pretrigger_blocks)
        )
        logging.debug(
            "High Sample post trig sampling blocks: "
            + str(high_sample_posttrigger_blocks)
        )
        logging.debug(
            "Low Sample post trig sampling blocks: "
            + str(low_sample_posttrigger_blocks)
        )

        # Calculate the high sample and low sample trigger times
        # This is a 32 bit number, consisting of:
        # 2 bits padding
        # 10 bits for low gain delay
        # 10 bits for mid gain delay
        # 10 bits for high gain delay
        time_of_flight_sample_delay_field = (
            packet.data["IDX__TXHDRSAMPDELAY"].raw_value >> 2
        )
        # Retrieve the delays for the different triggers
        low_gain_delay = (time_of_flight_sample_delay_field) & self.TEN_BIT_MASK
        mid_gain_delay = (time_of_flight_sample_delay_field >> 10) & self.TEN_BIT_MASK
        high_gain_delay = (time_of_flight_sample_delay_field >> 20) & self.TEN_BIT_MASK
        logging.debug(f"High gain delay = {high_gain_delay} samples.")
        logging.debug(f"Mid gain delay = {mid_gain_delay} samples.")
        logging.debug(f"Low gain delay = {low_gain_delay} samples.")

        # Determine which of the following triggered the dust event
        low_sample_trigger_mode = packet.data["IDX__TXHDRLSTRIGMODE"].derived_value
        low_gain_trigger_mode = packet.data["IDX__TXHDRLGTRIGMODE"].derived_value
        mid_gain_trigger_mode = packet.data["IDX__TXHDRMGTRIGMODE"].derived_value
        high_gain_trigger_mode = packet.data["IDX__TXHDRHGTRIGMODE"].derived_value
        logging.debug("Packet low trigger mode = " + str(low_gain_trigger_mode))
        logging.debug("Packet mid trigger mode = " + str(mid_gain_trigger_mode))
        logging.debug("Packet high trigger mode = " + str(high_gain_trigger_mode))
        if low_sample_trigger_mode:
            logging.debug("Low sampling trigger mode enabled.")
        if low_gain_trigger_mode:
            logging.debug("Low gain TOF trigger mode enabled.")
        if mid_gain_trigger_mode:
            logging.debug("Mid gain TOF trigger mode enabled.")
        if high_gain_trigger_mode != 0:
            logging.debug("High gain trigger mode enabled.")

        # Determine unique event identifier
        accountability_id = packet.data["IDX__SCI0AID"].derived_value
        logging.debug(f"AID = {accountability_id}")
        # Determine compression
        compression = bool(packet.data["IDX__SCI0COMP"].raw_value)
        logging.debug(f"Rice compression enabled = {compression}")

    def _parse_high_sample_waveform(self, waveform_raw: str):
        """Process the high sample waveform.

        Parse a binary string representing a high sample waveform
        Data arrives in 32 bit chunks, divided up into:
            * 2 bits of padding
            * 3x10 bits of integer data.

        The very last 4 numbers are bad usually, so remove those
        """
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:2")  # skip 2
            ints += w.readlist(["uint:10"] * 3)
        return ints[:-4]  # Remove last 4 numbers

    def _parse_low_sample_waveform(self, waveform_raw: str):
        """Process the low sample waveform.

        Parse a binary string representing a low sample waveform
        Data arrives in 32 bit chunks, divided up into:
            * 8 bits of padding
            * 2x12 bits of integer data.
        """
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:8")  # skip 8
            ints += w.readlist(["uint:12"] * 2)
        return ints

    def _calc_low_sample_resolution(self, num_samples: int):
        """Calculate the resolution of the low samples.

        Calculates the low sample time array based on the number
        of samples of data taken.

        Multiply a linear array by the sample rate
        Subtract the calculated trigger time
        """
        time_low_sr_init = np.linspace(0, num_samples, num_samples)
        time_low_sr_data = (
            self.LOW_SAMPLE_RATE * time_low_sr_init - self.low_sample_trigger_time
        )
        return time_low_sr_data

    def _calc_high_sample_resolution(self, num_samples: int):
        """Calculate the resolution of high samples.

        Calculates the high sample time array based on the number
        of samples of data taken.

        Multiply a linear array by the sample rate
        Subtract the calculated trigger time
        """
        time_high_sr_init = np.linspace(0, num_samples, num_samples)
        time_high_sr_data = (
            self.HIGH_SAMPLE_RATE * time_high_sr_init - self.high_sample_trigger_time
        )
        return time_high_sr_data

    def parse_packet(self, packet):
        """Parse IDEX data packets to populate bit strings.

        Parameters
        ----------
            packet: A single science data packet for one of the 6
                    IDEX observables
        """
        scitype = packet.data["IDX__SCI0TYPE"].raw_value
        raw_science_bits = packet.data["IDX__SCI0RAW"].raw_value
        self._append_raw_data(scitype, raw_science_bits)

    def _append_raw_data(self, scitype, bits):
        """Append data to the appropriate bit string.

        This function determines which variable to append the bits
        to, given a specific scitype.
        """
        if scitype == 2:
            self.TOF_High_bits += bits
        elif scitype == 4:
            self.TOF_Low_bits += bits
        elif scitype == 8:
            self.TOF_Mid_bits += bits
        elif scitype == 16:
            self.Target_Low_bits += bits
        elif scitype == 32:
            self.Target_High_bits += bits
        elif scitype == 64:
            self.Ion_Grid_bits += bits
        else:
            logging.warning("Unknown science type received: [%s]", scitype)

    def process(self):
        """Process the raw data into a xarray.Dataset.

        To be called after all packets for the IDEX event have been parsed
        Parses the binary data into numpy integer arrays, and combines them
        into an xarray.Dataset object.

        Returns
        -------
        xarray.Dataset
            A Dataset object containing the data from a single impact

        """
        # Process the 6 primary data variables
        tof_high_xr = xr.DataArray(
            name="TOF_High",
            data=[self._parse_high_sample_waveform(self.TOF_High_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
        )
        tof_low_xr = xr.DataArray(
            name="TOF_Low",
            data=[self._parse_high_sample_waveform(self.TOF_Low_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
        )
        tof_mid_xr = xr.DataArray(
            name="TOF_Mid",
            data=[self._parse_high_sample_waveform(self.TOF_Mid_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
        )
        target_high_xr = xr.DataArray(
            name="Target_High",
            data=[self._parse_low_sample_waveform(self.Target_High_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        target_low_xr = xr.DataArray(
            name="Target_Low",
            data=[self._parse_low_sample_waveform(self.Target_Low_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        ion_grid_xr = xr.DataArray(
            name="Ion_Grid",
            data=[self._parse_low_sample_waveform(self.Ion_Grid_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        # Determine the 3 coordinate variables
        epoch_xr = xr.DataArray(name="Epoch", data=[self.impact_time], dims=("Epoch"))

        time_low_sr_xr = xr.DataArray(
            name="Time_Low_SR",
            data=[self._calc_low_sample_resolution(len(target_low_xr[0]))],
            dims=("Epoch", "Time_Low_SR_dim"),
        )

        time_high_sr_xr = xr.DataArray(
            name="Time_High_SR",
            data=[self._calc_high_sample_resolution(len(tof_low_xr[0]))],
            dims=("Epoch", "Time_High_SR_dim"),
        )

        # Combine to return a dataset object
        return xr.Dataset(
            data_vars={
                "TOF_Low": tof_low_xr,
                "TOF_High": tof_high_xr,
                "TOF_Mid": tof_mid_xr,
                "Target_High": target_high_xr,
                "Target_Low": target_low_xr,
                "Ion_Grid": ion_grid_xr,
            },
            coords={
                "Epoch": epoch_xr,
                "Time_Low_SR": time_low_sr_xr,
                "Time_High_SR": time_high_sr_xr,
            },
        )
