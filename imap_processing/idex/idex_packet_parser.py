import logging

import bitstring
import numpy as np
import xarray as xr

# from lasp_packets import xtcedef, parser
from space_packet_parser import parser, xtcedef

from imap_processing import packet_definition_directory

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
    """
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
     ---------
        >>> # Print out the data in a L0 file
        >>> from imap_processing.idex.idex_packet_parser import IDEXPacketParser
        >>> l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
        >>> l0_data = IDEXPacketParser(l0_file)
        >>> print(l0_data.data)

    """

    def __init__(self, packet_file: str):
        """
        This function takes in a local l0 pkts file and performs all of the decom work
        directly in __init__().

        Parameters
        -----------
            packet_file (str):  The path and filename to the L0 file to read

        Notes
        -----
            Currently assumes one L0 file will generate exactly one l1a file
        """

        xtce_file = f"{packet_definition_directory}/idex_packet_definition.xml"
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
    def __init__(self, header_packet):
        """
        This function initializes a raw dust event, with an FPGA Header Packet from IDEX

        Parameters:
            header_packet:  The FPGA metadata event header

        """
        self.impact_time = (
            header_packet.data["SHCOARSE"].derived_value
            + TWENTY_MICROSECONDS * header_packet.data["SHFINE"].derived_value
        )
        self._log_packet_info(header_packet)

        # Init the binary data received from future packets
        self.TOF_High_bits = ""
        self.TOF_Mid_bits = ""
        self.TOF_Low_bits = ""
        self.Target_Low_bits = ""
        self.Target_High_bits = ""
        self.Ion_Grid_bits = ""

    def _log_packet_info(self, packet):
        """
        This function exists solely to log the parameters in the L0
        FPGA header packet for new dust events, nothing here should affect the data
        """
        event_num = packet.data["IDX__SCI0EVTNUM"].derived_value
        logging.debug(f"^*****Event header {event_num}******^")
        logging.debug(
            f"Timestamp = {self.impact_time} seconds since epoch \
              (Midnight January 1st, 2012)"
        )
        # Extract the 17-22-bit integer (usually 8)
        lspretrigblocks = (packet.data["IDX__TXHDRBLOCKS"].derived_value >> 16) & 0b1111
        # Extract the next 4-bit integer (usually 8)
        lsposttrigblocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 12
        ) & 0b1111
        # Extract the next 6 bits integer (usually 32)
        hspretrigblocks = (
            packet.data["IDX__TXHDRBLOCKS"].derived_value >> 6
        ) & 0b111111
        # Extract the first 6 bits (usually 32)
        hsposttrigblocks = (packet.data["IDX__TXHDRBLOCKS"].derived_value) & 0b111111
        logging.debug("HS pre trig sampling blocks: " + str(hspretrigblocks))
        logging.debug("LS pre trig sampling blocks: " + str(lspretrigblocks))
        logging.debug("HS post trig sampling blocks: " + str(hsposttrigblocks))
        logging.debug("LS post trig sampling blocks: " + str(lsposttrigblocks))
        tof_delay = (
            packet.data["IDX__TXHDRSAMPDELAY"].raw_value >> 2
        )  # First two bits are padding
        mask = 0b1111111111
        lgdelay = (tof_delay) & mask
        mgdelay = (tof_delay >> 10) & mask
        hgdelay = (tof_delay >> 20) & mask
        logging.debug(f"High gain delay = {hgdelay} samples.")
        logging.debug(f"Mid gain delay = {mgdelay} samples.")
        logging.debug(f"Low gain delay = {lgdelay} samples.")
        if (
            packet.data["IDX__TXHDRLSTRIGMODE"].derived_value != 0
        ):  # If this was a LS (Target Low Gain) trigger
            self.Triggerorigin = "LS"
            logging.debug("Low sampling trigger mode enabled.")
        logging.debug(
            "Packet low trigger mode = "
            + str(packet.data["IDX__TXHDRLGTRIGMODE"].derived_value)
        )
        logging.debug(
            "Packet mid trigger mode = "
            + str(packet.data["IDX__TXHDRMGTRIGMODE"].derived_value)
        )
        logging.debug(
            "Packet high trigger mode = "
            + str(packet.data["IDX__TXHDRHGTRIGMODE"].derived_value)
        )
        if packet.data["IDX__TXHDRLGTRIGMODE"].derived_value != 0:
            logging.debug("Low gain TOF trigger mode enabled.")
            self.Triggerorigin = "LG"
        if packet.data["IDX__TXHDRMGTRIGMODE"].derived_value != 0:
            logging.debug("Mid gain TOF trigger mode enabled.")
            self.Triggerorigin = "MG"
        if packet.data["IDX__TXHDRHGTRIGMODE"].derived_value != 0:
            logging.debug("High gain trigger mode enabled.")
            self.Triggerorigin = "HG"
        logging.debug(
            f"AID = {packet.data['IDX__SCI0AID'].derived_value}"
        )  # Instrument event number
        logging.debug(
            f"Event number = {packet.data['IDX__SCI0EVTNUM'].raw_value}"
        )  # Event number out of how many events constitute the file
        logging.debug(
            f"Rice compression enabled = {bool(packet.data['IDX__SCI0COMP'].raw_value)}"
        )

    def _parse_high_sample_waveform(self, waveform_raw: str):
        """
        Parse a binary string representing a high sample waveform
        Data arrives in 10 bit chunks
        """
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:2")  # skip 2
            ints += w.readlist(["uint:10"] * 3)
        return ints[:-4]

    def _parse_low_sample_waveform(self, waveform_raw: str):
        """
        Parse a binary string representing a low sample waveform
        Data arrives in 12 bit chunks
        """
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:8")  # skip 2
            ints += w.readlist(["uint:12"] * 2)
        return ints

    def parse_packet(self, packet):
        """
        This function parses IDEX data packets to populate bit strings

        Parameters
        ----------
            packet: A single science data packet for one of the 6
                    IDEX observables
        """

        scitype = packet.data["IDX__SCI0TYPE"].raw_value
        raw_science_bits = packet.data["IDX__SCI0RAW"].raw_value
        self._append_raw_data(scitype, raw_science_bits)

    def _append_raw_data(self, scitype, bits):
        """
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

    def process(self):
        """
        To be called after all packets for the IDEX event have been parsed
        Parses the binary data into numpy integer arrays, and combines them
        into an xarray.Dataset object

        Returns
        -------
        xarray.Dataset
            A Dataset object containing the data from a single impact

        TODO
        ----
        * high_sample_rate and low_sample_rate must be multiplied by
          some scale factor that will be decided upon in the future
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
            data=[self._parse_low_sample_waveform(self.Target_High_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        ion_grid_xr = xr.DataArray(
            name="Ion_Grid",
            data=[self._parse_low_sample_waveform(self.Target_High_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        # Determine the 3 coordinate variables
        epoch_xr = xr.DataArray(name="Epoch", data=[self.impact_time], dims=("Epoch"))
        time_low_sr_xr = xr.DataArray(
            name="Time_Low_SR",
            data=[np.linspace(0, len(ion_grid_xr[0]), len(ion_grid_xr[0]))],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        time_high_sr_xr = xr.DataArray(
            name="Time_High_SR",
            data=[np.linspace(0, len(tof_low_xr[0]), len(tof_low_xr[0]))],
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
