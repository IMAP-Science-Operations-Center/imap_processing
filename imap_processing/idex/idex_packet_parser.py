import logging

import bitstring
import numpy as np
from space_packet_parser import parser, xtcedef

from imap_processing import packet_definition_directory

TWENTY_MICROSECONDS = 20 * (10 ** (-6))


class IDEXPacketParser:
    """
    This class encapsulates the decom work needed to decom a daily file of IDEX data
    received from the POC.  The class is instantiated with a reference to a L0 file as
    it exists on the local file system.

    Parameters:
        filename (str):  The path to the CDF file to read

     Example:
        >>> # Print out the data in a single packet
        >>> from imap_processing.idex.idex_packet_parser import IDEXPacketParser
        >>> l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
        >>> l0_data = IDEXPacketParser(l0_file)
        >>> print(l0_data.epochs)
        >>> print(l0_data.data)
    """

    def __init__(self, idex_packet_file: str):
        """
        This function takes in a local l0 pkts file and performs all of the decom work
        directly in __init__().

        Parameters:
            idex_packet_file (str):  The path and filename to the L0 file to read

        Returns:
            Populates the {object}.data, where data is a dictionary in the form
            data[(variable name)] = 1D or 2D array of data

            Populates the {object}.coords, where these represent the coordinates that
            the data uses.

        Notes:
            There are a few things that must be determined in the future:
            * high_sample_rate and low_sample_rate must be multiplied by
              some scale factor that will be decided upon in the future
            * IDEX_Trigger is a placeholder variable for now
        """

        idex_xtce = f"{packet_definition_directory}/idex_packet_definition.xml"
        idex_definition = xtcedef.XtcePacketDefinition(xtce_document=idex_xtce)
        idex_parser = parser.PacketParser(idex_definition)

        idex_binary_data = bitstring.ConstBitStream(filename=idex_packet_file)
        idex_packet_generator = idex_parser.generator(idex_binary_data)

        epochs = {}

        self.coords = {}
        self.data = {}
        self.scitype_to_names = {
            2: "TOF_High",
            4: "TOF_Low",
            8: "TOF_Mid",
            16: "Target_Low",
            32: "Target_High",
            64: "Ion_Grid",
        }

        for packet in idex_packet_generator:
            if "IDX__SCI0TYPE" in packet.data:
                scitype = packet.data["IDX__SCI0TYPE"].raw_value
                event_number = packet.data["IDX__SCI0EVTNUM"].derived_value
                if scitype == 1:
                    epochs[event_number] = (
                        packet.data["SHCOARSE"].derived_value
                        + TWENTY_MICROSECONDS * packet.data["SHFINE"].derived_value
                    )
                    self._log_packet_info(epochs, packet)  # These are for our logs
                if scitype in self.scitype_to_names:
                    if scitype not in self.data:
                        self.data.update({scitype: {}})
                    if event_number not in self.data[scitype]:
                        self.data[scitype][event_number] = packet.data[
                            "IDX__SCI0RAW"
                        ].raw_value
                    else:
                        self.data[scitype][event_number] += packet.data[
                            "IDX__SCI0RAW"
                        ].raw_value

        # Parse the waveforms according to the scitype present
        # (high gain and low gain channels encode waveform data differently).
        datastore = {}
        low_sample_rate = []
        high_sample_rate = []
        for scitype in self.data:
            datastore[self.scitype_to_names[scitype]] = []
            for event in self.data[scitype]:
                datastore[self.scitype_to_names[scitype]].append(
                    self._parse_waveform_data(self.data[scitype][event], scitype)
                )
                if self.scitype_to_names[scitype] == "Target_Low":
                    low_sample_rate.append(
                        np.linspace(
                            0,
                            len(datastore["Target_Low"][0]),
                            len(datastore["Target_Low"][0]),
                        )
                    )
                if self.scitype_to_names[scitype] == "TOF_Low":
                    high_sample_rate.append(
                        np.linspace(
                            0,
                            len(datastore["TOF_Low"][0]),
                            len(datastore["TOF_Low"][0]),
                        )
                    )

        self.coords["Epoch"] = list(epochs.values())
        self.coords["Time_Low_SR"] = low_sample_rate
        self.coords["Time_High_SR"] = high_sample_rate
        self.data = datastore
        self.epochs = epochs

    def _log_packet_info(self, epochs, pkt):
        event_num = pkt.data["IDX__SCI0EVTNUM"].derived_value
        logging.info(f"^*****Event header {event_num}******^")
        logging.info(
            f"Timestamp = {epochs[event_num]} seconds since epoch \
              (Midnight January 1st, 2012)"
        )
        # Extract the 17-22-bit integer (usually 8)
        lspretrigblocks = (pkt.data["IDX__TXHDRBLOCKS"].derived_value >> 16) & 0b1111
        # Extract the next 4-bit integer (usually 8)
        lsposttrigblocks = (pkt.data["IDX__TXHDRBLOCKS"].derived_value >> 12) & 0b1111
        # Extract the next 6 bits integer (usually 32)
        hspretrigblocks = (pkt.data["IDX__TXHDRBLOCKS"].derived_value >> 6) & 0b111111
        # Extract the first 6 bits (usually 32)
        hsposttrigblocks = (pkt.data["IDX__TXHDRBLOCKS"].derived_value) & 0b111111
        logging.info("HS pre trig sampling blocks: " + str(hspretrigblocks))
        logging.info("LS pre trig sampling blocks: " + str(lspretrigblocks))
        logging.info("HS post trig sampling blocks: " + str(hsposttrigblocks))
        logging.info("LS post trig sampling blocks: " + str(lsposttrigblocks))
        tof_delay = (
            pkt.data["IDX__TXHDRSAMPDELAY"].raw_value >> 2
        )  # First two bits are padding
        mask = 0b1111111111
        lgdelay = (tof_delay) & mask
        mgdelay = (tof_delay >> 10) & mask
        hgdelay = (tof_delay >> 20) & mask
        logging.info(f"High gain delay = {hgdelay} samples.")
        logging.info(f"Mid gain delay = {mgdelay} samples.")
        logging.info(f"Low gain delay = {lgdelay} samples.")
        if (
            pkt.data["IDX__TXHDRLSTRIGMODE"].derived_value != 0
        ):  # If this was a LS (Target Low Gain) trigger
            self.Triggerorigin = "LS"
            logging.info("Low sampling trigger mode enabled.")
        logging.info(
            "Packet low trigger mode = "
            + str(pkt.data["IDX__TXHDRLGTRIGMODE"].derived_value)
        )
        logging.info(
            "Packet mid trigger mode = "
            + str(pkt.data["IDX__TXHDRMGTRIGMODE"].derived_value)
        )
        logging.info(
            "Packet high trigger mode = "
            + str(pkt.data["IDX__TXHDRHGTRIGMODE"].derived_value)
        )
        if pkt.data["IDX__TXHDRLGTRIGMODE"].derived_value != 0:
            logging.info("Low gain TOF trigger mode enabled.")
            self.Triggerorigin = "LG"
        if pkt.data["IDX__TXHDRMGTRIGMODE"].derived_value != 0:
            logging.info("Mid gain TOF trigger mode enabled.")
            self.Triggerorigin = "MG"
        if pkt.data["IDX__TXHDRHGTRIGMODE"].derived_value != 0:
            logging.info("High gain trigger mode enabled.")
            self.Triggerorigin = "HG"
        logging.info(
            f"AID = {pkt.data['IDX__SCI0AID'].derived_value}"
        )  # Instrument event number
        logging.info(
            f"Event number = {pkt.data['IDX__SCI0EVTNUM'].raw_value}"
        )  # Event number out of how many events constitute the file
        logging.info(
            f"Rice compression enabled = {bool(pkt.data['IDX__SCI0COMP'].raw_value)}"
        )

    def _parse_high_sample_waveform(self, waveform_raw: str):
        """Parse a binary string representing a high gain waveform"""
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:2")  # skip 2
            ints += w.readlist(["uint:10"] * 3)
        return ints

    def _parse_low_sample_waveform(self, waveform_raw: str):
        """Parse a binary string representing a low gain waveform"""
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read("pad:8")  # skip 2
            ints += w.readlist(["uint:12"] * 2)
        return ints

    def _parse_waveform_data(self, waveform: str, scitype: int):
        """
        Chooses waveform parsing function depending on the sample
        rate of the variables
        """
        logging.info(f"Parsing waveform for scitype={scitype}")
        if self.scitype_to_names[scitype] in ("TOF_High", "TOF_Low", "TOF_Mid"):
            return self._parse_high_sample_waveform(waveform)
        else:
            return self._parse_low_sample_waveform(waveform)
