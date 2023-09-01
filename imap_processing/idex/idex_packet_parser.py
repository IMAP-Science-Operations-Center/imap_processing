import logging

import bitstring
import numpy as np
import xarray as xr
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
            Populates {object}.data, where data is an xarray DataSet

        Notes:
            Currently assumes one L0 file will generate exactly one l1a file

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

        self.coords = {}
        self.data = {}
        self.dust_events = {}
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
                    time_of_impact = (
                        packet.data["SHCOARSE"].derived_value
                        + TWENTY_MICROSECONDS * packet.data["SHFINE"].derived_value
                    )
                    self._log_packet_info(
                        time_of_impact, packet
                    )  # These are for our logs
                    self.dust_events[event_number] = IDEXRawDustEvent(time_of_impact)
                if (
                    scitype in self.scitype_to_names
                ):  # Populate the IDEXRawDustEvent with 1's and 0's
                    raw_science_data = packet.data["IDX__SCI0RAW"].raw_value
                    self.dust_events[event_number].append_raw_data(
                        scitype, raw_science_data
                    )

        # Parse the waveforms according to the scitype present
        # (high gain and low gain channels encode waveform data differently).
        processed_dust_impact_list = []
        for event_number in self.dust_events:
            processed_dust_impact_list.append(self.dust_events[event_number].process())

        self.data = xr.concat(processed_dust_impact_list, dim="Epoch")

    def _log_packet_info(self, time_of_impact, pkt):
        """
        This function exists solely to log the parameters in the L0
        packet, nothing here should affect the data
        """
        event_num = pkt.data["IDX__SCI0EVTNUM"].derived_value
        logging.info(f"^*****Event header {event_num}******^")
        logging.info(
            f"Timestamp = {time_of_impact} seconds since epoch \
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


class IDEXRawDustEvent:
    def __init__(self, epoch: float):
        """
        This function initializes a raw dust event

        Parameters:
            epoch (float):  Impact time in seconds since January 1st 2012 Midnight UTC

        """
        self.impact_time = epoch
        self.TOF_High_bits = ""
        self.TOF_Mid_bits = ""
        self.TOF_Low_bits = ""
        self.Target_Low_bits = ""
        self.Target_High_bits = ""
        self.Ion_Grid_bits = ""

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
        return ints

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

    def append_raw_data(self, scitype, bits):
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
        time_low_sr_xr = xr.DataArray(  # name='Time_Low_SR',
            data=[np.linspace(0, len(ion_grid_xr[0]), len(ion_grid_xr[0]))],
            dims=("Epoch", "Time_Low_SR_dim"),
        )
        time_high_sr_xr = xr.DataArray(  # name='Time_High_SR',
            data=[np.linspace(0, len(tof_low_xr[0]), len(tof_low_xr[0]))],
            dims=("Epoch", "Time_High_SR_dim"),
        )

        # Return a DataSet object
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
