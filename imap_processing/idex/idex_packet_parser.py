"""Decommutates IDEX CCSDS packets.

This module contains code to decommutate IDEX packets and creates xarrays to
support creation of L1 data products.
"""

import dataclasses
import logging
from collections import namedtuple
from enum import IntEnum

import numpy as np
import xarray as xr
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.idex import idex_cdf_attrs

logger = logging.getLogger(__name__)


class Scitype(IntEnum):
    """IDEX Science Type."""

    FIRST_PACKET = 1
    TOF_HIGH = 2
    TOF_LOW = 4
    TOF_MID = 8
    TARGET_LOW = 16
    TARGET_HIGH = 32
    ION_GRID = 64


"""
Creates a large dictionary of values from the FPGA header
that need to be captured into the CDF file.  They are lumped together because
they share similar attributes.

Notes about the variables are set here, acting as comments and will also be
placed into the CDF in the VAR_NOTES attribute.
"""
TriggerDescription = namedtuple(
    "TriggerDescription",
    ["name", "packet_name", "num_bits", "field", "notes", "label", "units"],
)
trigger_description_dict = {
    trigger.name: trigger
    for trigger in [
        TriggerDescription(
            "event_number",
            "IDX__TXHDREVTNUM",
            16,
            "Event Number",
            "The unique number assigned to the impact by the FPGA",
            "Event #",
            "",
        ),
        TriggerDescription(
            "tof_high_trigger_level",
            "IDX__TXHDRHGTRIGLVL",
            10,
            "TOF High Trigger Level",
            "Trigger level for the TOF High Channel",
            "Level",
            "",
        ),
        TriggerDescription(
            "tof_high_trigger_num_max_1_2",
            "IDX__TXHDRHGTRIGNMAX12",
            11,
            "TOF High Double Pulse Max Samples",
            (
                "Maximum number of samples between pulse 1 and 2 for TOF "
                "High double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_high_trigger_num_min_1_2",
            "IDX__TXHDRHGTRIGNMIN12",
            11,
            "TOF High Double Pulse Min Samples",
            (
                "Minimum number of samples between pulse 1 and 2 for TOF High "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_high_trigger_num_min_1",
            "IDX__TXHDRHGTRIGNMIN1",
            8,
            "TOF High Pulse 1 Min Samples",
            (
                "Minimum number of samples for pulse 1 for TOF High single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_high_trigger_num_max_1",
            "IDX__TXHDRHGTRIGNMAX1",
            8,
            "TOF High Pulse 1 Max Samples",
            (
                "Maximum number of samples for pulse 1 for TOF High single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_high_trigger_num_min_2",
            "IDX__TXHDRHGTRIGNMIN2",
            8,
            "TOF High Pulse 2 Min Samples",
            (
                "Minimum number of samples for pulse 2 for TOF High single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_high_trigger_num_max_2",
            "IDX__TXHDRHGTRIGNMAX2",
            8,
            "TOF High Pulse 2 Max Samples",
            (
                "Maximum number of samples for pulse 2 for TOF High single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_level",
            "IDX__TXHDRLGTRIGLVL",
            10,
            "TOF Low Trigger Level",
            "Trigger level for the TOF Low Channel",
            "Level",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_max_1_2",
            "IDX__TXHDRLGTRIGNMAX12",
            11,
            "TOF Low Double Pulse Max Samples",
            (
                "Maximum number of samples between pulse 1 and 2 for TOF Low "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_min_1_2",
            "IDX__TXHDRLGTRIGNMIN12",
            11,
            "TOF Low Double Pulse Min Samples",
            (
                "Minimum number of samples between pulse 1 and 2 for TOF Low "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_min_1",
            "IDX__TXHDRLGTRIGNMIN1",
            8,
            "TOF Low Pulse 1 Min Samples",
            (
                "Minimum number of samples for pulse 1 for TOF Low single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_max_1",
            "IDX__TXHDRLGTRIGNMAX1",
            8,
            "TOF Low Pulse 1 Max Samples",
            (
                "Maximum number of samples for pulse 1 for TOF Low single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_min_2",
            "IDX__TXHDRLGTRIGNMIN2",
            8,
            "TOF Low Pulse 2 Min Samples",
            (
                "Minimum number of samples for pulse 2 for TOF Low single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_low_trigger_num_max_2",
            "IDX__TXHDRLGTRIGNMAX2",
            16,
            "TOF Low Pulse 2 Max Samples",
            (
                "Maximum number of samples for pulse 2 for TOF Low single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_level",
            "IDX__TXHDRMGTRIGLVL",
            10,
            "TOF Mid Trigger Level",
            "Trigger level for the TOF Mid Channel",
            "Level",
            "# Samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_max_1_2",
            "IDX__TXHDRMGTRIGNMAX12",
            11,
            "TOF Mid Double Pulse Max Samples",
            (
                "Maximum number of samples between pulse 1 and 2 for TOF Mid "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_min_1_2",
            "IDX__TXHDRMGTRIGNMIN12",
            11,
            "TOF Mid Double Pulse Min Samples",
            (
                "Minimum number of samples between pulse 1 and 2 for TOF Mid "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_min_1",
            "IDX__TXHDRMGTRIGNMIN1",
            8,
            "TOF Mid Pulse 1 Min Samples",
            (
                "Minimum number of samples for pulse 1 for TOF Mid single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_max_1",
            "IDX__TXHDRMGTRIGNMAX1",
            8,
            "TOF Mid Pulse 1 Max Samples",
            (
                "Maximum number of samples for pulse 1 for TOF Mid single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_min_2",
            "IDX__TXHDRMGTRIGNMIN2",
            8,
            "TOF Mid Pulse 2 Min Samples",
            (
                "Minimum number of samples for pulse 2 for TOF Mid single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "tof_mid_trigger_num_max_2",
            "IDX__TXHDRMGTRIGNMAX2",
            8,
            "TOF Mid Pulse 2 Max Samples",
            (
                "Maximum number of samples for pulse 2 for TOF Mid single and "
                "double pulse triggering"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "low_sample_coincidence_mode_blocks",
            "IDX__TXHDRLSTRIGCMBLOCKS",
            3,
            "LS Coincidence Mode Blocks",
            (
                "Number of blocks coincidence window is enabled after "
                "low sample trigger"
            ),
            "# Blocks",
            "Blocks",
        ),
        TriggerDescription(
            "low_sample_trigger_polarity",
            "IDX__TXHDRLSTRIGPOL",
            1,
            "LS Trigger Polarity",
            "The trigger polarity for low sample (0 = normal, 1 = inverted) ",
            "Polarity",
            "",
        ),
        TriggerDescription(
            "low_sample_trigger_level",
            "IDX__TXHDRLSTRIGLVL",
            12,
            "LS Trigger Level",
            "Trigger level for the low sample",
            "Level",
            "",
        ),
        TriggerDescription(
            "low_sample_trigger_num_min",
            "IDX__TXHDRLSTRIGNMIN",
            8,
            "LS Trigger Min Num Samples",
            (
                "The minimum number of samples above/below the trigger level for "
                "triggering the low sample"
            ),
            "# Samples",
            "samples",
        ),
        TriggerDescription(
            "low_sample_trigger_mode",
            "IDX__TXHDRLSTRIGMODE",
            1,
            "LS Trigger Mode Enabled",
            "Low sample trigger mode (0=disabled, 1=enabled)",
            "Mode",
            "",
        ),
        TriggerDescription(
            "tof_low_trigger_mode",
            "IDX__TXHDRLSTRIGMODE",
            1,
            "TOF Low Trigger Mode Enabled",
            "TOF Low trigger mode (0=disabled, 1=enabled)",
            "Mode",
            "",
        ),
        TriggerDescription(
            "tof_mid_trigger_mode",
            "IDX__TXHDRMGTRIGMODE",
            1,
            "TOF Mid Trigger Mode Enabled",
            "TOF Mid trigger mode (0=disabled, 1=enabled)",
            "Mode",
            "",
        ),
        TriggerDescription(
            "tof_high_trigger_mode",
            "IDX__TXHDRHGTRIGMODE",
            2,
            "TOF High Trigger Mode Enabled",
            (
                "TOF High trigger mode (0=disabled, 1=threshold mode, "
                "2=single pulse mode, 3=double pulse mode)"
            ),
            "Mode",
            "",
        ),
        TriggerDescription(
            "detector_voltage",
            "IDX__TXHDRHVPSHKCH0",
            12,
            "Detector Voltage",
            (
                "Last measurement in raw dN for processor board signal: "
                "Detector Voltage"
            ),
            "Voltage",
            "dN",
        ),
        TriggerDescription(
            "sensor_voltage",
            "IDX__TXHDRHVPSHKCH1",
            12,
            "Sensor Voltage",
            (
                "Last measurement in raw dN for processor board signal: "
                "Sensor Voltage "
            ),
            "Voltage",
            "dN",
        ),
        TriggerDescription(
            "target_voltage",
            "IDX__TXHDRHVPSHKCH2",
            12,
            "Target Voltage",
            (
                "Last measurement in raw dN for processor board signal: "
                "Target Voltage"
            ),
            "Voltage",
            "dN",
        ),
        TriggerDescription(
            "reflectron_voltage",
            "IDX__TXHDRHVPSHKCH3",
            12,
            "Reflectron Voltage",
            (
                "Last measurement in raw dN for processor board signal: "
                "Reflectron Voltage"
            ),
            "Voltage",
            "dN",
        ),
        TriggerDescription(
            "rejection_voltage",
            "IDX__TXHDRHVPSHKCH4",
            12,
            "Rejection Voltage",
            (
                "Last measurement in raw dN for processor board signal: "
                "Rejection Voltage"
            ),
            "Voltage",
            "dN",
        ),
        TriggerDescription(
            "detector_current",
            "IDX__TXHDRHVPSHKCH5",
            12,
            "Detector Current",
            (
                "Last measurement in raw dN for processor board signal: "
                "Detector Current "
            ),
            "Current",
            "dN",
        ),
    ]
}


class PacketParser:
    """IDEX packet parsing class.

    Encapsulates the decom work needed to decom a daily file of IDEX data
    received from the POC.  The class is instantiated with a reference to a L0 file as
    it exists on the local file system.

    Attributes
    ----------
        data (xarray.Dataset): An object containing all of the relevant L1 data

    TODO
    ----
        * Add method to generate quicklook

    Examples
    --------
        >>> # Print out the data in a L0 file
        >>> from imap_processing.idex.idex_packet_parser import PacketParser
        >>> l0_file = "imap_processing/tests/idex/imap_idex_l0_20230725_v01-00.pkts"
        >>> l1_data = PacketParser(l0_file)
        >>> l1_data.write_l1_cdf()

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

        dust_events = {}
        with open(packet_file, "rb") as binary_data:
            packet_generator = packet_parser.generator(binary_data)
            for packet in packet_generator:
                if "IDX__SCI0TYPE" in packet.data:
                    scitype = packet.data["IDX__SCI0TYPE"].raw_value
                    event_number = packet.data["IDX__SCI0EVTNUM"].derived_value
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
                        dust_events[event_number].parse_packet(packet)
                else:
                    logger.warning(f"Unhandled packet received: {packet}")

        processed_dust_impact_list = [
            dust_event.process() for dust_event in dust_events.values()
        ]

        self.data = xr.concat(processed_dust_impact_list, dim="Epoch")
        self.data.attrs = idex_cdf_attrs.idex_l1_global_attrs.output()


class RawDustEvent:
    """Encapsulate IDEX Raw Dust Event.

    Encapsulates the work needed to convert a single dust event into a
    processed XArray Dateset object.

    Attributes
    ----------
    None

    Methods
    -------
    __init__(space_packet_parser.ParsedPacket):
        Initialize a raw dust event, with an FPGA Header Packet from IDEX.
    parse_packet(space_packet_parser.ParsedPacket):
        Parse IDEX data packets to populate bit strings.
    process():
        Generates an xarray.Dataset object after all packets are parsed
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
        self.impact_time = 0
        self._set_impact_time(header_packet)

        # The actual trigger time for the low and high sample rate in
        # microseconds since the impact time
        self.low_sample_trigger_time = 0
        self.high_sample_trigger_time = 0
        self._set_sample_trigger_times(header_packet)
        # Iterate through the trigger description dictionary and pull out the values
        self.trigger_values = {
            trigger.name: header_packet.data[trigger.packet_name].raw_value
            for trigger in trigger_description_dict.values()
        }
        logger.debug(
            f"trigger_values:\n{self.trigger_values}"
        )  # Log values here in case of error

        # Initialize the binary data received from future packets
        self.TOF_High_bits = ""
        self.TOF_Mid_bits = ""
        self.TOF_Low_bits = ""
        self.Target_Low_bits = ""
        self.Target_High_bits = ""
        self.Ion_Grid_bits = ""

    def _set_impact_time(self, packet):
        """Calculate the datetime64 from the FPGA header information.

        We are given the MET seconds, we need convert it to UTC.

        Parameters
        ----------
        packet : space_packet_parser.ParsedPacket
            The IDEX FPGA header packet

        TODO
        ----
        This conversion is temporary for now, and will need SPICE in the future.
        IDEX has set the time launch to Jan 1 2012 for calibration testing.

        """
        # Number of seconds since epoch (nominally the launch time)
        seconds_since_launch = packet.data["SHCOARSE"].derived_value
        # Number of 20 microsecond "ticks" since the last second
        num_of_20_microsecond_increments = packet.data["SHFINE"].derived_value
        # Number of microseconds since the last second
        microseconds_since_last_second = 20 * num_of_20_microsecond_increments
        # Get the datetime of Jan 1 2012 as the start date
        launch_time = np.datetime64("2012-01-01T00:00:00.000000000")

        self.impact_time = (
            launch_time
            + np.timedelta64(seconds_since_launch, "s")
            + np.timedelta64(microseconds_since_last_second, "us")
        )

    def _set_sample_trigger_times(self, packet):
        """Calculate the actual sample trigger time.

        Determines how many samples of data are included before the dust impact
        triggered the insturment.

        Parameters
        ----------
            packet : space_packet_parser.ParsedPacket
                The IDEX FPGA header packet info

        Notes
        -----
            A "sample" is one single data point.

            A "block" is ~1.969 microseconds of data collection (8/4.0625).
            The only time that a block of data matters is in this function.

            Because the low sample data are taken every 1/4.0625 microseconds,
            there are 8 samples in one block of data.

            Because the high sample data are taken every 1/260 microseconds,
            there are 512 samples in one block of High Sample data.

            The header has information about the number of blocks before triggering,
            rather than the number of samples before triggering.

        """
        # Retrieve the number of samples of high gain delay
        high_gain_delay = packet.data["IDX__TXHDRADC0IDELAY"].raw_value

        # Retrieve number of low/high sample pretrigger blocks
        num_low_sample_pretrigger_blocks = packet.data[
            "IDX__TXHDRLSPREBLOCKS"
        ].derived_value
        num_high_sample_pretrigger_blocks = packet.data[
            "IDX__TXHDRHSPREBLOCKS"
        ].derived_value

        # Calculate the low and high sample trigger times based on the high gain delay
        # and the number of high sample/low sample pretrigger blocks
        self.low_sample_trigger_time = (
            self.LOW_SAMPLE_RATE
            * (num_low_sample_pretrigger_blocks + 1)
            * self.NUMBER_SAMPLES_PER_LOW_SAMPLE_BLOCK
            - self.HIGH_SAMPLE_RATE * high_gain_delay
        )
        self.high_sample_trigger_time = (
            self.HIGH_SAMPLE_RATE
            * (num_high_sample_pretrigger_blocks + 1)
            * self.NUMBER_SAMPLES_PER_HIGH_SAMPLE_BLOCK
        )

    def _parse_high_sample_waveform(self, waveform_raw: str):
        """Process the high sample waveform.

        Parse a binary string representing a high sample waveform
        Data arrives in 32 bit chunks, divided up into:
            * 2 bits of padding
            * 3x10 bits of integer data.

        The very last 4 numbers are bad usually, so remove those
        """
        ints = []
        for i in range(0, len(waveform_raw), 32):
            # 32 bit chunks, divided up into 2, 10, 10, 10
            # skip first two bits
            ints += [
                int(waveform_raw[i + 2 : i + 12], 2),
                int(waveform_raw[i + 12 : i + 22], 2),
                int(waveform_raw[i + 22 : i + 32], 2),
            ]
        return ints[:-4]  # Remove last 4 numbers

    def _parse_low_sample_waveform(self, waveform_raw: str):
        """Process the low sample waveform.

        Parse a binary string representing a low sample waveform
        Data arrives in 32 bit chunks, divided up into:
            * 8 bits of padding
            * 2x12 bits of integer data.
        """
        ints = []
        for i in range(0, len(waveform_raw), 32):
            ints += [
                int(waveform_raw[i + 8 : i + 20], 2),
                int(waveform_raw[i + 20 : i + 32], 2),
            ]
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
        # Gather the huge number of trigger info metadata
        trigger_vars = {}
        for var, value in self.trigger_values.items():
            trigger_description = trigger_description_dict[var]
            trigger_vars[var] = xr.DataArray(
                name=var,
                data=[value],
                dims=("Epoch"),
                attrs=dataclasses.replace(
                    idex_cdf_attrs.trigger_base,
                    catdesc=trigger_description.notes,
                    fieldname=trigger_description.field,
                    var_notes=trigger_description.notes,
                    validmax=2**trigger_description.num_bits - 1,
                    label_axis=trigger_description.label,
                    units=trigger_description.units,
                ).output(),
            )

        # Process the 6 primary data variables
        tof_high_xr = xr.DataArray(
            name="TOF_High",
            data=[self._parse_high_sample_waveform(self.TOF_High_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
            attrs=idex_cdf_attrs.tof_high_attrs.output(),
        )
        tof_low_xr = xr.DataArray(
            name="TOF_Low",
            data=[self._parse_high_sample_waveform(self.TOF_Low_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
            attrs=idex_cdf_attrs.tof_low_attrs.output(),
        )
        tof_mid_xr = xr.DataArray(
            name="TOF_Mid",
            data=[self._parse_high_sample_waveform(self.TOF_Mid_bits)],
            dims=("Epoch", "Time_High_SR_dim"),
            attrs=idex_cdf_attrs.tof_mid_attrs.output(),
        )
        target_high_xr = xr.DataArray(
            name="Target_High",
            data=[self._parse_low_sample_waveform(self.Target_High_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
            attrs=idex_cdf_attrs.target_high_attrs.output(),
        )
        target_low_xr = xr.DataArray(
            name="Target_Low",
            data=[self._parse_low_sample_waveform(self.Target_Low_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
            attrs=idex_cdf_attrs.target_low_attrs.output(),
        )
        ion_grid_xr = xr.DataArray(
            name="Ion_Grid",
            data=[self._parse_low_sample_waveform(self.Ion_Grid_bits)],
            dims=("Epoch", "Time_Low_SR_dim"),
            attrs=idex_cdf_attrs.ion_grid_attrs.output(),
        )

        # Determine the 3 coordinate variables
        epoch_xr = xr.DataArray(
            name="Epoch",
            data=[self.impact_time],
            dims=("Epoch"),
            attrs=ConstantCoordinates.EPOCH,
        )

        time_low_sr_xr = xr.DataArray(
            name="Time_Low_SR",
            data=[self._calc_low_sample_resolution(len(target_low_xr[0]))],
            dims=("Epoch", "Time_Low_SR_dim"),
            attrs=idex_cdf_attrs.low_sr_attrs.output(),
        )

        time_high_sr_xr = xr.DataArray(
            name="Time_High_SR",
            data=[self._calc_high_sample_resolution(len(tof_low_xr[0]))],
            dims=("Epoch", "Time_High_SR_dim"),
            attrs=idex_cdf_attrs.high_sr_attrs.output(),
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
            }
            | trigger_vars,
            coords={
                "Epoch": epoch_xr,
                "Time_Low_SR": time_low_sr_xr,
                "Time_High_SR": time_high_sr_xr,
            },
        )
