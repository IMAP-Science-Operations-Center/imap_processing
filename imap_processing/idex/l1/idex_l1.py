"""
Perform IDEX l1 Processing.

This module processes decommutated IDEX packets and creates l1 data products.
"""

import logging
from collections import namedtuple
from enum import IntEnum

import numpy as np
import space_packet_parser
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.idex.l0.idex_l0 import decom_packets

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
    ["name", "packet_name"],
)
trigger_description_dict = {
    trigger.name: trigger
    for trigger in [
        TriggerDescription("event_number", "IDX__TXHDREVTNUM"),
        TriggerDescription("tof_high_trigger_level", "IDX__TXHDRHGTRIGLVL"),
        TriggerDescription("tof_high_trigger_num_max_1_2", "IDX__TXHDRHGTRIGNMAX12"),
        TriggerDescription("tof_high_trigger_num_min_1_2", "IDX__TXHDRHGTRIGNMIN12"),
        TriggerDescription("tof_high_trigger_num_min_1", "IDX__TXHDRHGTRIGNMIN1"),
        TriggerDescription("tof_high_trigger_num_max_1", "IDX__TXHDRHGTRIGNMAX1"),
        TriggerDescription("tof_high_trigger_num_min_2", "IDX__TXHDRHGTRIGNMIN2"),
        TriggerDescription("tof_high_trigger_num_max_2", "IDX__TXHDRHGTRIGNMAX2"),
        TriggerDescription("tof_low_trigger_level", "IDX__TXHDRLGTRIGLVL"),
        TriggerDescription("tof_low_trigger_num_max_1_2", "IDX__TXHDRLGTRIGNMAX12"),
        TriggerDescription("tof_low_trigger_num_min_1_2", "IDX__TXHDRLGTRIGNMIN12"),
        TriggerDescription("tof_low_trigger_num_min_1", "IDX__TXHDRLGTRIGNMIN1"),
        TriggerDescription("tof_low_trigger_num_max_1", "IDX__TXHDRLGTRIGNMAX1"),
        TriggerDescription("tof_low_trigger_num_min_2", "IDX__TXHDRLGTRIGNMIN2"),
        TriggerDescription("tof_low_trigger_num_max_2", "IDX__TXHDRLGTRIGNMAX2"),
        TriggerDescription("tof_mid_trigger_level", "IDX__TXHDRMGTRIGLVL"),
        TriggerDescription("tof_mid_trigger_num_max_1_2", "IDX__TXHDRMGTRIGNMAX12"),
        TriggerDescription("tof_mid_trigger_num_min_1_2", "IDX__TXHDRMGTRIGNMIN12"),
        TriggerDescription("tof_mid_trigger_num_min_1", "IDX__TXHDRMGTRIGNMIN1"),
        TriggerDescription("tof_mid_trigger_num_max_1", "IDX__TXHDRMGTRIGNMAX1"),
        TriggerDescription("tof_mid_trigger_num_min_2", "IDX__TXHDRMGTRIGNMIN2"),
        TriggerDescription("tof_mid_trigger_num_max_2", "IDX__TXHDRMGTRIGNMAX2"),
        TriggerDescription("low_sample_coincidence_mode_blocks", "IDX__TXHDRLSTRIGCMBLOCKS"), # noqa
        TriggerDescription("low_sample_trigger_polarity", "IDX__TXHDRLSTRIGPOL"),
        TriggerDescription("low_sample_trigger_level", "IDX__TXHDRLSTRIGLVL"),
        TriggerDescription("low_sample_trigger_num_min", "IDX__TXHDRLSTRIGNMIN"),
        TriggerDescription("low_sample_trigger_mode", "IDX__TXHDRLSTRIGMODE"),
        TriggerDescription("tof_low_trigger_mode", "IDX__TXHDRLSTRIGMODE"),
        TriggerDescription("tof_mid_trigger_mode", "IDX__TXHDRMGTRIGMODE"),
        TriggerDescription("tof_high_trigger_mode", "IDX__TXHDRHGTRIGMODE"),
        TriggerDescription("detector_voltage", "IDX__TXHDRHVPSHKCH0"),
        TriggerDescription("sensor_voltage", "IDX__TXHDRHVPSHKCH1"),
        TriggerDescription("target_voltage", "IDX__TXHDRHVPSHKCH2"),
        TriggerDescription("reflectron_voltage", "IDX__TXHDRHVPSHKCH3"),
        TriggerDescription("rejection_voltage", "IDX__TXHDRHVPSHKCH4"),
        TriggerDescription("detector_current", "IDX__TXHDRHVPSHKCH5"),
    ]
}  # fmt: skip


def get_idex_attrs(data_version: str) -> ImapCdfAttributes:
    """
    Load in CDF attributes for IDEX instrument.

    Parameters
    ----------
    data_version : str
        Data version for CDF filename, in the format "vXXX".

    Returns
    -------
    idex_attrs : ImapCdfAttributes
        Imap object with l1a attribute files loaded in.
    """
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs("idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l1")
    idex_attrs.add_global_attribute("Data_version", data_version)
    return idex_attrs


class PacketParser:
    """
    IDEX packet parsing class.

    Encapsulates the decom work needed to decom a daily file of IDEX data
    received from the POC.  The class is instantiated with a reference to a L0 file as
    it exists on the local file system.

    Parameters
    ----------
    packet_file : str
        The path and filename to the L0 file to read.
    data_version : str
        The version of the data product being created.

    Methods
    -------
    TODO : Add method to generate quicklook plots

    Examples
    --------
    .. code-block:: python

        from imap_processing.idex.idex_packet_parser import PacketParser
        l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20230725_v001.pkts"
        l1_data = PacketParser(l0_file, data_version)
        l1_data.write_l1_cdf()
    """

    def __init__(self, packet_file: str, data_version: str) -> None:
        """
        Read a l0 pkts file and perform all of the decom work.

        Parameters
        ----------
        packet_file : str
          The path and filename to the L0 file to read.
        data_version : str
            The version of the data product being created.

        Notes
        -----
            Currently assumes one L0 file will generate exactly one l1a file.
        """
        decom_packet_list = decom_packets(packet_file)

        dust_events = {}
        for packet in decom_packet_list:
            if "IDX__SCI0TYPE" in packet.data:
                scitype = packet.data["IDX__SCI0TYPE"].raw_value
                event_number = packet.data["IDX__SCI0EVTNUM"].derived_value
                if scitype == Scitype.FIRST_PACKET:
                    # Initial packet for new dust event
                    # Further packets will fill in data
                    dust_events[event_number] = RawDustEvent(packet, data_version)
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

        self.data = xr.concat(processed_dust_impact_list, dim="epoch")
        idex_attrs = get_idex_attrs(data_version)
        self.data.attrs = idex_attrs.get_global_attributes("imap_idex_l1_sci")


class RawDustEvent:
    """
    Encapsulate IDEX Raw Dust Event.

    Encapsulates the work needed to convert a single dust event into a
    processed XArray Dateset object.

    Parameters
    ----------
    header_packet : space_packet_parser.parser.Packet
        The FPGA metadata event header.
    data_version : str
            The version of the data product being created.

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

    Methods
    -------
    _set_impact_time(packet)
    _set_sample_trigger_times(packet)
    _parse_high_sample_waveform(waveform_raw)
    _parse_low_sample_waveform(waveform_raw)
    _calc_low_sample_resolution(num_samples)
    _calc_high_sample_resolution(num_samples)
    parse_packet(packet)
    process()
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

    def __init__(
        self, header_packet: space_packet_parser.parser.Packet, data_version: str
    ) -> None:
        """
        Initialize a raw dust event, with an FPGA Header Packet from IDEX.

        The values we care about are:

        self.impact_time - When the impact occurred.
        self.low_sample_trigger_time - When the low sample stuff actually triggered.
        self.high_sample_trigger_time - When the high sample stuff actually triggered.

        Parameters
        ----------
        header_packet : space_packet_parser.parser.Packet
            The FPGA metadata event header.
        data_version : str
            Data version for CDF filename, in the format "vXXX".
        """
        # Calculate the impact time in seconds since epoch
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

        self.cdf_attrs = get_idex_attrs(data_version)

    def _set_impact_time(self, packet: space_packet_parser.parser.Packet) -> None:
        """
        Calculate the datetime64 from the FPGA header information.

        We are given the MET seconds, we need convert it to UTC.

        Parameters
        ----------
        packet : space_packet_parser.parser.Packet
            The IDEX FPGA header packet.

        Notes
        -----
        Todo
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
        met = seconds_since_launch + microseconds_since_last_second * 1e-6

        self.impact_time = met_to_j2000ns(
            met, reference_epoch=np.datetime64("2012-01-01T00:00:00.000000000")
        )

    def _set_sample_trigger_times(
        self, packet: space_packet_parser.parser.Packet
    ) -> None:
        """
        Calculate the actual sample trigger time.

        Determines how many samples of data are included before the dust impact
        triggered the instrument.

        Parameters
        ----------
        packet : space_packet_parser.parser.Packet
            The IDEX FPGA header packet info.

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

    def _parse_high_sample_waveform(self, waveform_raw: str) -> list[int]:
        """
        Will process the high sample waveform.

        Parse a binary string representing a high sample waveform.
        Data arrives in 32 bit chunks, divided up into:
            * 2 bits of padding
            * 3x10 bits of integer data.

        The very last 4 numbers are bad usually, so remove those.

        Parameters
        ----------
        waveform_raw : str
            The binary string representing the high sample waveform.

        Returns
        -------
        ints : list
            List of the high sample waveform.
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

    def _parse_low_sample_waveform(self, waveform_raw: str) -> list[int]:
        """
        Will process the low sample waveform.

        Parse a binary string representing a low sample waveform
        Data arrives in 32 bit chunks, divided up into:
            * 8 bits of padding
            * 2x12 bits of integer data.

        Parameters
        ----------
        waveform_raw : str
            The binary string representing the low sample waveform.

        Returns
        -------
        ints : list
            List of processed low sample waveform.
        """
        ints = []
        for i in range(0, len(waveform_raw), 32):
            ints += [
                int(waveform_raw[i + 8 : i + 20], 2),
                int(waveform_raw[i + 20 : i + 32], 2),
            ]
        return ints

    def _calc_low_sample_resolution(self, num_samples: int) -> np.ndarray:
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
        time_low_sr_data : numpy.ndarray
            Low time sample data array.
        """
        time_low_sr_init = np.linspace(0, num_samples, num_samples)
        time_low_sr_data = (
            self.LOW_SAMPLE_RATE * time_low_sr_init - self.low_sample_trigger_time
        )
        return time_low_sr_data

    def _calc_high_sample_resolution(self, num_samples: int) -> np.ndarray:
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
        time_high_sr_data : numpy.ndarray
            High sample time data array.
        """
        time_high_sr_init = np.linspace(0, num_samples, num_samples)
        time_high_sr_data = (
            self.HIGH_SAMPLE_RATE * time_high_sr_init - self.high_sample_trigger_time
        )
        return time_high_sr_data

    def parse_packet(self, packet: space_packet_parser.parser.Packet) -> None:
        """
        Parse IDEX data packets to populate bit strings.

        Parameters
        ----------
        packet : space_packet_parser.parser.Packet
            A single science data packet for one of the 6.
            IDEX observables.
        """
        scitype = packet.data["IDX__SCI0TYPE"].raw_value
        raw_science_bits = packet.data["IDX__SCI0RAW"].raw_value
        self._append_raw_data(scitype, raw_science_bits)

    def _append_raw_data(self, scitype: Scitype, bits: str) -> None:
        """
        Append data to the appropriate bit string.

        This function determines which variable to append the bits
        to, given a specific scitype.

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

    def process(self) -> xr.Dataset:
        """
        Will process the raw data into a xarray.Dataset.

        To be called after all packets for the IDEX event have been parsed
        Parses the binary data into numpy integer arrays, and combines them
        into a xarray.Dataset object.

        Returns
        -------
        xarray.Dataset
            A Dataset object containing the data from a single impact.
        """
        # Create object for CDF attrs
        idex_attrs = self.cdf_attrs

        # Gather the huge number of trigger info metadata
        trigger_vars = {}
        for var, value in self.trigger_values.items():
            trigger_description = trigger_description_dict[var]
            trigger_vars[var] = xr.DataArray(
                name=var,
                data=[value],
                dims=("epoch"),
                attrs=idex_attrs.get_variable_attributes(trigger_description.name),
            )

        # Process the 6 primary data variables
        tof_high_xr = xr.DataArray(
            name="TOF_High",
            data=[self._parse_high_sample_waveform(self.TOF_High_bits)],
            dims=("epoch", "time_high_ssr_dim"),
            # attrs=idex_cdf_attrs.tof_high_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("tof_high_attrs"),
        )
        tof_low_xr = xr.DataArray(
            name="TOF_Low",
            data=[self._parse_high_sample_waveform(self.TOF_Low_bits)],
            dims=("epoch", "time_high_sr"),
            # attrs=idex_cdf_attrs.tof_low_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("tof_low_attrs"),
        )
        tof_mid_xr = xr.DataArray(
            name="TOF_Mid",
            data=[self._parse_high_sample_waveform(self.TOF_Mid_bits)],
            dims=("epoch", "time_high_sr"),
            # attrs=idex_cdf_attrs.tof_mid_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("tof_mid_attrs"),
        )
        target_high_xr = xr.DataArray(
            name="Target_High",
            data=[self._parse_low_sample_waveform(self.Target_High_bits)],
            dims=("epoch", "time_low_sr"),
            # attrs=idex_cdf_attrs.target_high_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("target_high_attrs"),
        )
        target_low_xr = xr.DataArray(
            name="Target_Low",
            data=[self._parse_low_sample_waveform(self.Target_Low_bits)],
            dims=("epoch", "time_low_sr"),
            # attrs=idex_cdf_attrs.target_low_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("target_low_attrs"),
        )
        ion_grid_xr = xr.DataArray(
            name="Ion_Grid",
            data=[self._parse_low_sample_waveform(self.Ion_Grid_bits)],
            dims=("epoch", "time_low_sr"),
            # attrs=idex_cdf_attrs.ion_grid_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("ion_grid_attrs"),
        )

        # Determine the 3 coordinate variables
        epoch_xr = xr.DataArray(
            name="epoch",
            data=[self.impact_time],
            dims=("epoch"),
            # attrs=ConstantCoordinates.EPOCH,
            attrs=idex_attrs.get_variable_attributes("epoch"),
        )

        time_low_sr_xr = xr.DataArray(
            name="time_low_sr",
            data=[self._calc_low_sample_resolution(len(target_low_xr[0]))],
            dims=("epoch", "time_low_sr_dim"),
            # attrs=idex_cdf_attrs.low_sr_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("low_sr_attrs"),
        )

        time_high_sr_xr = xr.DataArray(
            name="time_high_sr",
            data=[self._calc_high_sample_resolution(len(tof_low_xr[0]))],
            dims=("epoch", "time_high_sr_dim"),
            # attrs=idex_cdf_attrs.high_sr_attrs.output(),
            attrs=idex_attrs.get_variable_attributes("high_sr_attrs"),
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
                "epoch": epoch_xr,
                "time_low_sr": time_low_sr_xr,
                "time_high_sr": time_high_sr_xr,
            },
        )
