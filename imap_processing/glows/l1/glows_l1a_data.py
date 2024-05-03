"""Contains data classes to support GLOWS L1A processing."""

import struct
from dataclasses import InitVar, dataclass, field

from imap_processing.glows import __version__
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0
from imap_processing.glows.utils.constants import DirectEvent, TimeTuple


@dataclass
class StatusData:
    """Data structure for GLOWS status data, also known as "data_every_second".

    This is used to generate the housekeeping info for each direct event from the
    compressed structure in the first 40 bytes of each direct event data field.

    Each DirectEventL1A instance covers one second of direct events data. Each second
    has metadata associated with it, which is described in this class. The first 40
    bytes of each direct event grouping is used to create this class. A second of
    direct events data may span multiple packets, but each second only has one set of
    StatusData attributes.

    Attributes must match byte_attribute_mapping in generate_status_data.

    Attributes
    ----------
    imap_sclk_last_pps: int
        IMAP seconds for last PPS
    glows_sclk_last_pps: int
        GLOWS seconds for last PPS
    glows_ssclk_last_pps: int
        GLOWS subseconds for last PPS
    imap_sclk_next_pps: int
        IMAP seconds for next PPS
    catbed_heater_active: int
        Flag - heater active
    spin_period_valid: int
        Flag - spin phase valid
    spin_phase_at_next_pps_valid: int
        Flag - spin phase at next PPS valid
    spin_period_source: int
        Flag - Spin period source
    spin_period: int
        Uint encoded spin period value
    spin_phase_at_next_pps: int
        Uint encoded next spin phase value
    number_of_completed_spins: int
        Number of spins, from onboard
    filter_temperature: int
        Uint encoded temperature
    hv_voltage: int
        Uint encoded voltage
    glows_time_on_pps_valid: int
        Flag - is glows time valid
    time_status_valid: int
        Flag - valid time status
    housekeeping_valid: int
        Flag - valid housekeeping
    is_pps_autogenerated: int
        Flag
    hv_test_in_progress: int
        Flag
    pulse_test_in_progress: int
        Flag
    memory_error_detected: int
        Flag
    """

    imap_sclk_last_pps: int
    glows_sclk_last_pps: int
    glows_ssclk_last_pps: int
    imap_sclk_next_pps: int
    catbed_heater_active: int
    spin_period_valid: int
    spin_phase_at_next_pps_valid: int
    spin_period_source: int
    spin_period: int
    spin_phase_at_next_pps: int
    number_of_completed_spins: int
    filter_temperature: int
    hv_voltage: int
    glows_time_on_pps_valid: int
    time_status_valid: int
    housekeeping_valid: int
    is_pps_autogenerated: int
    hv_test_in_progress: int
    pulse_test_in_progress: int
    memory_error_detected: int

    def __init__(self, general_data_subset: bytearray):
        """Generate the flag and encoded information from 40 bytes of direct event data.

        The 40 bytes also includes one extra byte of padding at the end.

        Parameters
        ----------
        general_data_subset: bytearray
            40 bytes containing the information for general data (data_every_second).
        """
        byte_attribute_mapping = {
            "imap_sclk_last_pps": 4,
            "glows_sclk_last_pps": 4,
            "glows_ssclk_last_pps": 4,
            "imap_sclk_next_pps": 4,
            "catbed_heater_active": 1,
            "spin_period_valid": 1,
            "spin_phase_at_next_pps_valid": 1,
            "spin_period_source": 1,
            "spin_period": 2,
            "spin_phase_at_next_pps": 2,
            "number_of_completed_spins": 4,
            "filter_temperature": 2,
            "hv_voltage": 2,
            "glows_time_on_pps_valid": 1,
            "time_status_valid": 1,
            "housekeeping_valid": 1,
            "is_pps_autogenerated": 1,
            "hv_test_in_progress": 1,
            "pulse_test_in_progress": 1,
            "memory_error_detected": 1,
        }

        prev_byte = 0

        for item in byte_attribute_mapping.items():
            self.__setattr__(
                item[0],
                int.from_bytes(
                    general_data_subset[prev_byte : prev_byte + item[1]], "big"
                ),
            )
            prev_byte = prev_byte + item[1]


@dataclass
class HistogramL1A:
    """Data structure for GLOWS Histogram Level 1A data.

    Attributes
    ----------
    l0: InitVar[HistogramL0]
        HistogramL0 Data class containing the raw data from the histogram packet. This
        is only used to create the class and cannot be accessed from an instance.
    histograms: list[int]
        List of histogram data values
    flight_software_version: int
        Version of the flight software used to generate the data. Part of block header.
    ground_software_version: str
        Version of the ground software used to process the data. Part of block header.
    pkts_file_name: str
        Name of the packet file used to generate the data. Part of block header.
    seq_count_in_pkts_file: int
        Sequence count in the packet file, equal to SRC_SEQ_CTR Part of block header.
    last_spin_id: int
        ID of the last spin in block (computed with start spin and offset)
    imap_start_time: tuple[int, int]
        IMAP start time for the block, in the form (seconds, subseconds)
    imap_time_offset: tuple[int, int]
        IMAP end time offset for the block, in the form (seconds, subseconds). In
        algorithm document as "imap_end_time_offset"
    glows_start_time: tuple[int, int]
        GLOWS start time for the block, in the form (seconds, subseconds)
    glows_time_offset: tuple[int, int]
        GLOWS end time offset for the block, in the form (seconds, subseconds). In
        algorithm document as "glows_end_time_offset"
    number_of_spins_per_block: int
        Number of spins in the block, from L0.SPINS
    number_of_bins_per_histogram: int
        Number of bins in the histogram, from L0.NBINS
    number_of_events: int
        Number of events in the block, from L0.EVENTS
    filter_temperature_average: int
        Average filter temperature in the block, from L0.TEMPAVG. Uint encoded.
    filter_temperature_variance: int
        Variance of filter temperature in the block, from L0.TEMPVAR. Uint encoded.
    hv_voltage_average: int
        Average HV voltage in the block, from L0.HVAVG. Uint encoded.
    hv_voltage_variance: int
        Variance of HV voltage in the block, from L0.HVVAR. Uint encoded.
    spin_period_average: int
        Average spin period in the block, from L0.SPAVG. Uint encoded.
    spin_period_variance: int
        Variance of spin period in the block, from L0.SPAVG. Uint encoded.
    pulse_length_average: int
        Average pulse length in the block, from L0.ELAVG. Uint encoded.
    pulse_length_variance: int
        Variance of pulse length in the block, from L0.ELVAR. Uint encoded.
    flags: dict
        Dictionary containing "flags_set_onboard" from L0, and "is_generated_on_ground",
        which is set to "False" for decommed packets.
    """

    l0: InitVar[HistogramL0]
    histograms: list[int] = None
    # next four are in block header
    flight_software_version: int = None
    ground_software_version: str = None
    pkts_file_name: str = None
    seq_count_in_pkts_file: int = None
    last_spin_id: int = None
    imap_start_time: TimeTuple = None
    imap_time_offset: TimeTuple = None
    glows_start_time: TimeTuple = None
    glows_time_offset: TimeTuple = None
    # Following variables are copied from L0
    number_of_spins_per_block: int = None
    number_of_bins_per_histogram: int = None
    number_of_events: int = None
    filter_temperature_average: int = None
    filter_temperature_variance: int = None
    hv_voltage_average: int = None
    hv_voltage_variance: int = None
    spin_period_average: int = None
    spin_period_variance: int = None
    pulse_length_average: int = None
    pulse_length_variance: int = None
    flags: dict = None

    def __post_init__(self, l0: HistogramL0):
        """Set the attributes based on the given L0 histogram data.

        This includes generating a block header and converting the time attributes from
        HistogramL0 into TimeTuple pairs.
        """
        self.histograms = list(l0.HISTOGRAM_DATA)

        self.flight_software_version = l0.SWVER
        self.ground_software_version = __version__
        self.pkts_file_name = l0.packet_file_name
        # note: packet number is seq_count (per apid!) field in CCSDS header
        self.seq_count_in_pkts_file = l0.ccsds_header.SRC_SEQ_CTR

        # use start ID and offset to calculate the last spin ID in the block
        self.last_spin_id = l0.STARTID + l0.ENDID

        # TODO: This sanity check should probably exist in the final code. However,
        # the emulator code does not properly set these values.
        # if self.l0.ENDID != self.l0.SPINS:
        #     raise ValueError(f"Inconsistency between L0 spin-numbering field ENDID "
        #                      f"[{self.l0.ENDID}] and histogram parameter field SPINS "
        #                      f"[{self.l0.SPINS}]")

        # Create time tuples based on second and subsecond pairs
        self.imap_start_time = TimeTuple(l0.SEC, l0.SUBSEC)
        self.imap_time_offset = TimeTuple(l0.OFFSETSEC, l0.OFFSETSUBSEC)
        self.glows_start_time = TimeTuple(l0.GLXSEC, l0.GLXSUBSEC)
        self.glows_time_offset = TimeTuple(l0.GLXOFFSEC, l0.GLXOFFSUBSEC)

        # In L1a, these are left as unit encoded values.
        self.number_of_spins_per_block = l0.SPINS
        self.number_of_bins_per_histogram = l0.NBINS
        self.number_of_events = l0.EVENTS
        self.filter_temperature_average = l0.TEMPAVG
        self.filter_temperature_variance = l0.TEMPVAR
        self.hv_voltage_average = l0.HVAVG
        self.hv_voltage_variance = l0.HVVAR
        self.spin_period_average = l0.SPAVG
        self.spin_period_variance = l0.SPAVG
        self.pulse_length_average = l0.ELAVG
        self.pulse_length_variance = l0.ELVAR

        # Flags
        self.flags = {
            "flags_set_onboard": l0.FLAGS,
            "is_generated_on_ground": False,
        }


@dataclass
class DirectEventL1A:
    """Data structure for GLOWS Histogram Level 1A data.

    This includes steps for merging multiple Direct Event packets into one class,
    so this class may span multiple packets. This is determined by the SEQ and LEN,
    by each packet having an incremental SEQ until LEN number of packets.

    Block header information is retrieved from l0:
    {
        "flight_software_version" = l0.ccsds_header.VERSION
        "ground_software_version" = __version__
        "pkts_file_name" = l0.packet_file_name
        "seq_count_in_pkts_file" = l0.ccsds_header.SRC_SEQ_CTR
    }

    Attributes
    ----------
    l0: DirectEventL0
        Level 0 data. In the case of multiple L0 direct events, this is the first L0
        data class in the sequence. This is used to verify all events in the sequence
        match.
    de_data: bytearray
        Bytearray of raw DirectEvent data, which is converted into direct_events
    most_recent_seq: int
        The most recent sequence added to the L1A dataclass - for counting gaps
    missing_seq: list[int]
        Any missing sequence counts in the data. Should be an empty array in normal
        operation
    status_data: StatusData
        StatusData generated from the first 40 bytes of direct events data. This
        includes information on flags and ancillary housekeeping info from the
        spacecraft.
    direct_events: list[DirectEvent]
        List of DirectEvent objects, which is created when the final level 0 packet in
        the sequence is added to de_data. Defaults to None.

    Methods
    -------
    append
        Add another Level0 instance
    """

    l0: DirectEventL0
    de_data: bytearray = field(repr=False)  # Do not include in prints
    most_recent_seq: int
    missing_seq: list[int]
    status_data: StatusData = None
    direct_events: list[DirectEvent] = None

    def __init__(self, level0: DirectEventL0):
        self.l0 = level0
        self.most_recent_seq = self.l0.SEQ
        self.de_data = bytearray(level0.DE_DATA)
        self.missing_seq = []

        if level0.LEN == 1:
            self._process_de_data()

    def append(self, second_l0: DirectEventL0):
        """Merge an additional direct event packet to this DirectEventL1A class.

        Direct event data can span multiple packets, as marked by the SEQ and LEN
        attributes. This method will add the next piece of data in the sequence
        to this data class. The two packets are compared with
        DirectEventL0.sequence_match_check. If they don't match, the method throws
        a ValueError.

        If the sequence is broken, the missing sequence numbers are added to
        missing_seq. Once the last value in the sequence is reached, the data is
        processed from raw bytes to useful information.

        Parameters
        ----------
        second_l0: DirectEventL0
            Additional L0 packet to add to the DirectEventL1A class
        """
        # if SEQ is missing or if the sequence is out of order, do not continue.
        if not second_l0.SEQ or second_l0.SEQ < self.most_recent_seq:
            raise ValueError(
                f"Sequence for direct event L1A is out of order or "
                f"incorrect. Attempted to append sequence counter "
                f"{second_l0.SEQ} after {self.most_recent_seq}."
            )

        # Track any missing sequence counts
        if second_l0.SEQ != self.most_recent_seq + 1:
            self.missing_seq.extend(range(self.most_recent_seq + 1, second_l0.SEQ))

        # Determine if new L0 packet matches existing L0 packet
        match = self.l0.within_same_sequence(second_l0)

        # TODO: Should this raise an error? Log? something else?
        if not match:
            raise ValueError(
                f"While attempting to merge L0 packet {second_l0} "
                f"with {self.l0} mismatched values"
                f"were found. "
            )

        self.de_data.extend(bytearray(second_l0.DE_DATA))

        self.most_recent_seq = second_l0.SEQ
        # if this is the last packet in the sequence, process the DE data
        # TODO: What if the last packet never arrives?
        if self.l0.LEN == self.most_recent_seq + 1:
            self._process_de_data()

    def _process_de_data(self):
        """
        Process direct event bytes.

        Once the packets are complete, create the status data table from the first 40
        bytes in de_data, and the direct events from the remaining bytes.
        """
        self.status_data = StatusData(self.de_data[:40])
        self.direct_events = self._generate_direct_events(self.de_data[40:])

    def _generate_direct_events(self, direct_events: bytearray):
        """Generate the list of direct events from the raw bytearray.

        First, the starting timestamp is created from the first 8 bytes in the direct
        event array. Then, the remaining events are processed based on a marker in the
        first two bits of each section. If the marker is 0, it is uncompressed, and
        the event is processed from the following 7 bytes. If it is 1, it is compressed
        to two bytes, and if it is 2, the direct event is compressed to 3 bytes.

        Parameters
        ----------
        direct_events: bytearray
            bytearray containing direct event data

        Returns
        -------
        processed_events: list[DirectEvent]
            An array containing DirectEvent objects

        """
        # read the first direct event, which is always uncompressed
        current_event = self._build_uncompressed_event(direct_events[:8])
        processed_events = [current_event]

        i = 8
        while i < len(direct_events) - 1:
            first_byte = int(direct_events[i])
            i += 1
            # Remove first two bits, which are used to mark compression
            oldest_diff = first_byte & 0x3F
            marker = first_byte >> 6

            if (
                marker == 0x0 and i < len(direct_events) - 7
            ):  # uncompressed time stamp (8-bytes)
                rest_bytes = direct_events[i : i + 7]
                i += 7
                part = bytearray([oldest_diff])
                part.extend(rest_bytes)
                current_event = self._build_uncompressed_event(part)

            elif (
                marker == 0x2 and i < len(direct_events) - 1
            ):  # 2-byte compression of timedelta
                current_event = self._build_compressed_event(
                    direct_events[i : i + 2], oldest_diff, current_event.timestamp
                )
                i += 2

            elif (
                marker == 0x3 and i < len(direct_events) - 2
            ):  # 3-byte compression of timedelta
                current_event = self._build_compressed_event(
                    direct_events[i : i + 3], oldest_diff, current_event.timestamp
                )
                i += 3

            else:  # wrong-marker or hitting-the-buffer-end case
                raise IndexError(
                    f"Error: Unexpected marker {marker} or out of bounds index {i} for "
                    f"direct events list of length {len(direct_events)}. Unable to "
                    f"process direct events."
                )

            processed_events.append(current_event)

        return processed_events

    def _build_compressed_event(
        self, raw: bytearray, oldest_diff: int, previous_time: TimeTuple
    ) -> "DirectEvent":
        """Build direct event from data with timestamps compressed as timedeltas.

        This process requires adding onto a previous timestamp to create a new
        timestamp. If raw is three bytes, the three byte method of compression is used,
        if raw is two bytes, then the two byte method is used. Any other length raises
        a ValueError.

        Parameters
        ----------
        raw: bytearray
            Raw 2 or 3 byte compressed data to process
        oldest_diff: int
            last 6 bits of the byte immediately before raw
        previous_time: TimeTuple
            The previous timestamp to build off of

        Returns
        -------
        DirectEvent built by the input data

        """
        if len(raw) == 2:
            rest = raw[0]

            diff = oldest_diff << 8 | rest
            length = raw[1]

        elif len(raw) == 3:
            rest = int.from_bytes(raw[0:2], "big")
            diff = oldest_diff << 16 | rest
            length = int(raw[2])

        else:
            raise ValueError(
                f"Incorrect length {len(raw)} for {raw}, expecting 2 or 3"
                f"bit compressed direct event data"
            )

        subseconds = previous_time.subseconds + diff
        seconds = previous_time.seconds

        return DirectEvent(TimeTuple(seconds, subseconds), length, False)

    def _build_uncompressed_event(self, raw: bytearray) -> DirectEvent:
        """Build direct event from raw binary 8-byte array.

        This method assumes that the raw binary contains uncompressed timestamps.

        Parameters
        ----------
        raw: bytearray
            8 bytes of data to build the event with

        Returns
        -------
        DirectEvent object built from raw
        """
        if len(raw) != 8:
            raise ValueError(
                f"Incorrect length {len(raw)} for {raw}, expecting 8 bytes of "
                f"uncompressed direct event data"
            )

        values = struct.unpack(">II", raw)

        seconds = values[0]

        # subsecond encoding on the least significant 21 bits
        subseconds = values[1] & 0x1FFFFF

        timestamp = TimeTuple(seconds, subseconds)
        # first byte encodes the impulse length
        impulse_length = (values[1] >> 24) & 0xFF

        # KPLabs says it is set by FPGA and currently not used by AppSW at all
        multi_event = bool((values[1] >> 23) & 0b1)
        return DirectEvent(timestamp, impulse_length, multi_event)
