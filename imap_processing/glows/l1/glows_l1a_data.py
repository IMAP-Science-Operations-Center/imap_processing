import struct
from collections import namedtuple
from dataclasses import dataclass

from imap_processing.glows import version
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0

# Spacecraft clock time, a float divided into seconds and subseconds
TimeTuple = namedtuple("TimeTuple", "seconds subseconds")


@dataclass(frozen=True)
class GlowsConstants:
    """
    Constants for GLOWS which can be used across different levels or classes.

    Attributes
    ----------
    SUBSECOND_LIMIT: int
        subsecond limit for GLOWS clock (and consequently also onboard-interpolated
        IMAP clock)
    SCAN_CIRCLE_ANGULAR_RADIUS: float
        angular radius of IMAP/GLOWS scanning circle [deg]
    """

    SUBSECOND_LIMIT: int = 2_000_000
    SCAN_CIRCLE_ANGULAR_RADIUS: float = 75.0


@dataclass
class DirectEvent:
    """
    DirectEvent() class for IMAP/GLOWS.

    Authors: Marek Strumik, maro@cbk.waw.pl, Maxine Hartnett
    """

    timestamp: TimeTuple
    impulse_length: int
    multi_event: bool = False


@dataclass
class HistogramL1A:
    """Data structure for GLOWS Histogram Level 1A data.

    Attributes
    ----------
    l0: HistogramL0
        HistogramL0 Data class containing the raw data from the histogram packet
    histograms: list[int]
        List of histogram data values
    block_header: dict
        Header for L1A
    last_spin_id: int
        ID of the last spin in block (computed with start spin and offset)
    imap_start_time: tuple[int, int]
        IMAP start time for the block, in the form (seconds, subseconds)
    imap_end_time_offset: tuple[int, int]
        IMAP end time offset for the block, in the form (seconds, subseconds)
    glows_start_time: tuple[int, int]
        GLOWS start time for the block, in the form (seconds, subseconds)
    glows_end_time_offset: tuple[int, int]
        GLOWS end time offset for the block, in the form (seconds, subseconds)
    flags: dict
        Dictionary containing "flags_set_onboard" from L0, and "is_generated_on_ground",
        which is set to "False" for decommed packets.
    """

    l0: HistogramL0
    histograms: list[int]
    block_header: dict
    last_spin_id: int
    imap_start_time: TimeTuple
    imap_end_time_offset: TimeTuple
    glows_start_time: TimeTuple
    glows_end_time_offset: TimeTuple
    flags: dict

    def _set_l1a_data(self):
        """Set all the Level 1a attributes which are different from level 0.

        This sets all the attributes in the class except "l0", "histograms", and
         "block_header"
        """
        # use start ID and offset to calculate the last spin ID in the block
        self.last_spin_id = self.l0.STARTID + self.l0.ENDID

        # TODO: This sanity check should probably exist in the final code. However,
        # the emulator code does not properly set these values.
        # if self.l0.ENDID != self.l0.SPINS:
        #     raise ValueError(f"Inconsistency between L0 spin-numbering field ENDID "
        #                      f"[{self.l0.ENDID}] and histogram parameter field SPINS "
        #                      f"[{self.l0.SPINS}]")

        # Create time tuples based on second and subsecond pairs
        self.imap_start_time = TimeTuple(self.l0.SEC, self.l0.SUBSEC)
        self.imap_end_time_offset = TimeTuple(self.l0.OFFSETSEC, self.l0.OFFSETSUBSEC)
        self.glows_start_time = TimeTuple(self.l0.GLXSEC, self.l0.GLXSUBSEC)
        self.glows_end_time_offset = TimeTuple(self.l0.GLXOFFSEC, self.l0.GLXOFFSUBSEC)

        # Flags
        self.flags = {
            "flags_set_onboard": self.l0.FLAGS,
            "is_generated_on_ground": False,
        }

    def _set_block_header(self):
        """Create the block header using software version info."""
        self.block_header = {
            "flight_software_version": self.l0.SWVER,
            "ground_software_version": version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }

    def __init__(self, level0: HistogramL0):
        """Convert the level 0 histogram data into a usable L1A list."""
        self.l0 = level0
        self.histograms = self._convert_histogram_data(self.l0.HISTOGRAM_DATA)
        self._set_l1a_data()
        self._set_block_header()

    def _convert_histogram_data(self, binary_hist_data: str) -> list[int]:
        """Convert the raw histogram data into a list.

        This method converts a binary number into a list of histogram values by
        splitting up the raw binary value into 8-bit segments.

        Parameters
        ----------
        binary_hist_data : str
            Raw data read from the packet, in binary format.

        Returns
        -------
        histograms: list[int]
            List of binned histogram data
        """
        # Convert the histogram data from a large raw string into a list of 8 bit values
        histograms = []
        for i in range(8, len(binary_hist_data), 8):
            histograms.append(int(binary_hist_data[i - 8 : i], 2))

        if len(histograms) != self.l0.EVENTS:
            raise ValueError(
                f"Histogram packet is lacking bins. Expected a count of "
                f"{self.l0.EVENTS}, actually received {len(histograms)}"
            )

        return histograms


@dataclass
class DirectEventL1A:
    """Data structure for GLOWS Histogram Level 1A data.

    This includes steps for merging multiple Direct Event packets into one class,
    so this class may span multiple packets. This is determined by the SEQ and LEN,
    by each packet having an incremental SEQ until LEN number of packets.

    Attributes
    ----------
    l0: DirectEventL0
    header: dict
    de_data: bin
    """

    l0: DirectEventL0
    header: dict
    de_data: bytearray
    most_recent_seq: int
    missing_seq: list[int]
    general_data: dict
    direct_events: list[DirectEvent]

    def __init__(self, level0: DirectEventL0):
        self.l0 = level0
        self.most_recent_seq = self.l0.SEQ
        self.de_data = bytearray(level0.DE_DATA)

        self.block_header = {
            "ground_software_version": version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }

        if level0.LEN == 1:
            self._process_de_data()

    def __post_init__(self):
        """Initialize mutable attribute."""
        self.missing_seq = []

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
        match = self.l0.sequence_match_check(second_l0)

        # TODO: Should this raise an error? Log? something else?
        if not match:
            raise ValueError(
                f"While attempting to merge L0 packet {second_l0} "
                f"into L1A packet {self.__repr__()}, mismatched values "
                f"were found. "
            )

        self.de_data.extend(bytearray(second_l0.DE_DATA))

        self.most_recent_seq = second_l0.SEQ
        # if this is the last packet in the sequence, process the DE data
        # TODO: What if the last packet never arrives?
        if self.l0.LEN == self.most_recent_seq + 1:
            self._process_de_data()

    def _process_de_data(self):
        self._generate_general_data(self.de_data[:40])
        self._generate_direct_events(self.de_data[40:])

    def _generate_general_data(self, general_data_subset: bytearray):
        """Once all the packets are in the dataclass, process the dataclass.

        This sets all the values for the data_every_second attribute.

        Attributes
        ----------
        general_data_subset: bytearray
            40 bytes containing the information for general data (data_every_second)
        """
        # Copied from GLOWS code provided 11/6. Author: Marek Strumik <maro@cbk.waw.pl>
        self.general_data = dict()
        self.general_data["imap_sclk_last_pps"] = int.from_bytes(
            general_data_subset[0:4], "big"
        )
        self.general_data["glows_sclk_last_pps"] = int.from_bytes(
            general_data_subset[4:8], "big"
        )
        self.general_data["glows_ssclk_last_pps"] = int.from_bytes(
            general_data_subset[8:12], "big"
        )
        self.general_data["imap_sclk_next_pps"] = int.from_bytes(
            general_data_subset[12:16], "big"
        )
        self.general_data["catbed_heater_active"] = bool(general_data_subset[16])
        self.general_data["spin_period_valid"] = bool(general_data_subset[17])
        self.general_data["spin_phase_at_next_pps_valid"] = bool(
            general_data_subset[18]
        )
        self.general_data["spin_period_source"] = bool(general_data_subset[19])
        self.general_data["spin_period"] = int.from_bytes(
            general_data_subset[20:22], "big"
        )
        self.general_data["spin_phase_at_next_pps"] = int.from_bytes(
            general_data_subset[22:24], "big"
        )
        self.general_data["number_of_completed_spins"] = int.from_bytes(
            general_data_subset[24:28], "big"
        )
        self.general_data["filter_temperature"] = int.from_bytes(
            general_data_subset[28:30], "big"
        )
        self.general_data["hv_voltage"] = int.from_bytes(
            general_data_subset[30:32], "big"
        )
        self.general_data["glows_time_on_pps_valid"] = bool(general_data_subset[32])
        self.general_data["time_status_valid"] = bool(general_data_subset[33])
        self.general_data["housekeeping_valid"] = bool(general_data_subset[34])
        self.general_data["is_pps_autogenerated"] = bool(general_data_subset[35])
        self.general_data["hv_test_in_progress"] = bool(general_data_subset[36])
        self.general_data["pulse_test_in_progress"] = bool(general_data_subset[37])
        self.general_data["memory_error_detected"] = bool(general_data_subset[38])

        # data every second has padding, so it is exactly 40 bytes

    def _generate_direct_events(self, direct_events: bytearray):
        """Generate the list of direct events from the raw bytearray."""
        # read the first direct event, which is always uncompressed
        current_event = self._build_uncompressed_event(direct_events[:8])
        self.direct_events = [current_event]

        i = 8
        while i < len(direct_events) - 1:
            first_byte = int(direct_events[i])
            i += 1
            oldest_diff = first_byte & 0x3F
            marker = first_byte >> 6
            previous_time = current_event.timestamp

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
                rest = int(direct_events[i])
                i += 1
                diff = oldest_diff << 8 | rest
                length = int(direct_events[i])
                i += 1
                current_event = self._build_compressed_event(
                    diff, length, previous_time
                )

            elif (
                marker == 0x3 and i < len(direct_events) - 2
            ):  # 3-byte compression of timedelta
                rest = int.from_bytes(direct_events[i : i + 2], "big")
                i += 2
                diff = oldest_diff << 16 | rest
                length = int(direct_events[i])
                i += 1
                current_event = self._build_compressed_event(
                    diff, length, previous_time
                )

            else:  # wrong-marker or hitting-the-buffer-end case
                raise Exception(
                    "error create_l1a_from_l0_data(): unexpected marker %d or end of "
                    "data stream %d %d" % (marker, i, len(direct_events))
                )

            # sanity check
            assert current_event.timestamp.subseconds < GlowsConstants.SUBSECOND_LIMIT

            self.direct_events.append(current_event)

        # TODO: Generate direct events from the rest of de_data

    def _build_compressed_event(
        self, diff: int, impulse_length: int, previous_time: TimeTuple
    ) -> "DirectEvent":
        """Build direct event from data with timestamps compressed as timedeltas.

        This process requires adding onto a previous timestamp to create a new
        timestamp.

        Parameters
        ----------
        diff: int
            offset for the timestamp in subseconds
        impulse_length: int
            the impulse length
        previous_time: TimeTuple
            The previous timestamp to build off of

        Returns
        -------
        DirectEvent built by the input data

        """
        subseconds = previous_time.subseconds + diff
        seconds = previous_time.seconds

        if subseconds >= GlowsConstants.SUBSECOND_LIMIT:
            seconds += 1
            subseconds -= GlowsConstants.SUBSECOND_LIMIT

        return DirectEvent(TimeTuple(seconds, subseconds), impulse_length, False)

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
        assert len(raw) == 8
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
