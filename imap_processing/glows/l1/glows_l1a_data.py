import struct
from dataclasses import dataclass
from typing import ClassVar

from imap_processing.glows import version
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0
from imap_processing.glows.utils.constants import DirectEvent, GlowsConstants, TimeTuple


class DataEverySecond:
    """Mapping of name to byte for dataeverysecond structure in GLOWS L1A.

    This is used to generate the housekeeping info for each direct event from the
    compressed structure in the first 40 bytes of each direct event data field.

    Attributes
    ----------
    mapping: dict
        Mapping of value name to the number of bytes in the compressed data

    """

    mapping: ClassVar[dict] = {
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

    def __init__(self, level0: HistogramL0):
        """Set the attributes based on the given L0 histogram data.

        This includes generating a block header and converting the time attributes from
        HistogramL0 into TimeTuple pairs.
        """
        self.l0 = level0
        self.histograms = list(self.l0.HISTOGRAM_DATA)

        self.block_header = {
            "flight_software_version": self.l0.SWVER,
            "ground_software_version": version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }

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
    block_header: dict
    de_data: bytearray
    most_recent_seq: int
    missing_seq: list[int]
    status_data: dict = None
    direct_events: list[DirectEvent] = None

    def __init__(self, level0: DirectEventL0):
        self.l0 = level0
        self.most_recent_seq = self.l0.SEQ
        self.de_data = bytearray(level0.DE_DATA)
        self.missing_seq = []

        self.block_header = {
            "ground_software_version": version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }

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
        self.status_data = self._generate_status_data(self.de_data[:40])
        self._generate_direct_events(self.de_data[40:])

    def _generate_status_data(self, general_data_subset: bytearray):
        """Once all the packets are in the dataclass, process the dataclass.

        This sets all the values for the data_every_second attribute.

        Attributes
        ----------
        general_data_subset: bytearray
            40 bytes containing the information for general data (data_every_second)
        """
        # Copied from GLOWS code provided 11/6. Author: Marek Strumik <maro@cbk.waw.pl>
        data_every_second = dict()
        prev_byte = 0
        for item in DataEverySecond.mapping.items():
            data_every_second[item[0]] = int.from_bytes(
                general_data_subset[prev_byte : prev_byte + item[1]], "big"
            )
            prev_byte = prev_byte + item[1]

        return data_every_second
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
                raise Exception(
                    "error create_l1a_from_l0_data(): unexpected marker %d or end of "
                    "data stream %d %d" % (marker, i, len(direct_events))
                )

            # sanity check
            assert current_event.timestamp.subseconds < GlowsConstants.SUBSECOND_LIMIT

            self.direct_events.append(current_event)

        # TODO: Generate direct events from the rest of de_data

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

        if subseconds >= GlowsConstants.SUBSECOND_LIMIT:
            seconds += 1
            subseconds -= GlowsConstants.SUBSECOND_LIMIT

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
