from dataclasses import dataclass
from typing import List
import struct

from imap_processing.glows import version
from imap_processing.glows.l0.glows_l0_data import DirectEventL0, HistogramL0


@dataclass(frozen=True)
class GlowsConstants:
    # subsecond limit for GLOWS clock (and consequently also onboard-interpolated IMAP
    # clock)
    SUBSECOND_LIMIT = 2_000_000

    # angular radius of IMAP/GLOWS scanning circle [deg]
    SCAN_CIRCLE_ANGULAR_RADIUS = 75.0


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
    imap_start_time: tuple[int, int]
    imap_end_time_offset: tuple[int, int]
    glows_start_time: tuple[int, int]
    glows_end_time_offset: tuple[int, int]
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
        self.imap_start_time = (self.l0.SEC, self.l0.SUBSEC)
        self.imap_end_time_offset = (self.l0.OFFSETSEC, self.l0.OFFSETSUBSEC)
        self.glows_start_time = (self.l0.GLXSEC, self.l0.GLXSUBSEC)
        self.glows_end_time_offset = (self.l0.GLXOFFSEC, self.l0.GLXOFFSUBSEC)

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
    """Data structure for GLOWS Histogram Level 1A data. This assumes that the
    multi-part DE packets are merged together into a single DirectEventL1A instance.

    This means there may be multiple DirectEventL0 packets.

    Attributes
    ----------
    l0: DirectEventL0
    header: dict
    imap_start_time_seconds: int
    packet_count: int
    seq_number: int
    de_data: bin
    """

    l0: DirectEventL0
    header: dict
    de_data: bytearray
    most_recent_seq: int
    missing_seq: List[int]
    data_every_second: dict

    def __init__(self, level0: DirectEventL0):
        self.l0 = level0
        self._set_block_header()
        self.most_recent_seq = self.l0.SEQ
        self.de_data = bytearray(level0.DE_DATA)

        if level0.LEN == 1:
            self._process_de_data()

    def __post_init__(self):
        self.missing_seq = []

    def merge_multi_event_packets(
        self, secondl0: DirectEventL0, current_seq_counter: int
    ):
        # Track any missing sequence counts
        if current_seq_counter != self.most_recent_seq + 1:
            self.missing_seq.extend(
                range(self.most_recent_seq + 1, current_seq_counter)
            )

        # Determine if new L0 packet matches existing L0 packet
        match = self.l0.sequence_match_check(secondl0)

        # TODO: Should this raise an error? Log? something else?
        if not match:
            raise ValueError(
                f"While attempting to merge L0 packet {secondl0} "
                f"into L1A packet {self.__repr__()}, mismatched values "
                f"were found. "
            )

        self.de_data.extend(bytearray(secondl0.DE_DATA))

        # if this is the last packet in the sequence, process the DE data
        # TODO: What if the last packet never arrives?
        if self.l0.LEN == current_seq_counter - 1:
            self._process_de_data()

    def _process_de_data(self):
        """

        Returns
        -------

        """
        # Copied from GLOWS code provided 11/6. Author: Marek Strumik <maro@cbk.waw.pl>
        self.data_every_second = dict()
        self.data_every_second["imap_sclk_last_pps"] = int.from_bytes(
            self.de_data[0:4], "big"
        )
        self.data_every_second["glows_sclk_last_pps"] = int.from_bytes(
            self.de_data[4:8], "big"
        )
        self.data_every_second["glows_ssclk_last_pps"] = int.from_bytes(
            self.de_data[8:12], "big"
        )
        self.data_every_second["imap_sclk_next_pps"] = int.from_bytes(
            self.de_data[12:16], "big"
        )
        self.data_every_second["catbed_heater_active"] = (
            False if int(self.de_data[16]) == 0 else True
        )
        self.data_every_second["spin_period_valid"] = (
            False if int(self.de_data[17]) == 0 else True
        )
        self.data_every_second["spin_phase_at_next_pps_valid"] = (
            False if int(self.de_data[18]) == 0 else True
        )
        self.data_every_second["spin_period_source"] = (
            False if int(self.de_data[19]) == 0 else True
        )
        self.data_every_second["spin_period"] = int.from_bytes(
            self.de_data[20:22], "big"
        )
        self.data_every_second["spin_phase_at_next_pps"] = int.from_bytes(
            self.de_data[22:24], "big"
        )
        self.data_every_second["number_of_completed_spins"] = int.from_bytes(
            self.de_data[24:28], "big"
        )
        self.data_every_second["filter_temperature"] = int.from_bytes(
            self.de_data[28:30], "big"
        )
        self.data_every_second["hv_voltage"] = int.from_bytes(
            self.de_data[30:32], "big"
        )
        self.data_every_second["glows_time_on_pps_valid"] = (
            False if int(self.de_data[32]) == 0 else True
        )
        self.data_every_second["time_status_valid"] = (
            False if int(self.de_data[33]) == 0 else True
        )
        self.data_every_second["housekeeping_valid"] = (
            False if int(self.de_data[34]) == 0 else True
        )
        self.data_every_second["is_pps_autogenerated"] = (
            False if int(self.de_data[35]) == 0 else True
        )
        self.data_every_second["hv_test_in_progress"] = (
            False if int(self.de_data[36]) == 0 else True
        )
        self.data_every_second["pulse_test_in_progress"] = (
            False if int(self.de_data[37]) == 0 else True
        )
        self.data_every_second["memory_error_detected"] = (
            False if int(self.de_data[38]) == 0 else True
        )

    def _set_block_header(self):
        """Create the block header using software version info."""
        self.block_header = {
            "ground_software_version": version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }


@dataclass(frozen=True)
class DirectEvent:
    """
    DirectEvent() class for IMAP/GLOWS.

    Author: Marek Strumik, maro@cbk.waw.pl
    """

    seconds: int
    subseconds: int
    impulse_length: int
    multi_event: bool = False

    def build_event_from_uncompressed_data(raw) -> "DirectEvent":
        """
        Build direct event from raw binary 8-byte array assuming that it contains uncompressed timestamps
        """
        assert len(raw) == 8
        values = struct.unpack(">II", raw)
        return DirectEvent(
            seconds=values[0],
            subseconds=values[1]
            & 0x1FFFFF,  # subsecond encoding on the least significant 21 bits
            impulse_length=(values[1] >> 24)
            & 0xFF,  # first byte encodes the impulse length
            multi_event=bool(
                (values[1] >> 23) & 0b1
            ),  # KPLabs says it is set by FPGA and currently not used by AppSW at all
        )

    def build_event_from_compressed_data(
        diff: int, length: int, current_event: "DirectEvent"
    ) -> "DirectEvent":
        """
        Build direct event assuming that it contains timestamps compressed as timedeltas
        """
        subseconds = current_event.subseconds + diff
        seconds = current_event.seconds

        if subseconds >= GlowsConstants.SUBSECOND_LIMIT:
            seconds += 1
            subseconds -= GlowsConstants.SUBSECOND_LIMIT

        return DirectEvent(
            seconds=seconds,
            subseconds=subseconds,
            impulse_length=length,
            multi_event=False,
        )