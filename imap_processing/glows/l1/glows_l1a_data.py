from dataclasses import dataclass

from imap_processing.glows.l0.glows_l0_data import HistogramL0


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
            "ground_software_version": self.l0.ground_sw_version,
            "pkts_file_name": self.l0.packet_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            "seq_count_in_pkts_file": self.l0.ccsds_header.SRC_SEQ_CTR,
        }

    def __post_init__(self):
        """Convert the level 0 histogram data into a usable L1A list."""
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

        if len(histograms) != 3599:
            raise ValueError(
                f"Histogram packet is lacking bins. Expected a count of 3599, "
                f"actually received {len(histograms)}"
            )

        return histograms
