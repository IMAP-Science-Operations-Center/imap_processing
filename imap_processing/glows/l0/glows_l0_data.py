"""Contains data classes to support GLOWS L0 processing."""

from dataclasses import dataclass, field

from imap_processing.ccsds.ccsds_data import CcsdsData


@dataclass
class GlowsL0:
    """
    Data structure for common values across histogram and direct events data.

    Attributes
    ----------
    ground_sw_version : str
        Ground software version
    packet_file_name : str
        File name of the source packet
    ccsds_header : CcsdsData
        CCSDS header data
    """

    ground_sw_version: str
    packet_file_name: str
    ccsds_header: CcsdsData


@dataclass
class HistogramL0(GlowsL0):
    """
    Data structure for storing GLOWS histogram packet data.

    Attributes
    ----------
    MET : int
        CCSDS Packet Time Stamp (coarse time)
    STARTID : int
        Histogram Start ID
    ENDID : int
        Histogram End ID
    FLAGS : int
        Histogram flags
    SWVER : int
        Version of SW used for generation
    SEC : int
        Block start time (IMAP), seconds
    SUBSEC : int
        Block start time (IMAP), subseconds
    OFFSETSEC : int
        Block end time (IMAP), seconds
    OFFSETSUBSEC : int
        Block end time (IMAP), subseconds
    GLXSEC : int
        Block start time (GLOWS), seconds
    GLXSUBSEC : int
        Block start time (GLOWS), Subseconds
    GLXOFFSEC : int
        Block end time (GLOWS), seconds
    GLXOFFSUBSEC : int
        Block end time (GLOWS), subseconds
    SPINS : int
        Number of spins
    NBINS : int
        Number of bins
    TEMPAVG : int
        Mean filter temperature
    TEMPVAR : int
        Variance of filter temperature
    HVAVG : int
        Mean CEM voltage
    HVVAR : int
        Variance of CEM voltage
    SPAVG : int
        Mean spin period
    SPVAR : int
        Variance of spin period
    ELAVG : int
        Mean length of event impulse
    ELVAR : int
        Variance of event-impulse length
    EVENTS : int
        Number of events
    HISTOGRAM_DATA : bytes
        Raw binary format histogram data
    ground_sw_version : str
        Ground software version
    packet_file_name : str
        File name of the source packet
    ccsds_header : CcsdsData
        CCSDS header data
    """

    MET: int
    STARTID: int
    ENDID: int
    FLAGS: int
    SWVER: int
    SEC: int
    SUBSEC: int
    OFFSETSEC: int
    OFFSETSUBSEC: int
    GLXSEC: int
    GLXSUBSEC: int
    GLXOFFSEC: int
    GLXOFFSUBSEC: int
    SPINS: int
    NBINS: int
    TEMPAVG: int
    TEMPVAR: int
    HVAVG: int
    HVVAR: int
    SPAVG: int
    SPVAR: int
    ELAVG: int
    ELVAR: int
    EVENTS: int
    HISTOGRAM_DATA: bytes


@dataclass
class DirectEventL0(GlowsL0):
    """
    Data structure for storing GLOWS direct event packet data.

    Attributes
    ----------
    MET : int
        CCSDS Packet Time Stamp (coarse time)
    SEC : int
        Data IMAP timestamp, seconds counter.
    LEN : int
        Number of packets in data set.
    SEQ : int
        Packet sequence in data set.
    DE_DATA : bytearray
        Raw direct event data (compressed)
    ground_sw_version : str
        Ground software version
    packet_file_name : str
        File name of the source packet
    ccsds_header : CcsdsData
        CCSDS header data

    Methods
    -------
    within_same_sequence(other)
    """

    MET: int
    SEC: int
    LEN: int
    SEQ: int
    DE_DATA: bytearray = field(repr=False)  # Do not include in print

    def __post_init__(self) -> None:
        """Convert from string to bytearray if DE_DATA is a string of ones and zeros."""
        if isinstance(self.DE_DATA, str):
            # Convert string output from space_packet_parser to bytearray
            self.DE_DATA = bytearray(
                int(self.DE_DATA, 2).to_bytes(len(self.DE_DATA) // 8, "big")
            )

    def within_same_sequence(self, other: "DirectEventL0") -> bool:
        """
        Compare fields for L0 which should be the same for packets within one sequence.

        This method compares the IMAP time (SEC) and packet length (LEN) fields.

        Parameters
        ----------
        other : DirectEventL0
            Another instance of DirectEventL0 to compare to.

        Returns
        -------
        bool
            True if the SEC and LEN fields match, False otherwise.
        """
        if not isinstance(other, DirectEventL0):
            return False

        # Time and overall packet length should match
        # TODO: What other fields need to match?
        return self.SEC == other.SEC and self.LEN == other.LEN
