from dataclasses import dataclass, fields

from imap_processing.ccsds.ccsds_data import CcsdsData


@dataclass
class GlowsL0:
    """Data structure for common values across histogram and direct events data.

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
    """Data structure for storing GLOWS histogram packet data.

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

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Initialize data class with a packet of histogram data.

        Parameters
        ----------
        packet
            Packet generated from space_packet_parser. Should be a type NamedTuple
            with header and data fields.
        software_version: str
            The version of the ground software being used
        packet_file_name: str
            The filename of the packet
        """
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))

        attributes = [field.name for field in fields(self)]

        # For each item in packet, assign it to the matching attribute in the class.
        for key, item in packet.data.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            if key in attributes:
                setattr(self, key, value)
            else:
                raise KeyError(
                    f"Did not find matching attribute in Histogram data class for "
                    f"{key}"
                )


@dataclass
class DirectEventL0(GlowsL0):
    """Data structure for storing GLOWS direct event packet data.

    Attributes
    ----------
    MET : int
        CCSDS Packet Time Stamp (coarse time)
    SEC : int
        Data timestamp, seconds counter.
    LEN : int
        Number of packets in data set.
    SEQ : int
        Packet sequence in data set.
    ground_sw_version : str
        Ground software version
    packet_file_name : str
        File name of the source packet
    ccsds_header : CcsdsData
        CCSDS header data
    """

    MET: int
    SEC: int
    LEN: int
    SEQ: int

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Initialize data class with a packet of direct event data.

        Parameters
        ----------
        packet
            Packet generated from space_packet_parser. Should be a type NamedTuple
            with header and data fields.
        software_version: str
            The version of the ground software being used
        packet_file_name: str
            The filename of the packet
        """
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))

        attributes = [field.name for field in fields(self)]

        # For each item in packet, assign it to the matching attribute in the class.
        for key, item in packet.data.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            if key in attributes:
                setattr(self, key, value)
            else:
                raise KeyError(
                    f"Did not find matching attribute in Direct events data class for "
                    f"{key}"
                )
