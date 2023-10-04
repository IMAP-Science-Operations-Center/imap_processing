from dataclasses import dataclass, fields


@dataclass
class HistogramL0:
    """Data structure for storing GLOWS histogram packet data.

    Parameters
    ----------
    packet : tuple[list]
        Histogram packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    SHCOARSE : int
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
    HISTOGRAM_DATA : bin
        Raw binary format histogram data
    """

    SHCOARSE: int
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
    HISTOGRAM_DATA: bin

    def __init__(self, packet):
        """Initialize data class with a packet of histogram data.

        Parameters
        ----------
        packet
            Packet generated from space_packet_parser. Should be a type NamedTuple
            with header and data fields.
        """
        # Loop through each attribute in the class
        for field in fields(self):
            key = field.name
            # Look for the key in the packet, and assign it to the attribute
            try:
                item = packet.data[key]
                value = (
                    item.derived_value
                    if item.derived_value is not None
                    else item.raw_value
                )
                setattr(self, key, value)
            except KeyError as err:
                raise KeyError(
                    f"Did not find expected field {key} in packet "
                    f"{packet.header} for histogram L0 decom."
                ) from err


@dataclass
class DirectEventL0:
    """Data structure for storing GLOWS direct event packet data.

    Parameters
    ----------
        packet : tuple[list]
            Direct event packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    SHCOARSE : int
        CCSDS Packet Time Stamp (coarse time)
    SEC : int
        Data timestamp, seconds counter.
    LEN : int
        Number of packets in data set.
    SEQ : int
        Packet sequence in data set.

    """

    SHCOARSE: int
    SEC: int
    LEN: int
    SEQ: int

    def __init__(self, packet):
        """Initialize data class with a packet of direct event data.

        Parameters
        ----------
        packet
            Packet generated from space_packet_parser. Should be a type NamedTuple
            with header and data fields.
        """
        # Loop through each attribute in the class
        for field in fields(self):
            key = field.name
            # Look for the key in the packet, and assign it to the attribute
            try:
                item = packet.data[key]
                value = (
                    item.derived_value
                    if item.derived_value is not None
                    else item.raw_value
                )
                setattr(self, key, value)
            except KeyError as err:
                raise KeyError(
                    f"Did not find expected field {key} in packet "
                    f"{packet.header} for histogram L0 decom."
                ) from err
