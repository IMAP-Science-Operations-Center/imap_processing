class HistogramL0:
    """Data structure for storing GLOWS histogram packet data.

    Parameters
    ----------
    packet : tuple[list]
        Histogram packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    packet_keys : tuple[str]
        Data names from the packet
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
    HISTOGRAM_DATA : int
        List of histogram data values
    """

    def __repr__(self):
        """Print the data.

        Returns
        -------
        String representation of GlowsHistL0
        """
        output = "{"
        for key in self.packet_keys:
            output += f"{key}: {getattr(self, key)}, "
        return output + "}"

    def __init__(self, packet):
        self.packet_keys = []
        for key, value in packet.data.items():
            if key != "HISTOGRAM_DATA":
                setattr(self, key, value.derived_value)
            else:
                setattr(self, key, self._convert_histogram_data(value.raw_value))

            self.packet_keys.append(key)

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


class DirectEventL0:
    """Data structure for storing GLOWS direct event packet data.

    Parameters
    ----------
        packet : tuple[list]
            Direct event packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    packet_keys : tuple[str]
        Data names from the packet
    SHCOARSE : int
        CCSDS Packet Time Stamp (coarse time)
    SEC : int
        Data timestamp, seconds counter.
    LEN : int
        Number of packets in data set.
    SEQ : int
        Packet sequence in data set.

    """

    def __repr__(self):
        """Print the data.

        Returns
        -------
        String representation of GlowsDeL0
        """
        output = "{"
        for key in self.packet_keys:
            output += f"{key}: {getattr(self, key)}, "
        return output + "}"

    def __init__(self, packet):
        self.packet_keys = []
        for key, value in packet.data.items():
            setattr(self, key, value.derived_value)
            self.packet_keys.append(key)
