class HistogramL0:
    """Data structure for storing GLOWS histogram packet data.

    Parameters
    ----------
    packet : tuple[list]
    Histogram packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    L0_KEYS : tuple[str]
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
        Block start time (IMAP), seconds<
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

    L0_KEYS: tuple[str] = (
        "SHCOARSE",
        "STARTID",
        "ENDID",
        "FLAGS",
        "SWVER",
        "SEC",
        "SUBSEC",
        "OFFSETSEC",
        "OFFSETSUBSEC",
        "GLXSEC",
        "GLXSUBSEC",
        "GLXOFFSEC",
        "GLXOFFSUBSEC",
        "SPINS",
        "NBINS",
        "TEMPAVG",
        "TEMPVAR",
        "HVAVG",
        "HVVAR",
        "SPAVG",
        "SPVAR",
        "ELAVG",
        "ELVAR",
        "EVENTS",
        "HISTOGRAM_DATA",
    )

    def __repr__(self):
        """Print the data.

        Returns
        -------
        String representation of GlowsHistL0
        """
        output = "{"
        for key in self.L0_KEYS:
            output += f"{key}: {getattr(self, key)}, "
        return output + "}"

    def __init__(self, packet):
        for key in self.L0_KEYS:
            if key != "HISTOGRAM_DATA":
                setattr(self, key, packet.data[key].derived_value)
            else:
                setattr(
                    self, key, self._convert_histogram_data(packet.data[key].raw_value)
                )
            if getattr(self, key) is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")

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
    L0_KEYS : tuple[str]
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

    L0_KEYS: tuple[str] = ("SHCOARSE", "SEC", "LEN", "SEQ")

    def __repr__(self):
        """Print the data.

        Returns
        -------
        String representation of GlowsDeL0
        """
        output = "{"
        for key in self.L0_KEYS:
            output += f"{key}: {getattr(self, key)}, "
        return output + "}"

    def __init__(self, packet):
        for key in self.L0_KEYS:
            setattr(self, key, packet.data[key].derived_value)
            if getattr(self, key) is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")
