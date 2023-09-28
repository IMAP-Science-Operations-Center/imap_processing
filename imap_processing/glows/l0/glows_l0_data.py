class GlowsHistL0:
    """Data structure for storing GLOWS histogram packet data.

    Parameters
    ----------
    packet : tuple[list]
        Histogram packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    L0_KEYS : tuple[str]
        Data names from the packet
    data: dict
        The data from the packet
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
        return str(self.data)

    def __init__(self, packet):
        self.data = self.process_packet(packet)

    def process_packet(self, packet) -> dict:
        """Process the GLOWS histogram packet yielded by space_packet_parser.

        Parameters
        ----------
        packet
            Histogram packet from space_packet_parser

        Returns
        -------
        data: dict
            Data dictionary created from the packet, using L0_KEYS
        """
        data = {}
        for key in self.L0_KEYS:
            if key != "HISTOGRAM_DATA":
                data[key] = packet.data[key].derived_value
            else:
                data[key] = self._convert_histogram_data(packet.data[key].raw_value)
            if data[key] is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")
        return data

    def _convert_histogram_data(self, binary_hist_data: str) -> list[int]:
        """Convert the raw histogram data into a list.

         This method converts a binary number into a list of histogram values by
         splitting up the raw binary value into 8-bit segments.

        Parameters
        ----------
        binary_hist_data: Raw data read from the packet, in binary format.

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


class GlowsDeL0:
    """Data structure for storing GLOWS direct event packet data.

    Parameters
    ----------
    packet : tuple[list]
        Direct event packet yielded from space_packet_parser.generate_packets.

    Attributes
    ----------
    L0_KEYS : tuple[str]
        Data names from the packet
    data: dict
        The data from the packet
    """

    L0_KEYS: tuple[str] = ("SHCOARSE", "SEC", "LEN", "SEQ")

    def __repr__(self):
        """Print the data.

        Returns
        -------
        String representation of GlowsHistL0
        """
        return str(self.data)

    def __init__(self, packet):
        self.data = self.process_packet(packet)

    def process_packet(self, packet):
        """Process the GLOWS direct event packet, from space_packet_parser.

        Parameters
        ----------
        packet
            Direct event packet from space_packet_parser

        Returns
        -------
        data: dict
            Data dictionary created from the packet, using L0_KEYS
        """
        data = {}
        for key in self.L0_KEYS:
            data[key] = packet.data[key].derived_value
            if data[key] is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")

        return data
