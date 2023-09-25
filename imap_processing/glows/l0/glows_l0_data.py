# TODO: Write pretty printing methods


class GlowsHistL0:
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

    def __init__(self, packet):
        self.data = self.process_packet(packet)

    def process_packet(self, packet):
        data = {}
        for key in self.L0_KEYS:
            if key != "HISTOGRAM_DATA":
                data[key] = packet.data[key].derived_value
            else:
                data[key] = self.convert_histogram_data(packet.data[key].raw_value)
            if data[key] is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")
        return data

    def convert_histogram_data(self, bin_hist_data: str) -> list[int]:
        """Convert the raw histogram data into a list by splitting up the raw binary
        value into 8-bit segments
        Parameters
        ----------
        bin_hist_data: Raw data read from the packet, in binary format

        Returns
        -------
        List of histogram data
        """
        # Convert the histogram data from a large raw string into a list of 8 bit values
        histograms = []
        for i in range(8, len(bin_hist_data), 8):
            histograms.append(int(bin_hist_data[i - 8 : i], 2))

        if len(histograms) != 3599:
            raise ValueError(
                f"Histogram packet is lacking bins. Expected a count of 3599, "
                f"actually received {len(histograms)}"
            )

        return histograms


class GlowsDeL0:
    L0_KEYS: tuple[str] = ("SHCOARSE", "SEC", "LEN", "SEQ")

    def __init__(self, packet):
        self.data = self.process_packet(packet)

    def process_packet(self, packet):
        data = {}
        for key in self.L0_KEYS:
            data[key] = packet.data[key].derived_value
            if data[key] is None:
                raise ValueError(f"Missing data for {key} in {packet.data}")

        return data
