from dataclasses import dataclass

from imap_processing.glows.l0.glows_l0_data import HistogramL0


@dataclass
class HistogramL1A:
    """Data structure for GLOWS Histogram Level 1A data.

    Attributes
    ----------
    l0: HistogramL0
        Data class containing the raw data from the histogram packet
    histograms: list[int]
        List of histogram data values
    """

    l0: HistogramL0
    histograms: list[int]

    def __post_init__(self):
        """Convert the level 0 histogram data into a usable L1A list."""
        self.histograms = self._convert_histogram_data(self.l0.HISTOGRAM_DATA)

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
