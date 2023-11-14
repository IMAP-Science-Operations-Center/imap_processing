"""Decompress CoDICE science data.

For CoDICE, there are 3 forms of compression:
    1. Table-based lossy compression A (24 -> 8-bit)
    2. Table-based lossy compression B (24 -> 8 bit)
    3. LZMA lossless compression

Only one lossy option can be selected in cases of lossy + lossless compression.
Thus, there are 6 possibly compression algorithms:
    1. No compression
    2. Lossy A only
    3. Lossy B only
    4. Lossless only
    5. Lossy A + lossless
    6. Lossy B + lossless

In the case of (5) and (6), the data is first run through lossy compression, and
then the result is run through lossless compression. Thus, to decompress, one
must apply lossless decompression first, then lossy decompression

References
----------
    This information was provided via email from Greg Dunn on Oct 23, 2023
"""

import lzma

from . import LOSSY_A_TABLE, LOSSY_B_TABLE


class CodiceDecompression:
    """Main class that contains implementations of the decompression algorithms.

    When an algorithm is applied, the ``decompressed_value`` attribute is set or
    updated to contain the 24- or 32-bit decompressed value.

    Attributes
    ----------
    compressed_value : int
        The 8-bit compressed value to decompress
    algorithm : str
        The algorithm to apply. Supported algorithms include 'no compression',
        'lossyA', 'lossyB', 'lossless', 'lossyA+lossless', and 'lossyB+lossless'
    decompressed_value : int or None
        The 24- or 32-bit decompressed value, or None if the algorithm is not
        supported or hasn't been applied yet.
    """

    def __init__(self, compressed_value, algorithm):
        self.compressed_value = compressed_value
        self.algorithm = algorithm
        self.decompressed_value = None

    def _apply_lossy_a(self, value):
        """Apply 8-bit to 32-bit Lossy A decompression algorithm.

        The Lossy A algorithm uses a lookup table imported into this module.

        Parameters
        ----------
        value : int
            The compressed 8-bit value
        """
        self.decompressed_value = LOSSY_A_TABLE[value]

    def _apply_lossy_b(self, value):
        """Apply 8-bit to 32-bit Lossy B decompression algorithm.

        The Lossy B algorithm uses a lookup table imported into this module.

        Parameters
        ----------
        value : int
            The compressed 8-bit value
        """
        self.decompressed_value = LOSSY_B_TABLE[value]

    def _apply_lzma_lossless(self, value):
        """Apply LZMA lossless decompression algorithm.

        Parameters
        ----------
        value : int
            The compressed 8-bit value
        """
        self.decompressed_value = lzma.decompress(value)
        self.decompressed_value = int.from_bytes(
            self.decompressed_value, byteorder="big"
        )

    def decompress(self):
        """Decompress the value.

        Apply the appropriate decompression algorithm(s) based on the value
        of the ``algorithm`` attribute. One or more individual algorithms may be
        applied to a given compressed value.
        """
        if self.algorithm == "no compression":
            self.decompressed_value = self.compressed_value
        elif self.algorithm == "lossyA":
            self._apply_lossy_a(self.compressed_value)
        elif self.algorithm == "lossyB":
            self._apply_lossy_b(self.compressed_value)
        elif self.algorithm == "lossless":
            self._apply_lzma_lossless(self.compressed_value)
        elif self.algorithm == "lossyA+lossless":
            self._apply_lzma_lossless(self.compressed_value)
            self._apply_lossy_a(self.decompressed_value)
        elif self.algorithm == "lossyB+lossless":
            self._apply_lzma_lossless(self.compressed_value)
            self._apply_lossy_b(self.decompressed_value)
