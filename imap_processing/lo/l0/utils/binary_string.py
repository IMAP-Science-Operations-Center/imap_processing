"""Class for binary data strings."""


class BinaryString:
    """
    Iterate through a binary string.

    The binary string class retains the current
    binary string bit position and updates it
    each time the string is accessed.

    Parameters
    ----------
    bin : str
        Binary string.

    Attributes
    ----------
    bin : str
        Binary string.
    bit_pos : int
        Current position in the binary string.

    Methods
    -------
    next_bits(num_bits)
        Returns the next x bits in the binary string
        using the current bit position as the reference point.
        The bit position is updated to be the bit immediately
        following the last returned bit.
    """

    bin: str
    bit_pos: int

    def __init__(self, bin: str) -> None:
        self.bin = bin
        self.bit_pos = 0

    def next_bits(self, num_bits: int) -> str:
        """
        Return the next x bits in the binary string.

        The bit position updated to be the bit immediately
        following the last returned bit.

        Parameters
        ----------
        num_bits : int
            The number of bits you want to read off the binary string.

        Returns
        -------
        bits : str
            A binary string containing the next x bits that were requested.
        """
        bits = self.bin[self.bit_pos : self.bit_pos + num_bits]
        self.bit_pos += num_bits
        return bits
