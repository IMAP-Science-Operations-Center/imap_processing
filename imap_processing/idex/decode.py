"""
Decompress IDEX raw wavelength data.

Originally written by Corinne Wuerthner.
A compressed data frame starts on a byte boundary, and begins with the 8-bit sequence
0xF5 to indicate the start of a frame.
A frame contains one or more subframes of compressed data. Subframes are packed as
tightly as possible, and are not aligned to any byte or word boundary. The compression
parameters are varied for each subframe to optimize the data compression. Within a
subframe the same predictor, and the same Rice parameter is used for all the
samples in that subframe. The predictor and rice parameter is allowed to vary
from subframe to subframe.

The start of each subframe consists of the following bits:
    00: Constant value
        The predictor assumes that every sample in the frame is equal to the
        first sample in the frame. In this case, the first sample is stored as
        the original NBIT binary value and no other samples are included in the
        frame.  The constant value predictor will only be selected for a
        subframe where every sample is the same value.

    01: Verbatim
        The predictor assumes that the data is not well correlated, and simply
        stores every sample in the frame using the original NBIT binary
        representation. This predictor minimizes the impact of data expansion
        for uncorrelated data samples where the golomb-rice coding would
        actually make the encoded data significantly larger than the original
        data set.

    10: Linear predictor #1
        This predictor assumes that each sample will have the same value as the
        sample immediately before it (sample X(n) = sampleX(n-1)). The predicted
        value is subtracted from the actual value of the sample and only the
        error (residual) between the actual value, and the predicted value is
        stored.  This predictor requires a single uncompressed sample as a
        “warmup sample” to be used in predicting the other samples in the
        subframe.

    11: Linear Predictor #2
        This predictor assumes that sample X(n) = 2*X(n-1)-X(n-2). This
        predictor calculates the slope of the signal based on the two previous
        values (slope = X(n-1)-X(n-2)) and predicts the value of X(n) will
        follow on a straight line drawn between the two previous points.
        X(n) = X(n-1) + (X(n-1)-X(n-2)) = 2*X(n-1)-X(n-2).

M bits for the rice parameter k. The bit width of k is equal to ceiling(log2(NBITS)).
For 10 bits, the bit width of k is 4 because log2(10) = 3.32 and ceiling(3.32) = 4.
The rice parameter k is stored in the bit stream as an unsigned binary value. The rice
parameter is omitted from the bit stream if the predictor type is set to “00” or '01'.
The next NBITS contain the raw binary value of the first sample in the subframe.
"""

# sub_frame_size is the compression block size
SUB_FRAME_SIZE = 64


def _decode_sub_frame(
    bits: str,
    bp: int,
    psel: int,
    k: int,
    n_bits: int,
) -> tuple[list[int], int]:
    """
    Decode a subframe of compressed data.

    Parameters
    ----------
    bits : str
        Raw waveform binary string.
    bp : int
        Current position to start reading from bits.
    psel : int
        Predictor select value.
    k : int
        Rice parameter used to divide each re-mapped residual value into two parts.
    n_bits : int
        Expected number of bits per sample. Either 10 or 12.

    Returns
    -------
    tuple[list, int]
        Decompressed subframe as a list of integers and the bit position.
    """
    sample_count = 0
    sub_frame_data = []

    while (sample_count < SUB_FRAME_SIZE) and bp < len(bits):
        if sample_count == 0:
            # For every subframe, the first sample is always uncompressed.
            # Read warmup sample
            d1, bp = read_bits(bits, bp, n_bits)

            sub_frame_data.append(d1)
            sample_count += 1

            # A 'psel' value of zero assumes that every sample in the frame is equal
            # to the first sample in the frame. In this case, the first sample is
            # stored as the original NBIT binary value
            if psel == 0:
                sub_frame_data.extend([d1] * (SUB_FRAME_SIZE - 1))
                sample_count = SUB_FRAME_SIZE

        # A 'psel' value of 1 assumes that the data is not well correlated, and
        # simply stores every sample in the frame using the original NBIT binary
        # representation.
        # A 'psel' value of 3 requires two uncompressed 'warm-up' samples.
        elif (psel == 1) or ((sample_count == 1) and (psel == 3)):
            d1, bp = read_bits(bits, bp, n_bits)
            sub_frame_data.append(d1)
            sample_count += 1

        else:
            # The rice parameter (k) is used to divide each re-mapped residual value
            # into two parts. The least significant k bits of the value are called
            # the remainder (r). The other part of the value (not included in the
            # remainder) is called the quotient (q).
            # The remapped quotient is unary encoded by including a number of 0 bits
            # equal to the value of the quotient, followed by a single '1' bit.
            q = bits[bp:].find("1")
            bp += q + 1
            # If the value of the quotient is equal to or larger than 47, then a
            # special symbol is used to denote that this particular residual value
            # is not rice encoded, but that this special symbol is followed by the
            # raw binary representation of the residual value using a (N_BITS+2)
            # bit binary number. This special symbol is simply 47 zeros followed
            # by a one.
            if q == 47:
                d1, bp = read_bits(bits, bp, n_bits + 2, True)
            else:
                if q % 2 == 1:
                    q = int(-((q + 1) / 2))
                else:
                    q = int(q / 2)

                r, bp = read_bits(bits, bp, k + 1)
                d1 = (q << (k + 1)) + r

            if psel == 2:
                d1 = d1 + sub_frame_data[sample_count - 1]
            elif (sample_count > 1) and (psel == 3):
                d1 = (
                    d1
                    + 2 * sub_frame_data[(sample_count - 1)]
                    - sub_frame_data[(sample_count - 2)]
                )

            if (d1 > 2**n_bits) or (d1 < -(2**n_bits)):
                raise ValueError(
                    f"Overflow Error while decoding subframe "
                    f"{k=}, {q=}, {r=}, {d1=}\n"
                    f"DataOut = {sub_frame_data}"
                )

            sub_frame_data.append(d1)
            sample_count += 1

    return sub_frame_data, bp


def rice_decode(compressed_data: str, nbit10: bool, sample_count: int) -> list[int]:
    """
    Decode compressed IDEX wavelength data using linear prediction and Golomb-RICE.

    Parameters
    ----------
    compressed_data : str
        Binary string representation of the raw waveform.
    nbit10 : bool
        If nbit10 is true, then the samples are expected to be 10 bits each, and if
        nbit10 is false, then the samples are expected to be 12 bits each.
    sample_count : int
        The total number of samples to be decompressed.

    Returns
    -------
    list[int]
        Decompressed data as a list of integers.
    """
    # Constants:
    k_bits = 4
    n_bits = 10 if nbit10 else 12

    # frame_size is the expected amount of data
    frame_size = sample_count
    sub_frame_per_frame = frame_size / SUB_FRAME_SIZE

    bits = compressed_data
    out_data: list[int] = []
    sub_frame_count = 0
    bp = 0
    # Decode all subframes
    while bp < len(bits) and (sub_frame_count < sub_frame_per_frame):
        # The next two bits are the predictor select bits
        psel, bp = read_bits(bits, bp, 2)
        if psel > 1:
            k, bp = read_bits(bits, bp, k_bits)
        else:
            k = 0

        sub_frame_data, bp = _decode_sub_frame(bits, bp, psel, k, n_bits)

        out_data.extend(sub_frame_data)
        sub_frame_count += 1

    if bp < len(bits) and (len(out_data) < frame_size):
        raise ValueError("End of file reached before", frame_size, "samples decoded")

    return out_data


def read_bits(
    bits: str, bit_pointer: int, read_num: int, signed: bool = False
) -> tuple[int, int]:
    """
    Read bits from a binary string and convert to an int.

    Parameters
    ----------
    bits : str
       Binary string to read from.
    bit_pointer : int
       Current position in binary string.
    read_num : int
       Number of bits to read.
    signed : bool
       If signed is True, convert bits to a signed int. Default is False.

    Returns
    -------
    value : int
       Value of bits read.
    bit_pointer : int
       Bit position after reading.
    """
    value = int(bits[bit_pointer : bit_pointer + read_num], 2)
    if signed and bits[bit_pointer] == "1":
        # If signed and is negative convert value
        value = value - 2**read_num
    bit_pointer += read_num
    return value, bit_pointer
