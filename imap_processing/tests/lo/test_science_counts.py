import numpy as np

from imap_processing.lo.l0.lo_science import decompress


def test_decompress_8_to_16_bit():
    # Arrange
    # 174 in binary (= 4071 decompressed)
    idx0 = "10101110"
    # 20 in binary (= 20 decompressed)
    idx1 = "00010100"
    bin_str = idx0 + idx1
    bits_per_index = 8
    section_start = 0
    section_length = 16
    expected = [4071, 20]

    # Act
    out = decompress(bin_str, bits_per_index, section_start, section_length)

    # Assert
    np.testing.assert_equal(out, expected)


def test_decompress_12_to_16_bit():
    # Arrange
    # 3643 in binary (= 35800 decompressed)
    idx0 = "111000111011"
    # 20 in binary (= 20 decompressed
    idx1 = "000000010100"
    bin_str = idx0 + idx1
    bits_per_index = 12
    section_start = 0
    section_length = 24
    expected = [35800, 20]

    # Act
    out = decompress(bin_str, bits_per_index, section_start, section_length)

    # Assert
    np.testing.assert_equal(out, expected)
