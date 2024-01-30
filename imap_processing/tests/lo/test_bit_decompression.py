import pytest

from imap_processing.lo.l0.utils.bit_decompression import Decompress, decompress_int


def test_decompress_int_8_to_16_bit():
    """Test decompressing integer from 8 to 16 bits."""
    ## Arrange
    decompression = Decompress.DECOMPRESS8TO16
    compressed_value = 209
    expected_decompressed_int = 9537

    ## Act
    decompressed_int = decompress_int(compressed_value, decompression)

    ## Assert
    assert expected_decompressed_int == decompressed_int


def test_decompress_int_8_to_12_bit():
    """Test decompressing integer from 8 to 12 bits."""
    ## Arrange
    decompression = Decompress.DECOMPRESS8TO12
    compressed_value = 205
    expected_decompressed_int = 1229

    ## Act
    decompressed_int = decompress_int(compressed_value, decompression)

    ## Assert
    assert expected_decompressed_int == decompressed_int


def test_decompress_int_12_to_16_bit():
    """Test decompressing integer from 12 to 16 bits."""
    ## Arrange
    decompression = Decompress.DECOMPRESS12TO16
    compressed_value = 4068
    expected_decompressed_int = 62908

    ## Act
    decompressed_int = decompress_int(compressed_value, decompression)

    ## Assert
    assert expected_decompressed_int == decompressed_int


def test_decompress_int_invalid_decompression():
    """Test receiving an invalid decompression input."""
    ## Arrange
    decompression = "invalid decompression"
    compressed_value = 4068

    ## Act / Assert
    with pytest.raises(ValueError, match="Invalid decompression method"):
        decompress_int(compressed_value, decompression)
