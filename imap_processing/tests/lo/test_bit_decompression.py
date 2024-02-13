import pytest

from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
    decompress_int,
)


@pytest.mark.parametrize(
    ("decompression", "compressed_value", "expected_decompressed_int"),
    [
        (Decompress.DECOMPRESS8TO16, 209, 9537),
        (Decompress.DECOMPRESS8TO12, 205, 1229),
        (Decompress.DECOMPRESS12TO16, 4068, 62908),
    ],
)
def test_decompress_int(decompression, compressed_value, expected_decompressed_int):
    """Test decompression for 8to16, 8to12, 12to16."""

    ## Act
    decompressed_int = decompress_int(
        compressed_value, decompression, DECOMPRESSION_TABLES
    )

    ## Assert
    assert expected_decompressed_int == decompressed_int


def test_decompress_int_invalid_decompression():
    """Test receiving an invalid decompression input."""
    ## Arrange
    decompression = "invalid decompression"
    compressed_value = 4068

    ## Act / Assert
    with pytest.raises(ValueError, match="Invalid decompression method"):
        decompress_int(compressed_value, decompression, DECOMPRESSION_TABLES)
