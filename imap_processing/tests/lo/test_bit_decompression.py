import pytest

from imap_processing.lo.l0.utils.bit_decompression import Decompress, decompress_int


def test_decompress_int_8_to_16_bit():
    decompression = Decompress.DECOMPRESS8TO16
    compressed_value = 209
    expected_decompressed_int = 9537

    decompressed_int = decompress_int(compressed_value, decompression)

    assert expected_decompressed_int == decompressed_int


def test_decompress_int_8_to_12_bit():
    decompression = Decompress.DECOMPRESS8TO12
    compressed_value = 205
    expected_decompressed_int = 1229

    decompressed_int = decompress_int(compressed_value, decompression)

    assert expected_decompressed_int == decompressed_int

def test_decompress_int_12_to_16_bit():
    decompression = Decompress.DECOMPRESS12TO16
    compressed_value = 4068
    expected_decompressed_int = 62908

    decompressed_int = decompress_int(compressed_value, decompression)

    assert expected_decompressed_int == decompressed_int

def test_decompress_int_invalid_decompression():
    decompression = "invalid decompression"
    compressed_value = 4068

    with pytest.raises(ValueError):
        decompressed_int = decompress_int(compressed_value, decompression)

