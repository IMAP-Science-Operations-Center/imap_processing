"""Tests the decompression algorithms for CoDICE science data"""

import lzma
from enum import IntEnum

import numpy as np
import pytest

from imap_processing.codice.codice_l1a_de import unpack_bits
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CoDICECompression

# Test the algorithms using input value of 234 (picked randomly)
lzma_bytes = lzma.compress((234).to_bytes(1, byteorder="big"))
# LZMA_EXAMPLE = "".join(format(byte, "08b") for byte in lzma_bytes)
TEST_DATA = [
    (b"\xea", CoDICECompression.NO_COMPRESSION, [234]),
    (b"\xea", CoDICECompression.LOSSY_A, [217087]),
    (b"\xea", CoDICECompression.LOSSY_B, [1376255]),
    (lzma_bytes, CoDICECompression.LOSSLESS, [234]),
    (lzma_bytes, CoDICECompression.LOSSY_A_LOSSLESS, [217087]),
    (lzma_bytes, CoDICECompression.LOSSY_B_LOSSLESS, [1376255]),
]


@pytest.mark.parametrize(
    ("compressed_binary", "algorithm", "expected_result"), TEST_DATA
)
def test_decompress(
    compressed_binary: str, algorithm: IntEnum, expected_result: list[int]
):
    """Tests the ``decompress`` function

    Parameters
    ----------
    compressed_binary : str
        The compressed binary string to test decompression on
    algorithm : IntEnum
        The algorithm to use in decompression
    expected_result : list[int]
        The expected, decompressed value
    """

    decompressed_value = decompress(compressed_binary, algorithm)
    assert decompressed_value == expected_result


def test_decompress_raises():
    """Tests that the ``decompress`` function raises with an unknown algorithm"""

    with pytest.raises(ValueError, match="some_unsupported_algorithm"):
        decompress("11101010", "some_unsupported_algorithm")


def test_unpack_bits():
    """Test that 64-bits is unpacked in LSB order correctly."""
    test_data = np.array([0x3, 0x9F], dtype=np.uint64)
    bit_chunks = {
        "c": {"bit_length": 52},
        "b": {"bit_length": 7},
        "a": {"bit_length": 5},
    }

    unpacked_fields = unpack_bits(bit_chunks, test_data)
    expected_unpacked = {
        "a": np.array([0, 0], dtype=np.uint64),
        "b": np.array([0, 0], dtype=np.uint64),
        "c": np.array([3, 159], dtype=np.uint64),
    }
    assert all(
        np.array_equal(unpacked_fields[key], expected_unpacked[key])
        for key in bit_chunks
    )
