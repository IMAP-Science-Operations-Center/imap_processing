"""Tests the decompression algorithms for CoDICE science data"""

import lzma
from enum import IntEnum

import pytest

from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CoDICECompression

# Test the algorithms using input value of 234 (picked randomly)
lzma_bytes = lzma.compress((234).to_bytes(1, byteorder="big"))
# LZMA_EXAMPLE = "".join(format(byte, "08b") for byte in lzma_bytes)
TEST_DATA = [
    (b"\xea", CoDICECompression.NO_COMPRESSION, [234]),
    (b"\xea", CoDICECompression.LOSSY_A, [221184]),
    (b"\xea", CoDICECompression.LOSSY_B, [1441792]),
    (lzma_bytes, CoDICECompression.LOSSLESS, [234]),
    (lzma_bytes, CoDICECompression.LOSSY_A_LOSSLESS, [221184]),
    (lzma_bytes, CoDICECompression.LOSSY_B_LOSSLESS, [1441792]),
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
