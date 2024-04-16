"""Tests the decompression algorithms for CoDICE science data"""

import lzma
from enum import IntEnum

import pytest

from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import CoDICECompression

# Test the algorithms using input value of 234 (picked randomly)
LZMA_EXAMPLE = lzma.compress((234).to_bytes(1, byteorder="big"))
TEST_DATA = [
    (234, CoDICECompression.NO_COMPRESSION, 234),
    (234, CoDICECompression.LOSSY_A, 221184),
    (234, CoDICECompression.LOSSY_B, 1441792),
    (LZMA_EXAMPLE, CoDICECompression.LOSSLESS, 234),
    (LZMA_EXAMPLE, CoDICECompression.LOSSY_A_LOSSLESS, 221184),
    (LZMA_EXAMPLE, CoDICECompression.LOSSY_B_LOSSLESS, 1441792),
]


@pytest.mark.parametrize(
    ("compressed_value", "algorithm", "expected_result"), TEST_DATA
)
def test_decompress(compressed_value: int, algorithm: IntEnum, expected_result: int):
    """Tests the ``decompress`` function

    Parameters
    ----------
    compressed_value : int
        The compressed value to test decompression on
    algorithm : IntEnum
        The algorithm to use in decompression
    expected_result : int
        The expected, decompressed value
    """

    decompressed_value = decompress(compressed_value, algorithm)
    assert decompressed_value == expected_result


def test_decompress_raises():
    """Tests that the ``decompress`` function raises with an unknown algorithm"""

    with pytest.raises(ValueError, match="some_unsupported_algorithm"):
        decompress(234, "some_unsupported_algorithm")
