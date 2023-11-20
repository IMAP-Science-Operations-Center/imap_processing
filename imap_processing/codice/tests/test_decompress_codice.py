"""Tests the decompression algorithms for CoDICE science data"""

import lzma

import pytest

from imap_processing.codice.l0.decompress_codice import decompress
from imap_processing.codice.utils.codice_utils import CoDICECompression

# Test the algorithms using input value of 234 (picked randomly)
LZMA_EXAMPLE = lzma.compress((234).to_bytes(1, byteorder="big"))
TEST_DATA = [
    (234, CoDICECompression.NO_COMPRESSION, 234),
    (234, CoDICECompression.LOSSY_A, 221184),
    (234, CoDICECompression.LOSSY_B, 1441792),
    (LZMA_EXAMPLE, CoDICECompression.LOSSLESS, 234),
    (LZMA_EXAMPLE, CoDICECompression.LOSSY_A_LOSSLESS, 221184),
    (LZMA_EXAMPLE, CoDICECompression.LOSSY_B_LOSSLESS, 1441792),
    pytest.param(None, "some_unsupported_algorithm", None, marks=pytest.mark.xfail()),
]


@pytest.mark.parametrize(
    ("compressed_value", "algorithm", "expected_result"), TEST_DATA
)
def test_decompress(compressed_value, algorithm, expected_result):
    """Tests the ``decompress`` function"""

    decompressed_value = decompress(compressed_value, algorithm)
    assert decompressed_value == expected_result
