"""Tests the decompression algorithms for CoDICE science data"""

import lzma

import pytest

from imap_processing.codice.l0.decompress_codice import decompress

# Test the algorithms using input value of 234 (picked randomly)
LZMA_EXAMPLE = lzma.compress((234).to_bytes())
TEST_DATA = [
    (234, "no compression", 234),
    (234, "lossyA", 221184),
    (234, "lossyB", 1441792),
    (LZMA_EXAMPLE, "lossless", 234),
    (LZMA_EXAMPLE, "lossyA+lossless", 221184),
    (LZMA_EXAMPLE, "lossyB+lossless", 1441792),
    pytest.param(None, "some_unsupported_algorithm", None, marks=pytest.mark.xfail()),
]


@pytest.mark.parametrize(
    ("compressed_value", "algorithm", "expected_result"), TEST_DATA
)
def test_decompress(compressed_value, algorithm, expected_result):
    """Tests the ``decompress`` function"""

    decompressed_value = decompress(compressed_value, algorithm)
    assert decompressed_value == expected_result
