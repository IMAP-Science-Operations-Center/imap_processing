"""Tests the decompression algorithms for CoDICE science data"""

import pytest

from imap_processing.codice.l0.decompress_codice import CodiceDecompression

# The value of 234 compressed using LZMA
LZMA_EXAMPLE = (
    b"\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3"
    b"\x01\x00\x00\xea\x00\x00\x00\x00\x15g\x1b\xad\x02.eJ\x00\x01\x19\x01\xa5,"
    b"\x81\xcc\x1f\xb6\xf3}\x01\x00\x00\x00\x00\x04YZ"
)

# Test the algorithms using input value of 234 (picked randomly)
TEST_DATA = [
    (234, "no compression", 234),
    (234, "lossyA", 221184),
    (234, "lossyB", 1441792),
    (LZMA_EXAMPLE, "lossless", 234),
    (LZMA_EXAMPLE, "lossyA+lossless", 221184),
    (LZMA_EXAMPLE, "lossyB+lossless", 1441792),
    (None, "some_unsupported_algorithm", None),
]


@pytest.mark.parametrize(
    ("compressed_value", "algorithm", "expected_result"), TEST_DATA
)
def test_decompress(compressed_value, algorithm, expected_result):
    """Test the ``decompress`` method"""

    cd = CodiceDecompression(compressed_value, algorithm)
    cd.decompress()
    assert cd.decompressed_value == expected_result
