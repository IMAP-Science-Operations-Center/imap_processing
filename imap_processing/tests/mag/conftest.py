"""Shared modules for MAG tests"""

from pathlib import Path

import pytest

from imap_processing.mag.l1a.mag_l1a import mag_l1a


@pytest.fixture()
def validation_l1a():
    current_directory = Path(__file__).parent
    test_file = current_directory / "validation" / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l1a = mag_l1a(test_file, "v000")
    return l1a
