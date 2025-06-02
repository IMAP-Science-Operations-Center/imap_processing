"""Pytest plugin module for test data paths."""

import pytest

from imap_processing import imap_module_directory


@pytest.fixture
def sc_packet_path():
    """Returns the spacecraft packet directory."""
    packet_path = (
        imap_module_directory / "tests" / "ialirt" / "data" / "l0" / "apid_478.bin"
    )
    xtce_ialirt_path = (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )

    return packet_path, xtce_ialirt_path
