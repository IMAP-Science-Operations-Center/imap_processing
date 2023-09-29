"""Pytest plugin module for test data paths"""
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def packet_path():
    """Returns the raw packet directory.
    """
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ialirt' / 'tests' / 'test_data' / 'l0' / \
        'IALiRT Raw Packet Telemetry.txt'


@pytest.fixture()
def xtce_ialirt_path():
    """Returns the xtce auxilliary directory.
    """
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ialirt' / 'packet_definitions' \
        / "ialirt.xml"
