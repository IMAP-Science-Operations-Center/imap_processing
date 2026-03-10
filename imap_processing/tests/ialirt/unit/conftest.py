"""Pytest plugin module for test data paths."""

import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf


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


@pytest.fixture
def swapi_postlaunch_sc_packet_path():
    """Returns the spacecraft packet directory."""
    xtce_ialirt_path = (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )

    directory = imap_module_directory / "tests" / "ialirt" / "data" / "l0"
    filenames = [
        "iois_1_packets_2025_344_05_57_56",
        "iois_1_packets_2025_344_05_59_58",
    ]
    return tuple(directory / fname for fname in filenames), xtce_ialirt_path


@pytest.fixture
def ialirt_mag_test_l1d_data():
    """Returns the MAG I-ALiRT calibration dataset."""
    cal_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "imap_mag_ialirt-calibration_20250101_v002.cdf"
    )

    calibration_data = load_cdf(cal_path)

    return calibration_data


@pytest.fixture
def ialirt_mag_test_l1d_data_postlaunch():
    """Returns the MAG I-ALiRT calibration dataset."""
    cal_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "imap_mag_ialirt-calibration_20250926_v002.cdf"
    )

    calibration_data = load_cdf(cal_path)

    return calibration_data
