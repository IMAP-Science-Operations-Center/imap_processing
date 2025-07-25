"""Pytest plugin module for test data paths."""

from unittest import mock

import pytest
from imap_data_access.processing_input import AncillaryInput

from imap_processing import imap_module_directory
from imap_processing.ancillary.ancillary_dataset_combiner import MagAncillaryCombiner


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

    with mock.patch(
        "imap_processing.ancillary.ancillary_dataset_combiner.AncillaryFilePath.construct_path",
        return_value=cal_path,
    ):
        processing = AncillaryInput(cal_path.name)
        calibration_data = MagAncillaryCombiner(processing, "20250101").combined_dataset

    return calibration_data
