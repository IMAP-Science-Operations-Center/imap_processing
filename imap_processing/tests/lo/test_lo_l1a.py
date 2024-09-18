from pathlib import Path

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.lo.l1a.lo_l1a import lo_l1a


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize(
    ("dependency", "expected_logical_source"),
    [
        (Path("imap_lo_l0_de_20100101_v001.pkts"), "imap_lo_l1a_de"),
        (
            Path("imap_lo_l0_spin_20100101_v001.pkt"),
            "imap_lo_l1a_spin",
        ),
    ],
)
def test_lo_l1a(dependency, expected_logical_source):
    # Act
    output_dataset = lo_l1a(dependency, "001")

    # Assert
    assert expected_logical_source == output_dataset.attrs["Logical_source"]


def test_lo_l1a_dataset():
    # Arrange
    dependency = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l0_raw_20240627_v001.pkts"
    )

    histogram_fields = [
        "SHCOARSE",
        "START_A",
        "START_C",
        "STOP_B0",
        "STOP_B3",
        "TOF0_COUNT",
        "TOF1_COUNT",
        "TOF2_COUNT",
        "TOF3_COUNT",
        "TOF0_TOF1",
        "TOF0_TOF2",
        "TOF1_TOF2",
        "SILVER",
        "DISC_TOF0",
        "DISC_TOF1",
        "DISC_TOF2",
        "DISC_TOF3",
        "POS0",
        "POS1",
        "POS2",
        "POS3",
        "HYDROGEN",
        "OXYGEN",
    ]
    hist_fields_lower = [field.lower() for field in histogram_fields]

    # Act
    output_datasets = lo_l1a(dependency, "001")

    # Assert
    np.testing.assert_array_equal(hist_fields_lower, output_datasets[0].data_vars)
