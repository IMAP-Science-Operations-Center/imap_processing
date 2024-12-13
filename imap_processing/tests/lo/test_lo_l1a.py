import numpy as np

from imap_processing import imap_module_directory
from imap_processing.lo.l1a.lo_l1a import lo_l1a


def test_lo_l1a():
    # Act
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    expected_logical_source = ["imap_lo_l1a_histogram", "imap_lo_l1a_de"]
    output_dataset = lo_l1a(dependency, "001")

    # Assert
    for dataset, logical_source in zip(output_dataset, expected_logical_source):
        assert logical_source == dataset.attrs["Logical_source"]


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
