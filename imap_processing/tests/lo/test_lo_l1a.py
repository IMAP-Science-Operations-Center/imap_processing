import numpy as np
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.lo.l1a.lo_l1a import lo_l1a


def test_lo_l1a():
    # Act
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    expected_logical_source = [
        "imap_lo_l1a_spin",
        "imap_lo_l1a_histogram",
        "imap_lo_l1a_de",
    ]
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
    np.testing.assert_array_equal(hist_fields_lower, output_datasets[1].data_vars)


def test_validate_spin_data():
    # Arrange
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    validation_path = (
        imap_module_directory / "tests/lo/validation_data/"
        "Instrument_FM1_T104_R129_20240803_ILO_SPIN_EU.csv"
    )
    validation_data = pd.read_csv(validation_path)

    spin_fields = [
        "shcoarse",
        "num_completed",
        "acq_start_sec",
        "acq_start_subsec",
        "acq_end_sec",
        "acq_end_subsec",
        "start_sec_spin",
        "start_subsec_spin",
        "esa_neg_dac_spin",
        "esa_pos_dac_spin",
        "valid_period_spin",
        "valid_phase_spin",
        "period_source_spin",
    ]
    # The validation data contains a duplicate set of columns for the same values.
    # They are formatted as column_prefix_<spin> and column_prefix[<spin>]
    # adding a condition to remove the one with [ before combining those columns into
    # a list
    bad_fields = [col for col in validation_data.columns if "[" in col]
    validation_data = validation_data.drop(bad_fields, axis=1)

    # The validation contains columns for each of the 28 spins in the packet, so these
    # need to be combined into a single list for the comparison
    for field in spin_fields:
        matching_columns = [
            col for col in validation_data.columns if col.startswith(field.upper())
        ]
        if len(matching_columns) > 1:
            validation_data[field.upper()] = validation_data[
                matching_columns
            ].values.tolist()
            validation_data = validation_data.drop(matching_columns, axis=1)

    # Act
    output_dataset = lo_l1a(dependency, "001")

    # Assert
    for field in spin_fields:
        np.testing.assert_array_equal(
            output_dataset[0][field], validation_data[field.upper()].values.tolist()
        )
