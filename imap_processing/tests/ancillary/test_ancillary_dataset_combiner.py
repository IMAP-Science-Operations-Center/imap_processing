from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from imap_data_access.processing_input import AncillaryInput

from imap_processing.ancillary.ancillary_dataset_combiner import (
    AncillaryCombiner,
    MagAncillaryCombiner,
    TimestampedData,
)
from imap_processing.cdf.utils import load_cdf


@pytest.fixture
def mocks():
    with mock.patch(
        "imap_processing.ancillary.ancillary_dataset_combiner.cdf_to_xarray"
    ) as read_cdf:
        mocks = {
            "read_cdf": read_cdf,
        }
        yield mocks


@pytest.fixture
def mag_calibration_dataset():
    imap_dir = Path(__file__).parent.parent.parent.parent
    cal_file = (
        imap_dir
        / "imap_processing"
        / "tests"
        / "mag"
        / "validation"
        / "calibration"
        / "imap_mag_l2-calibration-matrices_20251017_v004.cdf"
    )
    calibration_data = load_cdf(cal_file)

    return calibration_data


@pytest.fixture
def ancillary_input():
    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251017_20251023_v003.cdf",
        "imap_mag_l2-calibration-matrices_20251020_20251022_v001.cdf",
        "imap_mag_l2-calibration-matrices_20251020_20251021_v004.cdf",
        "imap_mag_l2-calibration-matrices_20251015_20251018_v005.cdf",
        "imap_mag_l2-calibration-matrices_20251016_20251016_v006.cdf",
        "imap_mag_l2-calibration-matrices_20251025_20251025_v002.cdf",
    )
    return input_example


def test_mag_ancillary_converter(mocks, mag_calibration_dataset):
    # Simple case, no overlap
    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251017_20251020_v003.cdf",
        "imap_mag_l2-calibration-matrices_20251020_20251021_v004.cdf",
    )

    mocks["read_cdf"].return_value = mag_calibration_dataset

    output = MagAncillaryCombiner(input_example, "20251031")
    expected_epoch = [
        np.datetime64("2025-10-17"),
        np.datetime64("2025-10-18"),
        np.datetime64("2025-10-19"),
        np.datetime64("2025-10-20"),
        np.datetime64("2025-10-21"),
    ]
    assert np.array_equal(output.combined_dataset["epoch"].data, expected_epoch)
    expected_versions = [3, 3, 3, 4, 4]

    assert np.array_equal(
        output.combined_dataset["input_file_version"].data, expected_versions
    )


def test_ancillary_converter_overlaps(mocks, mag_calibration_dataset):
    # Simple case, no overlap
    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251017_20251023_v003.cdf",
        "imap_mag_l2-calibration-matrices_20251020_20251021_v004.cdf",
    )

    mocks["read_cdf"].return_value = mag_calibration_dataset

    output = AncillaryCombiner(input_example, "20251031")
    expected_epoch = [
        np.datetime64("2025-10-17"),
        np.datetime64("2025-10-18"),
        np.datetime64("2025-10-19"),
        np.datetime64("2025-10-20"),
        np.datetime64("2025-10-21"),
        np.datetime64("2025-10-22"),
        np.datetime64("2025-10-23"),
    ]

    assert np.array_equal(output.combined_dataset["epoch"].data, expected_epoch)
    expected_versions = [3, 3, 3, 4, 4, 3, 3]

    assert np.array_equal(
        output.combined_dataset["input_file_version"].data, expected_versions
    )


def test_timestamped_data(mocks, mag_calibration_dataset, ancillary_input):
    data = [
        TimestampedData(
            np.datetime64("2025-10-17"),
            np.datetime64("2025-10-23"),
            xr.Dataset(),
            "v003",
        ),
        TimestampedData(
            np.datetime64("2025-10-20"),
            np.datetime64("2025-10-22"),
            xr.Dataset(),
            "v001",
        ),
        TimestampedData(
            np.datetime64("2025-10-20"),
            np.datetime64("2025-10-21"),
            xr.Dataset(),
            "v004",
        ),
        TimestampedData(
            np.datetime64("2025-10-15"),
            np.datetime64("2025-10-18"),
            xr.Dataset(),
            "v005",
        ),
    ]

    mocks["read_cdf"].return_value = mag_calibration_dataset

    output = MagAncillaryCombiner(ancillary_input, "20251031")

    for index, d in enumerate(data):
        assert d.start_time == output.timestamped_data[index].start_time
        assert d.end_time == output.timestamped_data[index].end_time
        assert d.version == output.timestamped_data[index].version

    new_file = "imap_mag_l2-calibration-matrices_20251017_20251020_v099.cdf"
    timestamped_output = output.convert_to_timestamped_data(new_file)
    assert timestamped_output.start_time == np.datetime64("2025-10-17")
    assert timestamped_output.end_time == np.datetime64("2025-10-20")
    assert (
        timestamped_output.dataset.data_vars.keys()
        == output.convert_file_to_dataset(new_file).data_vars.keys()
    )
    assert timestamped_output.version == "v099"


def test_mag_edge_cases(mocks, mag_calibration_dataset, ancillary_input):
    mocks["read_cdf"].return_value = mag_calibration_dataset

    output = MagAncillaryCombiner(ancillary_input, "20251031")

    expected_epoch_range = [
        np.datetime64("2025-10-15"),
        np.datetime64("2025-10-16"),
        np.datetime64("2025-10-17"),
        np.datetime64("2025-10-18"),
        np.datetime64("2025-10-19"),
        np.datetime64("2025-10-20"),
        np.datetime64("2025-10-21"),
        np.datetime64("2025-10-22"),
        np.datetime64("2025-10-23"),
        np.datetime64("2025-10-24"),
        np.datetime64("2025-10-25"),
    ]
    expected_output_versions = [5, 6, 5, 5, 3, 4, 4, 3, 3, 0, 2]

    assert np.array_equal(output.combined_dataset["epoch"].data, expected_epoch_range)
    assert np.array_equal(
        output.combined_dataset["input_file_version"].data, expected_output_versions
    )


def test_no_end_date(mocks, mag_calibration_dataset):
    mocks["read_cdf"].return_value = mag_calibration_dataset
    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251019_20251021_v003.cdf",
        "imap_mag_l2-calibration-matrices_20251020_v002.cdf",
    )

    output = AncillaryCombiner(input_example, "20251023")
    expected_epochs = [
        np.datetime64("2025-10-19"),
        np.datetime64("2025-10-20"),
        np.datetime64("2025-10-21"),
        np.datetime64("2025-10-22"),
        np.datetime64("2025-10-23"),
    ]
    expected_versions = [3, 3, 3, 2, 2]

    assert np.array_equal(output.combined_dataset["epoch"].data, expected_epochs)
    assert np.array_equal(
        output.combined_dataset["input_file_version"].data, expected_versions
    )
