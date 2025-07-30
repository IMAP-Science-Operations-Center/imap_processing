import json
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from imap_data_access.processing_input import AncillaryInput

from imap_processing.ancillary.ancillary_dataset_combiner import (
    AncillaryCombiner,
    GlowsAncillaryCombiner,
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
        / "imap_mag_l2-calibration_20251017_v004.cdf"
    )
    calibration_data = load_cdf(cal_file)

    return calibration_data


@pytest.fixture
def glows_ancillary_filepath():
    imap_dir = Path(__file__).parent.parent.parent.parent
    filepath = imap_dir / "imap_processing" / "glows" / "ancillary"

    return filepath


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


def test_glows_excluded_regions_combiner(glows_ancillary_filepath):
    file_path = (
        glows_ancillary_filepath
        / "imap_glows_map-of-excluded-regions_20250923_v002.dat"
    )

    # Test the convert_file_to_dataset method directly
    combiner = GlowsAncillaryCombiner(
        [], "20250925"
    )  # Empty list to avoid file parsing
    dataset = combiner.convert_file_to_dataset(file_path)

    assert dataset is not None
    assert "ecliptic_longitude_deg" in dataset.data_vars
    assert "ecliptic_latitude_deg" in dataset.data_vars
    assert dataset["ecliptic_longitude_deg"].dims == ("region",)
    assert dataset["ecliptic_latitude_deg"].dims == ("region",)


def test_glows_uv_sources_combiner(glows_ancillary_filepath):
    file_path = (
        glows_ancillary_filepath / "imap_glows_map-of-uv-sources_20250923_v002.dat"
    )

    # Test the convert_file_to_dataset method directly
    combiner = GlowsAncillaryCombiner(
        [], "20250925"
    )  # Empty list to avoid file parsing
    dataset = combiner.convert_file_to_dataset(file_path)

    assert dataset is not None
    assert "object_name" in dataset.data_vars
    assert "ecliptic_longitude_deg" in dataset.data_vars
    assert "ecliptic_latitude_deg" in dataset.data_vars
    assert "angular_radius_for_masking" in dataset.data_vars
    assert dataset["object_name"].dims == ("source",)


def test_glows_suspected_transients_combiner(glows_ancillary_filepath):
    file_path = (
        glows_ancillary_filepath / "imap_glows_suspected-transients_20250923_v002.dat"
    )

    # Test the convert_file_to_dataset method directly
    combiner = GlowsAncillaryCombiner(
        [], "20250925"
    )  # Empty list to avoid file parsing
    dataset = combiner.convert_file_to_dataset(file_path)

    assert dataset is not None
    assert "l1b_unique_block_identifier" in dataset.data_vars
    assert "histogram_mask_array" in dataset.data_vars
    assert dataset["l1b_unique_block_identifier"].dims == ("time_block",)


def test_glows_exclusions_by_instr_team_combiner(glows_ancillary_filepath):
    file_path = (
        glows_ancillary_filepath
        / "imap_glows_exclusions-by-instr-team_20250923_v002.dat"
    )

    # Test the convert_file_to_dataset method directly
    combiner = GlowsAncillaryCombiner(
        [], "20250925"
    )  # Empty list to avoid file parsing
    dataset = combiner.convert_file_to_dataset(file_path)

    assert dataset is not None
    assert "l1b_unique_block_identifier" in dataset.data_vars
    assert "histogram_mask_array" in dataset.data_vars
    assert dataset["l1b_unique_block_identifier"].dims == ("time_block",)

    # Test with mocked construct_path to simulate full file path workflow
    with patch(
        "imap_data_access.AncillaryFilePath.construct_path", return_value=str(file_path)
    ):
        combiner = GlowsAncillaryCombiner(
            ["imap_glows_exclusions-by-instr-team_20250923_v002.dat"], "20250925"
        )
        assert len(combiner.timestamped_data) == 1
        assert combiner.timestamped_data[0].version == "v002"


def test_ancillary_combiner_empty_input():
    """Test AncillaryCombiner with empty input list."""
    combiner = AncillaryCombiner([], "20251031")
    assert len(combiner.timestamped_data) == 0
    assert len(combiner.combined_dataset.data_vars) == 0


def test_ancillary_combiner_string_date():
    """Test AncillaryCombiner with string date format."""
    with mock.patch(
        "imap_processing.ancillary.ancillary_dataset_combiner.cdf_to_xarray"
    ) as mock_cdf:
        mock_dataset = xr.Dataset({"test_var": ([], 42)})
        mock_cdf.return_value = mock_dataset

        input_files = ["imap_mag_l2-calibration-matrices_20251017_v001.cdf"]
        combiner = AncillaryCombiner(input_files, "20251031")  # String date

        assert combiner.expected_end_date == np.datetime64("2025-10-31")


def test_dataset_with_epoch_dimension_error(mocks, mag_calibration_dataset):
    """Test error when input dataset has epoch dimension."""
    # Create a dataset with epoch dimension
    dataset_with_epoch = mag_calibration_dataset.copy()
    dataset_with_epoch = dataset_with_epoch.expand_dims("epoch")

    mocks["read_cdf"].return_value = dataset_with_epoch

    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251017_v001.cdf",
    )

    with pytest.raises(ValueError, match="input dataset has epoch dimension"):
        AncillaryCombiner(input_example, "20251031")


def test_glows_json_file_processing():
    """Test GLOWS JSON file processing."""
    # Create a temporary JSON file
    json_data = {
        "active_bad_angle_flags": [True, False, True, False],
        "active_bad_time_flags": [True, True, False],
        "sunrise_offset": 0.5,
        "sunset_offset": -0.3,
        "thresholds": {"uv_source_limit": 5.0, "excluded_region_limit": 10.0},
        "simple_value": 42,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_data, f)
        temp_path = f.name

    try:
        combiner = GlowsAncillaryCombiner([], "20250925")
        dataset = combiner.convert_json_to_dataset(temp_path)

        # Check that JSON data was converted properly
        assert "active_bad_angle_flags" in dataset.data_vars
        assert "active_bad_time_flags" in dataset.data_vars
        assert "sunrise_offset" in dataset.data_vars
        assert "sunset_offset" in dataset.data_vars
        assert "thresholds_uv_source_limit" in dataset.data_vars
        assert "thresholds_excluded_region_limit" in dataset.data_vars
        assert "simple_value" in dataset.data_vars

        # Check values
        assert list(dataset["active_bad_angle_flags"].values) == [
            True,
            False,
            True,
            False,
        ]
        assert float(dataset["sunrise_offset"].values) == 0.5
        assert float(dataset["thresholds_uv_source_limit"].values) == 5.0

    finally:
        Path(temp_path).unlink()


def test_glows_unknown_file_type():
    """Test error for unknown GLOWS file type."""
    with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
        temp_path = f.name

    try:
        combiner = GlowsAncillaryCombiner([], "20250925")
        with pytest.raises(ValueError, match="Unknown GLOWS ancillary file type"):
            combiner.convert_file_to_dataset(temp_path)
    finally:
        Path(temp_path).unlink()


def test_convert_json_with_nested_lists():
    """Test JSON conversion with nested lists and complex structures."""
    json_data = {
        "list_data": [1, 2, 3, 4],
        "nested_dict": {"inner_list": [10, 20, 30], "inner_scalar": 99},
        "tuple_data": (5, 6, 7),
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_data, f)
        temp_path = f.name

    try:
        combiner = AncillaryCombiner([], "20251031")
        dataset = combiner.convert_json_to_dataset(temp_path)

        # Check list handling
        assert "list_data" in dataset.data_vars
        assert list(dataset["list_data"].values) == [1, 2, 3, 4]
        assert dataset["list_data"].dims == ("dim_list_data",)

        # Check nested dict flattening
        assert "nested_dict_inner_list" in dataset.data_vars
        assert "nested_dict_inner_scalar" in dataset.data_vars
        assert list(dataset["nested_dict_inner_list"].values) == [10, 20, 30]
        assert dataset["nested_dict_inner_list"].dims == ("dim_nested_dict_inner_list",)
        assert dataset["nested_dict_inner_scalar"].dims == ()

    finally:
        Path(temp_path).unlink()


def test_glows_ancillary_combiner_with_processing_input():
    """Test GlowsAncillaryCombiner with ProcessingInput instead of list."""
    input_files = AncillaryInput("imap_glows_map-of-excluded-regions_20250923_v002.dat")

    with patch("imap_data_access.AncillaryFilePath.construct_path") as mock_path:
        # Create a temporary excluded regions file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".dat",
            prefix="imap_glows_map-of-excluded-regions",
            delete=False,
        ) as f:
            f.write("# longitude latitude\n")
            f.write("10.0 20.0\n")
            f.write("30.0 40.0\n")
            temp_path = f.name

        try:
            mock_path.return_value = temp_path
            combiner = GlowsAncillaryCombiner(input_files, "20250925")

            assert len(combiner.timestamped_data) == 1
            assert (
                "ecliptic_longitude_deg"
                in combiner.timestamped_data[0].dataset.data_vars
            )

        finally:
            Path(temp_path).unlink()
