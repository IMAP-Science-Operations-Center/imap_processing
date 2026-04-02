from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from imap_data_access.processing_input import ProcessingInputCollection

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import write_cdf
from imap_processing.cli import Mag
from imap_processing.mag.constants import FILLVAL, DataMode
from imap_processing.mag.l1d.mag_l1d import mag_l1d
from imap_processing.mag.l1d.mag_l1d_data import MagL1d, MagL1dConfiguration
from imap_processing.mag.l2.mag_l2_data import ValidFrames
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator


@pytest.fixture
def fake_mag_spin_data(spice_test_data_path, use_test_spin_data_csv):
    """Generate fake spin dataframe for testing"""
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    use_test_spin_data_csv([fake_spin_path])
    return fake_spin_path


@pytest.fixture
def norm_dataset(mag_test_l2_data):
    dataset = mag_l1a_dataset_generator(165)
    epoch_vals = np.arange(165)
    vectors_per_second_attr = "0:2,4000000000:4"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = "imap_mag_l1c_norm-mago"
    vectors = np.array([[i, i, i, 2] for i in range(1, 166)])
    dataset["vectors"].data = vectors

    return dataset


@pytest.fixture
def mag_l1d_test_class(mag_test_l1d_data):
    fake_data = mag_l1a_dataset_generator(155)

    day = np.datetime64("2000-01-01")
    config = MagL1dConfiguration(mag_test_l1d_data, day)

    # Skip post-init processing
    l1d = MagL1d.__new__(MagL1d)

    l1d.vectors = fake_data["vectors"].data[:, :3]
    l1d.epoch = fake_data["epoch"].data
    l1d.range = fake_data["vectors"].data[:, 3]
    l1d.global_attributes = {}
    l1d.quality_flags = np.zeros(len(fake_data["epoch"].data))
    l1d.quality_bitmask = np.zeros(len(fake_data["epoch"].data))
    l1d.data_mode = DataMode.BURST
    l1d.magi_epoch = fake_data["epoch"].data + 1
    l1d.magi_vectors = fake_data["vectors"].data[:, :3] + 10
    l1d.magi_range = fake_data["vectors"].data[:, 3]
    l1d.config = config
    l1d.spin_offsets = None
    l1d.magnitude = np.zeros(len(fake_data["epoch"].data))
    l1d.frame = ValidFrames.SRF

    return l1d


def test_mag_l1d(mag_test_l1d_data, norm_dataset, furnish_kernels, fake_mag_spin_data):
    norm_magi = norm_dataset.copy()
    norm_magi.attrs["Logical_source"] = "imap_mag_l1c_norm-magi"
    burst_magi = norm_dataset.copy()
    burst_magi.attrs["Logical_source"] = "imap_mag_l1c_burst-magi"
    burst_mago = norm_dataset.copy()
    burst_mago.attrs["Logical_source"] = "imap_mag_l1c_burst-mago"

    with (
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l2.mag_l2_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.ttj2000ns_to_met",
            side_effect=lambda *args, **kwargs: args[0],
        ),
    ):
        l1d = mag_l1d(
            [norm_dataset, norm_magi, burst_magi, burst_mago],
            mag_test_l1d_data,
            np.datetime64("2000-01-01"),
        )
    # Should have: 4 norm frames + 4 burst frames + spin offsets + 2 gradiometry offsets

    frame = l1d[0].attrs["Logical_source"].split("-")[-1].lower()
    assert len(l1d) == 11
    assert f"b_{frame}" in l1d[0].data_vars

    # Check that expected logical sources are present
    logical_sources = [ds.attrs.get("Logical_source", "") for ds in l1d]

    # Should include ancillary files
    assert "imap_mag_l1d_spin-offsets" in logical_sources
    assert "imap_mag_l1d_gradiometry-offsets-norm" in logical_sources
    assert "imap_mag_l1d_gradiometry-offsets-burst" in logical_sources


@pytest.mark.parametrize("data_mode", ["norm", "burst"])
def test_mag_l1d_attributes(
    mag_test_l1d_data,
    norm_dataset,
    furnish_kernels,
    fake_mag_spin_data,
    data_mode,
):
    """Test that L1D datasets have correct attributes based on frame and mode."""
    # L1D always requires normal mode MAGO and MAGI datasets
    norm_mago = norm_dataset.copy()
    norm_mago.attrs["Logical_source"] = "imap_mag_l1c_norm-mago"

    norm_magi = norm_dataset.copy()
    norm_magi.attrs["Logical_source"] = "imap_mag_l1c_norm-magi"

    input_datasets = [norm_mago, norm_magi]

    # If testing burst mode, add burst datasets as well
    if data_mode == "burst":
        burst_mago = norm_dataset.copy()
        burst_mago.attrs["Logical_source"] = "imap_mag_l1c_burst-mago"

        burst_magi = norm_dataset.copy()
        burst_magi.attrs["Logical_source"] = "imap_mag_l1c_burst-magi"

        input_datasets.extend([burst_mago, burst_magi])

    with (
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l2.mag_l2_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.ttj2000ns_to_met",
            side_effect=lambda *args, **kwargs: args[0],
        ),
    ):
        l1d_datasets = mag_l1d(
            input_datasets,
            mag_test_l1d_data,
            np.datetime64("2000-01-01"),
        )

    # Filter out ancillary datasets and select only datasets matching the data_mode
    science_datasets = [
        ds
        for ds in l1d_datasets
        if "spin-offsets" not in ds.attrs.get("Logical_source", "")
        and "gradiometry-offsets" not in ds.attrs.get("Logical_source", "")
        and f"l1d_{data_mode}-" in ds.attrs.get("Logical_source", "")
    ]

    # Verify we have the expected number of datasets for the mode
    # Each mode produces 4 frames: SRF, DSRF, GSE, RTN
    assert len(science_datasets) == 4, (
        f"Expected 4 L1D {data_mode} datasets, got {len(science_datasets)}"
    )

    for dataset in science_datasets:
        assert "Logical_source" in dataset.attrs
        assert "Data_type" in dataset.attrs
        assert dataset.attrs["Logical_source"].startswith(f"imap_mag_l1d_{data_mode}-")

        # Verify that data_level is correctly set to "l1d" in logical source
        logical_source_parts = dataset.attrs["Logical_source"].split("_")
        assert logical_source_parts[2] == "l1d", (
            f"Expected data_level 'l1d' in Logical_source, "
            f"got '{logical_source_parts[2]}'"
        )

        frame = dataset.attrs["Logical_source"].split("-")[-1].upper()

        vectors_attrs = dataset[f"b_{frame.lower()}"].attrs
        assert "DICT_KEY" in vectors_attrs

        assert f"CoordinateSystemName:{frame}" in vectors_attrs["DICT_KEY"]

        assert "magnitude" in dataset.data_vars
        assert "range" in dataset.data_vars
        assert dataset["magnitude"].attrs["UNITS"] == "nT"
        assert dataset["range"].attrs["DICT_KEY"] == (
            "SPASE>Support>SupportQuantity:InstrumentMode"
        )

    # Test that write_cdf can be called on all datasets
    with patch("imap_processing.cdf.utils.xarray_to_cdf") as mock_xarray_to_cdf:
        for dataset in l1d_datasets:
            write_cdf(dataset)

        # Verify xarray_to_cdf was called for each dataset
        assert mock_xarray_to_cdf.call_count == len(l1d_datasets)

    # Test that Mag.post_processing can be called on the datasets
    mag_processor = Mag(
        data_level="l1d",
        data_descriptor="all",
        dependency_str="[]",
        start_date="20000101",
        repointing=None,
        version="v001",
        upload_to_sdc=False,
    )

    mock_dependencies = ProcessingInputCollection()

    with patch("imap_processing.cdf.utils.xarray_to_cdf"):
        mag_processor.post_processing(l1d_datasets, mock_dependencies)


def test_offset_vector():
    # offsets are a vector of shape (2, 4, 3)
    offsets = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[-1, -1, -1], [-2, -2, -2], [-3, -3, -3], [-4, -4, -4]],
        ]
    )
    test_vector = np.array([1, 2, 3, 3])

    expected_vector = [5, 6, 7, 3]
    output_vector = MagL1d.apply_calibration_offset_single_vector(
        test_vector, offsets, False
    )

    assert np.array_equal(expected_vector, output_vector)

    test_vector = np.array([1, 2, 3, 0])
    expected_vector = [0, 1, 2, 0]
    output_vector = MagL1d.apply_calibration_offset_single_vector(
        test_vector, offsets, True
    )
    assert np.array_equal(expected_vector, output_vector)


def test_calculate_spin_offsets(
    mag_l1d_test_class, fake_mag_spin_data, furnish_kernels
):
    # Expected to fail, need to merge spice fix
    x_vectors = np.arange(1, 156)
    y_vectors = np.arange(156, 1, -1)
    mag_l1d_test_class.vectors[:, 0] = x_vectors
    mag_l1d_test_class.vectors[:, 1] = y_vectors
    mag_l1d_test_class.frame = ValidFrames.SRF
    mag_l1d_test_class.epoch = mag_l1d_test_class.epoch * 1e9

    kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_130.tf",
        "imap_science_120.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    # Spins have a length of 15
    mag_l1d_test_class.config.spin_count_calibration = 2
    with (
        furnish_kernels(kernels),
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.ttj2000ns_to_met",
            side_effect=lambda *args, **kwargs: args[0] / 1e9,
        ),
    ):
        offsets = mag_l1d_test_class.calculate_spin_offsets()

    expected_epochs = np.array([15.0, 45.0, 75.0, 105.0, 135.0]) * 1e9
    assert np.array_equal(offsets["epoch"].data, expected_epochs)

    indices_to_average = [[15, 45], [45, 75], [75, 105], [105, 135], [135, 150]]
    expected_x_avg = []
    expected_y_avg = []

    for start_index, end_index in indices_to_average:
        expected_x_avg.append(np.nanmean(x_vectors[start_index:end_index]))
        expected_y_avg.append(np.nanmean(y_vectors[start_index:end_index]))

    np.testing.assert_allclose(offsets["x_offset"].data, expected_x_avg)
    np.testing.assert_allclose(offsets["y_offset"].data, expected_y_avg)


def test_apply_spin_offsets(mag_l1d_test_class, fake_mag_spin_data, furnish_kernels):
    vectors = np.zeros((155, 3))
    epoch = np.arange(155)

    spin_average_application_factor = 1.0

    offset_dataset = xr.Dataset()
    offset_dataset["epoch"] = xr.DataArray([15, 45, 90, 150])
    offset_dataset["x_offset"] = xr.DataArray([1, 2, 3, 4])
    offset_dataset["y_offset"] = xr.DataArray([-1, -2, -3, -4])

    expected_output = np.concatenate(
        (
            np.full((45, 3), [-1, 1, 0]),
            np.full((45, 3), [-2, 2, 0]),
            np.full((65, 3), [-3, 3, 0]),
        ),
        axis=0,
    )

    output_vectors = MagL1d.apply_spin_offsets(
        offset_dataset, epoch, vectors, spin_average_application_factor
    )

    assert mag_l1d_test_class.vectors.shape == expected_output.shape
    assert np.array_equal(output_vectors, expected_output)

    # Check that the spin average application factor is being applied
    offset_dataset["x_offset"].data = offset_dataset["x_offset"].data * 2
    offset_dataset["y_offset"].data = offset_dataset["y_offset"].data * 2

    output_vectors = MagL1d.apply_spin_offsets(offset_dataset, epoch, vectors, 0.5)
    assert np.array_equal(output_vectors, expected_output)


def test_calculate_gradiometry_offsets():
    mago_vectors = np.ones((10, 3)) * 10
    mago_epoch = np.arange(10) * 1e9

    magi_vectors = np.ones((10, 3)) * 5
    magi_epoch = mago_epoch + 500

    grad_ds = MagL1d.calculate_gradiometry_offsets(
        mago_vectors, mago_epoch, magi_vectors, magi_epoch
    )

    assert np.array_equal(grad_ds["epoch"].data, mago_epoch)
    assert np.array_equal(grad_ds["gradiometer_offsets"].data.shape, mago_vectors.shape)
    assert np.allclose(grad_ds["gradiometer_offsets"].data, np.full((10, 3), -5.0))

    # Test new fields
    expected_magnitude = np.linalg.norm(np.full((10, 3), -5.0), axis=1)
    assert np.allclose(grad_ds["gradiometer_offset_magnitude"].data, expected_magnitude)
    assert np.array_equal(
        grad_ds["quality_flags"].data, np.zeros(10)
    )  # All below default threshold

    # Test shapes
    assert grad_ds["gradiometer_offset_magnitude"].data.shape == (10,)
    assert grad_ds["quality_flags"].data.shape == (10,)


def test_apply_gradiometry_offsets():
    vectors = np.ones((5, 3)) * 10
    epoch = np.arange(5) * 1e9
    offset_dataset = xr.Dataset()
    offset_dataset["epoch"] = xr.DataArray(epoch)
    offset_dataset["gradiometer_offsets"] = xr.DataArray(np.full((5, 3), [1, 1, 2]))

    gradiometer_factor = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    output = MagL1d.apply_gradiometry_offsets(
        offset_dataset, vectors, gradiometer_factor
    )

    assert np.allclose(output[0] - vectors[0], np.dot([-1, -1, -2], gradiometer_factor))


def test_quality_flags_with_threshold():
    """Test that quality flags are set correctly when magnitude exceeds threshold."""
    mago_vectors = np.array(
        [[10, 10, 10], [10, 10, 10], [1, 1, 1], [1, 1, 1], [10, 10, 10]]
    )
    mago_epoch = np.arange(5) * 1e9

    magi_vectors = np.array([[5, 5, 5], [5, 5, 5], [0, 0, 0], [0, 0, 0], [5, 5, 5]])
    magi_epoch = mago_epoch + 500

    # Set threshold so that only the large differences trigger quality flags
    # Magnitude for [5,5,5] difference is ~8.66, for [1,1,1] difference is ~1.73
    quality_threshold = 5.0

    grad_ds = MagL1d.calculate_gradiometry_offsets(
        mago_vectors, mago_epoch, magi_vectors, magi_epoch, quality_threshold
    )

    # First, second, and fifth vectors should exceed threshold (magnitude ~8.66)
    # Third and fourth vectors should not exceed threshold (magnitude ~1.73)
    expected_flags = np.array([1, 1, 0, 0, 1])
    assert np.array_equal(grad_ds["quality_flags"].data, expected_flags)

    # Test shapes for 5 vectors
    assert grad_ds["gradiometer_offset_magnitude"].data.shape == (5,)
    assert grad_ds["quality_flags"].data.shape == (5,)


def test_skip_gradiometry(norm_dataset, mag_test_l1d_data, fake_mag_spin_data):
    # Set up test data with all_vectors_primary = 0 for MAGO dataset
    norm_magi = norm_dataset.copy()
    norm_magi.attrs["Logical_source"] = "imap_mag_l1c_norm-magi"
    burst_magi = norm_dataset.copy()
    burst_magi.attrs["Logical_source"] = "imap_mag_l1c_burst-magi"
    burst_mago = norm_dataset.copy()
    burst_mago.attrs["Logical_source"] = "imap_mag_l1c_burst-mago"

    # Set all_vectors_primary = 0 for MAGO dataset to disable gradiometry
    norm_dataset.attrs["all_vectors_primary"] = 0
    norm_magi.attrs["all_vectors_primary"] = 1  # MAGI doesn't matter for this check
    burst_mago.attrs["all_vectors_primary"] = 0
    burst_magi.attrs["all_vectors_primary"] = 1

    with (
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l2.mag_l2_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.ttj2000ns_to_met",
            side_effect=lambda *args, **kwargs: args[0] / 1e9,
        ),
        patch.object(MagL1d, "calculate_gradiometry_offsets") as mock_calc,
        patch.object(MagL1d, "apply_gradiometry_offsets") as mock_apply,
    ):
        mag_l1d(
            [norm_dataset, norm_magi, burst_magi, burst_mago],
            mag_test_l1d_data,
            np.datetime64("2000-01-01"),
        )

        # Verify the gradiometry methods were never called
        mock_calc.assert_not_called()
        mock_apply.assert_not_called()


def test_spin_offset_gap_handling():
    """Test improved gap handling in spin calculation."""
    # Test the logic for handling data gaps
    # Create a mock scenario with gaps in spin data

    # Mock spin data with a gap
    spin_phase = np.array([0.1, 0.2, 0.3, np.nan, np.nan, np.nan, 0.1, 0.2, 0.3])

    # Find transitions from nan to number and vice versa
    nan_to_number = np.where(np.diff(np.isnan(spin_phase)) != 0)[0] + 1

    # Should detect gap start at index 3 and gap end at index 6
    expected_transitions = np.array([3, 6])
    assert np.array_equal(nan_to_number, expected_transitions)

    # Test that we can identify gaps that span multiple spins
    gap_start_idx, gap_end_idx = 3, 6
    assert gap_end_idx > gap_start_idx  # Basic gap detection works


@patch("imap_processing.mag.imap_mag_sdc_configuration_v001.ALWAYS_OUTPUT_MAGO", False)
def test_mago_magi_swap_functionality(mag_l1d_test_class):
    """Test MAGO/MAGI swap functionality when ALWAYS_OUTPUT_MAGO is False."""
    mago_vectors = mag_l1d_test_class.vectors.copy()
    mago_epoch = mag_l1d_test_class.epoch.copy()
    magi_vectors = mag_l1d_test_class.magi_vectors.copy()
    magi_epoch = mag_l1d_test_class.magi_epoch.copy()

    with patch(
        "imap_processing.mag.l1d.mag_l1d_data.MagL2L1dBase.truncate_to_24h",
        return_value=None,
    ):
        # Call generate_dataset (this should swap to use MAGI data)
        attributes = ImapCdfAttributes()
        attributes.add_instrument_global_attrs("mag")
        attributes.add_instrument_variable_attrs("mag", "l2")

        result = mag_l1d_test_class.generate_dataset(
            attributes, np.datetime64("2000-01-01")
        )

    # After generate_dataset, original vectors should be restored
    assert np.array_equal(mag_l1d_test_class.vectors, mago_vectors)
    assert np.array_equal(mag_l1d_test_class.epoch, mago_epoch)

    assert np.array_equal(result[mag_l1d_test_class.frame.var_name].data, magi_vectors)
    assert np.array_equal(result["epoch"].data, magi_epoch)


@patch("imap_processing.mag.imap_mag_sdc_configuration_v001.ALWAYS_OUTPUT_MAGO", True)
def test_mago_magi_no_swap_functionality(mag_l1d_test_class):
    """Test MAGO/MAGI swap functionality when ALWAYS_OUTPUT_MAGO is False."""
    mago_vectors = mag_l1d_test_class.vectors.copy()
    mago_epoch = mag_l1d_test_class.epoch.copy()

    with patch(
        "imap_processing.mag.l1d.mag_l1d_data.MagL2L1dBase.truncate_to_24h",
        return_value=None,
    ):
        # Call generate_dataset (this should swap to use MAGI data)
        attributes = ImapCdfAttributes()
        attributes.add_instrument_global_attrs("mag")
        attributes.add_instrument_variable_attrs("mag", "l2")

        result = mag_l1d_test_class.generate_dataset(
            attributes, np.datetime64("2000-01-01")
        )

    # After generate_dataset, original vectors should be restored
    assert np.array_equal(mag_l1d_test_class.vectors, mago_vectors)
    assert np.array_equal(mag_l1d_test_class.epoch, mago_epoch)

    assert np.array_equal(result[mag_l1d_test_class.frame.var_name].data, mago_vectors)
    assert np.array_equal(result["epoch"].data, mago_epoch)


def test_enhanced_gradiometry_with_quality_flags_detailed():
    """Test enhanced gradiometry calculation with quality flags and magnitude."""
    # Test data with known differences
    mago_vectors = np.array(
        [
            [10, 10, 10],  # Large difference case
            [1, 1, 1],  # Small difference case
            [5, 5, 5],  # Medium difference case
        ]
    )
    mago_epoch = np.array([0, 1000000000, 2000000000])  # 1 second intervals

    magi_vectors = np.array(
        [
            [5, 5, 5],  # After interpolation: actual offsets will be computed
            [0.5, 0.5, 0.5],  # based on linear interpolation to MAGo timestamps
            [3, 3, 3],  # See expected_offsets calculation below
        ]
    )
    magi_epoch = mago_epoch + 500000000  # 0.5 second offset

    # Set threshold so medium and large differences exceed it
    quality_threshold = 3.5

    grad_ds = MagL1d.calculate_gradiometry_offsets(
        mago_vectors, mago_epoch, magi_vectors, magi_epoch, quality_threshold
    )

    # Test basic functionality
    assert "gradiometer_offsets" in grad_ds.data_vars
    assert "gradiometer_offset_magnitude" in grad_ds.data_vars
    assert "quality_flags" in grad_ds.data_vars

    # Test shapes
    assert grad_ds["gradiometer_offsets"].shape == (3, 3)
    assert grad_ds["gradiometer_offset_magnitude"].shape == (3,)
    assert grad_ds["quality_flags"].shape == (3,)

    # Test calculated values
    # After interpolation, MAGi values at MAGo timestamps:
    # t=0: [7.25, 7.25, 7.25], t=1: [2.75, 2.75, 2.75], t=2: [1.75, 1.75, 1.75]
    # Offsets = MAGi_interpolated - MAGo
    expected_offsets = np.array(
        [[-2.75, -2.75, -2.75], [1.75, 1.75, 1.75], [-3.25, -3.25, -3.25]]
    )
    expected_magnitudes = np.linalg.norm(expected_offsets, axis=1)
    expected_flags = np.array([1, 0, 1])

    np.testing.assert_allclose(
        grad_ds["gradiometer_offsets"].data, expected_offsets, rtol=1e-10
    )
    np.testing.assert_allclose(
        grad_ds["gradiometer_offset_magnitude"].data, expected_magnitudes, rtol=1e-10
    )
    assert np.array_equal(grad_ds["quality_flags"].data, expected_flags)


def test_rotate_frame_preserves_fillval_and_nan(mag_l1d_test_class):
    """Test that L1D rotate_frame preserves FILLVAL and NaN vectors."""
    mag_l1d_test_class.frame = ValidFrames.MAGO

    vectors = mag_l1d_test_class.vectors.copy()
    magi_vectors = mag_l1d_test_class.magi_vectors.copy()

    # Set some MAGO vectors to FILLVAL and NaN
    vectors[0] = [FILLVAL, FILLVAL, FILLVAL]
    vectors[2] = [np.nan, np.nan, np.nan]
    vectors[4] = [1.0, np.nan, 3.0]
    mag_l1d_test_class.vectors = vectors

    # Set some MAGI vectors to FILLVAL and NaN
    magi_vectors[1] = [FILLVAL, FILLVAL, FILLVAL]
    magi_vectors[3] = [np.nan, np.nan, np.nan]
    mag_l1d_test_class.magi_vectors = magi_vectors

    def mock_frame_transform(
        epoch_et,
        vecs,
        from_frame,
        to_frame,
        allow_spice_noframeconnect,
    ):
        return np.full(vecs.shape, 99.0)

    with patch(
        "imap_processing.mag.l1d.mag_l1d_data.frame_transform",
        side_effect=mock_frame_transform,
    ):
        mag_l1d_test_class.rotate_frame(ValidFrames.SRF)

    assert mag_l1d_test_class.frame == ValidFrames.SRF

    # MAGO: FILLVAL/NaN rows preserved as FILLVAL
    assert np.all(mag_l1d_test_class.vectors[0] == FILLVAL)
    assert np.all(mag_l1d_test_class.vectors[2] == FILLVAL)
    assert mag_l1d_test_class.vectors[4, 1] == FILLVAL
    # Normal MAGO vectors get rotated value
    assert np.all(mag_l1d_test_class.vectors[1] == 99.0)
    assert np.all(mag_l1d_test_class.vectors[3] == 99.0)

    # MAGI: FILLVAL/NaN rows preserved as FILLVAL
    assert np.all(mag_l1d_test_class.magi_vectors[1] == FILLVAL)
    assert np.all(mag_l1d_test_class.magi_vectors[3] == FILLVAL)
    # Normal MAGI vectors get rotated value
    assert np.all(mag_l1d_test_class.magi_vectors[0] == 99.0)
    assert np.all(mag_l1d_test_class.magi_vectors[2] == 99.0)


def test_rotate_frames(mag_l1d_test_class):
    # Reset to initial MAGO frame for this test
    mag_l1d_test_class.frame = ValidFrames.MAGO

    initial_vectors = mag_l1d_test_class.vectors.copy()
    initial_magi_vectors = mag_l1d_test_class.magi_vectors.copy()

    # Mock frame_transform to return identifiable transformed vectors
    def mock_frame_transform(
        epoch_et, vectors, from_frame, to_frame, allow_spice_noframeconnect
    ):
        if from_frame in [ValidFrames.MAGO.spice_frame, ValidFrames.MAGI.spice_frame]:
            return vectors + 100
        else:
            return vectors + 300

    with patch(
        "imap_processing.mag.l1d.mag_l1d_data.frame_transform",
        side_effect=mock_frame_transform,
    ) as mock_transform_l1d:
        target_frame = ValidFrames.SRF
        mag_l1d_test_class.rotate_frame(target_frame)

        # Verify frame_transform was called twice (once for MAGO, once for MAGI)
        assert mock_transform_l1d.call_count == 2

        # First call should be for MAGO vectors
        first_call_args = mock_transform_l1d.call_args_list[0]
        assert first_call_args[1]["from_frame"] == ValidFrames.MAGO.spice_frame
        assert first_call_args[1]["to_frame"] == ValidFrames.SRF.spice_frame

        # Second call should be for MAGI vectors
        second_call_args = mock_transform_l1d.call_args_list[1]
        assert second_call_args[1]["from_frame"] == ValidFrames.MAGI.spice_frame
        assert second_call_args[1]["to_frame"] == ValidFrames.SRF.spice_frame

        # MAGO frame and MAGi frame not necessarily different (and are now the same)

        # Check that MAGO vectors were transformed (+100)
        expected_mago_vectors = initial_vectors + 100
        np.testing.assert_array_equal(mag_l1d_test_class.vectors, expected_mago_vectors)

        # Check that MAGI vectors were transformed (+100)
        expected_magi_vectors = initial_magi_vectors + 100
        np.testing.assert_array_equal(
            mag_l1d_test_class.magi_vectors, expected_magi_vectors
        )

        # Check that frame was updated
        assert mag_l1d_test_class.frame == ValidFrames.SRF
