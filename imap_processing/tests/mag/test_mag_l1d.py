from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.mag.constants import DataMode
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
    offsets = mag_test_l2_data[1]
    dataset = mag_l1a_dataset_generator(3504)
    epoch_vals = offsets["epoch"].data
    vectors_per_second_attr = "0:2,4000000000:4"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = "imap_mag_l1c_norm-mago"
    vectors = np.array([[i, i, i, 2] for i in range(1, 3505)])
    dataset["vectors"].data = vectors

    return dataset


@pytest.fixture
def mag_l1d_test_class(mag_test_l1d_data, norm_dataset):
    fake_data = mag_l1a_dataset_generator(155)

    day = np.datetime64("2025-10-17")
    config = MagL1dConfiguration(mag_test_l1d_data, day)

    # Skip post-init processing
    l1d = MagL1d.__new__(MagL1d)

    l1d.vectors = fake_data["vectors"].data[:, :3]
    l1d.epoch = fake_data["epoch"].data
    l1d.range = fake_data["vectors"].data[:, 3]
    l1d.global_attributes = {}
    l1d.quality_flags = np.zeros(len(norm_dataset["epoch"].data))
    l1d.quality_bitmask = np.zeros(len(norm_dataset["epoch"].data))
    l1d.data_mode = DataMode.BURST
    l1d.magi_vectors = fake_data["vectors"].data[:, :3]
    l1d.magi_range = fake_data["vectors"].data[:, 3]
    l1d.config = config
    l1d.spin_offsets = None
    l1d.magnitude = None

    return l1d


def test_mag_l1d(mag_test_l1d_data, norm_dataset):
    with (
        patch(
            "imap_processing.mag.l2.mag_l2_data.frame_transform",
            side_effect=lambda *args, **kwargs: args[1],
        ),
        patch(
            "imap_processing.mag.l1d.mag_l1d_data.MagL1d.calculate_spin_offsets",
            side_effect=lambda *args, **kwargs: None,
        ),
    ):
        l1d = mag_l1d(
            mag_test_l1d_data,
            norm_dataset,
            norm_dataset,
            np.datetime64("2025-10-17"),
        )
    assert "vectors" in l1d[0].data_vars


def test_offset_vector():
    # offsets are a vector of shape (2, 4, 3)
    raise NotImplementedError


def test_calculate_spin_offsets(
    mag_l1d_test_class, fake_mag_spin_data, furnish_kernels
):
    x_vectors = np.arange(1, 156)
    y_vectors = np.arange(156, 1, -1)
    mag_l1d_test_class.vectors[:, 0] = x_vectors
    mag_l1d_test_class.vectors[:, 1] = y_vectors
    mag_l1d_test_class.frame = ValidFrames.SRF

    kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_wkcp.tf",
        "imap_science_0001.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    # Spins have a length of 15
    mag_l1d_test_class.config.spin_count_calibration = 2
    with furnish_kernels(kernels):
        offsets = mag_l1d_test_class.calculate_spin_offsets()

    expected_epochs = [15, 45, 90, 150]
    assert np.array_equal(offsets["epoch"].data, expected_epochs)

    # pull out the valid full spins from the test data (last few are fudging to get a
    # chunk from 150-155)
    valid_spins = [
        [15, 30],
        [30, 45],
        [45, 60],
        [75, 90],
        [90, 105],
        [135, 150],
        [150, 155],
        [155, 155],
    ]

    expected_x_avg = []
    expected_y_avg = []
    for index in range(0, len(valid_spins), 2):
        x_spin = x_vectors[valid_spins[index][0] : valid_spins[index + 1][1]]

        expected_x_avg.append(np.nanmean(x_spin))
        y_spin = y_vectors[valid_spins[index][0] : valid_spins[index + 1][1]]
        expected_y_avg.append(np.nanmean(y_spin))

    np.testing.assert_allclose(offsets["x_offset"].data, expected_x_avg)
    np.testing.assert_allclose(offsets["y_offset"].data, expected_y_avg)


def test_apply_spin_offsets(mag_l1d_test_class, fake_mag_spin_data, furnish_kernels):
    x_vectors = np.zeros(155)
    y_vectors = np.zeros(155)
    z_vectors = np.zeros(155)

    mag_l1d_test_class.vectors[:, 0] = x_vectors
    mag_l1d_test_class.vectors[:, 1] = y_vectors
    mag_l1d_test_class.vectors[:, 2] = z_vectors
    mag_l1d_test_class.frame = ValidFrames.SRF

    mag_l1d_test_class.config.spin_count_calibration = 2

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

    mag_l1d_test_class.spin_offsets = offset_dataset
    output_vectors = mag_l1d_test_class.apply_spin_offsets(mag_l1d_test_class.vectors)

    assert mag_l1d_test_class.vectors.shape == expected_output.shape
    assert np.array_equal(output_vectors, expected_output)

    # Check that the spin average application factor is being applied
    offset_dataset["x_offset"].data = offset_dataset["x_offset"].data * 2
    offset_dataset["y_offset"].data = offset_dataset["y_offset"].data * 2

    mag_l1d_test_class.config.spin_average_application_factor = 0.5
    output_vectors = mag_l1d_test_class.apply_spin_offsets(mag_l1d_test_class.vectors)
    assert np.array_equal(output_vectors, expected_output)
