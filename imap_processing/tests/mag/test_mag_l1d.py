import numpy as np
import pytest
from unittest.mock import patch

from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1d.mag_l1d import mag_l1d
from imap_processing.mag.l1d.mag_l1d_data import MagL1d
from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration
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
    fake_data = mag_l1a_dataset_generator(20)

    day = np.datetime64("2025-10-17")
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
    l1d.offsets = mag_test_l1d_data["offsets"].data
    l1d.mago_calibration = retrieve_matrix_from_l2_calibration(
        mag_test_l1d_data, day, use_mago=True
    )
    l1d.magi_calibration = retrieve_matrix_from_l2_calibration(
        mag_test_l1d_data, day, use_mago=False
    )
    l1d.spin_offsets = None
    l1d.magnitude = None

    return l1d

def test_mag_l1d(mag_test_l1d_data, norm_dataset):

    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        side_effect=lambda *args, **kwargs: args[1],
    ):
        l1d = mag_l1d(
            mag_test_l1d_data,
            norm_dataset,
            norm_dataset,
            np.datetime64("2025-10-17"),
        )
    assert "vectors" in l1d[0].data_vars

def test_offset_vector():
    test_vector = [1.0, 2.0, 3.0, 0]
    # offsets are a vector of shape (2, 4, 3)



def test_spin_averaging_calculation(mag_l1d_test_class, fake_mag_spin_data, furnish_kernels):
    kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_wkcp.tf",
        "imap_science_0001.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        mag_l1d_test_class.rotate_frame(ValidFrames.SRF)
        mag_l1d_test_class.calculate_spin_offsets()
        assert mag_l1d_test_class.spin_offsets is not None
