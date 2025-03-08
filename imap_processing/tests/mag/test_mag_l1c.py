import numpy as np
import pytest
import xarray as xr

from imap_processing.mag.l1c.interpolation_methods import InterpolationFunction
from imap_processing.mag.l1c.mag_l1c import generate_timeline, mag_l1c, find_gaps, \
    generate_missing_timestamps, find_all_gaps, interpolate_gaps



@pytest.fixture(scope="module")
def mag_l1b_dataset():
    epoch = xr.DataArray(np.arange(0.1, 5.1, step=0.5)*1e9, name="epoch", dims=["epoch"])
    direction = xr.DataArray(np.arange(4), name="direction", dims=["direction"])
    vectors = xr.DataArray(
        np.array([[i, i, i, 2] for i in range(1, 11)]),
        dims=["epoch", "direction"],
        coords={"epoch": epoch, "direction": direction},
    )

    vectors[0, :] = np.array([1, 1, 1, 0])

    output_dataset = xr.Dataset(
        coords={"epoch": epoch, "direction": direction},
    )
    output_dataset["vectors"] = vectors

    return output_dataset

def test_interpolate_gaps(mag_l1b_dataset):
    timeline = np.array([0, 0.5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]) * 1e9
    print(mag_l1b_dataset)
    gaps = np.array([[2, 4]]) * 1e9

    output = interpolate_gaps(mag_l1b_dataset, gaps, timeline, InterpolationFunction.linear)
    # TODO add range back in
    expected_output = np.array([[5.8, 5.8, 5.8], [6.8, 6.8, 6.8], [7.8, 7.8, 7.8]])

    assert np.allclose(output, expected_output)


def test_mag_attributes(mag_l1b_dataset):
    # Fixture from test_mag_l1b.py, since L1A and L1B are very similar
    mag_l1b_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-mago"]

    output = mag_l1c(mag_l1b_dataset, mag_l1b_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-mago"

    mag_l1b_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-magi"]

    output = mag_l1c(mag_l1b_dataset, mag_l1b_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-magi"

    assert output.attrs["Data_level"] == "L1C"


def test_find_all_gaps():
    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 5, 5.5]) * 1e9
    expected_timeline = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])

    vectors_per_second_attr = "0:2"
    output = find_all_gaps(epoch_test, vectors_per_second_attr)
    expected_gaps = np.array([[2, 5]]) * 1e9
    print(output)
    assert np.array_equal(output, expected_gaps)

    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 4, 4.25, 4.5, 4.75, 5.5]) * 1e9
    vectors_per_second_attr = "0:2,4000000000:4"
    expected_timeline = np.array(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5]
    )
    expected_gaps = np.array([[2, 4], [4.75, 5.5]]) * 1e9
    output = find_all_gaps(epoch_test, vectors_per_second_attr)
    assert np.array_equal(output, expected_gaps)

def test_find_gaps():
    # Test should be in ns
    epoch_test = np.array([0, 0.5, 2, 3.5]) * 1e9
    gaps = find_gaps(epoch_test, 2)
    expected_return = np.array([[0.5, 2], [2, 3.5]]) * 1e9

    assert np.array_equal(gaps, expected_return)

    epoch_test = np.array([0, 0.5, 2, 2.5, 3, 4, 4.5, 5]) * 1e9
    gaps = find_gaps(epoch_test, 2)
    expected_return = np.array([[0.5, 2], [3, 4]]) * 1e9

    assert np.array_equal(gaps, expected_return)

    epoch_test = np.array([0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 3]) * 1e9
    gaps = find_gaps(epoch_test, 4)
    expected_return = np.array([[0.5, 1], [2, 3]]) * 1e9

    assert np.array_equal(gaps, expected_return)


def test_generate_timeline():
    epoch_test = np.array([0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 3]) * 1e9
    vecsec = "0:4"
    expected_output = np.array([0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]) * 1e9
    output = generate_timeline(epoch_test, vecsec)

    assert np.array_equal(output, expected_output)

    epoch_test = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) * 1e9
    vecsec = "1000000000:2"

    output = generate_timeline(epoch_test, vecsec)
    print(output)
    assert np.array_equal(output, epoch_test)

    epoch_test = np.array([1, 1.5, 2, 2.5, 3, 4, 4.5, 5]) * 1e9
    vecsec = "1000000000:2"

    expected_output = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) * 1e9
    output = generate_timeline(epoch_test, vecsec)
    assert np.array_equal(output, expected_output)