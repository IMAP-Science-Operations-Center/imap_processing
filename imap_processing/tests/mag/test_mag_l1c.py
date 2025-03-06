import numpy as np
import pytest
import xarray as xr

from imap_processing.mag.l1c.mag_l1c import mag_l1c, generate_timeline


@pytest.fixture(scope="module")
def mag_l1b_dataset():
    epoch = xr.DataArray(np.arange(20), name="epoch", dims=["epoch"])
    direction = xr.DataArray(np.arange(4), name="direction", dims=["direction"])
    vectors = xr.DataArray(
        np.zeros((20, 4)),
        dims=["epoch", "direction"],
        coords={"epoch": epoch, "direction": direction},
    )

    vectors[0, :] = np.array([1, 1, 1, 0])

    output_dataset = xr.Dataset(
        coords={"epoch": epoch, "direction": direction},
    )
    output_dataset["vectors"] = vectors

    return output_dataset


def test_mag_attributes(mag_l1b_dataset):
    # Fixture from test_mag_l1b.py, since L1A and L1B are very similar
    mag_l1b_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-mago"]

    output = mag_l1c(mag_l1b_dataset, mag_l1b_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-mago"

    mag_l1b_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-magi"]

    output = mag_l1c(mag_l1b_dataset, mag_l1b_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-magi"

    assert output.attrs["Data_level"] == "L1C"


def test_generate_timeline():
    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 5, 5.5])
    expected_timeline = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])

    vectors_per_second_attr = "0:2"
    output = generate_timeline(epoch_test, vectors_per_second_attr)

    print(output)

    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 4, 4.25, 4.5, 4.75, 5])
    vectors_per_second_attr = "0:2,4:4"
    expected_timeline = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5])
    output = generate_timeline(epoch_test, vectors_per_second_attr)



