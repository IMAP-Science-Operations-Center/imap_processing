import numpy as np
import pytest
import xarray as xr

from imap_processing.mag.l1c.mag_l1c import mag_l1c


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
