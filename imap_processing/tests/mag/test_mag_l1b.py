from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.mag.l1b.mag_l1b import mag_l1b, mag_l1b_processing


@pytest.fixture(scope="module")
def mag_l1a_dataset():
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


def test_mag_processing(mag_l1a_dataset):
    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]

    mag_l1b = mag_l1b_processing(mag_l1a_dataset)

    np.testing.assert_allclose(
        mag_l1b["vectors"][0].values, [2.29819857, 2.22914442, 2.24950008, 0]
    )
    np.testing.assert_allclose(mag_l1b["vectors"][1].values, [0, 0, 0, 0])

    assert mag_l1b["vectors"].values.shape == mag_l1a_dataset["vectors"].values.shape

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-magi"]

    mag_l1b = mag_l1b_processing(mag_l1a_dataset)

    np.testing.assert_allclose(
        mag_l1b["vectors"][0].values, [2.27615106, 2.22638234, 2.24382211, 0]
    )
    np.testing.assert_allclose(mag_l1b["vectors"][1].values, [0, 0, 0, 0])

    assert mag_l1b["vectors"].values.shape == mag_l1a_dataset["vectors"].values.shape


def test_mag_attributes(mag_l1a_dataset):
    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]

    output = mag_l1b(mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1b_norm-mago"

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_burst-magi"]

    output = mag_l1b(mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1b_burst-magi"

    assert output.attrs["Data_level"] == "L1B"


def test_cdf_output():
    l1a_cdf = load_cdf(
        Path(__file__).parent / "imap_mag_l1a_burst-magi_20231025_v001.cdf"
    )
    l1b_dataset = mag_l1b(l1a_cdf, "v001")

    output_path = write_cdf(l1b_dataset)

    assert Path.exists(output_path)
