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
    compression = xr.DataArray(np.arange(2), name="compression", dims=["compression"])

    direction_label = xr.DataArray(
        direction.values.astype(str),
        name="direction_label",
        dims=["direction_label"],
    )

    compression_label = xr.DataArray(
        compression.values.astype(str),
        name="compression_label",
        dims=["compression_label"],
    )

    vectors = xr.DataArray(
        np.zeros((20, 4)),
        dims=["epoch", "direction"],
        coords={"epoch": epoch, "direction": direction},
    )
    compression_flags = xr.DataArray(
        np.zeros((20, 2), dtype=np.int8), dims=["epoch", "compression"]
    )

    vectors[0, :] = np.array([1, 1, 1, 0])

    output_dataset = xr.Dataset(
        coords={"epoch": epoch, "direction": direction, "compression": compression},
    )
    output_dataset["vectors"] = vectors
    output_dataset["compression_flags"] = compression_flags
    output_dataset["direction_label"] = direction_label
    output_dataset["compression_label"] = compression_label

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
        Path(__file__).parent / "imap_mag_l1a_norm-magi_20251017_v001.cdf"
    )
    l1b_dataset = mag_l1b(l1a_cdf, "v001")

    output_path = write_cdf(l1b_dataset)

    assert Path.exists(output_path)


def test_mag_compression_scale(mag_l1a_dataset):
    test_calibration = np.array(
        [
            [2.2972202, 0.0, 0.0],
            [0.00348625, 2.23802879, 0.0],
            [-0.00250788, -0.00888437, 2.24950008],
        ]
    )
    mag_l1a_dataset["vectors"][0, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][1, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][2, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][3, :] = np.array([1, 1, 1, 0])

    mag_l1a_dataset["compression_flags"][0, :] = np.array([1, 16], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][1, :] = np.array([0, 0], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][2, :] = np.array([1, 18], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][3, :] = np.array([1, 14], dtype=np.int8)

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]
    output = mag_l1b(mag_l1a_dataset, "v001")

    calibrated_vectors = np.matmul(np.array([1, 1, 1]), test_calibration)
    # 16 bit width is the standard
    assert np.allclose(output["vectors"].data[0][:3], calibrated_vectors)
    # uncompressed data is uncorrected
    assert np.allclose(output["vectors"].data[1][:3], calibrated_vectors)

    # width of 18 should be multiplied by 1/4
    scaled_vectors = calibrated_vectors * 1 / 4
    # should be corrected
    assert np.allclose(output["vectors"].data[2][:3], scaled_vectors)

    # width of 14 should be multiplied by 4
    scaled_vectors = calibrated_vectors * 4
    assert np.allclose(output["vectors"].data[3][:3], scaled_vectors)
