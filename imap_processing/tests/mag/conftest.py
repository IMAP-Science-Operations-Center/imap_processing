"""Shared modules for MAG tests"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.mag.l1a.mag_l1a import mag_l1a


@pytest.fixture()
def validation_l1a():
    current_directory = Path(__file__).parent
    test_file = current_directory / "validation" / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l1a = mag_l1a(test_file, "v000")
    return l1a


def mag_l1a_dataset_generator(length):
    epoch = xr.DataArray(np.arange(length), name="epoch", dims=["epoch"])
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
        np.zeros((length, 4)),
        dims=["epoch", "direction"],
        coords={"epoch": epoch, "direction": direction},
    )
    compression_flags = xr.DataArray(
        np.zeros((length, 2), dtype=np.int8), dims=["epoch", "compression"]
    )

    output_dataset = xr.Dataset(
        coords={"epoch": epoch, "direction": direction, "compression": compression},
    )
    output_dataset["vectors"] = vectors
    output_dataset["compression_flags"] = compression_flags
    output_dataset["direction_label"] = direction_label
    output_dataset["compression_label"] = compression_label
    output_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]

    return output_dataset


@pytest.fixture()
def mag_test_calibration_data():
    imap_dir = Path(__file__).parent
    cal_file = imap_dir / "validation" / "imap_calibration_mag_20240229_v01.cdf"
    calibration_data = load_cdf(cal_file)
    return calibration_data


def generate_test_epoch(end, vectors_per_second, starting_point=0, gaps=None):
    spacing = 1 / vectors_per_second[0]
    output = np.array([])
    prev = starting_point
    if gaps:
        for index, gap in enumerate(gaps):
            if len(vectors_per_second) != 1:
                spacing = 1 / vectors_per_second[index]
            output = np.concatenate(
                (output, np.arange(prev, gap[0] + spacing, step=spacing) * 1e9)
            )
            prev = gap[1]
        spacing = 1 / vectors_per_second[-1]
    output = np.concatenate(
        (output, np.arange(prev, end + spacing, step=spacing) * 1e9)
    )

    return output
