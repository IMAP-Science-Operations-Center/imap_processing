"""Shared modules for MAG tests"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.mag.constants import VecSec
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.spice.time import TTJ2000_EPOCH


@pytest.fixture
def validation_l1a():
    current_directory = Path(__file__).parent
    test_file = current_directory / "validation" / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l1a = mag_l1a(test_file)
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


@pytest.fixture
def mag_test_l1b_calibration_data():
    imap_dir = Path(__file__).parent
    cal_file = (
        imap_dir
        / "validation"
        / "calibration"
        / "imap_mag_l1b-calibration_20240229_v001.cdf"
    )
    calibration_data = load_cdf(cal_file)
    return calibration_data


@pytest.fixture
def mag_test_l2_data():
    imap_dir = Path(__file__).parent
    cal_file = (
        imap_dir
        / "validation"
        / "calibration"
        / "imap_mag_l2-calibration-matrices_20251017_v004.cdf"
    )
    calibration_data = load_cdf(cal_file)

    offsets_data = load_cdf(
        imap_dir
        / "validation"
        / "calibration"
        / "imap_mag_l2-offsets-norm_20251017_20251017_v001.cdf"
    )

    return calibration_data, offsets_data


def mag_generate_l1b_from_csv(df, logical_source):
    length = len(df.index)
    dataset = mag_l1a_dataset_generator(length)

    dataset["vectors"].data = np.array(df[["x", "y", "z", "range"]])
    dataset["compression_flags"].data = np.array(
        df[["compression", "compression_width"]]
    )

    epoch = [np.datetime64(t) - np.datetime64(TTJ2000_EPOCH) for t in df["t"]]
    epoch_ns = [(e / np.timedelta64(1, "ns")).astype(np.int64) for e in epoch]
    dataset.coords["epoch"] = xr.DataArray(epoch_ns, name="epoch", dims=["epoch"])

    dataset.attrs["Logical_source"] = logical_source
    dataset.attrs["vectors_per_second"] = f"{epoch_ns[0]}:2"

    return dataset


def generate_test_epoch(
    end, vectors_per_second: list[VecSec], starting_point=0, gaps=None
):
    spacing = 1 / vectors_per_second[0].value
    output = np.array([])
    prev = starting_point
    if gaps:
        for index, gap in enumerate(gaps):
            if len(vectors_per_second) != 1:
                spacing = 1 / vectors_per_second[index].value
            output = np.concatenate(
                (output, np.arange(prev, gap[0] + spacing, step=spacing) * 1e9)
            )
            prev = gap[1]
        spacing = 1 / vectors_per_second[-1].value
    output = np.concatenate(
        (output, np.arange(prev, end + spacing, step=spacing) * 1e9)
    )

    return output
