from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from cdflib.epochs import CDFepoch
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_processing.cli import Mag


def _tt2000(time: str) -> np.int64:
    date_str, clock_str = time.split("T")
    year, month, day = [int(value) for value in date_str.split("-")]
    hour_minute_second, fraction = clock_str.split(".")
    hour, minute, second = [int(value) for value in hour_minute_second.split(":")]
    nanosecond = int(fraction.ljust(9, "0"))
    return np.int64(
        CDFepoch.compute_tt2000(
            [
                year,
                month,
                day,
                hour,
                minute,
                second,
                nanosecond // 1_000_000,
                (nanosecond // 1_000) % 1_000,
                nanosecond % 1_000,
            ]
        )
    )


def _mag_dataset(logical_source: str, epoch_times: list[str]) -> xr.Dataset:
    epochs = np.array([_tt2000(time) for time in epoch_times], dtype=np.int64)
    return xr.Dataset(
        coords={"epoch": xr.DataArray(epochs, name="epoch", dims=["epoch"])},
        attrs={"Logical_source": logical_source},
    )


def _mag_processor(data_level: str = "l1b") -> Mag:
    return Mag(
        data_level=data_level,
        data_descriptor="burst-mago",
        dependency_str="[]",
        start_date="20251220",
        repointing=None,
        version="v001",
        upload_to_sdc=False,
    )


def _dependencies() -> ProcessingInputCollection:
    return ProcessingInputCollection(
        ScienceInput("imap_mag_l1a_burst-mago_20251220_v001.cdf")
    )


def test_mag_l1b_epoch_validation_accepts_buffer_boundaries():
    dataset = _mag_dataset(
        "imap_mag_l1b_burst-mago",
        ["2025-12-19T23:30:00.000000000", "2025-12-21T00:30:00.000000000"],
    )

    with patch(
        "imap_processing.cli.write_cdf", return_value=Path("output.cdf")
    ) as mock_write_cdf:
        products = _mag_processor().post_processing([dataset], _dependencies())

    assert products == [Path("output.cdf")]
    mock_write_cdf.assert_called_once_with(dataset)


def test_mag_l1b_epoch_validation_rejects_wrong_day_before_write():
    dataset = _mag_dataset(
        "imap_mag_l1b_burst-mago",
        ["2025-12-29T23:59:58.667793920", "2025-12-30T00:00:00.000000000"],
    )

    with (
        patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        pytest.raises(
            ValueError,
            match=(
                "imap_mag_l1b_burst-mago.*Start_date 20251220.*"
                "imap_mag_l1a_burst-mago_20251220_v001.cdf"
            ),
        ),
    ):
        _mag_processor().post_processing([dataset], _dependencies())

    mock_write_cdf.assert_not_called()


@pytest.mark.parametrize(
    "logical_source",
    [
        "imap_mag_l1d_norm-srf",
        "imap_mag_l2_norm-srf",
    ],
)
def test_mag_day_truncated_outputs_reject_epoch_outside_utc_day(logical_source):
    dataset = _mag_dataset(
        logical_source,
        ["2025-12-19T23:59:59.999999999", "2025-12-20T12:00:00.000000000"],
    )

    data_level = logical_source.split("_")[2]
    with (
        patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        pytest.raises(ValueError, match=f"{logical_source}.*Start_date 20251220"),
    ):
        _mag_processor(data_level).post_processing([dataset], _dependencies())

    mock_write_cdf.assert_not_called()


def test_mag_l1d_ancillary_outputs_skip_epoch_validation():
    dataset = _mag_dataset(
        "imap_mag_l1d_spin-offsets",
        ["2025-12-30T00:00:00.000000000"],
    )
    dataset.attrs["Start_date"] = "20251220"
    dataset.attrs["Parents"] = ["imap_mag_l1c_norm-mago_20251220_v001.cdf"]

    _mag_processor("l1d")._validate_output_dataset(dataset)
