"""Unit tests for CoDICE table-id grouping helpers."""

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.codice.utils import process_by_table_id


def test_process_by_table_id_keeps_voltage_table_1d():
    """Shared support data should not be expanded across epoch groups."""

    unpacked_dataset = xr.Dataset(
        data_vars={
            "view_id": ("epoch", np.array([0, 0], dtype=np.int16)),
            "pkt_apid": ("epoch", np.array([1156, 1156], dtype=np.int16)),
            "plan_id": ("epoch", np.array([7, 7], dtype=np.int16)),
            "plan_step": ("epoch", np.array([3, 3], dtype=np.int16)),
            "table_id": ("epoch", np.array([1, 2], dtype=np.int16)),
            "sample": ("epoch", np.array([11, 22], dtype=np.int16)),
        },
        coords={"epoch": np.array([20, 10], dtype=np.int64)},
    )

    voltage_table = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def _process_fn(group_ds, lut_file, table_id, view_id, apid, plan_id, plan_step):
        return xr.Dataset(
            data_vars={
                "sample": ("epoch", group_ds["sample"].values),
                "voltage_table": ("esa_step", voltage_table),
            },
            coords={"epoch": group_ds["epoch"].values, "esa_step": np.arange(3)},
        )

    processed = process_by_table_id(unpacked_dataset, Path("unused.json"), _process_fn)

    assert processed["epoch"].values.tolist() == [10, 20]
    assert processed["voltage_table"].dims == ("esa_step",)
    np.testing.assert_array_equal(processed["voltage_table"].values, voltage_table)
