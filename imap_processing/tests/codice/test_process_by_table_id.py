"""Unit tests for CoDICE table-id grouping helpers."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.codice.utils import (
    index_sci_lut_files,
    process_by_table_id,
    read_sci_lut,
)


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


def test_transition_uses_each_tables_source(tmp_path):
    """A day's old and new packets must use their own ancillary definitions."""

    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps({"description": "old", "3972558773": {"factor": 2}}))
    new.write_text(json.dumps({"description": "new", "3974360784": {"factor": 3}}))
    sources = index_sci_lut_files([new, old])
    ds = xr.Dataset(
        {
            "view_id": ("epoch", [0, 0, 0]),
            "pkt_apid": ("epoch", [1156, 1156, 1156]),
            "plan_id": ("epoch", [0, 0, 0]),
            "plan_step": ("epoch", [0, 0, 0]),
            "table_id": ("epoch", [3974360784, 3972558773, 3972558773]),
            "sample": ("epoch", [10, 20, 30]),
        },
        coords={"epoch": [30, 10, 20]},
    )

    def process(group, lut, table_id, *_):
        factor = read_sci_lut(lut, table_id)["factor"]
        return xr.Dataset(
            {
                "sample": group["sample"] * factor,
                "voltage_table": ("esa_step", [factor, factor * 2]),
            },
            coords={"epoch": group.epoch},
        )

    result = process_by_table_id(ds, sources, process)
    assert result.epoch.values.tolist() == [10, 20, 30]
    assert result["sample"].values.tolist() == [40, 60, 30]
    assert result["voltage_table"].dims == ("epoch", "esa_step")
    assert result["voltage_table"].values.tolist() == [[2, 4], [2, 4], [3, 6]]
    assert result["voltage_table"].attrs["DEPEND_0"] == "epoch"


def test_lut_index_identical_duplicates_and_conflicts(tmp_path):
    """Cumulative files may repeat a definition, but conflicting copies fail."""

    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps({"description": "old", "1": {"factor": 2}}))
    new.write_text(json.dumps({"description": "new", "1": {"factor": 2}, "2": {}}))
    assert index_sci_lut_files([old, new]) == {"1": old, "2": new}
    new.write_text(json.dumps({"1": {"factor": 3}}))
    with pytest.raises(
        ValueError, match="Conflicting SCI-LUT definitions for table ID 1"
    ):
        index_sci_lut_files([old, new])


def test_transition_missing_id_fails_before_processing():
    """Report all missing IDs without partly processing the dataset."""

    ds = xr.Dataset(
        {
            name: ("epoch", [1, 2])
            for name in ["view_id", "pkt_apid", "plan_id", "plan_step", "table_id"]
        },
        coords={"epoch": [0, 1]},
    )
    process = Mock()
    with pytest.raises(ValueError, match="table IDs: 2"):
        process_by_table_id(ds, {"1": Path("old.json")}, process)
    process.assert_not_called()
