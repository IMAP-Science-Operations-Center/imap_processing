import json

import numpy as np
import pytest

from imap_processing.codice.utils import (
    calculate_acq_time_per_step,
    get_collapse_pattern_shape,
    get_counters_aggregated_pattern,
)

pytestmark = pytest.mark.external_test_data


def test_codice_non_zero_patterns(codice_lut_path):
    """Test L1A collapse Lo and Hi non-zero patterns.

    This is mainly checking for expected row indices of non-zero
    of Lo and Hi collapse patterns. This doesn't check for unique
    values in rows and column. It returns shape of data as it is,
    (row, column). This is different from collapse pattern shape
    which is tested in `test_get_collapse_pattern_shape`.
    """
    sci_lut_path = codice_lut_path(descriptor="l1a-sci-lut")[0]

    sci_lut = json.loads(sci_lut_path.read_text())
    table_id = "3978152295"
    assert table_id in sci_lut

    collapse_lo = sci_lut[table_id]["collapse_lo"]

    # expected non-zero row indices for each collapse_lo matrix
    expected_lo_non_zero_rows = {
        "0": [1, 2, 3, 23, 24],
        # "1" aggregation is tested separately below
        "2": list(range(1, 25)),
        "3": [1, 2, 3, 23, 24, 28, 31],
        "4": list(range(4, 23)),
        "5": [1, 2, 3, 23, 24],
        "6": list(range(4, 23)),
        "7": [1, 2, 3, 23, 24],
        "8": list(range(4, 23)),
    }
    for key in collapse_lo.keys():
        if key == "1":
            continue
        # check matrix shape is uniform across all keys
        arr = np.array(collapse_lo[f"{key}"]["matrix"])
        assert arr.shape == (32, 12)

        # check non-zero row indices match expected
        non_zero_rows = np.where(arr.any(axis=1))[0].tolist()
        if key in expected_lo_non_zero_rows:
            assert non_zero_rows == expected_lo_non_zero_rows[key]

    # Test Lo aggregation separately as its structure is different
    # instrument counts stores data as each variable in a separate key
    key = "1"
    for variable_name in collapse_lo[key]["variables"]:
        arr = np.array(collapse_lo[key]["variables"][variable_name])
        assert arr.shape == (12,)

    hi_collapse = sci_lut[table_id]["collapse_hi"]

    # expected non-zero row indices for each collapse_hi matrix
    # actual non-zero rows observed in the JSON collapse_hi matrices
    expected_hi_non_zero_rows = {
        # Tested Hi aggregated separately below
        # "0": [0, 1, 2, 4, 5, 6, 9, 12, 13, 14, 15],
        "1": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "2": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "4": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "7": [0, 1, 2, 3, 4, 5],
        "9": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "10": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
    }
    for key in hi_collapse.keys():
        if key == "0":
            continue
        arr = np.array(hi_collapse[f"{key}"]["matrix"])
        assert arr.shape == (16, 24)
        non_zero_rows = np.where(arr.any(axis=1))[0].tolist()
        if key in expected_hi_non_zero_rows:
            assert non_zero_rows == expected_hi_non_zero_rows[key]

    # Test Hi aggregated separately as its structure is different
    key = "0"
    for variable_name in hi_collapse[key]["variables"]:
        arr = np.array(hi_collapse[key]["variables"][variable_name])
        assert arr.shape == (24,)


def test_get_collapse_pattern_shape(codice_lut_path):
    """Test collapse pattern shapes used to reshape data.

    Here, we expact the shape to be in this order:
        (num_spin_sectors, num_positions)
    """
    sci_lut_path = codice_lut_path(descriptor="l1a-sci-lut")[0]

    table_id = "3978152295"
    sci_lut_data = json.loads(sci_lut_path.read_text()).get(table_id)

    # Lo instrument counts - singles
    column_collapsed_example = get_collapse_pattern_shape(
        sci_lut_data,
        sensor_id=0,
        collapse_table_id=2,
    )
    assert column_collapsed_example == (6, 24)

    # Hi omni
    aggre_counts = get_collapse_pattern_shape(
        sci_lut_data, sensor_id=1, collapse_table_id=2
    )
    assert aggre_counts == (1,)

    # Hi aggregated counts
    collapsed_row_example = get_counters_aggregated_pattern(
        sci_lut_data,
        sensor_id=1,
        collapse_table_id=0,
    )
    for variable in collapsed_row_example.keys():
        # All Hi aggregated variables should have
        # collapsed to one column cell.
        assert collapsed_row_example[variable] == 1
    # LoSW priority
    row_collapsed_example = get_collapse_pattern_shape(
        sci_lut_data, sensor_id=0, collapse_table_id=4
    )
    assert row_collapsed_example == (12, 1)

    # Lo SW angular
    non_collapsed_example = get_collapse_pattern_shape(
        sci_lut_data, sensor_id=0, collapse_table_id=7
    )
    assert non_collapsed_example == (12, 5)


def test_acquisition_time(codice_lut_path):
    sci_lut_path = codice_lut_path(descriptor="l1a-sci-lut")[0]
    sci_lut_data = json.loads(sci_lut_path.read_text())
    table_id = "3978152295"
    low_stepping_tab = sci_lut_data[table_id]["lo_stepping_tab"]
    acq_time_per_step = calculate_acq_time_per_step(low_stepping_tab)
    expected_acq_times = np.array(
        [
            0.57870833,
            0.57870833,
            0.57870833,
            0.57870833,
            0.28935417,
            0.28935417,
            0.28935417,
            0.28935417,
            0.28935417,
            0.28935417,
            0.28935417,
            0.28935417,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.14467708,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.11574167,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            0.19290278,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    np.testing.assert_allclose(acq_time_per_step, expected_acq_times, rtol=1e-5)
