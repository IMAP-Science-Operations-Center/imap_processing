import json

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.codice.utils import (
    calculate_acq_time_per_step,
    get_collapse_pattern_shape,
)

pytestmark = pytest.mark.external_test_data


def test_codice_non_zero_patterns():
    """Test L1A collapse Lo and Hi non-zero patterns.

    This is mainly checking for expected row indices of non-zero
    of Lo and Hi collapse patterns. This doesn't check for unique
    values in rows and column. It returns shape of data as it is,
    (row, column). This is different from collapse pattern shape
    which is tested in `test_get_collapse_pattern_shape`.
    """
    l1a_sci_lut_path = (
        imap_module_directory
        / "tests/codice/data/l1a_lut"
        / "imap_codice_l1a-sci-lut_20251007_v001.json"
    )

    sci_lut = json.loads(l1a_sci_lut_path.read_text())
    table_id = "3952862729"
    assert table_id in sci_lut

    collapse_lo = sci_lut[table_id]["collapse_lo"]

    # expected non-zero row indices for each collapse_lo matrix
    expected_lo_non_zero_rows = {
        "0": [1, 2, 3, 23, 24],
        # "1" is tested separately below
        "2": list(range(1, 25)),
        "3": [1, 2, 3, 23, 24, 28, 31],
        "4": list(range(4, 23)),
        "5": [1, 2, 3, 23, 24],
        "6": list(range(4, 23)),
        "7": [1, 2, 3, 23, 24],
        "8": list(range(4, 23)),
    }
    for key in collapse_lo.keys():
        # instrument counts stores data as each variable in a separate key
        if key == "1" and "variables" in collapse_lo[key].keys():
            for variable_name in collapse_lo[key]["variables"]:
                arr = np.array(collapse_lo[key]["variables"][variable_name])
                assert arr.shape == (12,)
            continue

        # check matrix shape is uniform across all keys
        arr = np.array(collapse_lo[f"{key}"]["matrix"])
        assert arr.shape == (32, 12)

        # check non-zero row indices match expected
        non_zero_rows = np.where(arr.any(axis=1))[0].tolist()
        if key in expected_lo_non_zero_rows:
            assert non_zero_rows == expected_lo_non_zero_rows[key]

    hi_collapse = sci_lut[table_id]["collapse_hi"]

    # expected non-zero row indices for each collapse_hi matrix
    # actual non-zero rows observed in the JSON collapse_hi matrices
    expected_hi_non_zero_rows = {
        "0": [0, 1, 2, 4, 5, 6, 9, 12, 13, 14, 15],
        "1": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "2": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "4": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "7": [0, 1, 2, 3, 4, 5],
        "9": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        "10": [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
    }
    for key in hi_collapse.keys():
        arr = np.array(hi_collapse[f"{key}"]["matrix"])
        assert arr.shape == (16, 24)
        non_zero_rows = np.where(arr.any(axis=1))[0].tolist()
        if key in expected_hi_non_zero_rows:
            assert non_zero_rows == expected_hi_non_zero_rows[key]


def test_get_collapse_pattern_shape():
    """Test collapse pattern shapes used to reshape data.

    Here, we expact the shape to be in this order:
        (num_spin_sectors, num_positions)
    """
    lut_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_lut"
        / "imap_codice_l1a-sci-lut_20251007_v001.json"
    )
    table_id = "3952862729"
    sci_lut_data = json.loads(lut_file_path.read_text()).get(table_id)

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
    collapsed_row_example = get_collapse_pattern_shape(
        sci_lut_data,
        sensor_id=1,
        collapse_table_id=0,
    )
    assert collapsed_row_example == (1, 11)

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


def test_acquisition_time():
    sci_lut_path = (
        imap_module_directory
        / "tests/codice/data/l1a_lut"
        / "imap_codice_l1a-sci-lut_20251007_v001.json"
    )
    sci_lut_data = json.loads(sci_lut_path.read_text())
    table_id = "3952862729"
    low_stepping_tab = sci_lut_data[table_id]["lo_stepping_tab"]
    acq_time_per_step = calculate_acq_time_per_step(low_stepping_tab)
    expected_acq_times = (
        np.array(
            [
                578.70833333,
                578.70833333,
                578.70833333,
                578.70833333,
                289.35416667,
                289.35416667,
                289.35416667,
                289.35416667,
                289.35416667,
                289.35416667,
                289.35416667,
                289.35416667,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                192.90277778,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                144.67708333,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                115.74166667,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
                95.69438889,
            ]
        )
        / 1e3
    )
    np.testing.assert_allclose(acq_time_per_step, expected_acq_times, rtol=1e-5)
