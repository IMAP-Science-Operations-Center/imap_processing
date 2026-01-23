import numpy as np
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.lo import lo_ancillary

ANCILLARY_DIR = imap_module_directory / "tests/lo/test_anc"


def test_read_backgrounds():
    ancillary_file = (
        ANCILLARY_DIR / "imap_lo_hydrogen-background-small_20250101_20270101_v001.csv"
    )
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 240

    # spot check the first row
    first_row = df.iloc[0]
    np.testing.assert_array_equal(
        first_row,
        np.array(
            [
                pd.Timestamp("2025-01-01"),
                473389200.0,
                473472000.0,
                0,
                1,
                "Lo",
                0.0098,
                0.0089,
                0.0118,
                0.0113,
                0.0056,
                0.0008,
                0.0000,
                "rate",
            ],
            dtype=object,
        ),
    )


def test_read_geometric_factor():
    # NOTE: The geometric factors are stored in the main project repository
    #       rather than the test ancillary data because they should be rarely
    #       changed and are considered basically static reference data.

    ancillary_file = (
        imap_module_directory
        / "lo/ancillary_data/imap_lo_hydrogen-geometric-factor_v001.csv"
    )
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 72
    assert "esa_mode" in df.columns
    assert all(df.loc[:35, "esa_mode"] == 0)
    assert all(df.loc[36:, "esa_mode"] == 1)

    # spot check the first row
    np.testing.assert_array_equal(
        df.iloc[0],
        [
            1,
            1,
            0.01633,
            0.00028,
            5.35e-05,
            4.82e-06,
            2.20e-05,
            1.87e-06,
            5.32e-05,
            4.26e-06,
            7.00e-05,
            4.90e-05,
            0,
        ],
    )

    # Make sure we didn't skip any rows mistakenly
    first_hithroughput = df[df["esa_mode"] == 1].iloc[0]
    assert first_hithroughput["incident_E-Step"] == 1
    assert first_hithroughput["Observed_E-Step"] == 1


def test_read_efficiency_factor():
    ancillary_file = (
        ANCILLARY_DIR / "imap_lo_efficiency-factor-small_20250101_20270101_v001.csv"
    )
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 31
    assert "Comments" in df.columns

    # spot check the first row
    first_row = df.iloc[0]
    np.testing.assert_array_equal(
        first_row,
        np.array(
            [
                pd.Timestamp("2025-01-01"),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                "Hi Track HiThr",
            ],
            dtype=object,
        ),
    )

    # spot check the different row
    lo_track_hithr = df[df["Comments"] == "Lo Track HiThr"].iloc[0]
    np.testing.assert_array_equal(
        lo_track_hithr,
        np.array(
            [
                pd.Timestamp("2025-01-22"),
                0.93,
                0.93,
                0.93,
                0.93,
                0.93,
                0.92,
                0.92,
                "Lo Track HiThr",
            ],
            dtype=object,
        ),
    )


def test_read_sweep_table():
    ancillary_file = (
        ANCILLARY_DIR / "imap_lo_sweep-table-small_20250101_20260301_v001.csv"
    )
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 10

    # spot check the first row
    np.testing.assert_array_equal(
        df.iloc[0],
        np.array(
            [
                pd.Timestamp("2025-01-01"),
                "ISN",
                "HiRes Mode",
                473389200,
                473407618,
                "Lo",
                1,
                "HiRes",
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
            dtype=object,
        ),
    )


def test_read_esa_mode_lut():
    ancillary_file = ANCILLARY_DIR / "imap_lo_esa-mode-lut_v001.csv"
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 112
    assert "ESA_Mode" in df.columns
    # The first 36 rows are ESA mode 0 (HiRes), the second 36 are ESA mode 1 (HiThr)
    np.testing.assert_array_equal(
        df.iloc[0],
        np.array([0, 1, 1, 61.9, 30.5, 66, 67, "HiThr"], dtype=object),
    )


def test_read_pointing_file():
    ancillary_file = (
        ANCILLARY_DIR / "imap_lo_pointing-file-small_20250101_20271231_v001.csv"
    )
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 5

    np.testing.assert_array_equal(
        df.iloc[0], np.array([pd.Timestamp("2025-01-01"), 4, 105], dtype=object)
    )


def test_read_pointing_index_file():
    ancillary_file = ANCILLARY_DIR / "imap_lo_pointing-Index_20250101_20271231_v001.csv"
    df = lo_ancillary.read_ancillary_file(ancillary_file)
    assert len(df) == 15

    np.testing.assert_array_equal(
        df.iloc[0],
        np.array(
            [1, 60, 55, 65, pd.Timestamp("2027-07-26"), pd.Timestamp("2027-12-29")],
            dtype=object,
        ),
    )
