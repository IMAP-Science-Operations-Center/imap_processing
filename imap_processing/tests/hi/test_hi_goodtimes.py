"""Test coverage for imap_processing.hi.hi_goodtimes.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.hi.hi_goodtimes import (
    INTERVAL_DTYPE,
    CullCode,
    _add_sweep_indices,
    _apply_goodtimes_filters,
    _build_per_sweep_datasets,
    _compute_bins_for_cluster,
    _compute_median_and_sigma_per_esa,
    _compute_normalized_counts_per_sweep,
    _compute_qualified_counts_per_sweep,
    _find_current_pointing_index,
    _find_event_clusters,
    _get_sweep_indices,
    _identify_cull_pattern,
    create_goodtimes_dataset,
    hi_goodtimes,
    mark_bad_tdc_cal,
    mark_drf_times,
    mark_incomplete_spin_sets,
    mark_overflow_packets,
    mark_statistical_filter_0,
    mark_statistical_filter_1,
    mark_statistical_filter_2,
)
from imap_processing.quality_flags import ImapHiL1bDeFlags


@pytest.fixture
def mock_l1b_de():
    """Create a mock L1B Direct Event dataset for testing."""
    # Create 10 unique MET times, each appearing twice (paired)
    # Plus 2 unpaired MET times
    n_paired = 10

    # Paired METs: each appears twice
    paired_mets = np.arange(1000.0, 1000.0 + n_paired * 10, 10)
    esa_step_met = np.repeat(paired_mets, 2)

    # Add unpaired METs
    unpaired_mets = np.array([2000.0, 3000.0])
    esa_step_met = np.concatenate([esa_step_met, unpaired_mets])

    # ESA energy step cycles through values
    esa_energy_step = np.tile(np.arange(1, 11), len(esa_step_met) // 10 + 1)[
        : len(esa_step_met)
    ]

    ds = xr.Dataset(
        {
            "esa_step_met": (["epoch"], esa_step_met),
            "esa_step": (["epoch"], esa_energy_step.astype(np.uint8)),
        },
        attrs={
            "Logical_source": "imap_hi_l1b_45sensor-de",
            "Repointing": "repoint00042",
        },
    )
    return ds


@pytest.fixture
def goodtimes_instance(mock_l1b_de):
    """Create a goodtimes dataset for testing."""
    return create_goodtimes_dataset(mock_l1b_de)


class TestCullCode:
    """Test suite for CullCode IntEnum."""

    def test_cull_code_values(self):
        """Test CullCode enum values are bit flags (powers of 2)."""
        assert CullCode.GOOD == 0
        assert CullCode.INCOMPLETE_SPIN == 1
        assert CullCode.DRF == 2
        assert CullCode.BAD_TDC_CAL == 4
        assert CullCode.OVERFLOW == 8
        assert CullCode.STAT_FILTER_0 == 16
        assert CullCode.STAT_FILTER_1 == 32
        assert CullCode.STAT_FILTER_2 == 64

    def test_cull_code_is_int(self):
        """Test that CullCode values are integers."""
        assert isinstance(CullCode.GOOD, int)
        assert isinstance(CullCode.INCOMPLETE_SPIN, int)

    def test_cull_codes_can_be_combined(self):
        """Test that cull codes can be combined with bitwise OR."""
        combined = CullCode.INCOMPLETE_SPIN | CullCode.DRF
        assert combined == 3
        # Check individual flags can be extracted with bitwise AND
        assert combined & CullCode.INCOMPLETE_SPIN == CullCode.INCOMPLETE_SPIN
        assert combined & CullCode.DRF == CullCode.DRF
        assert combined & CullCode.BAD_TDC_CAL == 0


class TestGoodtimesFromL1bDe:
    """Test suite for create_goodtimes_dataset() from L1B DE."""

    def test_from_l1b_de_basic(self, mock_l1b_de):
        """Test basic creation from L1B DE data."""
        gt = create_goodtimes_dataset(mock_l1b_de)

        assert isinstance(gt, xr.Dataset)

    def test_from_l1b_de_keeps_unique_mets(self, mock_l1b_de):
        """Test that all unique METs are included."""
        gt = create_goodtimes_dataset(mock_l1b_de)

        # Should have 12 unique METs (10 paired + 2 unpaired)
        assert len(gt.coords["met"]) == 12

    def test_from_l1b_de_dimensions(self, goodtimes_instance):
        """Test that dimensions are correct."""
        assert "met" in goodtimes_instance.dims
        assert "spin_bin" in goodtimes_instance.dims
        assert goodtimes_instance.dims["spin_bin"] == 90

    def test_from_l1b_de_coordinates(self, goodtimes_instance):
        """Test that coordinates are set correctly."""
        assert "met" in goodtimes_instance.coords
        assert "spin_bin" in goodtimes_instance.coords

        # spin_bin should be 0-89
        np.testing.assert_array_equal(
            goodtimes_instance.coords["spin_bin"].values, np.arange(90)
        )

    def test_from_l1b_de_data_variables(self, goodtimes_instance):
        """Test that data variables are created."""
        assert "cull_flags" in goodtimes_instance.data_vars
        assert "esa_step" in goodtimes_instance.data_vars

    def test_from_l1b_de_cull_flags_initialized_to_zero(self, goodtimes_instance):
        """Test that cull_flags are initialized to 0 (good)."""
        assert np.all(goodtimes_instance["cull_flags"].values == 0)

    def test_from_l1b_de_cull_flags_shape(self, goodtimes_instance):
        """Test cull_flags array shape."""
        n_met = len(goodtimes_instance.coords["met"])
        assert goodtimes_instance["cull_flags"].shape == (n_met, 90)

    def test_from_l1b_de_esa_step_preserved(self, mock_l1b_de, goodtimes_instance):
        """Test that ESA step values are preserved for all unique METs."""
        # Get first occurrence of each unique MET
        met_all = mock_l1b_de["esa_step_met"].values
        unique_mets, first_indices = np.unique(met_all, return_index=True)
        expected_esa_steps = mock_l1b_de["esa_step"].values[first_indices]

        np.testing.assert_array_equal(
            goodtimes_instance["esa_step"].values, expected_esa_steps
        )

    def test_from_l1b_de_attributes(self, goodtimes_instance):
        """Test that attributes are set correctly."""
        assert goodtimes_instance.attrs["Sensor"] == "45sensor"
        assert goodtimes_instance.attrs["Repointing"] == "repoint00042"


class TestRemoveTimes:
    """Test suite for Goodtimes.mark_bad_times() method."""

    def test_mark_bad_times_single_met_all_bins(self, goodtimes_instance):
        """Test flagging a single MET with all bins."""
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        # Check that all bins for the first MET are flagged
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, :] == CullCode.INCOMPLETE_SPIN
        )

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[1:, :] == CullCode.GOOD)

    def test_mark_bad_times_single_met_specific_bins(self, goodtimes_instance):
        """Test flagging specific bins for a single MET."""
        met_val = goodtimes_instance.coords["met"].values[0]
        bins_to_flag = np.array([0, 1, 2, 10])
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=bins_to_flag, cull=CullCode.INCOMPLETE_SPIN
        )

        # Check that specified bins are flagged
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, bins_to_flag]
            == CullCode.INCOMPLETE_SPIN
        )

        # Check that other bins are still good
        other_bins = np.setdiff1d(np.arange(90), bins_to_flag)
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, other_bins] == CullCode.GOOD
        )

    def test_mark_bad_times_multiple_mets(self, goodtimes_instance):
        """Test flagging multiple METs."""
        met_vals = goodtimes_instance.coords["met"].values[:3]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_vals, bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        # Check that first 3 METs are flagged
        assert np.all(
            goodtimes_instance["cull_flags"].values[:3, :] == CullCode.INCOMPLETE_SPIN
        )

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[3:, :] == CullCode.GOOD)

    def test_mark_bad_times_time_range(self, goodtimes_instance):
        """Test flagging a time range."""
        met_vals = goodtimes_instance.coords["met"].values
        met_start = met_vals[2]
        met_end = met_vals[5]

        goodtimes_instance.goodtimes.mark_bad_times(
            met=(met_start, met_end), bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        # Check that METs 2-5 are flagged
        assert np.all(
            goodtimes_instance["cull_flags"].values[2:6, :] == CullCode.INCOMPLETE_SPIN
        )

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[:2, :] == CullCode.GOOD)
        assert np.all(goodtimes_instance["cull_flags"].values[6:, :] == CullCode.GOOD)

    def test_mark_bad_times_invalid_cull_code_zero(self, goodtimes_instance):
        """Test that cull code 0 raises ValueError."""
        met_val = goodtimes_instance.coords["met"].values[0]
        with pytest.raises(ValueError, match="Cull code must be non-zero"):
            goodtimes_instance.goodtimes.mark_bad_times(met=met_val, cull=0)

    def test_mark_bad_times_invalid_bin_indices(self, goodtimes_instance):
        """Test that invalid bin indices raise ValueError."""
        met_val = goodtimes_instance.coords["met"].values[0]

        # Test bin < 0
        with pytest.raises(ValueError, match="Spin bins must be in range"):
            goodtimes_instance.goodtimes.mark_bad_times(
                met=met_val, bins=np.array([-1, 0])
            )

        # Test bin >= 90
        with pytest.raises(ValueError, match="Spin bins must be in range"):
            goodtimes_instance.goodtimes.mark_bad_times(
                met=met_val, bins=np.array([89, 90])
            )

    def test_mark_bad_times_met_out_of_range(self, goodtimes_instance):
        """Test that MET outside valid range raises ValueError."""
        met_vals = goodtimes_instance.coords["met"].values
        # Use a value clearly beyond the valid range
        # (last MET + 2x the interval between last two METs)
        met_out_of_range = met_vals[-1] + 2000

        with pytest.raises(ValueError, match="MET value\\(s\\) "):
            goodtimes_instance.goodtimes.mark_bad_times(met=met_out_of_range)

    def test_mark_bad_times_combines_cull_codes(self, goodtimes_instance):
        """Test that cull codes are combined using bitwise OR."""
        met_val = goodtimes_instance.coords["met"].values[0]

        # Flag with INCOMPLETE_SPIN (1)
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=None, cull=CullCode.INCOMPLETE_SPIN
        )
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, :] == CullCode.INCOMPLETE_SPIN
        )

        # Add another cull code with bitwise OR (1 | 2 = 3)
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=None, cull=CullCode.DRF
        )
        expected = CullCode.INCOMPLETE_SPIN | CullCode.DRF  # 1 | 2 = 3
        assert np.all(goodtimes_instance["cull_flags"].values[0, :] == expected)


class TestGetGoodIntervals:
    """Test suite for Goodtimes.get_good_intervals() method."""

    def test_get_good_intervals_all_good(self, goodtimes_instance):
        """Test getting intervals when all times are good."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # When all cull flags are identical (all zeros), should merge into 1 interval
        assert len(intervals) == 1

        # Check interval structure
        assert intervals.dtype == INTERVAL_DTYPE

    def test_get_good_intervals_structure(self, goodtimes_instance):
        """Test interval structure and field names."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Check that all fields exist
        assert "met_start" in intervals.dtype.names
        assert "met_end" in intervals.dtype.names
        assert "spin_bin_low" in intervals.dtype.names
        assert "spin_bin_high" in intervals.dtype.names
        assert "n_bins" in intervals.dtype.names
        assert "esa_step_mask" in intervals.dtype.names
        assert "cull_value" in intervals.dtype.names

    def test_get_good_intervals_all_good_values(self, goodtimes_instance):
        """Test interval values when all bins are good."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Single interval spanning all METs with all bins good
        assert len(intervals) == 1
        interval = intervals[0]
        assert interval["spin_bin_low"] == 0
        assert interval["spin_bin_high"] == 89
        assert interval["n_bins"] == 90
        assert interval["cull_value"] == 0
        # met_start should be first MET, met_end should be last MET
        met_values = goodtimes_instance.coords["met"].values
        assert interval["met_start"] == met_values[0]
        assert interval["met_end"] == met_values[-1]

    def test_get_good_intervals_with_culled_bins(self, goodtimes_instance):
        """Test intervals when some bins are culled."""
        # Flag bins 0-20 for first MET only
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=np.arange(21), cull=CullCode.INCOMPLETE_SPIN
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # First MET has different pattern, creates separate intervals
        # First MET: 2 intervals (bins 0-20 culled, bins 21-89 good)
        # Remaining METs: 1 interval (all bins good)
        assert len(intervals) == 3

        # Check first interval (culled bins 0-20)
        assert intervals[0]["spin_bin_low"] == 0
        assert intervals[0]["spin_bin_high"] == 20
        assert intervals[0]["n_bins"] == 21
        assert intervals[0]["cull_value"] == CullCode.INCOMPLETE_SPIN

        # Check second interval (good bins 21-89)
        assert intervals[1]["spin_bin_low"] == 21
        assert intervals[1]["spin_bin_high"] == 89
        assert intervals[1]["n_bins"] == 69
        assert intervals[1]["cull_value"] == 0

    def test_get_good_intervals_with_gaps(self, goodtimes_instance):
        """Test intervals when bins have gaps in cull values."""
        # Flag bins 20-70 for first MET, leaving bins 0-19 and 71-89 as good
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=np.arange(20, 71), cull=CullCode.INCOMPLETE_SPIN
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # First MET has 3 regions (0-19 good, 20-70 culled, 71-89 good)
        # Remaining METs merged into 1 interval (all bins good)
        assert len(intervals) == 4

        # First MET intervals should have same met_start == met_end
        assert intervals[0]["met_start"] == intervals[0]["met_end"]
        assert intervals[1]["met_start"] == intervals[1]["met_end"]
        assert intervals[2]["met_start"] == intervals[2]["met_end"]

        # Check the three segments for first MET
        assert intervals[0]["spin_bin_low"] == 0
        assert intervals[0]["spin_bin_high"] == 19
        assert intervals[0]["cull_value"] == 0
        assert intervals[1]["spin_bin_low"] == 20
        assert intervals[1]["spin_bin_high"] == 70
        assert intervals[1]["cull_value"] == CullCode.INCOMPLETE_SPIN
        assert intervals[2]["spin_bin_low"] == 71
        assert intervals[2]["spin_bin_high"] == 89
        assert intervals[2]["cull_value"] == 0

    def test_get_good_intervals_all_bins_culled(self, goodtimes_instance):
        """Test intervals when all bins are culled for a MET."""
        # Flag all bins for first MET
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Should have 2 intervals: one for culled first MET, one for remaining METs
        assert len(intervals) == 2

        # First interval is the culled MET
        assert intervals[0]["cull_value"] == CullCode.INCOMPLETE_SPIN
        assert intervals[0]["spin_bin_low"] == 0
        assert intervals[0]["spin_bin_high"] == 89

        # Second interval is remaining good METs
        assert intervals[1]["cull_value"] == 0
        assert intervals[1]["met_start"] == goodtimes_instance.coords["met"].values[1]

    def test_get_good_intervals_empty(self):
        """Test intervals with empty goodtimes dataset."""
        # Create empty dataset
        gt = xr.Dataset(
            data_vars={
                "cull_flags": xr.DataArray(
                    np.zeros((0, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(np.array([], dtype=np.uint8), dims=["met"]),
            },
            coords={"met": np.array([]), "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 0},
        )

        intervals = gt.goodtimes.get_good_intervals()
        assert len(intervals) == 0

    def test_get_good_intervals_esa_step_mask(self, goodtimes_instance):
        """Test that ESA step mask includes all ESA steps in the interval."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Single interval should include all ESA steps from all METs
        assert len(intervals) == 1
        esa_step_mask = intervals[0]["esa_step_mask"]

        # Check that the mask has bits set for all unique ESA steps
        unique_esa_steps = set(goodtimes_instance["esa_step"].values)
        for esa_step in unique_esa_steps:
            bit_position = esa_step - 1  # ESA step 1 -> bit 0, etc.
            assert (esa_step_mask >> bit_position) & 1 == 1


class TestGetCullStatistics:
    """Test suite for Goodtimes.get_cull_statistics() method."""

    def test_get_cull_statistics_all_good(self, goodtimes_instance):
        """Test statistics when all bins are good."""
        stats = goodtimes_instance.goodtimes.get_cull_statistics()

        total_bins = len(goodtimes_instance.coords["met"]) * 90
        assert stats["total_bins"] == total_bins
        assert stats["good_bins"] == total_bins
        assert stats["culled_bins"] == 0
        assert stats["fraction_good"] == 1.0
        assert stats["cull_code_counts"] == {}

    def test_get_cull_statistics_with_culls(self, goodtimes_instance):
        """Test statistics after culling some bins."""
        # Flag first MET, all bins
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        stats = goodtimes_instance.goodtimes.get_cull_statistics()

        total_bins = len(goodtimes_instance.coords["met"]) * 90
        assert stats["total_bins"] == total_bins
        assert stats["good_bins"] == total_bins - 90
        assert stats["culled_bins"] == 90
        assert stats["fraction_good"] == (total_bins - 90) / total_bins
        assert stats["cull_code_counts"][CullCode.INCOMPLETE_SPIN] == 90

    def test_get_cull_statistics_multiple_cull_codes(self, goodtimes_instance):
        """Test statistics with multiple cull codes."""
        met_vals = goodtimes_instance.coords["met"].values

        # Flag first MET with LOOSE
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_vals[0], bins=None, cull=CullCode.INCOMPLETE_SPIN
        )

        # Flag second MET with code 2
        goodtimes_instance.goodtimes.mark_bad_times(met=met_vals[1], bins=None, cull=2)

        stats = goodtimes_instance.goodtimes.get_cull_statistics()

        assert stats["culled_bins"] == 180
        assert stats["cull_code_counts"][CullCode.INCOMPLETE_SPIN] == 90
        assert stats["cull_code_counts"][2] == 90


class TestToTxt:
    """Test suite for Goodtimes.to_txt() method."""

    def test_to_txt_creates_file(self, goodtimes_instance, tmp_path):
        """Test that to_txt creates a file."""
        output_path = tmp_path / "goodtimes.txt"
        result = goodtimes_instance.goodtimes.write_txt(output_path)

        assert result == output_path
        assert output_path.exists()

    def test_to_txt_format(self, goodtimes_instance, tmp_path):
        """Test the format of the output file."""
        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            lines = f.readlines()

        # Should have 1 line (all METs merged into single interval)
        assert len(lines) == 1

        # Check format of first line
        # Format: pointing met_start met_end bin_low bin_high sensor
        # esa_steps[10] cull_value
        parts = lines[0].strip().split()
        assert len(parts) == 17  # 6 base fields + 10 ESA step flags + cull_value
        assert parts[0] == "00042"  # pointing
        assert parts[5] == "45"  # sensor
        assert parts[16] == "0"  # cull_value (all good)

    def test_to_txt_values(self, goodtimes_instance, tmp_path):
        """Test the values in the output file."""
        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            line = f.readline()

        parts = line.strip().split()
        # Format: pointing met_start met_end bin_low bin_high sensor
        # esa_steps[10] cull_value
        pointing = parts[0]
        met_start = parts[1]
        met_end = parts[2]
        bin_low = parts[3]
        bin_high = parts[4]
        sensor = parts[5]
        esa_step_flags = parts[6:16]
        cull_value = parts[16]

        assert pointing == "00042"
        assert int(met_start) == int(goodtimes_instance.coords["met"].values[0])
        assert int(met_end) == int(goodtimes_instance.coords["met"].values[-1])
        assert int(bin_low) == 0
        assert int(bin_high) == 89
        assert sensor == "45"
        assert cull_value == "0"

        # Check ESA step flags - should have 1s for all unique ESA steps
        unique_esa_steps = set(goodtimes_instance["esa_step"].values)
        for i, flag in enumerate(esa_step_flags):
            esa_step = i + 1  # ESA steps are 1-indexed
            expected = "1" if esa_step in unique_esa_steps else "0"
            assert flag == expected

    def test_to_txt_with_culled_bins(self, goodtimes_instance, tmp_path):
        """Test output when some bins are culled."""
        # Flag bins 0-20 for first MET
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=np.arange(21), cull=CullCode.INCOMPLETE_SPIN
        )

        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            lines = f.readlines()

        # Should have 3 intervals: culled bins (0-20), good bins (21-89), remaining METs
        assert len(lines) == 3

        # First interval: culled bins 0-20
        parts = lines[0].strip().split()
        assert int(parts[3]) == 0  # bin_low
        assert int(parts[4]) == 20  # bin_high
        assert parts[16] == "1"  # cull_value (INCOMPLETE_SPIN)

        # Second interval: good bins 21-89
        parts = lines[1].strip().split()
        assert int(parts[3]) == 21  # bin_low
        assert int(parts[4]) == 89  # bin_high
        assert parts[16] == "0"  # cull_value (good)

    def test_to_txt_with_gaps(self, goodtimes_instance, tmp_path):
        """Test output when bins have gaps."""
        # Flag bins 20-70, leaving 0-19 and 71-89 as good
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.mark_bad_times(
            met=met_val, bins=np.arange(20, 71), cull=CullCode.INCOMPLETE_SPIN
        )

        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            lines = f.readlines()

        # Should have 4 lines (3 for first MET with gap pattern, 1 for remaining METs)
        assert len(lines) == 4

        # First three lines should be for same MET (first MET)
        parts1 = lines[0].strip().split()
        parts2 = lines[1].strip().split()
        parts3 = lines[2].strip().split()
        assert parts1[1] == parts2[1] == parts3[1]  # Same met_start

        # Check the regions: bins 0-19 (good), 20-70 (culled), 71-89 (good)
        np.testing.assert_array_equal(parts1[3:5], ["0", "19"])
        assert parts1[16] == "0"
        np.testing.assert_array_equal(parts2[3:5], ["20", "70"])
        assert parts2[16] == "1"
        np.testing.assert_array_equal(parts3[3:5], ["71", "89"])
        assert parts3[16] == "0"


class TestFinalizeDataset:
    """Test suite for GoodtimesAccessor.finalize_dataset() method."""

    def test_finalize_changes_dimension_to_epoch(self, goodtimes_instance):
        """Test that finalize changes primary dimension from met to epoch."""
        # Mock met_to_ttj2000ns to avoid SPICE dependency
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            # Return fake epoch values
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert "epoch" in finalized.dims
            assert "met" not in finalized.dims
            assert "spin_bin" in finalized.dims

    def test_finalize_adds_met_as_data_variable(self, goodtimes_instance):
        """Test that met coordinate becomes a data variable."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert "met" in finalized.data_vars
            assert "met" not in finalized.coords

    def test_finalize_preserves_met_values(self, goodtimes_instance):
        """Test that original MET values are preserved in data variable."""
        original_met = goodtimes_instance.coords["met"].values.copy()

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(100, 100 + len(original_met))

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            np.testing.assert_array_equal(finalized["met"].values, original_met)

    def test_finalize_converts_met_to_epoch(self, goodtimes_instance):
        """Test that met_to_ttj2000ns is called with MET values."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            # Return same number of epoch values as MET values
            n_mets = len(goodtimes_instance.coords["met"])
            mock_convert.return_value = np.arange(1000, 1000 + n_mets, dtype=np.int64)

            goodtimes_instance.goodtimes.finalize_dataset()

            # Verify conversion function was called
            mock_convert.assert_called_once()
            called_mets = mock_convert.call_args[0][0]
            np.testing.assert_array_equal(
                called_mets, goodtimes_instance.coords["met"].values
            )

    def test_finalize_adds_epoch_coordinate(self, goodtimes_instance):
        """Test that epoch coordinate is added with converted values."""
        fake_epochs = np.arange(100, 100 + len(goodtimes_instance.coords["met"]))

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = fake_epochs

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            np.testing.assert_array_equal(finalized.coords["epoch"].values, fake_epochs)

    def test_finalize_adds_spin_bin_label_coordinate(self, goodtimes_instance):
        """Test that spin_bin_label coordinate is added."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert "spin_bin_label" in finalized.coords
            assert len(finalized.coords["spin_bin_label"]) == 90
            assert finalized.coords["spin_bin_label"].values[0] == "0"
            assert finalized.coords["spin_bin_label"].values[89] == "89"

    def test_finalize_preserves_cull_flags_data(self, goodtimes_instance):
        """Test that cull_flags data is preserved."""
        # Mark some bins as bad
        goodtimes_instance.goodtimes.mark_bad_times(
            met=goodtimes_instance.coords["met"].values[0],
            bins=np.arange(10),
            cull=CullCode.INCOMPLETE_SPIN,
        )
        original_flags = goodtimes_instance["cull_flags"].values.copy()

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            np.testing.assert_array_equal(
                finalized["cull_flags"].values, original_flags
            )

    def test_finalize_preserves_esa_step_data(self, goodtimes_instance):
        """Test that esa_step data is preserved."""
        original_esa_step = goodtimes_instance["esa_step"].values.copy()

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            np.testing.assert_array_equal(
                finalized["esa_step"].values, original_esa_step
            )

    def test_finalize_adds_cdf_attributes_to_variables(self, goodtimes_instance):
        """Test that CDF attributes are added to all variables."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            # Check that variables have attributes
            assert len(finalized["cull_flags"].attrs) > 0
            assert len(finalized["met"].attrs) > 0
            assert len(finalized["esa_step"].attrs) > 0
            assert len(finalized.coords["epoch"].attrs) > 0
            assert len(finalized.coords["spin_bin"].attrs) > 0

    def test_finalize_adds_global_attributes(self, goodtimes_instance):
        """Test that global CDF attributes are added."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            # Check for required global attributes
            assert "Logical_source" in finalized.attrs
            assert "Data_type" in finalized.attrs

    def test_finalize_formats_logical_source(self, goodtimes_instance):
        """Test that Logical_source is properly formatted with sensor."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            # Should contain the sensor designation
            assert (
                "45sensor" in finalized.attrs["Logical_source"]
                or "45sensor" in finalized.attrs["Logical_source"]
            )
            # Should not contain template markers
            assert "{sensor}" not in finalized.attrs["Logical_source"]

    def test_finalize_preserves_original_dataset(self, goodtimes_instance):
        """Test that finalize doesn't modify the original dataset."""
        original_dims = set(goodtimes_instance.dims.keys())
        original_coords = set(goodtimes_instance.coords.keys())

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            # Call finalize but don't need to assign result
            goodtimes_instance.goodtimes.finalize_dataset()

            # Original should be unchanged
            assert set(goodtimes_instance.dims.keys()) == original_dims
            assert set(goodtimes_instance.coords.keys()) == original_coords
            assert "epoch" not in goodtimes_instance.coords

    def test_finalize_cull_flags_dimensions(self, goodtimes_instance):
        """Test that cull_flags has correct dimensions after finalization."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert finalized["cull_flags"].dims == ("epoch", "spin_bin")

    def test_finalize_esa_step_dimensions(self, goodtimes_instance):
        """Test that esa_step has correct dimensions after finalization."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert finalized["esa_step"].dims == ("epoch",)

    def test_finalize_met_dimensions(self, goodtimes_instance):
        """Test that met has correct dimensions after finalization."""
        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.arange(
                100, 100 + len(goodtimes_instance.coords["met"])
            )

            finalized = goodtimes_instance.goodtimes.finalize_dataset()

            assert finalized["met"].dims == ("epoch",)

    def test_finalize_with_empty_dataset(self):
        """Test finalize with an empty goodtimes dataset."""
        empty_ds = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((0, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(np.array([], dtype=np.uint8), dims=["met"]),
            },
            coords={"met": np.array([]), "spin_bin": np.arange(90)},
            attrs={"Sensor": "45sensor", "Pointing": 1},
        )

        with patch("imap_processing.hi.hi_goodtimes.met_to_ttj2000ns") as mock_convert:
            mock_convert.return_value = np.array([])

            finalized = empty_ds.goodtimes.finalize_dataset()

            assert len(finalized.coords["epoch"]) == 0
            assert finalized["cull_flags"].shape == (0, 90)


class TestIntervalDtype:
    """Test suite for INTERVAL_DTYPE."""

    def test_interval_dtype_fields(self):
        """Test that INTERVAL_DTYPE has correct fields."""
        field_names = INTERVAL_DTYPE.names
        assert "met_start" in field_names
        assert "met_end" in field_names
        assert "spin_bin_low" in field_names
        assert "spin_bin_high" in field_names
        assert "n_bins" in field_names
        assert "esa_step_mask" in field_names
        assert "cull_value" in field_names

    def test_interval_dtype_types(self):
        """Test that INTERVAL_DTYPE has correct field types."""
        assert INTERVAL_DTYPE["met_start"] == np.float64
        assert INTERVAL_DTYPE["met_end"] == np.float64
        assert INTERVAL_DTYPE["spin_bin_low"] == np.uint8
        assert INTERVAL_DTYPE["spin_bin_high"] == np.uint8
        assert INTERVAL_DTYPE["n_bins"] == np.uint8
        assert INTERVAL_DTYPE["esa_step_mask"] == np.uint16
        assert INTERVAL_DTYPE["cull_value"] == np.uint8


def _create_l1b_de_dataset(
    esa_step_met: list[float],
    last_spin_num: list[int],
    esa_step: list[int],
    ccsds_qf: list[int] | None = None,
    repointing: str = "repoint00001",
) -> xr.Dataset:
    """
    Helper function to create L1B DE datasets for testing.

    Parameters
    ----------
    esa_step_met : list[float]
        MET timestamps for each packet.
    last_spin_num : list[int]
        Last spin number (1-8) for each packet.
    esa_step : list[int]
        ESA energy step values for each packet.
    ccsds_qf : list[int] | None
        Quality flags for each packet. If None, all zeros (valid).
    repointing : str
        Repointing attribute value.

    Returns
    -------
    xr.Dataset
        Mock L1B DE dataset.
    """
    n_packets = len(esa_step_met)
    if ccsds_qf is None:
        ccsds_qf = [0] * n_packets

    return xr.Dataset(
        {
            "esa_step_met": (["epoch"], np.array(esa_step_met, dtype=np.float64)),
            "last_spin_num": (["epoch"], np.array(last_spin_num, dtype=np.uint8)),
            "esa_step": (["epoch"], np.array(esa_step, dtype=np.uint8)),
            "ccsds_qf": (["epoch"], np.array(ccsds_qf, dtype=np.uint8)),
        },
        attrs={
            "Logical_source": "imap_hi_l1b_45sensor-de",
            "Repointing": repointing,
        },
    )


class TestDropIncompleteSpinSets:
    """Test suite for mark_incomplete_spin_sets() function."""

    @pytest.fixture
    def l1b_de_complete_4th_spin(self):
        """Create L1B DE data with complete 4th spin cadence (last_spin_num 4,8)."""
        # 5 unique METs, each with 2 packets (last_spin_num 4 and 8)
        # 60 second intervals between METs (every 4th spin)
        n_mets = 5
        mets = np.arange(1000.0, 1000.0 + n_mets * 60, 60)

        esa_step_met = []
        last_spin_num = []
        esa_step = []

        for met in mets:
            # Add two packets per MET: last_spin_num 4 and 8
            esa_step_met.extend([met, met])
            last_spin_num.extend([4, 8])
            esa_step.extend([1, 1])

        return _create_l1b_de_dataset(
            esa_step_met, last_spin_num, esa_step, repointing="repoint00001"
        )

    @pytest.fixture
    def l1b_de_complete_2nd_spin(self):
        """Create L1B DE data with complete 2nd spin cadence (last_spin_num 2,4,6,8)."""
        # 3 unique METs, each with 4 packets
        # 30 second intervals between METs (every 2nd spin)
        n_mets = 3
        mets = np.arange(2000.0, 2000.0 + n_mets * 30, 30)

        esa_step_met = []
        last_spin_num = []
        esa_energy_step = []

        for met in mets:
            # Add four packets per MET: last_spin_num 2,4,6,8
            esa_step_met.extend([met] * 4)
            last_spin_num.extend([2, 4, 6, 8])
            esa_energy_step.extend([2] * 4)

        return _create_l1b_de_dataset(
            esa_step_met, last_spin_num, esa_energy_step, repointing="repoint00002"
        )

    @pytest.fixture
    def l1b_de_complete_every_spin(self):
        """Create L1B DE data with complete every spin cadence (last_spin_num 1-8)."""
        # 2 unique METs, each with 8 packets
        # 15 second intervals between METs (every spin)
        n_mets = 2
        mets = np.arange(3000.0, 3000.0 + n_mets * 15, 15)

        esa_step_met = []
        last_spin_num = []
        esa_energy_step = []

        for met in mets:
            # Add eight packets per MET: last_spin_num 1-8
            esa_step_met.extend([met] * 8)
            last_spin_num.extend(range(1, 9))
            esa_energy_step.extend([3] * 8)

        return _create_l1b_de_dataset(
            esa_step_met, last_spin_num, esa_energy_step, repointing="repoint00003"
        )

    @pytest.fixture
    def l1b_de_incomplete(self):
        """Create L1B DE data with incomplete 8-spin periods."""
        # 4 METs: 2 complete (4,8), 2 incomplete (missing spin 8)
        # 60 second intervals (every 4th spin cadence)
        mets = [1000.0, 1060.0, 1120.0, 1180.0]

        esa_step_met = []
        last_spin_num = []
        esa_energy_step = []

        # Complete METs
        for met in mets[:2]:
            esa_step_met.extend([met, met])
            last_spin_num.extend([4, 8])
            esa_energy_step.extend([1, 1])

        # Incomplete METs (only spin 4, missing spin 8)
        for met in mets[2:]:
            esa_step_met.append(met)
            last_spin_num.append(4)
            esa_energy_step.append(1)

        return _create_l1b_de_dataset(
            esa_step_met, last_spin_num, esa_energy_step, repointing="repoint00004"
        )

    @pytest.fixture
    def l1b_de_with_invalid_spins(self):
        """Create L1B DE data with ccsds_qf spin invalid flag set."""
        # 60 second intervals (every 4th spin cadence)
        mets = [1000.0, 1060.0]

        esa_step_met = []
        last_spin_num = []
        esa_energy_step = []
        ccsds_qf = []

        # First MET: complete but with spin invalid flag (bit 1 set = 0x02)
        esa_step_met.extend([mets[0], mets[0]])
        last_spin_num.extend([4, 8])
        esa_energy_step.extend([1, 1])
        ccsds_qf.extend(
            [ImapHiL1bDeFlags.BADSPIN, 0]
        )  # First packet has spin invalid flag

        # Second MET: complete and valid
        esa_step_met.extend([mets[1], mets[1]])
        last_spin_num.extend([4, 8])
        esa_energy_step.extend([1, 1])
        ccsds_qf.extend([0, 0])

        return _create_l1b_de_dataset(
            esa_step_met,
            last_spin_num,
            esa_energy_step,
            ccsds_qf=ccsds_qf,
            repointing="repoint00005",
        )

    def test_mark_incomplete_spin_sets_complete_4th_spin(
        self, l1b_de_complete_4th_spin
    ):
        """Test that complete 4th spin cadence is accepted."""
        gt = create_goodtimes_dataset(l1b_de_complete_4th_spin)
        mark_incomplete_spin_sets(gt, l1b_de_complete_4th_spin)

        # All times should still be good (no culling)
        assert np.all(gt["cull_flags"].values == CullCode.GOOD)

    def test_mark_incomplete_spin_sets_complete_2nd_spin(
        self, l1b_de_complete_2nd_spin
    ):
        """Test that complete 2nd spin cadence is accepted."""
        gt = create_goodtimes_dataset(l1b_de_complete_2nd_spin)
        mark_incomplete_spin_sets(gt, l1b_de_complete_2nd_spin)

        # All times should still be good (no culling)
        assert np.all(gt["cull_flags"].values == CullCode.GOOD)

    def test_mark_incomplete_spin_sets_complete_every_spin(
        self, l1b_de_complete_every_spin
    ):
        """Test that complete every-spin cadence is accepted."""
        gt = create_goodtimes_dataset(l1b_de_complete_every_spin)
        mark_incomplete_spin_sets(gt, l1b_de_complete_every_spin)

        # All times should still be good (no culling)
        assert np.all(gt["cull_flags"].values == CullCode.GOOD)

    def test_mark_incomplete_spin_sets_incomplete(self, l1b_de_incomplete):
        """Test that incomplete 8-spin periods are culled."""
        gt = create_goodtimes_dataset(l1b_de_incomplete)
        mark_incomplete_spin_sets(gt, l1b_de_incomplete)

        # First 2 METs should be good, last 2 should be culled
        assert np.all(gt["cull_flags"].values[0, :] == CullCode.GOOD)
        assert np.all(gt["cull_flags"].values[1, :] == CullCode.GOOD)
        assert np.all(gt["cull_flags"].values[2, :] == CullCode.INCOMPLETE_SPIN)
        assert np.all(gt["cull_flags"].values[3, :] == CullCode.INCOMPLETE_SPIN)

    def test_mark_incomplete_spin_sets_with_invalid_spins(
        self, l1b_de_with_invalid_spins
    ):
        """Test that times with spin invalid flag in ccsds_qf are culled."""
        gt = create_goodtimes_dataset(l1b_de_with_invalid_spins)
        mark_incomplete_spin_sets(gt, l1b_de_with_invalid_spins)

        # First MET should be culled (has spin invalid flag), second should be good
        assert np.all(gt["cull_flags"].values[0, :] == CullCode.INCOMPLETE_SPIN)
        assert np.all(gt["cull_flags"].values[1, :] == CullCode.GOOD)

    def test_mark_incomplete_spin_sets_no_de_packets(self):
        """Test that MET times with no DE packets are culled."""
        # Create L1B DE with packets at 1000.0 and 1120.0
        # (60 second intervals for 4th spin)
        l1b_de = _create_l1b_de_dataset(
            esa_step_met=[1000.0, 1000.0, 1120.0, 1120.0],
            last_spin_num=[4, 8, 4, 8],
            esa_step=[1, 1, 1, 1],
            repointing="repoint00006",
        )

        gt = create_goodtimes_dataset(l1b_de)

        # Manually add a MET time with no packets (insert in sorted order)
        # Original METs are [1000.0, 1120.0], insert 1060.0 at index 1
        new_met = np.insert(gt.coords["met"].values, 1, 1060.0)
        new_cull_flags = np.insert(
            gt["cull_flags"].values, 1, np.zeros((1, 90), dtype=np.uint8), axis=0
        )
        new_esa_step = np.insert(gt["esa_step"].values, 1, 1)

        gt = xr.Dataset(
            {
                "cull_flags": (["met", "spin_bin"], new_cull_flags),
                "esa_step": (["met"], new_esa_step),
            },
            coords={"met": new_met, "spin_bin": np.arange(90)},
            attrs=gt.attrs,
        )

        mark_incomplete_spin_sets(gt, l1b_de)

        # First and last METs should be good, middle one should be culled
        assert np.all(gt["cull_flags"].values[0, :] == CullCode.GOOD)
        assert np.all(
            gt["cull_flags"].values[1, :] == CullCode.INCOMPLETE_SPIN
        )  # No packets
        assert np.all(gt["cull_flags"].values[2, :] == CullCode.GOOD)

    def test_mark_incomplete_spin_sets_mixed_cadence(self):
        """Test that mixed/invalid cadence patterns are culled."""
        # Create packets with invalid pattern: has spins 4,8,1 (mixing cadences)
        l1b_de = _create_l1b_de_dataset(
            esa_step_met=[1000.0, 1000.0, 1000.0],
            last_spin_num=[4, 8, 1],  # Invalid - mixing cadences
            esa_step=[1, 1, 1],
            repointing="repoint00007",
        )

        gt = create_goodtimes_dataset(l1b_de)
        mark_incomplete_spin_sets(gt, l1b_de)

        # Should be culled (invalid pattern)
        assert np.all(gt["cull_flags"].values[0, :] == CullCode.INCOMPLETE_SPIN)

    def test_mark_incomplete_spin_sets_duplicate_spin_num(self):
        """Test that duplicate last_spin_num values are culled."""
        # Create packets with duplicate spin: has spins 4,4 (should be 4,8)
        l1b_de = _create_l1b_de_dataset(
            esa_step_met=[1000.0, 1000.0],
            last_spin_num=[4, 4],  # Duplicate - invalid
            esa_step=[1, 1],
            repointing="repoint00008",
        )

        gt = create_goodtimes_dataset(l1b_de)
        mark_incomplete_spin_sets(gt, l1b_de)

        # Should be culled (duplicate spin numbers)
        assert np.all(gt["cull_flags"].values[0, :] == CullCode.INCOMPLETE_SPIN)

    def test_mark_incomplete_spin_sets_custom_cull_code(self, l1b_de_incomplete):
        """Test that custom cull code is used."""
        gt = create_goodtimes_dataset(l1b_de_incomplete)
        custom_cull_code = 5
        mark_incomplete_spin_sets(gt, l1b_de_incomplete, cull_code=custom_cull_code)

        # Incomplete METs should be culled with custom code
        assert np.all(gt["cull_flags"].values[2, :] == custom_cull_code)
        assert np.all(gt["cull_flags"].values[3, :] == custom_cull_code)

    def test_mark_incomplete_spin_sets_preserves_good_times(self, l1b_de_incomplete):
        """Test that previously good times remain untouched."""
        gt = create_goodtimes_dataset(l1b_de_incomplete)

        # Manually mark first MET as culled with code 2
        gt["cull_flags"].values[0, :] = 2

        mark_incomplete_spin_sets(gt, l1b_de_incomplete)

        # Check that complete times are good
        assert np.all(gt["cull_flags"].values[1, :] == CullCode.GOOD)


class TestDropDrfTimes:
    """Test suite for mark_drf_times() function."""

    @pytest.fixture
    def goodtimes_for_drf(self):
        """Create a goodtimes dataset with METs spanning 2 hours."""
        # Create METs every 60 seconds for 2 hours (120 METs)
        n_mets = 120
        met_values = np.arange(1000.0, 1000.0 + n_mets * 60, 60)

        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((n_mets, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(np.ones(n_mets, dtype=np.uint8), dims=["met"]),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )
        return gt

    @pytest.fixture
    def hk_single_drf_transition(self):
        """Create HK data with one DRF transition from 1->0."""
        # HK packets every 60 seconds for 2 hours
        n_hk = 120
        shcoarse = np.arange(1000.0, 1000.0 + n_hk * 60, 60)

        # DRF active for first 30 minutes, then inactive
        # Transition at index 30 (MET 2800.0)
        fsw_thruster_warn = np.zeros(n_hk, dtype=np.uint8)
        fsw_thruster_warn[:30] = 1  # DRF active

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )
        return hk

    @pytest.fixture
    def hk_multiple_drf_transitions(self):
        """Create HK data with multiple DRF transitions."""
        # HK packets every 60 seconds for 2 hours
        n_hk = 120
        shcoarse = np.arange(1000.0, 1000.0 + n_hk * 60, 60)

        # Multiple DRF periods:
        # Active: 0-30, inactive: 30-60, active: 60-90, inactive: 90-120
        # Transitions at indices 30 and 90
        fsw_thruster_warn = np.zeros(n_hk, dtype=np.uint8)
        fsw_thruster_warn[0:30] = 1  # First DRF period
        fsw_thruster_warn[60:90] = 1  # Second DRF period

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )
        return hk

    @pytest.fixture
    def hk_no_drf(self):
        """Create HK data with no DRF activity."""
        n_hk = 120
        shcoarse = np.arange(1000.0, 1000.0 + n_hk * 60, 60)
        fsw_thruster_warn = np.zeros(n_hk, dtype=np.uint8)

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )
        return hk

    @pytest.fixture
    def hk_always_drf(self):
        """Create HK data with DRF always active (no transitions)."""
        n_hk = 120
        shcoarse = np.arange(1000.0, 1000.0 + n_hk * 60, 60)
        fsw_thruster_warn = np.ones(n_hk, dtype=np.uint8)

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )
        return hk

    @pytest.fixture
    def hk_empty(self):
        """Create empty HK data."""
        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([])),
                "fsw_thruster_warn": (["epoch"], np.array([], dtype=np.uint8)),
            }
        )
        return hk

    def test_mark_drf_times_single_transition(
        self, goodtimes_for_drf, hk_single_drf_transition
    ):
        """Test that a single DRF transition removes 30-minute window."""
        mark_drf_times(goodtimes_for_drf, hk_single_drf_transition)

        # Transition at index 30 (MET 2800.0)
        # Window: 2800 - 1800 = 1000 to 2800 (inclusive on both ends)
        # mark_bad_times uses (met_start, met_end) which includes both endpoints
        # So METs from 1000 to 2800 should be culled (indices 0-30)

        # Check that METs in the window are culled (indices 0-30)
        for i in range(31):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.DRF
            ), (
                f"MET at index {i} (value "
                f"{goodtimes_for_drf.coords['met'].values[i]}) should be culled"
            )

        # Check that METs after the window are good
        for i in range(31, len(goodtimes_for_drf.coords["met"])):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.GOOD
            ), f"MET at index {i} should be good"

    def test_mark_drf_times_multiple_transitions(
        self, goodtimes_for_drf, hk_multiple_drf_transitions
    ):
        """Test that multiple DRF transitions remove multiple windows."""
        mark_drf_times(goodtimes_for_drf, hk_multiple_drf_transitions)

        # First transition at index 30 (MET 2800.0)
        # Window: 2800 - 1800 = 1000 to 2800 (inclusive, so indices 0-30)

        # Second transition at index 90 (MET 6400.0)
        # Window: 6400 - 1800 = 4600 to 6400 (inclusive, so indices 60-90)

        # Check first window (indices 0-30)
        for i in range(31):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.DRF
            ), f"MET at index {i} should be culled (first window)"

        # Check between windows (indices 31-59, should be good)
        for i in range(31, 60):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.GOOD
            ), f"MET at index {i} should be good (between windows)"

        # Check second window (indices 60-90)
        for i in range(60, 91):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.DRF
            ), f"MET at index {i} should be culled (second window)"

        # Check after second window (indices 91+, should be good)
        for i in range(91, len(goodtimes_for_drf.coords["met"])):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.GOOD
            ), f"MET at index {i} should be good (after windows)"

    def test_mark_drf_times_no_drf(self, goodtimes_for_drf, hk_no_drf):
        """Test that no times are removed when DRF is never active."""
        mark_drf_times(goodtimes_for_drf, hk_no_drf)

        # All times should remain good
        assert np.all(goodtimes_for_drf["cull_flags"].values == CullCode.GOOD)

    def test_mark_drf_times_always_drf(self, goodtimes_for_drf, hk_always_drf):
        """Test that no times are removed when DRF is always active (no transitions)."""
        mark_drf_times(goodtimes_for_drf, hk_always_drf)

        # All times should remain good (no 1->0 transitions)
        assert np.all(goodtimes_for_drf["cull_flags"].values == CullCode.GOOD)

    def test_mark_drf_times_empty_hk(self, goodtimes_for_drf, hk_empty):
        """Test that function handles empty HK data gracefully."""
        # Should log warning and return without error
        mark_drf_times(goodtimes_for_drf, hk_empty)

        # All times should remain good
        assert np.all(goodtimes_for_drf["cull_flags"].values == CullCode.GOOD)

    def test_mark_drf_times_custom_cull_code(
        self, goodtimes_for_drf, hk_single_drf_transition
    ):
        """Test that custom cull code is used."""
        custom_cull_code = 5
        mark_drf_times(
            goodtimes_for_drf, hk_single_drf_transition, cull_code=custom_cull_code
        )

        # Check that culled times use custom code (indices 0-30)
        for i in range(31):
            assert np.all(
                goodtimes_for_drf["cull_flags"].values[i, :] == custom_cull_code
            ), f"MET at index {i} should use custom cull code"

    def test_mark_drf_times_overwrites_existing_culls(
        self, goodtimes_for_drf, hk_single_drf_transition
    ):
        """Test that existing cull flags are overwritten by DRF culling."""
        # Manually set some METs to a different cull code
        goodtimes_for_drf["cull_flags"].values[0:5, :] = 2

        mark_drf_times(goodtimes_for_drf, hk_single_drf_transition)

        # First 5 METs should now be DRF (overwritten via bitwise OR with existing 2)
        for i in range(5):
            assert np.all(goodtimes_for_drf["cull_flags"].values[i, :] == CullCode.DRF)

    def test_mark_drf_times_transition_at_start(self):
        """Test DRF transition near the start - window exactly at data start."""
        # Create goodtimes starting at a later time
        met_values = np.arange(2000.0, 4000.0, 60)
        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((len(met_values), 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                ),
                "esa_step": xr.DataArray(
                    np.ones(len(met_values), dtype=np.uint8), dims=["met"]
                ),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )

        # HK with DRF active for first 30 samples, then transition
        # Transition at index 30 gives window that exactly matches goodtimes start
        shcoarse = np.arange(2000.0, 4000.0, 60)
        fsw_thruster_warn = np.zeros(len(shcoarse), dtype=np.uint8)
        fsw_thruster_warn[0:30] = 1  # Active for first 30 samples

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )

        mark_drf_times(gt, hk)

        # Transition at index 30 (MET 3800.0)
        # Window: 3800 - 1800 = 2000 to 3800
        # This includes METs from 2000 to 3800 (indices 0-30)
        for i in range(31):
            assert np.all(gt["cull_flags"].values[i, :] == CullCode.DRF), (
                f"MET at index {i} should be culled"
            )

        # Rest should be good
        for i in range(31, len(met_values)):
            assert np.all(gt["cull_flags"].values[i, :] == CullCode.GOOD), (
                f"MET at index {i} should be good"
            )

    def test_mark_drf_times_transition_at_end(self):
        """Test DRF transition at the very end of HK data."""
        # Create goodtimes
        met_values = np.arange(1000.0, 3000.0, 60)
        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((len(met_values), 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                ),
                "esa_step": xr.DataArray(
                    np.ones(len(met_values), dtype=np.uint8), dims=["met"]
                ),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )

        # HK with DRF becoming active mid-way, then transition at end
        shcoarse = np.arange(1000.0, 3000.0, 60)
        fsw_thruster_warn = np.zeros(len(shcoarse), dtype=np.uint8)
        fsw_thruster_warn[-10:] = 1  # Active for last 10 samples
        fsw_thruster_warn[-1] = 0  # Transition at last sample

        hk = xr.Dataset(
            {
                "shcoarse": (["epoch"], shcoarse),
                "fsw_thruster_warn": (["epoch"], fsw_thruster_warn),
            }
        )

        mark_drf_times(gt, hk)

        # Transition at last index (MET ~2940)
        # Should remove 30-minute window before it
        # Most METs should still be good except the last ~30
        n_culled = np.sum(gt["cull_flags"].values[:, 0] == CullCode.DRF)
        assert n_culled > 0  # Some should be culled
        assert n_culled <= 31  # But not all (only last ~30 minutes)


class TestMarkBadTdcCal:
    """Test suite for mark_bad_tdc_cal() function."""

    @pytest.fixture
    def goodtimes_for_tdc(self):
        """Create a goodtimes dataset with METs spanning a range."""
        # Create METs every 50 seconds for 200 seconds (5 METs)
        n_mets = 5
        met_values = np.arange(1000.0, 1000.0 + n_mets * 50, 50)

        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((n_mets, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(np.ones(n_mets, dtype=np.uint8), dims=["met"]),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )
        return gt

    @pytest.fixture
    def diagfee_all_good(self):
        """Create DIAG_FEE dataset where all TDC calibrations pass."""
        # 4 DIAG_FEE packets, all with bit 1 set (=2, meaning calibration good)
        return xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1050, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
            }
        )

    @pytest.fixture
    def diagfee_tdc1_fails(self):
        """Create DIAG_FEE dataset where TDC1 fails at packet index 2."""
        # TDC1 fails at index 2 (bit 1 not set, so value 0)
        return xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1050, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (
                    ["epoch"],
                    np.array([2, 2, 0, 2]),
                ),  # fails at idx 2
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
            }
        )

    @pytest.fixture
    def diagfee_with_duplicate(self):
        """Create DIAG_FEE dataset with duplicate packets within 10 seconds."""
        # First two packets are within 10 seconds (should skip first)
        return xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1005, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (
                    ["epoch"],
                    np.array([0, 2, 2, 2]),
                ),  # First would fail but is skipped
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
            }
        )

    def test_mark_bad_tdc_cal_all_good(self, goodtimes_for_tdc, diagfee_all_good):
        """Test that no times are marked when all TDC calibrations pass."""
        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_all_good)

        # All times should remain good
        assert np.all(goodtimes_for_tdc["cull_flags"].values == CullCode.GOOD)

    def test_mark_bad_tdc_cal_tdc1_fails(self, goodtimes_for_tdc, diagfee_tdc1_fails):
        """Test that times are marked when TDC1 fails."""
        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_tdc1_fails)

        # TDC1 fails at packet 2 (MET 1100), should mark times from 1100 to 1150
        # goodtimes METs are [1000, 1050, 1100, 1150, 1200]
        # MET 1100 falls in window [1100, 1150), so MET 1100 should be culled
        met_values = goodtimes_for_tdc.coords["met"].values

        # MET 1100 (index 2) should be culled
        idx_1100 = np.where(met_values == 1100.0)[0][0]
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[idx_1100, :] == CullCode.BAD_TDC_CAL
        )

        # METs before 1100 should still be good
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[0, :] == CullCode.GOOD
        )  # 1000
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[1, :] == CullCode.GOOD
        )  # 1050

    def test_mark_bad_tdc_cal_skip_duplicate_packets(
        self, goodtimes_for_tdc, diagfee_with_duplicate
    ):
        """Test that duplicate DIAG_FEE packets within 10 seconds are skipped."""
        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_with_duplicate)

        # First packet (MET 1000) has TDC1 fail but should be skipped
        # because it's within 10 seconds of the next packet (MET 1005)
        # So all times should remain good
        assert np.all(goodtimes_for_tdc["cull_flags"].values == CullCode.GOOD)

    def test_mark_bad_tdc_cal_insufficient_packets(self, goodtimes_for_tdc):
        """Test that less than 2 packets logs warning and returns early."""
        # Create DIAG_FEE with only 1 packet
        diagfee_single = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000])),
                "tdc1_cal_ctrl_stat": (["epoch"], np.array([0])),  # Fails but ignored
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2])),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2])),
            }
        )

        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_single)

        # All times should remain good (no culling due to insufficient packets)
        assert np.all(goodtimes_for_tdc["cull_flags"].values == CullCode.GOOD)

    def test_mark_bad_tdc_cal_tdc2_fails(self, goodtimes_for_tdc):
        """Test that times are marked when TDC2 fails."""
        diagfee_tdc2_fails = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1050, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc2_cal_ctrl_stat": (
                    ["epoch"],
                    np.array([2, 0, 2, 2]),
                ),  # fails at idx 1
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
            }
        )

        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_tdc2_fails)

        # TDC2 fails at packet 1 (MET 1050), should mark times from 1050 to 1100
        met_values = goodtimes_for_tdc.coords["met"].values

        # MET 1050 (index 1) should be culled
        idx_1050 = np.where(met_values == 1050.0)[0][0]
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[idx_1050, :] == CullCode.BAD_TDC_CAL
        )

    def test_mark_bad_tdc_cal_tdc3_fails(self, goodtimes_for_tdc):
        """Test that times are marked when TDC3 fails."""
        diagfee_tdc3_fails = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1050, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc3_cal_ctrl_stat": (
                    ["epoch"],
                    np.array([0, 2, 2, 2]),
                ),  # fails at idx 0
            }
        )

        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_tdc3_fails)

        # TDC3 fails at packet 0 (MET 1000), should mark times from 1000 to 1050
        # MET 1000 (index 0) should be culled
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[0, :] == CullCode.BAD_TDC_CAL
        )  # 1000

        # MET 1050 should be good (next DIAG_FEE packet starts good window)
        assert np.all(
            goodtimes_for_tdc["cull_flags"].values[1, :] == CullCode.GOOD
        )  # 1050

    def test_mark_bad_tdc_cal_custom_cull_code(
        self, goodtimes_for_tdc, diagfee_tdc1_fails
    ):
        """Test that custom cull code is used."""
        custom_cull_code = 5
        mark_bad_tdc_cal(
            goodtimes_for_tdc, diagfee_tdc1_fails, cull_code=custom_cull_code
        )

        # Check that culled times use custom code
        assert np.any(goodtimes_for_tdc["cull_flags"].values == custom_cull_code)

    def test_mark_bad_tdc_cal_last_packet_fails(self, goodtimes_for_tdc):
        """Test behavior when the last DIAG_FEE packet has TDC failure."""
        diagfee_last_fails = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([1000, 1050, 1100, 1150])),
                "tdc1_cal_ctrl_stat": (
                    ["epoch"],
                    np.array([2, 2, 2, 0]),
                ),  # fails at last
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([2, 2, 2, 2])),
            }
        )

        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_last_fails)

        # TDC1 fails at last packet (MET 1150), should mark all times >= 1150
        met_values = goodtimes_for_tdc.coords["met"].values

        # METs >= 1150 should be culled
        for i, met in enumerate(met_values):
            if met >= 1150:
                assert np.all(
                    goodtimes_for_tdc["cull_flags"].values[i, :] == CullCode.BAD_TDC_CAL
                )
            else:
                assert np.all(
                    goodtimes_for_tdc["cull_flags"].values[i, :] == CullCode.GOOD
                )

    def test_mark_bad_tdc_cal_empty_diagfee(self, goodtimes_for_tdc):
        """Test that function handles empty DIAG_FEE data gracefully."""
        diagfee_empty = xr.Dataset(
            {
                "shcoarse": (["epoch"], np.array([], dtype=np.float64)),
                "tdc1_cal_ctrl_stat": (["epoch"], np.array([], dtype=np.uint8)),
                "tdc2_cal_ctrl_stat": (["epoch"], np.array([], dtype=np.uint8)),
                "tdc3_cal_ctrl_stat": (["epoch"], np.array([], dtype=np.uint8)),
            }
        )

        # Should log warning and return without error
        mark_bad_tdc_cal(goodtimes_for_tdc, diagfee_empty)

        # All times should remain good
        assert np.all(goodtimes_for_tdc["cull_flags"].values == CullCode.GOOD)


class TestMarkOverflowPackets:
    """Test suite for mark_overflow_packets function."""

    @pytest.fixture
    def mock_config_df(self):
        """Create a mock calibration product configuration DataFrame."""
        # Create a minimal config with coincidence types
        # ABC1C2 = 15, ABC1 = 14, AB = 12
        data = {
            "coincidence_type_list": [("ABC1C2", "ABC1"), ("AB",)],
            "tof_ab_low": [0, 0],
            "tof_ab_high": [100, 100],
            "tof_ac1_low": [0, 0],
            "tof_ac1_high": [100, 100],
            "tof_bc1_low": [-50, -50],
            "tof_bc1_high": [50, 50],
            "tof_c1c2_low": [0, 0],
            "tof_c1c2_high": [100, 100],
        }
        df = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(
                [(1, 1), (2, 1)], names=["calibration_prod", "esa_energy_step"]
            ),
        )
        # Add coincidence_type_values column (converted from strings to ints)
        # ABC1C2=15, ABC1=14, AB=12
        df["coincidence_type_values"] = [(15, 14), (12,)]
        return df

    @pytest.fixture
    def mock_goodtimes(self):
        """Create a mock goodtimes dataset."""
        met_values = np.arange(1000.0, 1100.0, 10.0)
        return xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((len(met_values), 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                ),
                "esa_step": xr.DataArray(
                    np.ones(len(met_values), dtype=np.uint8), dims=["met"]
                ),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )

    def test_no_full_packets(self, mock_goodtimes, mock_config_df):
        """Test that no culling occurs when no packets are full."""
        # Create L1B DE with packets having < 664 events
        n_events = 100
        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.zeros(n_events, dtype=np.uint16)),
                "coincidence_type": (
                    ["event_met"],
                    np.full(n_events, 15, dtype=np.uint8),
                ),
            },
            coords={"event_met": np.linspace(1000.0, 1010.0, n_events)},
        )

        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)

        # No times should be culled
        assert np.all(mock_goodtimes["cull_flags"].values == 0)

    def test_full_packet_with_qualified_event(self, mock_goodtimes, mock_config_df):
        """Test that full packet with qualified final event is culled."""
        # Create L1B DE with one packet having exactly 664 events
        n_events = 664
        event_mets = np.linspace(1005.0, 1006.0, n_events)
        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.zeros(n_events, dtype=np.uint16)),
                # Final event has coincidence_type=15 (ABC1C2), which is qualified
                "coincidence_type": (
                    ["event_met"],
                    np.full(n_events, 15, dtype=np.uint8),
                ),
            },
            coords={"event_met": event_mets},
        )

        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)

        # MET ~1006 should be culled (maps to goodtimes MET 1000)
        # The MET 1000 bin should have all spin bins culled with OVERFLOW flag
        assert np.all(mock_goodtimes["cull_flags"].values[0, :] == CullCode.OVERFLOW)

    def test_full_packet_with_unqualified_event(self, mock_goodtimes, mock_config_df):
        """Test that full packet with unqualified final event is NOT culled."""
        # Create L1B DE with one packet having exactly 664 events
        n_events = 664
        event_mets = np.linspace(1005.0, 1006.0, n_events)
        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.zeros(n_events, dtype=np.uint16)),
                # Final event has coincidence_type=3 (not in any cal product)
                "coincidence_type": (
                    ["event_met"],
                    np.full(n_events, 3, dtype=np.uint8),
                ),
            },
            coords={"event_met": event_mets},
        )

        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)

        # No times should be culled since final event is unqualified
        assert np.all(mock_goodtimes["cull_flags"].values == 0)

    def test_multiple_full_packets(self, mock_goodtimes, mock_config_df):
        """Test handling of multiple full packets."""
        # Create L1B DE with two packets, each having 664 events
        n_events_per_packet = 664
        n_packets = 2

        ccsds_indices = np.concatenate(
            [np.full(n_events_per_packet, i, dtype=np.uint16) for i in range(n_packets)]
        )
        # Packet 0: final event qualified (15)
        # Packet 1: final event unqualified (3)
        coincidence_types = np.concatenate(
            [
                np.concatenate(
                    [np.full(n_events_per_packet - 1, 3, dtype=np.uint8), [15]]
                ),
                np.full(n_events_per_packet, 3, dtype=np.uint8),
            ]
        )
        event_mets = np.concatenate(
            [
                np.linspace(1005.0, 1006.0, n_events_per_packet),  # Packet 0
                np.linspace(1015.0, 1016.0, n_events_per_packet),  # Packet 1
            ]
        )

        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], ccsds_indices),
                "coincidence_type": (["event_met"], coincidence_types),
            },
            coords={"event_met": event_mets},
        )

        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)

        # Only packet 0's MET should be culled (MET 1000)
        # Packet 1 has unqualified final event, so MET 1010 should not be culled
        assert np.sum(mock_goodtimes["cull_flags"].values[0, :] > 0) == 90  # All bins
        assert np.all(mock_goodtimes["cull_flags"].values[1, :] == 0)  # MET 1010

    def test_empty_de_data(self, mock_goodtimes, mock_config_df):
        """Test handling of empty L1B DE data."""
        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.array([], dtype=np.uint16)),
                "coincidence_type": (["event_met"], np.array([], dtype=np.uint8)),
            },
            coords={"event_met": np.array([])},
        )

        # Should not raise, just return without culling
        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)
        assert np.all(mock_goodtimes["cull_flags"].values == 0)

    def test_custom_cull_code(self, mock_goodtimes, mock_config_df):
        """Test using a custom cull code."""
        n_events = 664
        event_mets = np.linspace(1005.0, 1006.0, n_events)
        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.zeros(n_events, dtype=np.uint16)),
                "coincidence_type": (
                    ["event_met"],
                    np.concatenate([np.full(n_events - 1, 3, dtype=np.uint8), [15]]),
                ),
            },
            coords={"event_met": event_mets},
        )

        custom_cull = 5
        mark_overflow_packets(
            mock_goodtimes, l1b_de, mock_config_df, cull_code=custom_cull
        )

        # Check that the custom cull code was used
        assert np.any(mock_goodtimes["cull_flags"].values == custom_cull)

    def test_final_event_is_last_in_list(self, mock_goodtimes, mock_config_df):
        """Test that the final event is the last one in the list for the packet."""
        n_events = 664
        event_mets = np.linspace(1005.0, 1006.0, n_events)

        # All events have unqualified type except the last one in the list
        coincidence_types = np.full(n_events, 3, dtype=np.uint8)
        coincidence_types[-1] = 12  # Last event is qualified

        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], np.zeros(n_events, dtype=np.uint16)),
                "coincidence_type": (["event_met"], coincidence_types),
            },
            coords={"event_met": event_mets},
        )

        mark_overflow_packets(mock_goodtimes, l1b_de, mock_config_df)

        # Should be culled because the final event (last in list) is qualified
        assert np.sum(mock_goodtimes["cull_flags"].values > 0) > 0


class TestGetSweepIndices:
    """Test suite for _get_sweep_indices() helper function."""

    def test_empty_array(self):
        """Test with empty input."""
        result = _get_sweep_indices(np.array([]))
        assert len(result) == 0
        assert result.dtype == np.int32

    def test_single_sweep(self):
        """Test with single complete ESA sweep (no transitions)."""
        esa_step = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = _get_sweep_indices(esa_step)

        # All should be in sweep 0
        np.testing.assert_array_equal(result, np.zeros(9, dtype=np.int32))

    def test_two_sweeps_standard_transition(self):
        """Test with two sweeps with standard 9->1 transition."""
        esa_step = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = _get_sweep_indices(esa_step)

        # First 9 should be sweep 0, next 9 should be sweep 1
        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32
        )
        np.testing.assert_array_equal(result, expected)

    def test_multiple_sweeps(self):
        """Test with multiple sweeps."""
        esa_step = np.array([3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3])
        result = _get_sweep_indices(esa_step)

        # Transitions at index 6->7 (9->1) and 15->16 (9->1)
        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=np.int32
        )
        np.testing.assert_array_equal(result, expected)

    def test_non_standard_transition(self):
        """Test with non-standard ESA step decrease (e.g., 5->2)."""
        esa_step = np.array([5, 6, 7, 8, 9, 2, 3, 4, 5])
        result = _get_sweep_indices(esa_step)

        # Transition at index 4->5 (9->2, diff=-7, negative so boundary)
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_no_decreases_only_increases(self):
        """Test with only increasing steps (no sweep boundaries)."""
        esa_step = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = _get_sweep_indices(esa_step)

        # All in sweep 0
        np.testing.assert_array_equal(result, np.zeros(9, dtype=np.int32))

    def test_constant_esa_step(self):
        """Test with constant ESA step (no transitions)."""
        esa_step = np.array([5, 5, 5, 5, 5])
        result = _get_sweep_indices(esa_step)

        # All in sweep 0
        np.testing.assert_array_equal(result, np.zeros(5, dtype=np.int32))


class TestAddSweepIndices:
    """Test suite for _add_sweep_indices() helper function."""

    def test_adds_coordinate(self):
        """Test that esa_sweep coordinate is added."""
        ds = xr.Dataset(
            {
                "ccsds_met": (["epoch"], np.array([1000.0, 1060.0, 1120.0])),
                "esa_step": (["epoch"], np.array([1, 2, 3], dtype=np.uint8)),
            },
            coords={"epoch": np.arange(3)},
        )

        result = _add_sweep_indices(ds)

        assert "esa_sweep" in result.coords
        assert result.coords["esa_sweep"].dims == ("epoch",)

    def test_coordinate_values(self):
        """Test that sweep indices are correctly calculated."""
        ds = xr.Dataset(
            {
                "ccsds_met": (["epoch"], np.arange(1000.0, 1000.0 + 18 * 60, 60)),
                "esa_step": (
                    ["epoch"],
                    np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 2).astype(np.uint8),
                ),
            },
            coords={"epoch": np.arange(18)},
        )

        result = _add_sweep_indices(ds)

        # First 9 should be sweep 0, next 9 should be sweep 1
        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32
        )
        np.testing.assert_array_equal(result.coords["esa_sweep"].values, expected)

    def test_preserves_original_data(self):
        """Test that original dataset variables are preserved."""
        ds = xr.Dataset(
            {
                "ccsds_met": (["epoch"], np.array([1000.0, 1060.0, 1120.0])),
                "esa_step": (["epoch"], np.array([1, 2, 1], dtype=np.uint8)),
                "other_var": (["epoch"], np.array([10, 20, 30])),
            },
            coords={"epoch": np.arange(3)},
        )

        result = _add_sweep_indices(ds)

        assert "ccsds_met" in result.data_vars
        assert "esa_step" in result.data_vars
        assert "other_var" in result.data_vars
        np.testing.assert_array_equal(
            result["ccsds_met"].values, ds["ccsds_met"].values
        )


class TestComputeNormalizedCountsPerSweep:
    """Test suite for _compute_normalized_counts_per_sweep() helper function."""

    def _create_test_dataset(
        self,
        n_sweeps: int = 2,
        n_esa_steps: int = 9,
        packets_per_esa_step: int = 2,
        events_per_packet: int = 10,
        tof_ab_range: tuple[int, int] = (-15, 15),
    ) -> xr.Dataset:
        """Create a test L1B DE dataset with esa_sweep coordinate."""
        n_packets = n_sweeps * n_esa_steps * packets_per_esa_step
        n_events = n_packets * events_per_packet

        # Create ESA steps: each step repeated packets_per_esa_step times per sweep
        # e.g., [1,1,2,2,3,3,...,9,9, 1,1,2,2,3,3,...,9,9] for 2 sweeps, 2 packets/step
        esa_step = np.tile(
            np.repeat(np.arange(1, n_esa_steps + 1), packets_per_esa_step), n_sweeps
        ).astype(np.uint8)

        # esa_energy_step same as esa_step for test purposes
        # (in real data they can differ)
        esa_energy_step = esa_step.copy()

        # Create METs with unique incrementing values for each packet
        ccsds_met = np.arange(1000.0, 1000.0 + n_packets * 60, 60)

        # Create events
        tof_ab = np.random.randint(tof_ab_range[0], tof_ab_range[1], n_events).astype(
            np.int32
        )
        coincidence_type = np.full(n_events, 12, dtype=np.uint8)  # AB = 12
        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )

        ds = xr.Dataset(
            {
                "ccsds_met": (["epoch"], ccsds_met),
                "esa_step": (["epoch"], esa_step),
                "esa_energy_step": (["epoch"], esa_energy_step),
                "tof_ab": (["event_met"], tof_ab),
                "coincidence_type": (["event_met"], coincidence_type),
                "ccsds_index": (["event_met"], ccsds_index),
            },
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(n_events),
            },
        )

        # Add sweep indices
        return _add_sweep_indices(ds)

    def test_output_dimensions(self):
        """Test that output has correct dimensions."""
        np.random.seed(42)
        ds = self._create_test_dataset(n_sweeps=2)

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        assert "esa_sweep" in result.dims
        assert "esa_energy_step" in result.dims
        assert "epoch" not in result.dims
        assert "event_met" not in result.dims

    def test_normalized_count_added(self):
        """Test that normalized_count variable is added."""
        np.random.seed(42)
        ds = self._create_test_dataset(n_sweeps=2)

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        assert "normalized_count" in result.data_vars
        assert result["normalized_count"].dims == ("esa_sweep",)

    def test_normalized_count_calculation(self):
        """Test that normalized counts are calculated correctly."""
        np.random.seed(42)
        # Create dataset where we know exact counts
        n_sweeps = 2
        n_esa_steps = 9
        events_per_packet = 10

        # All events within ±15ns
        ds = self._create_test_dataset(
            n_sweeps=n_sweeps,
            n_esa_steps=n_esa_steps,
            events_per_packet=events_per_packet,
            tof_ab_range=(-10, 10),
        )

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        # Allow some tolerance for randomness in tof_ab values
        assert len(result["normalized_count"]) == n_sweeps
        assert np.all(result["normalized_count"].values >= 0)

    def test_filters_by_tof_ab_limit(self):
        """Test that events outside tof_ab limit are excluded."""
        np.random.seed(42)
        # Create dataset with events outside limit
        ds = self._create_test_dataset(n_sweeps=2, tof_ab_range=(20, 100))

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        # All events have |tof_ab| > 15, so counts should be 0
        np.testing.assert_array_equal(result["normalized_count"].values, np.zeros(2))

    def test_preserves_epoch_variables(self):
        """Test that epoch-based variables are preserved."""
        np.random.seed(42)
        ds = self._create_test_dataset(n_sweeps=2)

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        assert "ccsds_met" in result.data_vars
        # esa_step becomes a coordinate (dimension) after unstack
        assert "esa_energy_step" in result.coords

    def test_removes_event_met_variables(self):
        """Test that event_met dimension variables are removed."""
        np.random.seed(42)
        ds = self._create_test_dataset(n_sweeps=2)

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        # Variables that were on event_met dimension should be gone
        assert "tof_ab" not in result.data_vars
        assert "coincidence_type" not in result.data_vars
        assert "ccsds_index" not in result.data_vars

    def test_raises_without_esa_sweep_coordinate(self):
        """Test that function raises error without esa_sweep coordinate."""
        ds = xr.Dataset(
            {
                "ccsds_met": (["epoch"], np.array([1000.0, 1060.0])),
                "esa_step": (["epoch"], np.array([1, 2], dtype=np.uint8)),
            },
            coords={"epoch": np.arange(2)},
        )

        with pytest.raises(ValueError, match="must have esa_sweep coordinate"):
            _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

    def test_multiple_sweeps(self):
        """Test with multiple sweeps."""
        np.random.seed(42)
        ds = self._create_test_dataset(n_sweeps=5)

        result = _compute_normalized_counts_per_sweep(ds, tof_ab_limit_ns=15)

        assert len(result["normalized_count"]) == 5
        assert result.dims["esa_sweep"] == 5


class TestStatisticalFilter0:
    """Test suite for mark_statistical_filter_0() integration tests."""

    @pytest.fixture
    def goodtimes_for_filter(self):
        """Create a goodtimes dataset for testing statistical filter 0."""
        # Create 2 complete ESA sweeps (9 METs each = 18 total)
        n_mets = 18
        met_values = np.arange(1000.0, 1000.0 + n_mets * 60, 60)

        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((n_mets, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(
                    np.tile(np.arange(1, 10), 2).astype(np.uint8), dims=["met"]
                ),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )
        return gt

    def _create_l1b_de_dataset(
        self,
        n_sweeps: int = 2,
        events_per_met: int = 10,
        tof_ab_range: tuple[int, int] = (-15, 15),
        base_met: float = 1000.0,
    ) -> xr.Dataset:
        """
        Create a mock L1B DE dataset with complete ESA sweeps.

        Parameters
        ----------
        n_sweeps : int
            Number of complete ESA sweeps (each sweep = 9 METs for ESA 1-9).
        events_per_met : int
            Number of events per MET.
        tof_ab_range : tuple[int, int]
            Range for random tof_ab values.
        base_met : float
            Base MET value for the dataset.

        Returns
        -------
        xr.Dataset
            Mock L1B DE dataset with complete ESA sweeps.
        """
        n_esa_steps = 9  # ESA steps 1-9
        n_packets = n_sweeps * n_esa_steps
        n_events = n_packets * events_per_met

        # Create ESA steps cycling through 1-9 for each sweep
        esa_step = np.tile(np.arange(1, n_esa_steps + 1), n_sweeps).astype(np.uint8)
        # esa_energy_step same as esa_step for test purposes
        esa_energy_step = esa_step.copy()

        # Create ccsds_met for packets
        ccsds_met = np.arange(base_met, base_met + n_packets * 60, 60, dtype=np.float64)

        # Create events distributed across packets
        tof_ab_values = np.random.randint(
            tof_ab_range[0], tof_ab_range[1], n_events
        ).astype(np.int32)
        coincidence_type = np.full(n_events, 12, dtype=np.uint8)  # AB coincidence
        ccsds_index = np.repeat(np.arange(n_packets), events_per_met).astype(np.uint16)

        return xr.Dataset(
            data_vars={
                "tof_ab": (["event_met"], tof_ab_values, {"FILLVAL": -2147483648}),
                "coincidence_type": (["event_met"], coincidence_type),
                "ccsds_index": (["event_met"], ccsds_index),
                "ccsds_met": (["epoch"], ccsds_met),
                "esa_step": (["epoch"], esa_step, {"FILLVAL": 255}),
                "esa_energy_step": (["epoch"], esa_energy_step, {"FILLVAL": 255}),
            },
            coords={
                "event_met": np.arange(n_events),
                "epoch": np.arange(n_packets),
            },
        )

    def test_passes_normal_sweeps(self, goodtimes_for_filter):
        """Test that similar counts across sweeps passes the filter."""
        np.random.seed(42)
        # Create 5 datasets with 2 sweeps each, similar event counts
        l1b_de_datasets = [
            self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10) for _ in range(5)
        ]

        mark_statistical_filter_0(
            goodtimes_for_filter, l1b_de_datasets, current_index=2
        )

        # All times should still be good (no sweeps exceed threshold)
        assert np.all(goodtimes_for_filter["cull_flags"].values == CullCode.GOOD)

    def test_fails_anomalous_sweep(self, goodtimes_for_filter):
        """Test that sweeps exceeding 150% median are marked as culled."""
        np.random.seed(42)
        l1b_de_datasets = []

        for i in range(5):
            if i == 2:  # Current pointing - create many more events
                ds = self._create_l1b_de_dataset(n_sweeps=2, events_per_met=50)
            else:
                ds = self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10)
            l1b_de_datasets.append(ds)

        mark_statistical_filter_0(
            goodtimes_for_filter, l1b_de_datasets, current_index=2
        )

        # Current sweeps have 5x the events, should be culled
        # Check that at least some METs are culled
        assert np.any(
            goodtimes_for_filter["cull_flags"].values == CullCode.STAT_FILTER_0
        )

    def test_insufficient_pointings(self, goodtimes_for_filter):
        """Test that fewer than min_pointings raises ValueError."""
        l1b_de_datasets = [
            self._create_l1b_de_dataset(),
            self._create_l1b_de_dataset(),
            self._create_l1b_de_dataset(),
        ]

        with pytest.raises(ValueError, match="At least 4 valid Pointings required"):
            mark_statistical_filter_0(
                goodtimes_for_filter, l1b_de_datasets, current_index=2
            )

    def test_current_index_out_of_range(self, goodtimes_for_filter):
        """Test that current_index out of range raises ValueError."""
        l1b_de_datasets = [self._create_l1b_de_dataset()] * 5

        with pytest.raises(ValueError, match="current_index.*out of range"):
            mark_statistical_filter_0(
                goodtimes_for_filter, l1b_de_datasets, current_index=10
            )

    def test_partial_sweep_culling(self, goodtimes_for_filter):
        """Test that only bad sweeps are culled, not entire Pointing."""
        np.random.seed(42)

        # Create current pointing with one normal sweep and one anomalous sweep
        n_esa_steps = 9
        n_packets = 2 * n_esa_steps  # 2 sweeps

        # First sweep: normal count (10 events/MET)
        # Second sweep: high count (100 events/MET)
        events_sweep1 = 10 * n_esa_steps
        events_sweep2 = 100 * n_esa_steps
        n_events = events_sweep1 + events_sweep2

        esa_step = np.tile(np.arange(1, 10), 2).astype(np.uint8)
        esa_energy_step = esa_step.copy()
        ccsds_met = np.arange(1000.0, 1000.0 + n_packets * 60, 60, dtype=np.float64)

        # Events for first sweep (packets 0-8)
        ccsds_index_1 = np.repeat(np.arange(n_esa_steps), 10).astype(np.uint16)
        # Events for second sweep (packets 9-17)
        ccsds_index_2 = np.repeat(np.arange(n_esa_steps, 2 * n_esa_steps), 100).astype(
            np.uint16
        )
        ccsds_index = np.concatenate([ccsds_index_1, ccsds_index_2])

        tof_ab = np.random.randint(-10, 10, n_events).astype(np.int32)
        coincidence_type = np.full(n_events, 12, dtype=np.uint8)

        current_ds = xr.Dataset(
            data_vars={
                "tof_ab": (["event_met"], tof_ab, {"FILLVAL": -2147483648}),
                "coincidence_type": (["event_met"], coincidence_type),
                "ccsds_index": (["event_met"], ccsds_index),
                "ccsds_met": (["epoch"], ccsds_met),
                "esa_step": (["epoch"], esa_step, {"FILLVAL": 255}),
                "esa_energy_step": (["epoch"], esa_energy_step, {"FILLVAL": 255}),
            },
            coords={
                "event_met": np.arange(n_events),
                "epoch": np.arange(n_packets),
            },
        )

        # Other pointings: all normal (10 events/MET)
        l1b_de_datasets = [
            self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10),
            self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10),
            current_ds,  # Current with mixed sweeps
            self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10),
            self._create_l1b_de_dataset(n_sweeps=2, events_per_met=10),
        ]

        mark_statistical_filter_0(
            goodtimes_for_filter, l1b_de_datasets, current_index=2
        )

        # First 9 METs (sweep 1) should be good, last 9 METs (sweep 2) should be bad
        first_sweep_flags = goodtimes_for_filter["cull_flags"].values[:9, :]
        second_sweep_flags = goodtimes_for_filter["cull_flags"].values[9:, :]

        assert np.all(first_sweep_flags == CullCode.GOOD)
        assert np.all(second_sweep_flags == CullCode.STAT_FILTER_0)


class TestIdentifyCullPattern:
    """Test suite for _identify_cull_pattern() convolution-based pattern detection."""

    def _create_test_data(
        self, n_sweeps: int = 10, n_esa_steps: int = 5
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Create test counts, median, and sigma DataArrays."""
        # Create counts array (esa_sweep x esa_energy_step)
        counts = xr.DataArray(
            np.zeros((n_sweeps, n_esa_steps)),
            dims=["esa_sweep", "esa_energy_step"],
            coords={
                "esa_sweep": np.arange(n_sweeps),
                "esa_energy_step": np.arange(1, n_esa_steps + 1),
            },
        )

        # Create median and sigma per ESA energy step (all valid)
        median = xr.DataArray(
            np.full(n_esa_steps, 10.0),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": np.arange(1, n_esa_steps + 1)},
        )
        sigma = xr.DataArray(
            np.full(n_esa_steps, 3),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": np.arange(1, n_esa_steps + 1)},
        )

        return counts, median, sigma

    def test_no_exceedances(self):
        """Test with counts below all thresholds."""
        counts, median, sigma = self._create_test_data()
        # All counts are 0, well below threshold of ~15 (10 + 1.8*3)

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        assert cull_mask.dims == ("esa_sweep", "esa_energy_step")
        assert not cull_mask.any()

    def test_consecutive_run_with_esa_neighbor(self):
        """Test finding 3+ consecutive high counts with ESA neighbor confirmation."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)
        # threshold = 10 + 1.8 * 3 = 15.4

        # Create 4 consecutive high counts at ESA step 3 (sweeps 2-5)
        counts.loc[2:5, 3] = 20
        # Also make ESA step 2 high at same time positions (neighbor at same time)
        counts.loc[2:5, 2] = 20

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Sweeps 2-5 at ESA 3 should be marked
        # (consecutive run with neighbor at same time)
        assert cull_mask.sel(esa_sweep=2, esa_energy_step=3).values
        assert cull_mask.sel(esa_sweep=3, esa_energy_step=3).values
        assert cull_mask.sel(esa_sweep=4, esa_energy_step=3).values
        assert cull_mask.sel(esa_sweep=5, esa_energy_step=3).values
        # ESA 2 should also be marked (consecutive run with ESA 3 as neighbor)
        assert cull_mask.sel(esa_sweep=2, esa_energy_step=2).values
        assert cull_mask.sel(esa_sweep=3, esa_energy_step=2).values
        assert cull_mask.sel(esa_sweep=4, esa_energy_step=2).values
        assert cull_mask.sel(esa_sweep=5, esa_energy_step=2).values

    def test_consecutive_run_no_esa_neighbor(self):
        """Test that consecutive run without ESA neighbor is not marked."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)

        # Create 4 consecutive high counts at ESA step 3, but no neighbors high
        counts.loc[2:5, 3] = 20

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Without ESA neighbor confirmation, consecutive runs alone don't trigger
        # (but extreme outliers at 5-sigma would - threshold = 10 + 5*3 = 25)
        assert not cull_mask.sel(esa_energy_step=3).any()

    def test_isolated_interval_marked(self):
        """Test that good interval surrounded by bad is marked."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)

        # Create pattern: bad - good - bad at ESA step 3
        # First create a setup that triggers consecutive culling
        counts.loc[0:3, 3] = 20  # 4 consecutive high
        counts.loc[0:3, 2] = 20  # ESA neighbor
        counts.loc[5:8, 3] = 20  # Another 4 consecutive high
        counts.loc[5:8, 2] = 20  # ESA neighbor
        # Sweep 4 at ESA 3 is good (low count) but surrounded by bad

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Sweep 4 should be marked as isolated
        assert cull_mask.sel(esa_sweep=4, esa_energy_step=3).values

    def test_extreme_outlier(self):
        """Test detection of extreme outliers (5-sigma)."""
        counts, median, sigma = self._create_test_data()
        # extreme threshold = 10 + 5 * 3 = 25

        # Single extreme outlier at sweep 5, ESA 3
        counts.loc[5, 3] = 30

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Only the extreme outlier should be marked
        assert cull_mask.sel(esa_sweep=5, esa_energy_step=3).values
        # Other positions should not be marked
        assert not cull_mask.sel(esa_sweep=4, esa_energy_step=3).values
        assert not cull_mask.sel(esa_sweep=6, esa_energy_step=3).values

    def test_nan_handling(self):
        """Test that NaN values in counts are handled correctly."""
        counts, median, sigma = self._create_test_data()

        # Set some counts to NaN
        counts.loc[3, 2] = np.nan

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # NaN positions should not be marked (treated as not exceeding)
        assert not cull_mask.sel(esa_sweep=3, esa_energy_step=2).values

    def test_returns_dataarray_with_correct_coords(self):
        """Test that returned mask has correct dimensions and coordinates."""
        counts, median, sigma = self._create_test_data(n_sweeps=8, n_esa_steps=4)

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        assert isinstance(cull_mask, xr.DataArray)
        assert cull_mask.dims == counts.dims
        np.testing.assert_array_equal(
            cull_mask.coords["esa_sweep"].values, counts.coords["esa_sweep"].values
        )
        np.testing.assert_array_equal(
            cull_mask.coords["esa_energy_step"].values,
            counts.coords["esa_energy_step"].values,
        )

    def test_consecutive_run_at_first_esa_edge(self):
        """Test that consecutive run at first ESA step passes neighbor check at edge."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)
        # threshold = 10 + 1.8 * 3 = 15.4

        # Create 4 consecutive high counts at ESA step 1 (first ESA step)
        counts.loc[2:5, 1] = 20
        # No ESA neighbor below (edge), but edge should pass the check

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Sweeps 2-5 at ESA 1 should be marked (edge passes neighbor check)
        assert cull_mask.sel(esa_sweep=2, esa_energy_step=1).values
        assert cull_mask.sel(esa_sweep=3, esa_energy_step=1).values
        assert cull_mask.sel(esa_sweep=4, esa_energy_step=1).values
        assert cull_mask.sel(esa_sweep=5, esa_energy_step=1).values

    def test_consecutive_run_at_last_esa_edge(self):
        """Test that consecutive run at last ESA step passes neighbor check at edge."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)
        # threshold = 10 + 1.8 * 3 = 15.4

        # Create 4 consecutive high counts at ESA step 5 (last ESA step)
        counts.loc[2:5, 5] = 20
        # No ESA neighbor above (edge), but edge should pass the check

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Sweeps 2-5 at ESA 5 should be marked (edge passes neighbor check)
        assert cull_mask.sel(esa_sweep=2, esa_energy_step=5).values
        assert cull_mask.sel(esa_sweep=3, esa_energy_step=5).values
        assert cull_mask.sel(esa_sweep=4, esa_energy_step=5).values
        assert cull_mask.sel(esa_sweep=5, esa_energy_step=5).values

    def test_orphan_not_marked_at_time_edge(self):
        """Test that positions at time edges are not marked as orphans."""
        counts, median, sigma = self._create_test_data(n_sweeps=10, n_esa_steps=5)

        # Create bad intervals at sweeps 1 and 3, leaving sweep 0 as "orphan-like"
        # But sweep 0 is at edge and should NOT be marked as orphan
        counts.loc[1:3, 3] = 20  # Consecutive run
        counts.loc[1:3, 2] = 20  # ESA neighbor

        cull_mask = _identify_cull_pattern(counts, median, sigma)

        # Sweep 0 should NOT be marked (edge, not a true orphan)
        assert not cull_mask.sel(esa_sweep=0, esa_energy_step=3).values
        # Sweeps 1-3 should be marked (consecutive with neighbor)
        assert cull_mask.sel(esa_sweep=1, esa_energy_step=3).values
        assert cull_mask.sel(esa_sweep=2, esa_energy_step=3).values
        assert cull_mask.sel(esa_sweep=3, esa_energy_step=3).values


class TestComputeQualifiedCountsPerSweep:
    """Test suite for _compute_qualified_counts_per_sweep() helper function."""

    def _create_test_dataset(
        self,
        n_packets: int = 10,
        events_per_packet: int = 5,
        coincidence_types: list[int] | None = None,
    ) -> xr.Dataset:
        """Create a test L1B DE dataset with esa_sweep coordinate."""
        n_events = n_packets * events_per_packet

        if coincidence_types is None:
            # Default: mix of types, 12 (AB) is qualified
            coincidence_types = [12, 4, 8, 12, 4] * (n_events // 5 + 1)
            coincidence_types = coincidence_types[:n_events]

        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )

        # Create ESA steps: 2 packets per ESA step, ESA steps 1-5
        esa_step = np.repeat(np.arange(1, n_packets // 2 + 1), 2)[:n_packets].astype(
            np.uint8
        )
        # esa_energy_step same as esa_step for test purposes
        esa_energy_step = esa_step.copy()

        ds = xr.Dataset(
            {
                "coincidence_type": (
                    ["event_met"],
                    np.array(coincidence_types, dtype=np.uint8),
                ),
                "ccsds_index": (["event_met"], ccsds_index),
                "ccsds_met": (
                    ["epoch"],
                    np.arange(1000.0, 1000.0 + n_packets * 60, 60),
                ),
                "esa_step": (["epoch"], esa_step),
                "esa_energy_step": (["epoch"], esa_energy_step),
            },
            coords={
                "event_met": np.arange(n_events),
                "epoch": np.arange(n_packets),
            },
        )

        # Add esa_sweep coordinate
        return _add_sweep_indices(ds)

    def test_sums_counts_per_eight_spin(self):
        """Test that counts are summed per 8-spin interval (esa_sweep, esa_step)."""
        # 10 packets, 2 packets per ESA step = 5 unique (esa_sweep, esa_step) combos
        # All in same sweep (no high-to-low transition), ESA steps 1-5
        ds = self._create_test_dataset(n_packets=10, events_per_packet=10)

        # Create qualified mask based on coincidence type 12
        qualified_mask = np.isin(ds["coincidence_type"].values, [12])

        result = _compute_qualified_counts_per_sweep(ds, qualified_mask)

        assert "qualified_count" in result.data_vars
        assert "esa_sweep" in result.dims
        assert "esa_energy_step" in result.dims

        # Each packet has 10 events, 4 are type 12 (from pattern [12,4,8,12,4] * 2)
        # 2 packets per 8-spin set = 8 qualified counts per (esa_sweep, esa_step)
        # Select only the valid ESA steps (1-5)
        for esa in range(1, 6):
            count = (
                result["qualified_count"].sel(esa_sweep=0, esa_energy_step=esa).values
            )
            assert count == 8

    def test_raises_without_coordinate(self):
        """Test that function raises error without esa_sweep coordinate."""
        ds = xr.Dataset(
            {
                "coincidence_type": (["event_met"], np.array([12, 4], dtype=np.uint8)),
                "ccsds_index": (["event_met"], np.array([0, 0], dtype=np.uint16)),
                "ccsds_met": (["epoch"], np.array([1000.0])),
                "esa_step": (["epoch"], np.array([1], dtype=np.uint8)),
            },
            coords={"event_met": np.arange(2), "epoch": np.arange(1)},
        )

        # Create qualified mask for coincidence type 12
        qualified_mask = np.isin(ds["coincidence_type"].values, [12])

        with pytest.raises(ValueError, match="must have esa_sweep coordinate"):
            _compute_qualified_counts_per_sweep(ds, qualified_mask)


class TestBuildPerSweepDatasets:
    """Test suite for _build_per_sweep_datasets() helper function."""

    def _create_test_dataset(
        self, n_packets: int = 18, base_met: float = 1000.0
    ) -> xr.Dataset:
        """Create a test L1B DE dataset with 2 packets per ESA step (9 ESA steps)."""
        events_per_packet = 10
        n_events = n_packets * events_per_packet

        # All events are type 12 (qualified)
        coincidence_types = np.full(n_events, 12, dtype=np.uint8)
        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )

        # 2 packets per ESA step: [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        esa_step = np.repeat(np.arange(1, 10), 2).astype(np.uint8)
        # esa_energy_step same as esa_step for test purposes
        esa_energy_step = esa_step.copy()

        return xr.Dataset(
            {
                "coincidence_type": (["event_met"], coincidence_types),
                "ccsds_index": (["event_met"], ccsds_index),
                "ccsds_met": (
                    ["epoch"],
                    np.arange(base_met, base_met + n_packets * 60, 60),
                ),
                "esa_step": (["epoch"], esa_step),
                "esa_energy_step": (["epoch"], esa_energy_step),
            },
            coords={
                "event_met": np.arange(n_events),
                "epoch": np.arange(n_packets),
            },
        )

    def test_builds_per_sweep_datasets(self):
        """Test that per-sweep datasets are built correctly."""
        ds = self._create_test_dataset()

        # Add qualified mask based on coincidence type 12 directly to dataset
        ds["qualified_mask"] = xr.DataArray(
            np.isin(ds["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        per_sweep_datasets = _build_per_sweep_datasets([ds])

        # Should have per-sweep dataset for index 0 with 2D structure
        assert 0 in per_sweep_datasets
        assert "esa_sweep" in per_sweep_datasets[0].dims
        assert "esa_energy_step" in per_sweep_datasets[0].dims
        # 9 ESA energy steps (1-9) in the data
        assert len(per_sweep_datasets[0].coords["esa_energy_step"]) == 9

        # Each (esa_sweep, esa_energy_step) should have 20 qualified counts
        # 2 packets per ESA step, 10 events each = 20 qualified counts per 8-spin
        for esa in range(1, 10):
            count = (
                per_sweep_datasets[0]["qualified_count"]
                .sel(esa_sweep=0, esa_energy_step=esa)
                .values
            )
            assert count == 20

    def test_multiple_datasets(self):
        """Test with multiple datasets."""
        ds1 = self._create_test_dataset(base_met=1000.0)
        ds2 = self._create_test_dataset(base_met=2000.0)

        # Add qualified masks directly to datasets
        ds1["qualified_mask"] = xr.DataArray(
            np.isin(ds1["coincidence_type"].values, [12]),
            dims=["event_met"],
        )
        ds2["qualified_mask"] = xr.DataArray(
            np.isin(ds2["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        per_sweep_datasets = _build_per_sweep_datasets([ds1, ds2])

        # Should have per-sweep datasets for both indices
        assert 0 in per_sweep_datasets
        assert 1 in per_sweep_datasets


class TestComputeMedianAndSigmaPerEsa:
    """Test suite for _compute_median_and_sigma_per_esa() helper function."""

    def test_basic_calculation(self):
        """Test basic median and sigma calculation."""
        # Create dataset with counts where median is 4 for each ESA energy step
        # Using 5 sweeps with counts [2, 4, 6, 4, 4] for ESA energy steps 1-9
        n_sweeps = 5
        counts_per_sweep = [2, 4, 6, 4, 4]
        counts_2d = np.zeros((n_sweeps, 10))  # ESA energy steps 0-9
        for sweep_idx, count in enumerate(counts_per_sweep):
            for esa in range(1, 10):
                counts_2d[sweep_idx, esa] = count

        per_sweep_datasets = {
            0: xr.Dataset(
                {
                    "qualified_count": (["esa_sweep", "esa_energy_step"], counts_2d),
                    "ccsds_met": (
                        ["esa_sweep", "esa_energy_step"],
                        np.full_like(counts_2d, 1000.0),
                    ),
                },
                coords={
                    "esa_sweep": np.arange(n_sweeps),
                    "esa_energy_step": np.arange(10),
                },
            )
        }

        median_per_esa, sigma_per_esa = _compute_median_and_sigma_per_esa(
            per_sweep_datasets
        )

        for esa in range(1, 10):
            assert median_per_esa.sel(esa_energy_step=esa).values == 4.0
            # sigma = round(sqrt(4 + 1)) = round(2.236) = 2
            assert sigma_per_esa.sel(esa_energy_step=esa).values == 2

    def test_zero_median_excluded(self):
        """Test that ESA energy steps with zero median are excluded."""
        # ESA energy step 1: all zeros, ESA energy step 2: median 4
        n_sweeps = 5
        counts_2d = np.zeros((n_sweeps, 10))
        # ESA 1 stays at 0
        # ESA 2 gets counts [2, 4, 6, 4, 4]
        for sweep_idx, count in enumerate([2, 4, 6, 4, 4]):
            counts_2d[sweep_idx, 2] = count

        per_sweep_datasets = {
            0: xr.Dataset(
                {
                    "qualified_count": (["esa_sweep", "esa_energy_step"], counts_2d),
                    "ccsds_met": (
                        ["esa_sweep", "esa_energy_step"],
                        np.full_like(counts_2d, 1000.0),
                    ),
                },
                coords={
                    "esa_sweep": np.arange(n_sweeps),
                    "esa_energy_step": np.arange(10),
                },
            )
        }

        median_per_esa, sigma_per_esa = _compute_median_and_sigma_per_esa(
            per_sweep_datasets
        )

        # ESA energy step 1 should have NaN median (zero counts excluded)
        assert np.isnan(median_per_esa.sel(esa_energy_step=1).values)
        # ESA energy step 2 should have valid median
        assert median_per_esa.sel(esa_energy_step=2).values == 4.0

    def test_empty_datasets_handled(self):
        """Test that empty datasets result in empty DataArrays."""
        # Empty per_sweep_datasets
        per_sweep_datasets: dict[int, xr.Dataset] = {}

        median_per_esa, sigma_per_esa = _compute_median_and_sigma_per_esa(
            per_sweep_datasets
        )

        assert len(median_per_esa) == 0
        assert len(sigma_per_esa) == 0


class TestStatisticalFilter1:
    """Test suite for mark_statistical_filter_1() integration tests."""

    @pytest.fixture
    def goodtimes_for_filter1(self):
        """Create a goodtimes dataset for testing statistical filter 1."""
        # Create 18 METs (2 complete ESA sweeps)
        n_mets = 18
        met_values = np.arange(1000.0, 1000.0 + n_mets * 60, 60)

        gt = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((n_mets, 90), dtype=np.uint8), dims=["met", "spin_bin"]
                ),
                "esa_step": xr.DataArray(
                    np.tile(np.arange(1, 10), 2).astype(np.uint8), dims=["met"]
                ),
            },
            coords={"met": met_values, "spin_bin": np.arange(90)},
            attrs={"sensor": "45sensor", "pointing": 1},
        )
        return gt

    def _create_l1b_de_dataset(
        self,
        n_packets: int = 18,
        events_per_packet: int = 10,
        base_met: float = 1000.0,
        coincidence_type: int = 12,
    ) -> xr.Dataset:
        """Create a mock L1B DE dataset."""
        n_events = n_packets * events_per_packet

        # ESA steps cycling 1-9 for each sweep
        esa_step = np.tile(np.arange(1, 10), n_packets // 9 + 1)[:n_packets].astype(
            np.uint8
        )
        # esa_energy_step same as esa_step for test purposes
        esa_energy_step = esa_step.copy()
        ccsds_met = np.arange(base_met, base_met + n_packets * 60, 60, dtype=np.float64)
        coincidence_types = np.full(n_events, coincidence_type, dtype=np.uint8)
        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )

        return xr.Dataset(
            data_vars={
                "coincidence_type": (["event_met"], coincidence_types),
                "ccsds_index": (["event_met"], ccsds_index),
                "ccsds_met": (["epoch"], ccsds_met),
                "esa_step": (["epoch"], esa_step),
                "esa_energy_step": (["epoch"], esa_energy_step),
            },
            coords={
                "event_met": np.arange(n_events),
                "epoch": np.arange(n_packets),
            },
        )

    def test_passes_normal_data(self, goodtimes_for_filter1):
        """Test that normal data with consistent counts passes the filter."""
        # Current pointing at index 2 must have METs matching goodtimes (1000.0-2020.0)
        l1b_de_datasets = [
            self._create_l1b_de_dataset(events_per_packet=10, base_met=0.0),
            self._create_l1b_de_dataset(events_per_packet=10, base_met=500.0),
            self._create_l1b_de_dataset(
                events_per_packet=10, base_met=1000.0
            ),  # Current
            self._create_l1b_de_dataset(events_per_packet=10, base_met=2500.0),
            self._create_l1b_de_dataset(events_per_packet=10, base_met=3500.0),
        ]

        # Add qualified masks directly to datasets
        for ds in l1b_de_datasets:
            ds["qualified_mask"] = xr.DataArray(
                np.isin(ds["coincidence_type"].values, [12]),
                dims=["event_met"],
            )

        mark_statistical_filter_1(
            goodtimes_for_filter1,
            l1b_de_datasets,
            current_index=2,
        )

        # All times should still be good
        assert np.all(goodtimes_for_filter1["cull_flags"].values == CullCode.GOOD)

    def test_fails_extreme_outlier(self, goodtimes_for_filter1):
        """Test that extreme outliers (>5-sigma) are marked as bad."""
        # Create datasets - current pointing at index 2 must have
        # METs matching goodtimes
        # goodtimes has METs from 1000.0 to ~2020.0 (18 METs, 60s apart)
        l1b_de_datasets = [
            self._create_l1b_de_dataset(events_per_packet=10, base_met=0.0),
            self._create_l1b_de_dataset(events_per_packet=10, base_met=500.0),
            self._create_l1b_de_dataset(
                events_per_packet=10, base_met=1000.0
            ),  # Current - matches goodtimes
            self._create_l1b_de_dataset(events_per_packet=10, base_met=2500.0),
            self._create_l1b_de_dataset(events_per_packet=10, base_met=3500.0),
        ]

        # Make current pointing have extreme counts for one interval
        current_ds = l1b_de_datasets[2]
        # Add many more events to first packet only
        extra_events = 100
        new_coincidence = np.concatenate(
            [
                current_ds["coincidence_type"].values,
                np.full(extra_events, 12, dtype=np.uint8),
            ]
        )
        new_ccsds_index = np.concatenate(
            [
                current_ds["ccsds_index"].values,
                np.zeros(extra_events, dtype=np.uint16),  # All to first packet
            ]
        )

        l1b_de_datasets[2] = xr.Dataset(
            data_vars={
                "coincidence_type": (["event_met"], new_coincidence),
                "ccsds_index": (["event_met"], new_ccsds_index),
                "ccsds_met": current_ds["ccsds_met"],
                "esa_step": current_ds["esa_step"],
                "esa_energy_step": current_ds["esa_energy_step"],
            },
            coords={
                "event_met": np.arange(len(new_coincidence)),
                "epoch": current_ds["epoch"],
            },
        )

        # Add qualified masks directly to datasets
        for ds in l1b_de_datasets:
            ds["qualified_mask"] = xr.DataArray(
                np.isin(ds["coincidence_type"].values, [12]),
                dims=["event_met"],
            )

        mark_statistical_filter_1(
            goodtimes_for_filter1,
            l1b_de_datasets,
            current_index=2,
        )

        # At least the first MET should be marked bad (extreme outlier)
        assert np.any(
            goodtimes_for_filter1["cull_flags"].values == CullCode.STAT_FILTER_1
        )

    def test_insufficient_pointings(self, goodtimes_for_filter1):
        """Test that fewer than min_pointings raises ValueError."""
        l1b_de_datasets = [
            self._create_l1b_de_dataset(),
            self._create_l1b_de_dataset(),
            self._create_l1b_de_dataset(),
        ]

        # Add qualified masks directly to datasets
        for ds in l1b_de_datasets:
            ds["qualified_mask"] = xr.DataArray(
                np.isin(ds["coincidence_type"].values, [12]),
                dims=["event_met"],
            )

        with pytest.raises(ValueError, match="At least 4 valid Pointings required"):
            mark_statistical_filter_1(
                goodtimes_for_filter1,
                l1b_de_datasets,
                current_index=1,
            )

    def test_current_index_out_of_range(self, goodtimes_for_filter1):
        """Test that current_index out of range raises ValueError."""
        l1b_de_datasets = [self._create_l1b_de_dataset() for _ in range(5)]

        # Add qualified masks directly to datasets
        for ds in l1b_de_datasets:
            ds["qualified_mask"] = xr.DataArray(
                np.isin(ds["coincidence_type"].values, [12]),
                dims=["event_met"],
            )

        with pytest.raises(ValueError, match="current_index.*out of range"):
            mark_statistical_filter_1(
                goodtimes_for_filter1,
                l1b_de_datasets,
                current_index=10,
            )


class TestFindEventClusters:
    """Test suite for _find_event_clusters() helper function."""

    def test_empty_array(self):
        """Test with empty input."""
        result = _find_event_clusters(np.array([]), min_events=3, max_time_delta=100)
        assert result == []

    def test_too_few_events(self):
        """Test with fewer events than min_events."""
        de_tags = np.array([10, 50])
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=100)
        assert result == []

    def test_events_too_spread(self):
        """Test with events spread beyond max_time_delta."""
        de_tags = np.array([0, 1000, 2000, 3000, 4000, 5000])
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=100)
        assert result == []

    def test_single_cluster(self):
        """Test detection of a single cluster."""
        de_tags = np.array([100, 110, 120, 130, 140, 150])
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=100)
        assert len(result) == 1
        assert result[0] == (0, 5)

    def test_multiple_clusters(self):
        """Test detection of multiple separate clusters."""
        de_tags = np.array([100, 110, 120, 1000, 1010, 1020])
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=50)
        assert len(result) == 2
        assert result[0] == (0, 2)
        assert result[1] == (3, 5)

    def test_cluster_merge(self):
        """Test that overlapping clusters are merged."""
        # Events that form overlapping windows
        de_tags = np.array([0, 10, 20, 30, 40, 50])
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=30)
        # All events should merge into one cluster
        assert len(result) == 1
        assert result[0] == (0, 5)

    def test_exact_threshold(self):
        """Test cluster detection at exact min_events threshold."""
        de_tags = np.array([0, 10, 20])  # Exactly 3 events within 20 ticks
        result = _find_event_clusters(de_tags, min_events=3, max_time_delta=20)
        assert len(result) == 1
        assert result[0] == (0, 2)


class TestComputeBinsForCluster:
    """Test suite for _compute_bins_for_cluster() helper function."""

    def test_basic_range(self):
        """Test basic bin range computation."""
        nominal_bins = np.array([40, 42, 44, 46])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=0, cluster_end=3, bin_padding=1
        )
        expected = np.arange(39, 48)  # 40-1 to 46+1
        np.testing.assert_array_equal(bins, expected)

    def test_wrapping_at_zero(self):
        """Test that bins wrap around at 0."""
        nominal_bins = np.array([0, 1, 2])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=0, cluster_end=2, bin_padding=2, n_bins=90
        )
        # Should wrap: -2, -1, 0, 1, 2, 3, 4 -> 88, 89, 0, 1, 2, 3, 4
        expected = np.array([88, 89, 0, 1, 2, 3, 4])
        np.testing.assert_array_equal(bins, expected)

    def test_wrapping_at_max(self):
        """Test that bins wrap around at n_bins."""
        nominal_bins = np.array([87, 88, 89])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=0, cluster_end=2, bin_padding=2, n_bins=90
        )
        # Should wrap: 85, 86, 87, 88, 89, 90, 91 -> 85, 86, 87, 88, 89, 0, 1
        expected = np.array([85, 86, 87, 88, 89, 0, 1])
        np.testing.assert_array_equal(bins, expected)

    def test_partial_cluster(self):
        """Test range computation for partial cluster."""
        nominal_bins = np.array([10, 20, 30, 40, 50])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=1, cluster_end=3, bin_padding=1
        )
        expected = np.arange(19, 42)  # 20-1 to 40+1
        np.testing.assert_array_equal(bins, expected)

    def test_no_wrapping_needed(self):
        """Test middle bins that don't need wrapping."""
        nominal_bins = np.array([44, 45, 46])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=0, cluster_end=2, bin_padding=1
        )
        expected = np.arange(43, 48)  # 44-1 to 46+1
        np.testing.assert_array_equal(bins, expected)

    def test_cluster_spanning_zero_boundary(self):
        """Test cluster that spans across the 0/89 boundary."""
        # Cluster bins span from 87 to 2 (wrapping around)
        nominal_bins = np.array([87, 88, 89, 0, 1, 2])
        bins = _compute_bins_for_cluster(
            nominal_bins, cluster_start=0, cluster_end=5, bin_padding=1, n_bins=90
        )
        # Should mark 86-89 and 0-3 (cluster 87-2 plus padding of 1)
        expected = np.array([86, 87, 88, 89, 0, 1, 2, 3])
        np.testing.assert_array_equal(bins, expected)


class TestStatisticalFilter2:
    """Test suite for mark_statistical_filter_2() function."""

    @pytest.fixture
    def goodtimes_for_filter2(self):
        """Create a goodtimes dataset for filter 2 testing."""
        n_mets = 10
        met_values = np.arange(1000.0, 1000.0 + n_mets * 120, 120)

        ds = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((n_mets, 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                ),
                "esa_step": xr.DataArray(
                    np.tile(np.arange(1, 11), 1)[:n_mets].astype(np.uint8),
                    dims=["met"],
                ),
            },
            coords={
                "met": met_values,
                "spin_bin": np.arange(90),
            },
            attrs={"sensor": "45sensor", "pointing": 1},
        )
        return ds

    def _create_l1b_de_for_filter2(
        self,
        n_packets: int = 10,
        events_per_packet: int = 20,
        base_met: float = 1000.0,
        esa_step: int = 1,
    ) -> xr.Dataset:
        """Create L1B DE dataset for filter 2 testing.

        Creates a dataset with proper epoch and event_met dimensions:
        - epoch: packet-level variables (ccsds_met, esa_step)
        - event_met: event-level variables (ccsds_index, event_met, etc.)
        """
        n_events = n_packets * events_per_packet

        # Spread events across packets
        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )

        # Packet-level METs (120 seconds apart)
        packet_mets = np.arange(base_met, base_met + n_packets * 120, 120)

        # Default: events spread out in time within each packet (no clusters)
        # Each packet spans ~120 seconds, events spread across that time
        event_met_values = np.zeros(n_events, dtype=np.float64)
        for i in range(n_packets):
            start_idx = i * events_per_packet
            end_idx = start_idx + events_per_packet
            # Events spread 0-100 seconds within each packet
            event_met_values[start_idx:end_idx] = packet_mets[i] + np.linspace(
                0, 100, events_per_packet
            )

        # All events are qualified type 12
        coincidence_type = np.full(n_events, 12, dtype=np.uint8)

        # Spread events across spin bins
        nominal_bin = np.tile(
            np.linspace(0, 89, events_per_packet).astype(np.uint8), n_packets
        )

        # All packets at same ESA step (single 8-spin set)
        packet_esa_steps = np.full(n_packets, esa_step, dtype=np.uint8)

        return xr.Dataset(
            {
                # Packet-level variables (epoch dimension)
                "ccsds_met": (["epoch"], packet_mets),
                "esa_step": (["epoch"], packet_esa_steps),
                # Event-level variables (event dimension)
                "ccsds_index": (["event_met"], ccsds_index),
                "coincidence_type": (["event_met"], coincidence_type),
                "nominal_bin": (["even_met"], nominal_bin),
            },
            coords={
                "epoch": np.arange(n_packets),
                "event_met": event_met_values,
            },
        )

    def test_no_qualified_events(self, goodtimes_for_filter2):
        """Test with no qualified events."""
        l1b_de = self._create_l1b_de_for_filter2()
        # Change all events to unqualified type
        l1b_de["coincidence_type"] = xr.DataArray(
            np.full(len(l1b_de["event_met"]), 4, dtype=np.uint8),
            dims=["event_met"],
        )

        # Add qualified mask directly to dataset - no events match type 12
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=10.0,
        )

        # No bins should be marked
        assert np.all(goodtimes_for_filter2["cull_flags"].values == 0)

    def test_no_clusters(self, goodtimes_for_filter2):
        """Test with qualified events but no clusters."""
        # Create l1b_de with different esa_steps per packet
        # This ensures events are in different 8-spin sets and don't get pooled
        n_packets = 10
        events_per_packet = 20
        n_events = n_packets * events_per_packet
        base_met = 1000.0

        ccsds_index = np.repeat(np.arange(n_packets), events_per_packet).astype(
            np.uint16
        )
        # Events spread out in time within each packet (no clusters)
        # Events at 0, 5, 10, ... 95 seconds - more than 0.2s apart
        packet_mets = np.arange(base_met, base_met + n_packets * 120, 120)
        event_met_values = np.zeros(n_events, dtype=np.float64)
        for i in range(n_packets):
            start_idx = i * events_per_packet
            end_idx = start_idx + events_per_packet
            event_met_values[start_idx:end_idx] = packet_mets[i] + np.linspace(
                0, 95, events_per_packet
            )
        coincidence_type = np.full(n_events, 12, dtype=np.uint8)
        nominal_bin = np.tile(
            np.linspace(0, 89, events_per_packet).astype(np.uint8), n_packets
        )
        # Each packet has a different esa_step (different 8-spin sets)
        packet_esa_steps = np.arange(1, n_packets + 1, dtype=np.uint8)

        l1b_de = xr.Dataset(
            {
                "ccsds_met": (["epoch"], packet_mets),
                "esa_step": (["epoch"], packet_esa_steps),
                "ccsds_index": (["event_met"], ccsds_index),
                "coincidence_type": (["event_met"], coincidence_type),
                "nominal_bin": (["event_met"], nominal_bin),
            },
            coords={
                "epoch": np.arange(n_packets),
                "event_met": event_met_values,
            },
        )

        # Add qualified mask directly to dataset
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=0.2,
        )

        # Events are spread out within each 8-spin set, no clusters should form
        assert np.all(goodtimes_for_filter2["cull_flags"].values == 0)

    def test_cluster_detected(self, goodtimes_for_filter2):
        """Test that a cluster is detected and bins are marked."""
        l1b_de = self._create_l1b_de_for_filter2(n_packets=1, events_per_packet=10)

        # Create a cluster: 6 events within 0.1 seconds, all at bins 40-45
        # Events at 0.01, 0.02, ..., 0.06 seconds (cluster) and
        # 10, 11, 12, 13 seconds (spread out)
        l1b_de["event_met"] = xr.DataArray(
            np.array(
                [
                    1000.01,
                    1000.02,
                    1000.03,
                    1000.04,
                    1000.05,
                    1000.06,
                    1010.0,
                    1011.0,
                    1012.0,
                    1013.0,
                ],
                dtype=np.float64,
            ),
            dims=["event_met"],
        )
        l1b_de["nominal_bin"] = xr.DataArray(
            np.array([40, 41, 42, 43, 44, 45, 10, 20, 30, 50], dtype=np.uint8),
            dims=["event_met"],
        )

        # Add qualified mask directly to dataset
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=0.1,
            bin_padding=1,
        )

        # Bins 39-46 should be marked for MET 1000.0 (first MET)
        cull_flags = goodtimes_for_filter2["cull_flags"].sel(met=1000.0).values
        assert np.all(cull_flags[39:47] == CullCode.STAT_FILTER_2)
        # Other bins should be unmarked
        assert np.all(cull_flags[:39] == 0)
        assert np.all(cull_flags[47:] == 0)

    def test_multiple_clusters_same_packet(self, goodtimes_for_filter2):
        """Test detection of multiple clusters in the same packet."""
        l1b_de = self._create_l1b_de_for_filter2(n_packets=1, events_per_packet=12)

        # Two clusters: one at 0.01-0.06s (bins 10-15), one at 10.0-10.05s (bins 70-75)
        l1b_de["event_met"] = xr.DataArray(
            np.array(
                [
                    1000.01,
                    1000.02,
                    1000.03,
                    1000.04,
                    1000.05,
                    1000.06,
                    1010.00,
                    1010.01,
                    1010.02,
                    1010.03,
                    1010.04,
                    1010.05,
                ],
                dtype=np.float64,
            ),
            dims=["event_met"],
        )
        l1b_de["nominal_bin"] = xr.DataArray(
            np.array([10, 11, 12, 13, 14, 15, 70, 71, 72, 73, 74, 75], dtype=np.uint8),
            dims=["event_met"],
        )

        # Add qualified mask directly to dataset
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=0.1,
            bin_padding=1,
        )

        cull_flags = goodtimes_for_filter2["cull_flags"].sel(met=1000.0).values
        # First cluster: bins 9-16
        assert np.all(cull_flags[9:17] == CullCode.STAT_FILTER_2)
        # Second cluster: bins 69-76
        assert np.all(cull_flags[69:77] == CullCode.STAT_FILTER_2)
        # Middle bins should be unmarked
        assert np.all(cull_flags[17:69] == 0)

    def test_bin_padding_with_wrapping(self, goodtimes_for_filter2):
        """Test that bin padding wraps at array boundaries."""
        l1b_de = self._create_l1b_de_for_filter2(n_packets=1, events_per_packet=6)

        # Cluster at bins 0-2 with padding=2: bins -2 to 4 wrap to
        # [88, 89, 0, 1, 2, 3, 4]
        # 6 events clustered within 0.1 seconds
        l1b_de["event_met"] = xr.DataArray(
            np.array(
                [1000.01, 1000.02, 1000.03, 1000.04, 1000.05, 1000.06], dtype=np.float64
            ),
            dims=["event_met"],
        )
        l1b_de["nominal_bin"] = xr.DataArray(
            np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8),
            dims=["event_met"],
        )

        # Add qualified mask directly to dataset
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=0.1,
            bin_padding=2,
        )

        cull_flags = goodtimes_for_filter2["cull_flags"].sel(met=1000.0).values
        # Bins 0-4 should be marked (cluster at 0-2 + padding of 2)
        assert np.all(cull_flags[0:5] == CullCode.STAT_FILTER_2)
        # Bins 88-89 should also be marked due to wrapping (bin -2 and -1)
        assert np.all(cull_flags[88:90] == CullCode.STAT_FILTER_2)
        # Middle bins should be unmarked
        assert np.all(cull_flags[5:88] == 0)
        # Check that no cull_flags were set on any other METs
        other_mets = goodtimes_for_filter2["cull_flags"].drop_sel(met=1000.0)
        assert np.all(other_mets.values == 0)

    def test_custom_parameters(self, goodtimes_for_filter2):
        """Test with custom min_events and max_time_delta."""
        l1b_de = self._create_l1b_de_for_filter2(n_packets=1, events_per_packet=10)

        # Create events: 4 close together (not enough for default min_events=6)
        # First 4 events at 0.01-0.04s (cluster), rest spread out
        l1b_de["event_met"] = xr.DataArray(
            np.array(
                [
                    1000.01,
                    1000.02,
                    1000.03,
                    1000.04,
                    1010.0,
                    1011.0,
                    1012.0,
                    1013.0,
                    1014.0,
                    1015.0,
                ],
                dtype=np.float64,
            ),
            dims=["event_met"],
        )
        l1b_de["nominal_bin"] = xr.DataArray(
            np.array([40, 41, 42, 43, 10, 20, 30, 50, 60, 70], dtype=np.uint8),
            dims=["event_met"],
        )

        # Add qualified mask directly to dataset
        l1b_de["qualified_mask"] = xr.DataArray(
            np.isin(l1b_de["coincidence_type"].values, [12]),
            dims=["event_met"],
        )

        # With min_events=4, should detect cluster
        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=4,
            max_time_delta=0.1,
            bin_padding=1,
        )

        cull_flags = goodtimes_for_filter2["cull_flags"].sel(met=1000.0).values
        assert np.all(cull_flags[39:45] == CullCode.STAT_FILTER_2)

    def test_only_qualified_events_contribute_to_clusters(self, goodtimes_for_filter2):
        """Test that only qualified events are used for cluster detection.

        This test verifies the filtering behavior by creating a scenario where:
        - Unqualified events (type 4) form a cluster if incorrectly included
        - Qualified events (type 12) are spread out and don't form a cluster
        - No cluster should be detected because only qualified events should be used
        """
        n_events = 12
        # Create base dataset structure with correct event_met dimension
        event_met_values = np.array(
            [
                # 6 unqualified events clustered together
                1000.01,
                1000.02,
                1000.03,
                1000.04,
                1000.05,
                1000.06,
                # 6 qualified events spread out (no cluster)
                1010.0,
                1020.0,
                1030.0,
                1040.0,
                1050.0,
                1060.0,
            ],
            dtype=np.float64,
        )

        # First 6 events are unqualified (type 4), last 6 are qualified (type 12)
        coincidence_type = np.array(
            [4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12], dtype=np.uint8
        )

        # All events at similar bins (so cluster would be detected if all included)
        nominal_bin = np.array(
            [40, 41, 42, 43, 44, 45, 40, 41, 42, 43, 44, 45], dtype=np.uint8
        )

        # Create ccsds_index - all events in same packet
        ccsds_index = np.zeros(n_events, dtype=np.uint16)

        l1b_de = xr.Dataset(
            {
                "ccsds_index": (["event_met"], ccsds_index),
                "coincidence_type": (["event_met"], coincidence_type),
                "nominal_bin": (["event_met"], nominal_bin),
                "ccsds_met": (["epoch"], np.array([1000.0])),
                "esa_step": (["epoch"], np.array([1], dtype=np.uint8)),
            },
            coords={
                "event_met": event_met_values,
                "epoch": np.arange(1),
            },
        )

        # Add qualified mask directly to dataset - only type 12 events are qualified
        qualified_mask = np.isin(l1b_de["coincidence_type"].values, [12])
        l1b_de["qualified_mask"] = xr.DataArray(qualified_mask, dims=["event_met"])

        # Verify our test setup: 6 unqualified, 6 qualified
        assert np.sum(~qualified_mask) == 6  # 6 unqualified
        assert np.sum(qualified_mask) == 6  # 6 qualified

        mark_statistical_filter_2(
            goodtimes_for_filter2,
            l1b_de,
            min_events=6,
            max_time_delta=0.1,
            bin_padding=1,
        )

        # No bins should be marked because:
        # - The 6 unqualified events form a cluster but should be filtered out
        # - The 6 qualified events are spread out and don't form a cluster
        cull_flags = goodtimes_for_filter2["cull_flags"].sel(met=1000.0).values
        assert np.all(cull_flags == 0), (
            "Bins were incorrectly marked - unqualified events may have been "
            "included in cluster detection"
        )


class TestFindCurrentPointingIndex:
    """Test suite for _find_current_pointing_index helper function."""

    def test_finds_current_index(self):
        """Test that current index is found correctly."""
        ds1 = MagicMock()
        ds1.attrs = {"Repointing": "repoint00001"}
        ds2 = MagicMock()
        ds2.attrs = {"Repointing": "repoint00002"}
        ds3 = MagicMock()
        ds3.attrs = {"Repointing": "repoint00003"}

        datasets = [ds1, ds2, ds3]
        current_index = _find_current_pointing_index(datasets, "repoint00002")

        assert current_index == 1

    def test_finds_first_matching_repointing(self):
        """Test that the first matching repointing is returned."""
        ds1 = MagicMock()
        ds1.attrs = {"Repointing": "repoint00005"}
        ds2 = MagicMock()
        ds2.attrs = {"Repointing": "repoint00005"}

        datasets = [ds1, ds2]
        current_index = _find_current_pointing_index(datasets, "repoint00005")

        assert current_index == 0

    def test_raises_when_repointing_not_found(self):
        """Test that ValueError is raised when repointing not found."""
        ds1 = MagicMock()
        ds1.attrs = {"Repointing": "repoint00001"}
        ds2 = MagicMock()
        ds2.attrs = {"Repointing": "repoint00002"}

        datasets = [ds1, ds2]
        with pytest.raises(ValueError, match="Could not find current repointing"):
            _find_current_pointing_index(datasets, "repoint00099")


class TestApplyGoodtimesFilters:
    """Test suite for _apply_goodtimes_filters helper function."""

    def test_loads_cal_config(self, tmp_path):
        """Test that cal config is loaded."""
        mock_goodtimes = MagicMock()
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "good_bins": 100,
            "total_bins": 100,
        }
        mock_l1b_de = MagicMock()
        mock_hk = MagicMock()
        mock_cal = {"coincidence_type_values": [{12}]}

        cal_path = tmp_path / "cal.csv"

        with (
            patch(
                "imap_processing.hi.utils.CalibrationProductConfig.from_csv"
            ) as mock_cal_load,
            patch("imap_processing.hi.hi_goodtimes.mark_incomplete_spin_sets"),
            patch("imap_processing.hi.hi_goodtimes.mark_drf_times"),
            patch("imap_processing.hi.hi_goodtimes.mark_overflow_packets"),
            patch("imap_processing.hi.hi_goodtimes.mark_bad_tdc_cal"),
            patch("imap_processing.hi.hi_goodtimes.mark_statistical_filter_0"),
            patch("imap_processing.hi.hi_goodtimes.mark_statistical_filter_1"),
            patch("imap_processing.hi.hi_goodtimes.mark_statistical_filter_2"),
        ):
            mock_cal_load.return_value = mock_cal

            _apply_goodtimes_filters(
                mock_goodtimes,
                [mock_l1b_de],
                current_index=0,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=cal_path,
            )

            mock_cal_load.assert_called_once_with(cal_path)

    def test_calls_all_filters(self, tmp_path):
        """Test that all 7 filters are called."""
        mock_goodtimes = MagicMock()
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "good_bins": 100,
            "total_bins": 100,
        }
        mock_l1b_de = MagicMock()
        mock_hk = MagicMock()
        mock_cal = {"coincidence_type_values": [{12}]}

        with (
            patch(
                "imap_processing.hi.utils.CalibrationProductConfig.from_csv",
                return_value=mock_cal,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes.mark_incomplete_spin_sets"
            ) as mock_f1,
            patch("imap_processing.hi.hi_goodtimes.mark_drf_times") as mock_f2,
            patch("imap_processing.hi.hi_goodtimes.mark_bad_tdc_cal") as mock_f3,
            patch("imap_processing.hi.hi_goodtimes.mark_overflow_packets") as mock_f4,
            patch(
                "imap_processing.hi.hi_goodtimes.mark_statistical_filter_0"
            ) as mock_f5,
            patch(
                "imap_processing.hi.hi_goodtimes.mark_statistical_filter_1"
            ) as mock_f6,
            patch(
                "imap_processing.hi.hi_goodtimes.mark_statistical_filter_2"
            ) as mock_f7,
        ):
            _apply_goodtimes_filters(
                mock_goodtimes,
                [mock_l1b_de],
                current_index=0,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=tmp_path / "cal.csv",
            )

            mock_f1.assert_called_once()
            mock_f2.assert_called_once()
            mock_f3.assert_called_once()
            mock_f4.assert_called_once()
            mock_f5.assert_called_once()
            mock_f6.assert_called_once()
            mock_f7.assert_called_once()

    def test_raises_statistical_filter_0_errors(self, tmp_path):
        """Test that ValueError from statistical filter 0 is raised."""
        mock_goodtimes = MagicMock()
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "good_bins": 100,
            "total_bins": 100,
        }
        mock_l1b_de = MagicMock()
        mock_hk = MagicMock()
        mock_cal = {"coincidence_type_values": [{12}]}

        with (
            patch(
                "imap_processing.hi.utils.CalibrationProductConfig.from_csv",
                return_value=mock_cal,
            ),
            patch("imap_processing.hi.hi_goodtimes.mark_incomplete_spin_sets"),
            patch("imap_processing.hi.hi_goodtimes.mark_drf_times"),
            patch("imap_processing.hi.hi_goodtimes.mark_bad_tdc_cal"),
            patch("imap_processing.hi.hi_goodtimes.mark_overflow_packets"),
            patch(
                "imap_processing.hi.hi_goodtimes.mark_statistical_filter_0",
                side_effect=ValueError("filter 0 error"),
            ),
        ):
            with pytest.raises(ValueError, match="filter 0 error"):
                _apply_goodtimes_filters(
                    mock_goodtimes,
                    [mock_l1b_de],
                    current_index=0,
                    l1b_hk=mock_hk,
                    l1a_diagfee=MagicMock(),
                    cal_product_config_path=tmp_path / "cal.csv",
                )

    def test_raises_statistical_filter_1_errors(self, tmp_path):
        """Test that ValueError from statistical filter 1 is raised."""
        mock_goodtimes = MagicMock()
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "good_bins": 100,
            "total_bins": 100,
        }
        mock_l1b_de = MagicMock()
        mock_hk = MagicMock()
        mock_cal = {"coincidence_type_values": [{12}]}

        with (
            patch(
                "imap_processing.hi.utils.CalibrationProductConfig.from_csv",
                return_value=mock_cal,
            ),
            patch("imap_processing.hi.hi_goodtimes.mark_incomplete_spin_sets"),
            patch("imap_processing.hi.hi_goodtimes.mark_drf_times"),
            patch("imap_processing.hi.hi_goodtimes.mark_bad_tdc_cal"),
            patch("imap_processing.hi.hi_goodtimes.mark_overflow_packets"),
            patch("imap_processing.hi.hi_goodtimes.mark_statistical_filter_0"),
            patch(
                "imap_processing.hi.hi_goodtimes.mark_statistical_filter_1",
                side_effect=ValueError("filter 1 error"),
            ),
        ):
            with pytest.raises(ValueError, match="filter 1 error"):
                _apply_goodtimes_filters(
                    mock_goodtimes,
                    [mock_l1b_de],
                    current_index=0,
                    l1b_hk=mock_hk,
                    l1a_diagfee=MagicMock(),
                    cal_product_config_path=tmp_path / "cal.csv",
                )


class TestHiGoodtimes:
    """Test suite for hi_goodtimes top-level function."""

    def test_raises_value_error_when_repoint_not_complete(self, tmp_path):
        """Test that ValueError is raised when repoint+3 has not occurred."""
        mock_repoint_df = pd.DataFrame(
            {
                "repoint_id": [1, 2, 3],
            }
        )
        mock_de = MagicMock()
        mock_hk = MagicMock()

        with patch(
            "imap_processing.hi.hi_goodtimes.get_repoint_data"
        ) as mock_get_repoint:
            mock_get_repoint.return_value = mock_repoint_df
            with pytest.raises(
                ValueError, match="Goodtimes cannot yet be processed for repoint00001"
            ):
                _ = hi_goodtimes(
                    current_repointing="repoint00001",
                    l1b_de_datasets=[mock_de],
                    l1b_hk=mock_hk,
                    l1a_diagfee=MagicMock(),
                    cal_product_config_path=tmp_path / "cal.csv",
                )

    def test_calls_find_current_index_when_repoint_complete(self, tmp_path):
        """Test that _find_current_pointing_index is called when repoint passes."""
        mock_repoint_df = pd.DataFrame({"repoint_id": list(range(1, 10))})
        mock_goodtimes = MagicMock()
        mock_goodtimes.attrs = {"sensor": "45sensor"}
        mock_goodtimes.__getitem__ = MagicMock()
        # Mock the goodtimes accessor methods
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "total_bins": 100,
            "good_bins": 80,
            "culled_bins": 20,
            "fraction_good": 0.8,
            "cull_code_counts": {},
        }
        mock_goodtimes.goodtimes.finalize_dataset.return_value = MagicMock()
        mock_datasets = [MagicMock() for _ in range(7)]
        mock_hk = MagicMock()

        with (
            patch(
                "imap_processing.hi.hi_goodtimes.get_repoint_data",
                return_value=mock_repoint_df,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes._find_current_pointing_index",
                return_value=3,
            ) as mock_find,
            patch(
                "imap_processing.hi.hi_goodtimes.create_goodtimes_dataset",
                return_value=mock_goodtimes,
            ),
            patch("imap_processing.hi.hi_goodtimes._apply_goodtimes_filters"),
        ):
            hi_goodtimes(
                current_repointing="repoint00004",
                l1b_de_datasets=mock_datasets,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=tmp_path / "cal.csv",
            )

            mock_find.assert_called_once_with(mock_datasets, "repoint00004")

    def test_marks_all_bad_when_incomplete_de_set(self, tmp_path):
        """Test that cull_flags are set when DE set is incomplete."""
        mock_repoint_df = pd.DataFrame({"repoint_id": list(range(1, 10))})
        mock_goodtimes = MagicMock()
        mock_goodtimes.attrs = {"sensor": "45sensor"}
        mock_cull_flags = MagicMock()
        mock_goodtimes.__getitem__ = MagicMock(return_value=mock_cull_flags)
        # Mock the goodtimes accessor methods
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "total_bins": 100,
            "good_bins": 0,
            "culled_bins": 100,
            "fraction_good": 0.0,
            "cull_code_counts": {1: 100},
        }
        mock_goodtimes.goodtimes.finalize_dataset.return_value = MagicMock()
        mock_datasets = [MagicMock() for _ in range(3)]  # Less than 7
        mock_hk = MagicMock()

        with (
            patch(
                "imap_processing.hi.hi_goodtimes.get_repoint_data",
                return_value=mock_repoint_df,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes._find_current_pointing_index",
                return_value=0,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes.create_goodtimes_dataset",
                return_value=mock_goodtimes,
            ),
        ):
            hi_goodtimes(
                current_repointing="repoint00001",
                l1b_de_datasets=mock_datasets,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=tmp_path / "cal.csv",
            )

            # Verify cull_flags were set to LOOSE (all bad)
            mock_goodtimes.__getitem__.assert_called_with("cull_flags")

    def test_calls_apply_filters_when_full_de_set(self, tmp_path):
        """Test that _apply_goodtimes_filters is called with 7 DE datasets."""
        mock_repoint_df = pd.DataFrame({"repoint_id": list(range(1, 10))})
        mock_goodtimes = MagicMock()
        mock_goodtimes.attrs = {"sensor": "45sensor"}
        # Mock the goodtimes accessor methods
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "total_bins": 100,
            "good_bins": 80,
            "culled_bins": 20,
            "fraction_good": 0.8,
            "cull_code_counts": {},
        }
        mock_goodtimes.goodtimes.finalize_dataset.return_value = MagicMock()
        mock_datasets = [MagicMock() for _ in range(7)]
        mock_hk = MagicMock()

        with (
            patch(
                "imap_processing.hi.hi_goodtimes.get_repoint_data",
                return_value=mock_repoint_df,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes._find_current_pointing_index",
                return_value=3,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes.create_goodtimes_dataset",
                return_value=mock_goodtimes,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes._apply_goodtimes_filters"
            ) as mock_apply,
        ):
            hi_goodtimes(
                current_repointing="repoint00004",
                l1b_de_datasets=mock_datasets,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=tmp_path / "cal.csv",
            )

            mock_apply.assert_called_once()

    def test_returns_datasets(self, tmp_path):
        """Test that hi_goodtimes returns list of datasets."""
        mock_repoint_df = pd.DataFrame({"repoint_id": list(range(1, 10))})
        mock_goodtimes = MagicMock()
        mock_goodtimes.attrs = {"sensor": "45sensor"}
        # Mock the goodtimes accessor methods
        mock_goodtimes.goodtimes.get_cull_statistics.return_value = {
            "total_bins": 100,
            "good_bins": 80,
            "culled_bins": 20,
            "fraction_good": 0.8,
            "cull_code_counts": {},
        }
        mock_finalized = MagicMock()
        mock_goodtimes.goodtimes.finalize_dataset.return_value = mock_finalized
        mock_datasets = [MagicMock() for _ in range(7)]
        mock_hk = MagicMock()

        with (
            patch(
                "imap_processing.hi.hi_goodtimes.get_repoint_data",
                return_value=mock_repoint_df,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes._find_current_pointing_index",
                return_value=3,
            ),
            patch(
                "imap_processing.hi.hi_goodtimes.create_goodtimes_dataset",
                return_value=mock_goodtimes,
            ),
            patch("imap_processing.hi.hi_goodtimes._apply_goodtimes_filters"),
        ):
            result = hi_goodtimes(
                current_repointing="repoint00004",
                l1b_de_datasets=mock_datasets,
                l1b_hk=mock_hk,
                l1a_diagfee=MagicMock(),
                cal_product_config_path=tmp_path / "cal.csv",
            )

            # Should return finalized dataset, not original
            assert result == [mock_finalized]
