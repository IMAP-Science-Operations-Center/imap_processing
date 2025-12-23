"""Test coverage for imap_processing.hi.hi_goodtimes.py"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.hi.hi_goodtimes import (
    INTERVAL_DTYPE,
    CullCode,
    create_goodtimes_dataset,
)


@pytest.fixture
def mock_l1a_de():
    """Create a mock L1A Direct Event dataset for testing."""
    # Create 10 unique MET times, each appearing twice (paired)
    # Plus 2 unpaired MET times
    n_paired = 10

    # Paired METs: each appears twice
    paired_mets = np.arange(1000.0, 1000.0 + n_paired * 10, 10)
    met_seconds = np.repeat(paired_mets.astype(int), 2)
    met_subseconds = np.zeros(len(met_seconds))

    # Add unpaired METs
    unpaired_mets = np.array([2000.0, 3000.0])
    met_seconds = np.concatenate([met_seconds, unpaired_mets.astype(int)])
    met_subseconds = np.concatenate([met_subseconds, np.zeros(len(unpaired_mets))])

    # ESA step cycles through values
    esa_step = np.tile(np.arange(1, 11), len(met_seconds) // 10 + 1)[: len(met_seconds)]

    ds = xr.Dataset(
        {
            "meta_seconds": (["epoch"], met_seconds),
            "meta_subseconds": (["epoch"], met_subseconds),
            "esa_step": (["epoch"], esa_step.astype(np.uint8)),
        },
        attrs={
            "Logical_source": "imap_hi_l1a_45sensor-de",
            "Repointing": "repoint00042",
        },
    )
    return ds


@pytest.fixture
def goodtimes_instance(mock_l1a_de):
    """Create a goodtimes dataset for testing."""
    return create_goodtimes_dataset(mock_l1a_de)


class TestCullCode:
    """Test suite for CullCode IntEnum."""

    def test_cull_code_values(self):
        """Test CullCode enum values."""
        assert CullCode.GOOD == 0
        assert CullCode.LOOSE == 1

    def test_cull_code_is_int(self):
        """Test that CullCode values are integers."""
        assert isinstance(CullCode.GOOD, int)
        assert isinstance(CullCode.LOOSE, int)


class TestGoodtimesFromL1aDe:
    """Test suite for Goodtimes.from_l1a_de() classmethod."""

    def test_from_l1a_de_basic(self, mock_l1a_de):
        """Test basic creation from L1A DE data."""
        gt = create_goodtimes_dataset(mock_l1a_de)

        assert isinstance(gt, xr.Dataset)

    def test_from_l1a_de_filters_unpaired_mets(self, mock_l1a_de):
        """Test that unpaired METs are filtered out."""
        gt = create_goodtimes_dataset(mock_l1a_de)

        # Should have 10 paired METs (20 total entries -> 10 unique paired)
        assert len(gt.coords["met"]) == 10

    def test_from_l1a_de_dimensions(self, goodtimes_instance):
        """Test that dimensions are correct."""
        assert "met" in goodtimes_instance.dims
        assert "spin_bin" in goodtimes_instance.dims
        assert goodtimes_instance.dims["spin_bin"] == 90

    def test_from_l1a_de_coordinates(self, goodtimes_instance):
        """Test that coordinates are set correctly."""
        assert "met" in goodtimes_instance.coords
        assert "spin_bin" in goodtimes_instance.coords

        # spin_bin should be 0-89
        np.testing.assert_array_equal(
            goodtimes_instance.coords["spin_bin"].values, np.arange(90)
        )

    def test_from_l1a_de_data_variables(self, goodtimes_instance):
        """Test that data variables are created."""
        assert "cull_flags" in goodtimes_instance.data_vars
        assert "esa_step" in goodtimes_instance.data_vars

    def test_from_l1a_de_cull_flags_initialized_to_zero(self, goodtimes_instance):
        """Test that cull_flags are initialized to 0 (good)."""
        assert np.all(goodtimes_instance["cull_flags"].values == 0)

    def test_from_l1a_de_cull_flags_shape(self, goodtimes_instance):
        """Test cull_flags array shape."""
        n_met = len(goodtimes_instance.coords["met"])
        assert goodtimes_instance["cull_flags"].shape == (n_met, 90)

    def test_from_l1a_de_esa_step_preserved(self, mock_l1a_de, goodtimes_instance):
        """Test that ESA step values are preserved for paired METs."""
        # Get first occurrence of each paired MET
        met_all = mock_l1a_de["meta_seconds"].values.astype(float)
        unique_mets, first_indices, counts = np.unique(
            met_all, return_index=True, return_counts=True
        )
        paired_mask = counts == 2
        expected_esa_steps = mock_l1a_de["esa_step"].values[first_indices[paired_mask]]

        np.testing.assert_array_equal(
            goodtimes_instance["esa_step"].values, expected_esa_steps
        )

    def test_from_l1a_de_attributes(self, goodtimes_instance):
        """Test that attributes are set correctly."""
        assert goodtimes_instance.attrs["sensor"] == "Hi45"
        assert goodtimes_instance.attrs["pointing"] == 42


class TestRemoveTimes:
    """Test suite for Goodtimes.remove_times() method."""

    def test_remove_times_single_met_all_bins(self, goodtimes_instance):
        """Test flagging a single MET with all bins."""
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=None, cull=CullCode.LOOSE
        )

        # Check that all bins for the first MET are flagged
        assert np.all(goodtimes_instance["cull_flags"].values[0, :] == CullCode.LOOSE)

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[1:, :] == CullCode.GOOD)

    def test_remove_times_single_met_specific_bins(self, goodtimes_instance):
        """Test flagging specific bins for a single MET."""
        met_val = goodtimes_instance.coords["met"].values[0]
        bins_to_flag = np.array([0, 1, 2, 10])
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=bins_to_flag, cull=CullCode.LOOSE
        )

        # Check that specified bins are flagged
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, bins_to_flag] == CullCode.LOOSE
        )

        # Check that other bins are still good
        other_bins = np.setdiff1d(np.arange(90), bins_to_flag)
        assert np.all(
            goodtimes_instance["cull_flags"].values[0, other_bins] == CullCode.GOOD
        )

    def test_remove_times_multiple_mets(self, goodtimes_instance):
        """Test flagging multiple METs."""
        met_vals = goodtimes_instance.coords["met"].values[:3]
        goodtimes_instance.goodtimes.remove_times(
            met=met_vals, bins=None, cull=CullCode.LOOSE
        )

        # Check that first 3 METs are flagged
        assert np.all(goodtimes_instance["cull_flags"].values[:3, :] == CullCode.LOOSE)

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[3:, :] == CullCode.GOOD)

    def test_remove_times_time_range(self, goodtimes_instance):
        """Test flagging a time range."""
        met_vals = goodtimes_instance.coords["met"].values
        met_start = met_vals[2]
        met_end = met_vals[5]

        goodtimes_instance.goodtimes.remove_times(
            met=(met_start, met_end), bins=None, cull=CullCode.LOOSE
        )

        # Check that METs 2-5 are flagged
        assert np.all(goodtimes_instance["cull_flags"].values[2:6, :] == CullCode.LOOSE)

        # Check that other METs are still good
        assert np.all(goodtimes_instance["cull_flags"].values[:2, :] == CullCode.GOOD)
        assert np.all(goodtimes_instance["cull_flags"].values[6:, :] == CullCode.GOOD)

    def test_remove_times_invalid_cull_code_zero(self, goodtimes_instance):
        """Test that cull code 0 raises ValueError."""
        met_val = goodtimes_instance.coords["met"].values[0]
        with pytest.raises(ValueError, match="Cull code must be non-zero"):
            goodtimes_instance.goodtimes.remove_times(met=met_val, cull=0)

    def test_remove_times_invalid_bin_indices(self, goodtimes_instance):
        """Test that invalid bin indices raise ValueError."""
        met_val = goodtimes_instance.coords["met"].values[0]

        # Test bin < 0
        with pytest.raises(ValueError, match="Spin bins must be in range"):
            goodtimes_instance.goodtimes.remove_times(
                met=met_val, bins=np.array([-1, 0])
            )

        # Test bin >= 90
        with pytest.raises(ValueError, match="Spin bins must be in range"):
            goodtimes_instance.goodtimes.remove_times(
                met=met_val, bins=np.array([89, 90])
            )

    def test_remove_times_met_out_of_range(self, goodtimes_instance):
        """Test that MET outside valid range raises ValueError."""
        met_vals = goodtimes_instance.coords["met"].values
        met_out_of_range = met_vals[-1] + 1000

        with pytest.raises(ValueError, match="MET value\\(s\\) "):
            goodtimes_instance.goodtimes.remove_times(met=met_out_of_range)

    def test_remove_times_overwrites_existing_cull(self, goodtimes_instance):
        """Test that new cull code overwrites existing one."""
        met_val = goodtimes_instance.coords["met"].values[0]

        # Flag with LOOSE
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=None, cull=CullCode.LOOSE
        )
        assert np.all(goodtimes_instance["cull_flags"].values[0, :] == CullCode.LOOSE)

        # Overwrite with a different cull code
        goodtimes_instance.goodtimes.remove_times(met=met_val, bins=None, cull=2)
        assert np.all(goodtimes_instance["cull_flags"].values[0, :] == 2)


class TestGetGoodIntervals:
    """Test suite for Goodtimes.get_good_intervals() method."""

    def test_get_good_intervals_all_good(self, goodtimes_instance):
        """Test getting intervals when all times are good."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Should have one interval per MET
        n_met = len(goodtimes_instance.coords["met"])
        assert len(intervals) == n_met

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
        assert "n_good_bins" in intervals.dtype.names
        assert "esa_step" in intervals.dtype.names

    def test_get_good_intervals_all_good_values(self, goodtimes_instance):
        """Test interval values when all bins are good."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # When all bins are good, should have bins 0-89
        for interval in intervals:
            assert interval["spin_bin_low"] == 0
            assert interval["spin_bin_high"] == 89
            assert interval["n_good_bins"] == 90
            assert interval["met_start"] == interval["met_end"]

    def test_get_good_intervals_with_culled_bins(self, goodtimes_instance):
        """Test intervals when some bins are culled."""
        # Flag bins 0-20 for first MET
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=np.arange(21), cull=CullCode.LOOSE
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # First interval should only have bins 21-89
        assert intervals[0]["spin_bin_low"] == 21
        assert intervals[0]["spin_bin_high"] == 89
        assert intervals[0]["n_good_bins"] == 69

    def test_get_good_intervals_with_gaps(self, goodtimes_instance):
        """Test intervals when good bins have gaps (wraparound)."""
        # Flag bins 20-70 for first MET, leaving bins 0-19 and 71-89 as good
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=np.arange(20, 71), cull=CullCode.LOOSE
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Should create 2 intervals for the first MET (bins split by gap)
        # Plus 9 more intervals for the remaining METs
        assert len(intervals) == 11

        # First two intervals should be for the same MET
        assert intervals[0]["met_start"] == intervals[1]["met_start"]

        # Check the two segments
        assert intervals[0]["spin_bin_low"] == 0
        assert intervals[0]["spin_bin_high"] == 19
        assert intervals[1]["spin_bin_low"] == 71
        assert intervals[1]["spin_bin_high"] == 89

    def test_get_good_intervals_all_bins_culled(self, goodtimes_instance):
        """Test intervals when all bins are culled for a MET."""
        # Flag all bins for first MET
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=None, cull=CullCode.LOOSE
        )

        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Should have 9 intervals (one per good MET, excluding the first)
        assert len(intervals) == 9

        # First interval should be for the second MET
        assert intervals[0]["met_start"] == goodtimes_instance.coords["met"].values[1]

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
            attrs={"sensor": "Hi45", "pointing": 0},
        )

        intervals = gt.goodtimes.get_good_intervals()
        assert len(intervals) == 0

    def test_get_good_intervals_esa_step_included(self, goodtimes_instance):
        """Test that ESA step is included in intervals."""
        intervals = goodtimes_instance.goodtimes.get_good_intervals()

        # Check that each interval has an ESA step
        for i, interval in enumerate(intervals):
            expected_esa_step = goodtimes_instance["esa_step"].values[i]
            assert interval["esa_step"] == expected_esa_step


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
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=None, cull=CullCode.LOOSE
        )

        stats = goodtimes_instance.goodtimes.get_cull_statistics()

        total_bins = len(goodtimes_instance.coords["met"]) * 90
        assert stats["total_bins"] == total_bins
        assert stats["good_bins"] == total_bins - 90
        assert stats["culled_bins"] == 90
        assert stats["fraction_good"] == (total_bins - 90) / total_bins
        assert stats["cull_code_counts"][CullCode.LOOSE] == 90

    def test_get_cull_statistics_multiple_cull_codes(self, goodtimes_instance):
        """Test statistics with multiple cull codes."""
        met_vals = goodtimes_instance.coords["met"].values

        # Flag first MET with LOOSE
        goodtimes_instance.goodtimes.remove_times(
            met=met_vals[0], bins=None, cull=CullCode.LOOSE
        )

        # Flag second MET with code 2
        goodtimes_instance.goodtimes.remove_times(met=met_vals[1], bins=None, cull=2)

        stats = goodtimes_instance.goodtimes.get_cull_statistics()

        assert stats["culled_bins"] == 180
        assert stats["cull_code_counts"][CullCode.LOOSE] == 90
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

        # Should have one line per interval (10 METs, all good)
        assert len(lines) == 10

        # Check format of first line
        parts = lines[0].strip().split()
        assert len(parts) == 7
        assert parts[0] == "00042"  # pointing
        assert parts[5] == "Hi45"  # sensor

    def test_to_txt_values(self, goodtimes_instance, tmp_path):
        """Test the values in the output file."""
        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            line = f.readline()

        parts = line.strip().split()
        pointing, met_start, met_end, bin_low, bin_high, sensor, esa_step = parts

        assert pointing == "00042"
        assert int(met_start) == int(goodtimes_instance.coords["met"].values[0])
        assert int(met_end) == int(goodtimes_instance.coords["met"].values[0])
        assert int(bin_low) == 0
        assert int(bin_high) == 89
        assert sensor == "Hi45"
        assert int(esa_step) == goodtimes_instance["esa_step"].values[0]

    def test_to_txt_with_culled_bins(self, goodtimes_instance, tmp_path):
        """Test output when some bins are culled."""
        # Flag bins 0-20 for first MET
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=np.arange(21), cull=CullCode.LOOSE
        )

        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            first_line = f.readline()

        parts = first_line.strip().split()
        bin_low = int(parts[3])
        bin_high = int(parts[4])

        # First interval should only include bins 21-89
        assert bin_low == 21
        assert bin_high == 89

    def test_to_txt_with_gaps(self, goodtimes_instance, tmp_path):
        """Test output when bins have gaps."""
        # Flag bins 20-70, leaving 0-19 and 71-89 as good
        met_val = goodtimes_instance.coords["met"].values[0]
        goodtimes_instance.goodtimes.remove_times(
            met=met_val, bins=np.arange(20, 71), cull=CullCode.LOOSE
        )

        output_path = tmp_path / "goodtimes.txt"
        goodtimes_instance.goodtimes.write_txt(output_path)

        with open(output_path) as f:
            lines = f.readlines()

        # Should have 11 lines (2 for first MET, 1 for each of 9 remaining METs)
        assert len(lines) == 11

        # First two lines should be for same MET
        parts1 = lines[0].strip().split()
        parts2 = lines[1].strip().split()
        assert parts1[1] == parts2[1]  # Same met_start

        # Check bin ranges
        assert int(parts1[3]) == 0
        assert int(parts1[4]) == 19
        assert int(parts2[3]) == 71
        assert int(parts2[4]) == 89


class TestIntervalDtype:
    """Test suite for INTERVAL_DTYPE."""

    def test_interval_dtype_fields(self):
        """Test that INTERVAL_DTYPE has correct fields."""
        field_names = INTERVAL_DTYPE.names
        assert "met_start" in field_names
        assert "met_end" in field_names
        assert "spin_bin_low" in field_names
        assert "spin_bin_high" in field_names
        assert "n_good_bins" in field_names
        assert "esa_step" in field_names

    def test_interval_dtype_types(self):
        """Test that INTERVAL_DTYPE has correct field types."""
        assert INTERVAL_DTYPE["met_start"] == np.float64
        assert INTERVAL_DTYPE["met_end"] == np.float64
        assert INTERVAL_DTYPE["spin_bin_low"] == np.uint8
        assert INTERVAL_DTYPE["spin_bin_high"] == np.uint8
        assert INTERVAL_DTYPE["n_good_bins"] == np.uint8
        assert INTERVAL_DTYPE["esa_step"] == np.uint8
