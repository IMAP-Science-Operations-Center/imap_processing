"""Test coverage for imap_processing.hi.utils.py"""

import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imap_processing.hi
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import (
    HIAPID,
    CalibrationProductConfig,
    CoincidenceBitmap,
    EsaEnergyStepLookupTable,
    compute_qualified_event_mask,
    create_dataset_variables,
    filter_events_by_coincidence,
    full_dataarray,
    get_bin_range_with_wrap,
    get_tof_window_mask,
    parse_sensor_number,
)


def test_hiapid():
    """Test coverage for HIAPID class"""
    hi_apid = HIAPID(754)
    assert isinstance(hi_apid, HIAPID)
    assert hi_apid.name == "H45_APP_NHK"
    assert hi_apid.sensor == "45sensor"

    hi_apid = HIAPID["H90_SCI_CNT"]
    assert hi_apid.value == 833
    assert hi_apid.sensor == "90sensor"


@pytest.mark.parametrize(
    "test_str, expected",
    [
        ("imap_hi_l1b_45sensor-de", 45),
        ("imap_hi_l1c_90sensor-pset_20250415_v001.cdf", 90),
        ("imap_hi_l1c_{number}sensor", None),
    ],
)
def test_parse_sensor_number(test_str, expected):
    """Test coverage for parse_sensor_number function"""
    if expected:
        sensor_number = parse_sensor_number(test_str)
        assert sensor_number == expected
    else:
        with pytest.raises(ValueError, match=r"String 'sensor\(45|90\)' not found.*"):
            _ = parse_sensor_number(test_str)


@pytest.mark.parametrize(
    "name, shape, fill_value, expected_shape",
    [
        ("despun_z", (1, 3), None, (1, 3)),
        ("hae_latitude", None, 0, (1, 360)),
        ("counts", None, None, (1, 10, 5, 360)),
    ],
)
def test_full_dataarray(name, shape, fill_value, expected_shape):
    """Test coverage for full_dataarray function"""
    coords = {
        "epoch": xr.DataArray(np.array([0])),
        "esa_energy_step": xr.DataArray(np.arange(10)),
        "calibration_prod": xr.DataArray(np.arange(5)),
        "spin_angle_bin": xr.DataArray(np.arange(360)),
    }
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_variable_attrs(instrument="hi", level=None)
    attrs = cdf_manager.get_variable_attributes(f"hi_pset_{name}")

    dataarray = full_dataarray(
        name, attrs, coords=coords, shape=shape, fill_value=fill_value
    )
    assert dataarray.data.shape == expected_shape
    expected_fill_value = fill_value if fill_value is not None else attrs["FILLVAL"]
    np.testing.assert_array_equal(dataarray.data, expected_fill_value)


@pytest.mark.parametrize(
    "var_names, shape, fill_value, lookup_str",
    [
        (["tof_ab", "tof_ac1"], 5, None, "hi_de_{0}"),
        (["hae_latitude"], (3, 5), 0, "hi_pset_{0}"),
    ],
)
def test_create_dataset_variables(var_names, shape, fill_value, lookup_str):
    """Test coverage for `imap_processing.hi.utils.create_dataset_variables`"""
    var_names = ["tof_ab", "tof_ac1", "tof_bc1"]
    l1b_de_vars = create_dataset_variables(
        var_names, shape, fill_value=fill_value, att_manager_lookup_str="hi_de_{0}"
    )
    assert len(l1b_de_vars) == len(var_names)
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    for var_name, data_array in l1b_de_vars.items():
        attrs = attr_mgr.get_variable_attributes(
            f"hi_de_{var_name}", check_schema=False
        )
        assert data_array.values.dtype == attrs["dtype"]
        if data_array.ndim == 1:
            assert data_array.size == shape
        else:
            assert data_array.shape == shape
        expected_fill_value = fill_value if fill_value is not None else attrs["FILLVAL"]
        np.testing.assert_array_equal(data_array, expected_fill_value)


@pytest.mark.parametrize(
    "sensor_hit_str, expected_val",
    [
        ("ABC1C2", 15),
        ("ABC1", 14),
        ("AB", 12),
        ("AC1C2", 11),
        ("AC1", 10),
        ("A", 8),
        ("BC1C2", 7),
        ("BC1", 6),
        ("B", 4),
        ("C1C2", 3),
        ("C1", 2),
    ],
)
def test_coincidence_type_string_to_int(sensor_hit_str, expected_val):
    """Test coverage for coincidence_type_string_to_int function"""
    assert CoincidenceBitmap.detector_hit_str_to_int(sensor_hit_str) == expected_val


class TestEsaEnergyStepLookupTable:
    """Test suite for EsaEnergyStepLookupTable class."""

    @pytest.fixture
    def empty_lookup(self):
        """Create an empty lookup table for testing."""
        return EsaEnergyStepLookupTable()

    @pytest.fixture
    def populated_lookup(self):
        """Create a lookup table with test data."""
        lookup = EsaEnergyStepLookupTable()
        # Columns are: [start_met, end_met, esa_step, esa_energy_step]
        entries = [
            (0.0, 10.0, 1, 100.0),
            (0.0, 10.0, 2, 200.0),
            (10.0, 20.0, 1, 150.0),
            (10.0, 20.0, 2, 250.0),
        ]
        for entry in entries:
            lookup.add_entry(*entry)
        return lookup

    def test_init(self, empty_lookup):
        """Test initialization of lookup table."""
        assert len(empty_lookup.df) == 0
        assert list(empty_lookup.df.columns) == [
            "start_met",
            "end_met",
            "esa_step",
            "esa_energy_step",
        ]
        assert empty_lookup._indexed is False

    def test_add_entry_single(self, empty_lookup):
        """Test adding a single entry."""
        empty_lookup.add_entry(0.0, 10.0, 1, 100.0)

        assert len(empty_lookup.df) == 1
        assert empty_lookup.df.iloc[0]["start_met"] == 0.0
        assert empty_lookup.df.iloc[0]["end_met"] == 10.0
        assert empty_lookup.df.iloc[0]["esa_step"] == 1
        assert empty_lookup.df.iloc[0]["esa_energy_step"] == 100.0
        assert empty_lookup._indexed is False

    def test_add_entry_multiple(self, empty_lookup):
        """Test adding multiple entries one by one."""
        empty_lookup.add_entry(0.0, 10.0, 1, 100.0)
        empty_lookup.add_entry(10.0, 20.0, 2, 200.0)

        assert len(empty_lookup.df) == 2
        assert empty_lookup._indexed is False

    def test_ensure_indexed(self, populated_lookup):
        """Test the indexing functionality."""
        # Initially not indexed
        assert populated_lookup._indexed is False

        # Call _ensure_indexed
        populated_lookup._ensure_indexed()
        assert populated_lookup._indexed is True

        # Check that data is sorted by start_met, then esa_step
        df = populated_lookup.df
        # np.lexsort uses the last key as the primary sort order
        np.testing.assert_array_equal(
            np.lexsort((df["esa_step"].values, df["start_met"].values)),
            np.arange(len(df)),
        )

    def test_query_scalar_found(self, populated_lookup):
        """Test scalar query that finds a match."""
        result = populated_lookup.query(5.0, 1)
        assert result == 100.0

        result = populated_lookup.query(12.0, 2)
        assert result == 250.0

    def test_query_scalar_not_found(self, populated_lookup):
        """Test scalar query that doesn't find a match."""
        # Query outside time range
        result = populated_lookup.query(25.0, 1)
        assert result == 255

        # Query with non-existent esa_step
        result = populated_lookup.query(5.0, 99)
        assert result == 255

    def test_query_array_found(self, populated_lookup):
        """Test array query with matches."""
        mets = [5.0, 12.0, 8.0]
        steps = [1, 2, 3]
        results = populated_lookup.query(mets, steps)

        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert results[0] == 100  # MET=5.0, step=1
        assert results[1] == 250  # MET=12.0, step=2
        assert results[2] == 255  # MET=8.0, step=3

    def test_query_array_all_not_found(self, populated_lookup):
        """Test array query where no entries are found."""
        mets = [25.0, 30.0]
        steps = [1, 2]
        results = populated_lookup.query(mets, steps)

        assert isinstance(results, np.ndarray)
        assert len(results) == 2
        assert results[0] == 255
        assert results[1] == 255

    def test_query_numpy_arrays(self, populated_lookup):
        """Test query with numpy arrays as input."""
        mets = np.array([5.0, 12.0])
        steps = np.array([1, 2])
        results = populated_lookup.query(mets, steps)

        assert isinstance(results, np.ndarray)
        assert len(results) == 2
        assert results[0] == 100.0
        assert results[1] == 250.0

    def test_query_mixed_scalar_array_raises_error(self, populated_lookup):
        """Test that mixing scalar and array inputs raises ValueError."""
        with pytest.raises(
            ValueError,
            match="query_met and esa_step must both be scalars or both be array-like",
        ):
            populated_lookup.query(5.0, [1, 2])

        with pytest.raises(
            ValueError,
            match="query_met and esa_step must both be scalars or both be array-like",
        ):
            populated_lookup.query([5.0, 12.0], 1)

    def test_query_different_length_arrays_raises_error(self, populated_lookup):
        """Test that arrays of different lengths raise ValueError."""
        with pytest.raises(
            ValueError,
            match="query_met and esa_step must have the same length "
            "when both are array-like",
        ):
            populated_lookup.query([5.0, 12.0], [1, 2, 3])

        with pytest.raises(
            ValueError,
            match="query_met and esa_step must have the same length "
            "when both are array-like",
        ):
            populated_lookup.query([5.0, 12.0, 8.0], [1, 2])

    def test_query_different_shape_arrays_raises_error(self, populated_lookup):
        """Test that arrays of different shapes raise ValueError."""
        mets = np.array([[5.0, 12.0]])  # Shape (1, 2)
        steps = np.array([1, 2])  # Shape (2,)

        with pytest.raises(
            ValueError,
            match="query_met and esa_step must have the same "
            "length when both are array-like",
        ):
            populated_lookup.query(mets, steps)

    def test_query_single_element_arrays(self, populated_lookup):
        """Test query with single-element arrays."""
        results = populated_lookup.query([5.0], [1])
        assert isinstance(results, np.ndarray)
        assert len(results) == 1
        assert results[0] == 100.0

    def test_query_calls_ensure_indexed(self, populated_lookup):
        """Test that query calls _ensure_indexed."""
        # Reset indexed flag
        populated_lookup._indexed = False

        # Call query
        populated_lookup.query(5.0, 1)

        # Check that indexing was performed
        assert populated_lookup._indexed is True

    def test_add_entry_resets_indexed_flag(self, populated_lookup):
        """Test that adding entries resets the indexed flag."""
        # Ensure it's indexed first
        populated_lookup._ensure_indexed()
        assert populated_lookup._indexed is True

        # Add entry
        populated_lookup.add_entry(20.0, 30.0, 4, 400.0)
        assert populated_lookup._indexed is False

    def test_edge_case_boundary_values(self, populated_lookup):
        """Test queries at exact boundary values."""
        # Test exact start time
        result = populated_lookup.query(0.0, 1)
        assert result == 100.0

        # Test exact end time
        result = populated_lookup.query(10.0, 1)
        # Should match both (0.0, 10.0, 1, 100.0) and (10.0, 20.0, 1, 150.0)
        assert result in [100.0, 150.0]


class TestCalibrationProductConfig:
    """
    All test coverage for the pd.DataFrame accessor extension "cal_prod_config".
    """

    def test_wrong_columns(self):
        """Test coverage for a dataframe with the wrong columns."""
        required_columns = (
            imap_processing.hi.utils.CalibrationProductConfig.required_columns
        )
        for exclude_column_name in required_columns:
            include_columns = set(required_columns) - {exclude_column_name}
            df = pd.DataFrame({col: [1, 2, 3] for col in include_columns})
            with pytest.raises(AttributeError, match="Required column*"):
                _ = df.cal_prod_config.number_of_products

    def test_from_csv(self, hi_test_cal_prod_config_path):
        """Test coverage for read_csv function."""
        df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
            hi_test_cal_prod_config_path
        )
        assert isinstance(df["coincidence_type_list"][0, 1], tuple)

    def test_added_coincidence_type_values_column(self, hi_test_cal_prod_config_path):
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert "coincidence_type_values" in df.columns
        for _, row in df.iterrows():
            for detect_string, val in zip(
                row["coincidence_type_list"],
                row["coincidence_type_values"],
                strict=False,
            ):
                assert val == CoincidenceBitmap.detector_hit_str_to_int(detect_string)

    def test_number_of_products(self, hi_test_cal_prod_config_path):
        """Test coverage for number of products accessor."""
        df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
            hi_test_cal_prod_config_path
        )
        assert df.cal_prod_config.number_of_products == 2

    def test_calibration_product_numbers(self, hi_test_cal_prod_config_path):
        """Test coverage for calibration_product_numbers accessor."""
        df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
            hi_test_cal_prod_config_path
        )
        cal_prod_numbers = df.cal_prod_config.calibration_product_numbers
        # The test config file has calibration products 0 and 1
        np.testing.assert_array_equal(cal_prod_numbers, np.array([0, 1]))
        # Verify it's a numpy array of integers
        assert isinstance(cal_prod_numbers, np.ndarray)
        assert cal_prod_numbers.dtype in [np.int32, np.int64]

    def test_calibration_product_numbers_arbitrary_values(self):
        """Test calibration_product_numbers with arbitrary non-sequential values."""
        # Create a temporary CSV with non-sequential calibration product numbers
        csv_content = """\
calibration_prod,esa_energy_step,geometric_factor,coincidence_type_list,tof_ab_low,tof_ab_high,tof_ac1_low,tof_ac1_high,tof_bc1_low,tof_bc1_high,tof_c1c2_low,tof_c1c2_high
10,1,0.00055,BC1C2,15,55,0,70,-50,10,5,25
10,2,0.00085,BC1C2,15,55,0,70,-50,10,5,25
5,1,0.00055,ABC1C2,15,55,0,70,-50,10,5,25
5,2,0.00085,ABC1C2,15,55,0,70,-50,10,5,25
100,1,0.00055,AC1,15,55,0,70,-50,10,5,25
100,2,0.00085,AC1,15,55,0,70,-50,10,5,25
        """

        df = CalibrationProductConfig.from_csv(io.StringIO(csv_content))
        cal_prod_numbers = df.cal_prod_config.calibration_product_numbers

        # Should return sorted unique calibration product numbers
        np.testing.assert_array_equal(cal_prod_numbers, np.array([5, 10, 100]))
        assert isinstance(cal_prod_numbers, np.ndarray)


class TestGetTofWindowMask:
    """Test suite for get_tof_window_mask function."""

    @pytest.fixture
    def mock_de_dataset(self):
        """Create a mock L1B DE dataset with TOF values."""
        n_events = 10
        return xr.Dataset(
            {
                "tof_ab": (
                    ["event_met"],
                    np.array([20, 50, 100, 30, 40, 60, 10, 80, 90, 55]),
                ),
                "tof_ac1": (
                    ["event_met"],
                    np.array([10, 30, -5, 50, 20, 40, 0, 60, 70, 35]),
                ),
                "tof_bc1": (
                    ["event_met"],
                    np.array([-30, 0, -20, 10, -40, -10, -50, 15, 20, 5]),
                ),
                "tof_c1c2": (
                    ["event_met"],
                    np.array([50, 60, 80, 30, 40, 70, 20, 90, 100, 55]),
                ),
            },
            coords={"event_met": np.arange(n_events, dtype=float)},
        )

    def test_all_tofs_in_window(self, mock_de_dataset):
        """Test that events with all TOFs in window pass."""
        # Use wide windows that include all values
        tof_windows = {
            "tof_ab": (0, 200),
            "tof_ac1": (-20, 100),
            "tof_bc1": (-100, 50),
            "tof_c1c2": (0, 200),
        }
        tof_fill_vals = {k: -9999 for k in tof_windows}
        mask = get_tof_window_mask(mock_de_dataset, tof_windows, tof_fill_vals)
        assert np.all(mask)

    def test_some_tofs_out_of_window(self, mock_de_dataset):
        """Test that events with TOFs outside window are filtered."""
        # tof_ab values: [20, 50, 100, 30, 40, 60, 10, 80, 90, 55]
        # Window (25, 75) should pass indices: 1, 3, 4, 5, 9 (values 50, 30, 40, 60, 55)
        tof_windows = {
            "tof_ab": (25, 75),
        }
        tof_fill_vals = {"tof_ab": -9999}
        mask = get_tof_window_mask(mock_de_dataset, tof_windows, tof_fill_vals)
        expected = np.array(
            [False, True, False, True, True, True, False, False, False, True]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_with_fill_values(self, mock_de_dataset):
        """Test that events with fill values pass the filter."""
        # Set some values to fill value
        fill_val = -9999
        mock_de_dataset["tof_ab"].values[0] = fill_val  # Was 20, now fill
        mock_de_dataset["tof_ab"].values[2] = fill_val  # Was 100, now fill

        tof_windows = {"tof_ab": (25, 75)}
        tof_fill_vals = {"tof_ab": fill_val}

        mask = get_tof_window_mask(mock_de_dataset, tof_windows, tof_fill_vals)
        # Events 0, 2 have fill values (pass), events 1, 3, 4, 5, 9 are in window
        expected = np.array(
            [True, True, True, True, True, True, False, False, False, True]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_multiple_tof_windows(self, mock_de_dataset):
        """Test with multiple TOF windows - all must pass."""
        # tof_ab:  [20, 50, 100, 30, 40, 60, 10, 80, 90, 55]
        # tof_ac1: [10, 30, -5, 50, 20, 40, 0, 60, 70, 35]
        tof_windows = {
            "tof_ab": (20, 80),  # Passes: 0,1,3,4,5,7,9 (not 2,6,8)
            "tof_ac1": (10, 60),  # Passes: 0,1,3,4,5,7,9 (not 2,6,8)
        }
        tof_fill_vals = {k: -9999 for k in tof_windows}
        mask = get_tof_window_mask(mock_de_dataset, tof_windows, tof_fill_vals)
        # Must pass both: 0, 1, 3, 4, 5, 7, 9
        expected = np.array(
            [True, True, False, True, True, True, False, True, False, True]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty_ds = xr.Dataset(
            {
                "tof_ab": (["event_met"], np.array([])),
                "tof_ac1": (["event_met"], np.array([])),
                "tof_bc1": (["event_met"], np.array([])),
                "tof_c1c2": (["event_met"], np.array([])),
            },
            coords={"event_met": np.array([])},
        )
        tof_windows = {"tof_ab": (0, 100)}
        mask = get_tof_window_mask(empty_ds, tof_windows, {})
        assert len(mask) == 0


class TestFilterEventsByCoincidence:
    """Test suite for filter_events_by_coincidence function."""

    @pytest.fixture
    def mock_de_dataset(self):
        """Create a mock L1B DE dataset with coincidence types."""
        # Coincidence bitmap: A=8, B=4, C1=2, C2=1
        # ABC1C2 = 15, ABC1 = 14, AB = 12, AC1 = 10, BC1 = 6, etc.
        return xr.Dataset(
            {
                "coincidence_type": (
                    ["event_met"],
                    np.array([15, 14, 12, 10, 6, 15, 8, 4, 2, 1]),
                ),
            },
            coords={"event_met": np.arange(10, dtype=float)},
        )

    def test_single_coincidence_type(self, mock_de_dataset):
        """Test filtering for a single coincidence type."""
        # Filter for ABC1C2 (15)
        mask = filter_events_by_coincidence(mock_de_dataset, [15])
        expected = np.array(
            [True, False, False, False, False, True, False, False, False, False]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_multiple_coincidence_types(self, mock_de_dataset):
        """Test filtering for multiple coincidence types."""
        # Filter for ABC1C2 (15) or ABC1 (14)
        mask = filter_events_by_coincidence(mock_de_dataset, [15, 14])
        expected = np.array(
            [True, True, False, False, False, True, False, False, False, False]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_no_matching_coincidence(self, mock_de_dataset):
        """Test when no events match the coincidence types."""
        # Filter for type 3 which doesn't exist
        mask = filter_events_by_coincidence(mock_de_dataset, [3])
        assert not np.any(mask)

    def test_all_matching_coincidence(self, mock_de_dataset):
        """Test when all events match the coincidence types."""
        all_types = [15, 14, 12, 10, 6, 8, 4, 2, 1]
        mask = filter_events_by_coincidence(mock_de_dataset, all_types)
        assert np.all(mask)

    def test_empty_coincidence_list(self, mock_de_dataset):
        """Test with empty coincidence type list."""
        mask = filter_events_by_coincidence(mock_de_dataset, [])
        assert not np.any(mask)

    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty_ds = xr.Dataset(
            {
                "coincidence_type": (["event_met"], np.array([], dtype=np.uint8)),
            },
            coords={"event_met": np.array([])},
        )
        mask = filter_events_by_coincidence(empty_ds, [15])
        assert len(mask) == 0


class TestGetBinRangeWithWrap:
    """Test suite for get_bin_range_with_wrap function."""

    def test_no_wrap_middle(self):
        """Test range in middle of bins (no wraparound)."""
        result = get_bin_range_with_wrap(
            first_bin=10, last_bin=20, n_bins=90, extend_by=1
        )
        expected = np.arange(9, 22)  # 10-1 to 20+1
        np.testing.assert_array_equal(result, expected)

    def test_no_wrap_with_larger_extension(self):
        """Test with larger extension value."""
        result = get_bin_range_with_wrap(
            first_bin=10, last_bin=20, n_bins=90, extend_by=3
        )
        expected = np.arange(7, 24)  # 10-3 to 20+3
        np.testing.assert_array_equal(result, expected)

    def test_wrap_at_end(self):
        """Test wraparound at high end (88 -> 0 boundary)."""
        result = get_bin_range_with_wrap(
            first_bin=87, last_bin=1, n_bins=90, extend_by=1
        )
        # Should get bins 86, 87, 88, 89, 0, 1, 2
        expected = np.array([86, 87, 88, 89, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_wrap_at_start(self):
        """Test wraparound near bin 0."""
        result = get_bin_range_with_wrap(
            first_bin=0, last_bin=5, n_bins=90, extend_by=1
        )
        # first-1 = -1 % 90 = 89, last+1 = 6
        # This should NOT wrap (89 > 6), so we get 89,0,1,2,3,4,5,6
        expected = np.array([89, 0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result, expected)

    def test_wrap_at_both_ends(self):
        """Test when first_bin is near end and last_bin is near start."""
        result = get_bin_range_with_wrap(
            first_bin=88, last_bin=2, n_bins=90, extend_by=1
        )
        # bot = 87, top = 3
        # Since 3 < 87, we wrap: [87, 88, 89] + [0, 1, 2, 3]
        expected = np.array([87, 88, 89, 0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_single_bin_range(self):
        """Test when first_bin equals last_bin."""
        result = get_bin_range_with_wrap(
            first_bin=45, last_bin=45, n_bins=90, extend_by=1
        )
        expected = np.array([44, 45, 46])
        np.testing.assert_array_equal(result, expected)

    def test_zero_extension(self):
        """Test with zero extension."""
        result = get_bin_range_with_wrap(
            first_bin=10, last_bin=15, n_bins=90, extend_by=0
        )
        expected = np.arange(10, 16)
        np.testing.assert_array_equal(result, expected)

    def test_different_n_bins(self):
        """Test with different number of bins."""
        result = get_bin_range_with_wrap(
            first_bin=350, last_bin=10, n_bins=360, extend_by=5
        )
        # bot = 345, top = 15
        # Since 15 < 345, we wrap: [345..359] + [0..15]
        expected = np.concatenate([np.arange(345, 360), np.arange(0, 16)])
        np.testing.assert_array_equal(result, expected)

    def test_adjacent_to_boundary(self):
        """Test bins adjacent to boundary (89 and 0)."""
        result = get_bin_range_with_wrap(
            first_bin=89, last_bin=89, n_bins=90, extend_by=1
        )
        # bot = 88, top = 0 (90 % 90)
        # Since 0 < 88, we wrap: [88, 89] + [0]
        expected = np.array([88, 89, 0])
        np.testing.assert_array_equal(result, expected)

    def test_full_spin_wrap(self):
        """Test wrapping that covers almost all bins."""
        result = get_bin_range_with_wrap(
            first_bin=85, last_bin=5, n_bins=90, extend_by=1
        )
        # bot = 84, top = 6
        # Since 6 < 84, we wrap
        expected = np.concatenate([np.arange(84, 90), np.arange(0, 7)])
        np.testing.assert_array_equal(result, expected)


class TestComputeQualifiedEventMask:
    """Test suite for compute_qualified_event_mask function."""

    @pytest.fixture
    def mock_cal_product_config(self):
        """Create a mock calibration product config DataFrame."""
        # Create a config with 2 calibration products, 2 ESA energy steps
        # Coincidence bitmap: A=8, B=4, C1=2, C2=1
        # ABC1C2=15, ABC1=14, AB=12
        data = {
            "coincidence_type_list": [
                ("ABC1C2", "ABC1"),  # cal_prod=1, esa_energy=1
                ("ABC1C2", "ABC1"),  # cal_prod=1, esa_energy=2
                ("AB",),  # cal_prod=2, esa_energy=1
                ("AB",),  # cal_prod=2, esa_energy=2
            ],
            "tof_ab_low": [10, 10, 10, 10],
            "tof_ab_high": [100, 100, 100, 100],
            "tof_ac1_low": [5, 5, 5, 5],
            "tof_ac1_high": [80, 80, 80, 80],
            "tof_bc1_low": [-50, -50, -50, -50],
            "tof_bc1_high": [50, 50, 50, 50],
            "tof_c1c2_low": [20, 20, 20, 20],
            "tof_c1c2_high": [120, 120, 120, 120],
        }
        index = pd.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (2, 1), (2, 2)],
            names=["calibration_prod", "esa_energy_step"],
        )
        df = pd.DataFrame(data, index=index)
        # Trigger the accessor to add coincidence_type_values column
        _ = df.cal_prod_config.number_of_products
        return df

    @pytest.fixture
    def mock_de_dataset(self):
        """Create a mock L1B DE dataset with events."""
        # 10 events with various coincidence types and TOF values
        # Coincidence bitmap: A=8, B=4, C1=2, C2=1
        # ABC1C2=15, ABC1=14, AB=12, A=8
        n_events = 10
        fill_val = -9999.0
        ds = xr.Dataset(
            {
                "coincidence_type": (
                    ["event_met"],
                    np.array([15, 14, 12, 8, 15, 14, 12, 8, 15, 12]),
                ),
                "tof_ab": (
                    ["event_met"],
                    np.array([50, 50, 50, 50, 200, 50, 50, 50, 50, 50]),
                ),  # Event 4 out of window
                "tof_ac1": (
                    ["event_met"],
                    np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30]),
                ),
                "tof_bc1": (
                    ["event_met"],
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                ),
                "tof_c1c2": (
                    ["event_met"],
                    np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50]),
                ),
            },
            coords={"event_met": np.arange(n_events, dtype=float)},
        )
        # Add FILLVAL attributes to TOF variables
        for tof_var in ["tof_ab", "tof_ac1", "tof_bc1", "tof_c1c2"]:
            ds[tof_var].attrs["FILLVAL"] = fill_val
        return ds

    def test_qualifies_with_both_coincidence_and_tof(
        self, mock_cal_product_config, mock_de_dataset
    ):
        """Events passing both coincidence and TOF checks qualify."""
        # All events at ESA energy step 1
        esa_energy_steps = np.ones(10, dtype=int)

        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        # Events with coincidence_type in [15, 14, 12] and TOF in window should pass
        # Event 4 has bad TOF (200, outside 10-100 window)
        # Events 3, 7 have coincidence_type=8 (A only, not in config)
        expected = np.array(
            [True, True, True, False, False, True, True, False, True, True]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_fails_coincidence_only(self, mock_cal_product_config, mock_de_dataset):
        """Events with wrong coincidence type don't qualify."""
        # All events at ESA energy step 1
        esa_energy_steps = np.ones(10, dtype=int)

        # Check events 3, 7 which have coincidence_type=8 (not in config)
        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        # Events 3 and 7 should not qualify
        assert mask[3] is np.False_
        assert mask[7] is np.False_

    def test_fails_tof_only(self, mock_cal_product_config, mock_de_dataset):
        """Events with valid coincidence but bad TOF don't qualify."""
        # All events at ESA energy step 1
        esa_energy_steps = np.ones(10, dtype=int)

        # Event 4 has coincidence_type=15 (valid) but tof_ab=200 (outside 10-100)
        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        assert mask[4] is np.False_

    def test_union_across_cal_products(self, mock_cal_product_config, mock_de_dataset):
        """Events qualify if they pass for ANY cal product."""
        esa_energy_steps = np.ones(10, dtype=int)

        # Event 2 has coincidence_type=12 (AB), valid for cal_prod 2
        # Event 0 has coincidence_type=15 (ABC1C2), valid for cal_prod 1
        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        assert mask[0]  # Qualifies for cal_prod 1
        assert mask[2]  # Qualifies for cal_prod 2

    def test_fill_values_pass_tof(self, mock_cal_product_config, mock_de_dataset):
        """Events with TOF fill values pass TOF check."""
        esa_energy_steps = np.ones(10, dtype=int)

        # Set event 4's TOF to fill value (it was failing due to high tof_ab)
        # The FILLVAL attribute is already set by the fixture
        fill_val = mock_de_dataset["tof_ab"].attrs["FILLVAL"]
        mock_de_dataset["tof_ab"].values[4] = fill_val

        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        # Event 4 should now pass (fill value passes TOF check)
        assert mask[4]

    def test_different_esa_energy_steps(self, mock_cal_product_config, mock_de_dataset):
        """Events match config based on their ESA energy step."""
        # Half events at ESA 1, half at ESA 2
        esa_energy_steps = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        # Events 0-4 should match ESA 1 config
        # Events 5-9 should match ESA 2 config
        # Event 4 still fails due to bad TOF
        # Events 7 fails due to bad coincidence type (8)
        expected = np.array(
            [True, True, True, False, False, True, True, False, True, True]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_no_matching_esa_energy_step(
        self, mock_cal_product_config, mock_de_dataset
    ):
        """Events with unmatched ESA energy step don't qualify."""
        # All events at ESA energy step 99 (not in config)
        esa_energy_steps = np.full(10, 99)

        mask = compute_qualified_event_mask(
            mock_de_dataset, mock_cal_product_config, esa_energy_steps
        )

        # No events should qualify
        assert not np.any(mask)

    def test_empty_dataset(self, mock_cal_product_config):
        """Test with empty dataset."""
        empty_ds = xr.Dataset(
            {
                "coincidence_type": (["event_met"], np.array([], dtype=np.uint8)),
                "tof_ab": (["event_met"], np.array([])),
                "tof_ac1": (["event_met"], np.array([])),
                "tof_bc1": (["event_met"], np.array([])),
                "tof_c1c2": (["event_met"], np.array([])),
            },
            coords={"event_met": np.array([])},
        )
        esa_energy_steps = np.array([])

        mask = compute_qualified_event_mask(
            empty_ds, mock_cal_product_config, esa_energy_steps
        )

        assert len(mask) == 0
