"""Test coverage for imap_processing.hi.utils.py"""

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
    create_dataset_variables,
    full_dataarray,
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
