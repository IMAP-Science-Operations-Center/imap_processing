"""Tests the L2 processing of CoDICE L1 data"""

from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imap_data_access import AncillaryInput, ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l2 import (
    add_dataset_attributes,
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    process_codice_l2,
    process_lo_species_intensity,
)
from imap_processing.codice.constants import (
    LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
)

pytestmark = pytest.mark.external_test_data

EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l2_hi-direct-events",
    "imap_codice_l2_lo-direct-events",
]


@pytest.fixture
def processing_dependencies(codice_lut_path):
    eff_file = "imap_codice_l2-lo-efficiency_20251008_v001.csv"
    gf_file = "imap_codice_l2-lo-gfactor_20251008_v001.csv"
    return ProcessingInputCollection(AncillaryInput(gf_file), AncillaryInput(eff_file))


@pytest.fixture
def mock_get_file_paths(codice_lut_path):
    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        mock_get_file_paths.side_effect = codice_lut_path
        yield mock_get_file_paths


@pytest.fixture
def mock_cdf_attrs():
    # Create a mock ImapCdfAttributes object
    cdf_attrs = MagicMock(spec=ImapCdfAttributes)
    cdf_attrs.get_global_attributes.return_value = {
        "global_attr_key": "global_attr_value"
    }
    cdf_attrs.get_variable_attributes.side_effect = lambda var, check_schema: {
        "var1": {"attr1": "value1"},
        "test-product-var2": {"attr2": "value2"},
    }[var]
    return cdf_attrs


@pytest.fixture
def mock_half_spin_lut(monkeypatch):
    """
    Mock HALF_SPIN_LUT for testing.
    Example:
      ESA steps 0–63 belong to half_spin=1
      ESA steps 64–127 belong to half_spin=2
    """
    mock_lut = {
        1: list(range(0, 64)),
        2: list(range(64, 128)),
    }
    monkeypatch.setattr(
        "imap_processing.codice.codice_l2.HALF_SPIN_LUT",
        mock_lut,
    )


def test_compute_geometric_factors_all_full_mode(mock_half_spin_lut):
    # rgfo_half_spin = 3 means all half_spin values (1 or 2) are < rgfo_half_spin
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([3, 3]))})
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "full" values everywhere
    expected = np.full((2, 128, 24), 0)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_all_reduced_mode(mock_half_spin_lut):
    # rgfo_half_spin = 0 means all half_spin values (>=1) are >= rgfo_half_spin
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([0]))})
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "reduced" values everywhere
    expected = np.full((1, 128, 24), 1)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_mixed(mock_half_spin_lut):
    # rgfo_half_spin = 2
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([2]))})
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # ESA steps 0-63 (half_spin=1) -> 1 < 2 → mode=full → 1
    # ESA steps 64-127 (half_spin=2) -> 2 !< 2 → mode=reduced → 0
    expected = np.repeat(np.array([[[0]] * 64 + [[1]] * 64]), 24, -1)
    np.testing.assert_array_equal(result, expected)


def test_add_dataset_attributes(mock_cdf_attrs):
    dataset_name = "imap_codice_l2_test-product"

    # Create a sample xarray.Dataset
    sample_dataset = xr.Dataset(
        {
            "var1": (["dim1"], [1, 2, 3]),
            "var2": (["dim1"], [4, 5, 6]),
            "var3": (["dim1"], [7, 8, 9]),
        }
    )

    # Patch the logger to capture error messages
    with patch("imap_processing.codice.codice_l2.logger") as mock_logger:
        # Call the function
        updated_dataset = add_dataset_attributes(
            sample_dataset, dataset_name, mock_cdf_attrs
        )

        # Assert global attributes are updated
        assert updated_dataset.attrs == {"global_attr_key": "global_attr_value"}

        # Assert variable attributes are updated

        # var1 should get attributes directly
        assert updated_dataset["var1"].attrs == {"attr1": "value1"}

        # var2 should get attributes with product descriptor prefix (test-product)
        assert updated_dataset["var2"].attrs == {"attr2": "value2"}

        # var3 should log an error since it doesn't have corresponding attributes
        assert updated_dataset["var3"].attrs == {}

        # Check logger error call for missing attributes
        mock_logger.error.assert_called_with(
            "Field 'var3' and 'test-product-var3' not found in attribute manager."
        )


def test_get_geometric_factor_lut(processing_dependencies, mock_get_file_paths):
    gfactor_lut = get_geometric_factor_lut(processing_dependencies)

    # Load the csv files directly to compare
    geometric_factors = pd.read_csv(
        processing_dependencies.get_file_paths("l2-lo-gfactor")[0]
    )
    full = (
        geometric_factors[geometric_factors["mode"] == "full"]
        .drop(["mode", "esa_step"], axis=1)
        .to_numpy()
    )
    reduced = (
        geometric_factors[geometric_factors["mode"] == "reduced"]
        .drop(["mode", "esa_step"], axis=1)
        .to_numpy()
    )

    # Test the shape is (modes, esa_steps, positions)
    np.testing.assert_array_equal(gfactor_lut["full"].shape, (128, 24))

    np.testing.assert_array_equal(gfactor_lut["full"], full)
    np.testing.assert_array_equal(gfactor_lut["reduced"], reduced)


def test_get_efficiency_lut(processing_dependencies, mock_get_file_paths):
    efficiency_lut = get_efficiency_lut(processing_dependencies)
    expected_colnames = ["esa_step", "product", "species"] + [
        f"position_{x}" for x in range(1, 25)
    ]

    for col in expected_colnames:
        assert col in efficiency_lut.columns, f"Missing column {col} in efficiency LUT"


def test_process_lo_species_intensity():
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814211100_v0.0.3.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    l1b_val_data_processed = l1b_val_data.copy()
    gf = np.ones((len(l1b_val_data.epoch), 128, 24)) * 2
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=np.ones((128, 5)) * 2,
    ):
        len_pos = 5
        process_lo_species_intensity(
            l1b_val_data_processed,
            LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
            gf,
            None,
            list(np.arange(0, len_pos)),
        )

    for var in LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES:
        assert var in l1b_val_data_processed, f"Missing variable {var} after processing"
        # Check that values are non-negative
        assert np.all(l1b_val_data_processed[var].values >= 0), (
            f"Variable {var} contains negative values"
        )
        # Check that values match expected calculation
        expected_intensity = (
            l1b_val_data[var]
            / (len_pos * 4 * l1b_val_data["energy_table"].data)[
                np.newaxis, :, np.newaxis
            ]
        )
        np.testing.assert_allclose(
            l1b_val_data_processed[var].values, expected_intensity.values, rtol=1e-5
        )


def test_process_lo_missing_species_intensity():
    l1b_val_data = xr.Dataset(
        {
            "epoch": ("epoch", np.ones(5)),
            "energy_table": (("esa_step",), np.ones(128) * 10),
        }
    )

    l1b_val_data_processed = l1b_val_data.copy()
    gf = np.ones((len(l1b_val_data.epoch), 128, 24)) * 2
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=np.ones((128, 5)) * 2,
    ):
        len_pos = 5
        process_lo_species_intensity(
            l1b_val_data_processed,
            LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
            gf,
            None,
            list(np.arange(0, len_pos)),
        )

    for var in LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES:
        assert var in l1b_val_data_processed, f"Missing variable {var} after processing"
        # Check that all the missing species are filled with NaNs
        assert not np.any(np.isfinite(l1b_val_data_processed[var].values)), (
            f"Variable {var} should be all NaNs"
        )


def test_codice_l2_sw_species_intensity(processing_dependencies, mock_get_file_paths):
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814211100_v0.0.3.cdf"
    )
    ds = process_codice_l2(l1b_val_data, processing_dependencies)
    ds.attrs["Data_version"] = "001"
    write_cdf(ds)


def test_codice_l2_nsw_species_intensity(processing_dependencies, mock_get_file_paths):
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-nsw-species_20250814211100_v0.0.3.cdf"
    )
    ds = process_codice_l2(l1b_val_data, processing_dependencies)
    ds.attrs["Data_version"] = "001"
    write_cdf(ds)
