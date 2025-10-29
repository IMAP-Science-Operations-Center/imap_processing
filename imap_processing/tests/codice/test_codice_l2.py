"""Tests the L2 processing of CoDICE L1 data"""

from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imap_data_access import AncillaryInput, ProcessingInputCollection, ScienceInput

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l2 import (
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    process_codice_l2,
    process_lo_angular_intensity,
    process_lo_species_intensity,
)
from imap_processing.codice.constants import (
    LO_NSW_ANGULAR_VARIABLE_NAMES,
    LO_SW_ANGULAR_VARIABLE_NAMES,
    LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
    SW_POSITIONS,
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
        # Ensure the side effect treats science inputs as L1B for these L2 tests
        mock_get_file_paths.side_effect = (
            lambda descriptor, data_type=None: codice_lut_path(
                descriptor, data_type="l1b"
            )
        )
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
        / "imap_codice_l1b_lo-sw-species_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    l1b_val_data_processed = l1b_val_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_val_data.epoch), 128, 24)) * 2,
        dims=("epoch", "energy_table", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(
            np.ones((128, 24)) * 2, dims=("energy_table", "inst_az")
        ),
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
    gf = xr.DataArray(
        np.ones((len(l1b_val_data.epoch), 128, 24)) * 2,
        dims=("epoch", "energy_table", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(
            np.ones((128, 24)) * 2, dims=("energy_table", "inst_az")
        ),
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


def test_process_lo_angular_intensity():
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-angular_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    l1b_val_data_processed = l1b_val_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_val_data.epoch), 128, 24)) * 2,
        dims=("epoch", "energy_table", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(
            np.ones((128, 24)) * 2, dims=("energy_table", "inst_az")
        ),
    ):
        l1b_val_data_processed = process_lo_angular_intensity(
            l1b_val_data_processed,
            LO_SW_ANGULAR_VARIABLE_NAMES,
            gf,
            None,
            SW_POSITIONS,
        )

    for var in LO_SW_ANGULAR_VARIABLE_NAMES:
        assert var in l1b_val_data_processed, f"Missing variable {var} after processing"
        # Check that values are non-negative
        assert np.all(l1b_val_data_processed[var].values >= 0), (
            f"Variable {var} contains negative values"
        )
        # Check shape
        expected_shape = (
            len(l1b_val_data.epoch),
            len(l1b_val_data.energy_table),
            len(l1b_val_data.spin_sector),
            3,  # 3 elevation angles map to 5 positions
        )
        np.testing.assert_allclose(
            expected_shape, l1b_val_data_processed[var].shape, rtol=1e-5
        )
        # Check that values match expected calculation
        expected_intensity = (
            l1b_val_data[var]
            / (4 * l1b_val_data["energy_table"].data)[
                np.newaxis, :, np.newaxis, np.newaxis
            ]
        )
        # convert pos to el
        expected_intensity = (
            expected_intensity.assign_coords(group=("inst_az", [0, 1, 2, 2, 1]))
            .groupby("group")
            .sum()
        )
        # Skip checking the first elevations. Those get reassigned and will be
        # validated below.
        np.testing.assert_allclose(
            l1b_val_data_processed[var].values[:, :, :, 1:],
            expected_intensity.values[:, :, :, 1:],
            rtol=1e-5,
        )
    # Check coords
    np.testing.assert_allclose(l1b_val_data_processed["elevation_angle"], [0, 15, 30])
    np.testing.assert_allclose(
        l1b_val_data_processed["spin_angle"], np.arange(24) * 15 + 7.5
    )


def test_codice_l2_sw_species_intensity(processing_dependencies, mock_get_file_paths):
    sci_input = ScienceInput("imap_codice_l1b_lo-sw-species_20250814_v007.cdf")
    processing_dependencies.add(sci_input)

    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / "imap_codice_l2_lo-sw-species_20250814_v007.cdf"
    )
    l2_val_data = load_cdf(l2_val_data)
    ds = process_codice_l2("lo-sw-species", processing_dependencies)
    for variable in l2_val_data.data_vars:
        processed_val = ds[variable].values
        np.testing.assert_allclose(
            processed_val,
            l2_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    ds.attrs["Data_version"] = "001"
    write_cdf(ds, istp=False)


def test_codice_l2_nsw_species_intensity(processing_dependencies, mock_get_file_paths):
    sci_input = ScienceInput("imap_codice_l1b_lo-nsw-species_20250814_v007.cdf")
    processing_dependencies.add(sci_input)
    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / "imap_codice_l2_lo-nsw-species_20250814_v007.cdf"
    )
    l2_val_data = load_cdf(l2_val_data)
    ds = process_codice_l2("lo-nsw-species", processing_dependencies)
    for variable in l2_val_data.data_vars:
        np.testing.assert_allclose(
            ds[variable].values,
            l2_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    ds.attrs["Data_version"] = "001"
    write_cdf(ds, istp=False)


def test_codice_l2_nsw_angular_intensity(processing_dependencies, mock_get_file_paths):
    sci_input = ScienceInput("imap_codice_l1b_lo-nsw-angular_20250814_v007.cdf")
    processing_dependencies.add(sci_input)
    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / "imap_codice_l2_lo-nsw-angular_20250814_v007.cdf"
    )
    l2_val_data = load_cdf(l2_val_data)
    ds = process_codice_l2("lo-nsw-angular", processing_dependencies)
    for variable in LO_NSW_ANGULAR_VARIABLE_NAMES:
        np.testing.assert_allclose(
            ds[variable].values,
            l2_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    ds.attrs["Data_version"] = "001"
    # TODO fix attrs
    write_cdf(ds, istp=False)


def test_codice_l2_sw_angular_intensity(processing_dependencies, mock_get_file_paths):
    sci_input = ScienceInput("imap_codice_l1b_lo-sw-angular_20250814_v007.cdf")
    processing_dependencies.add(sci_input)
    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / "imap_codice_l2_lo-sw-angular_20250814_v007.cdf"
    )
    l2_val_data = load_cdf(l2_val_data)
    ds = process_codice_l2("lo-sw-angular", processing_dependencies)
    for variable in LO_SW_ANGULAR_VARIABLE_NAMES:
        np.testing.assert_allclose(
            ds[variable].values,
            l2_val_data[variable].values,
            # TODO is 1e-4 ok?
            rtol=1e-4,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    ds.attrs["Data_version"] = "001"
    # TODO fix attrs
    write_cdf(ds, istp=False)
