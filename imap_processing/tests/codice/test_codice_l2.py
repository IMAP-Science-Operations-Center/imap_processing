"""Tests the L2 processing of CoDICE L1 data"""

from unittest import mock
from unittest.mock import MagicMock, patch

import cdflib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imap_data_access import AncillaryInput, ProcessingInputCollection
from sammi.validation import CDFValidator

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.codice.codice_l2 import (
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    get_hi_de_luts,
    get_mpq_calc_energy_conversion_vals,
    get_mpq_calc_tof_conversion_vals,
    process_codice_l2,
    process_lo_species_intensity,
)
from imap_processing.codice.constants import (
    LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
)
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)

pytestmark = pytest.mark.external_test_data

# epoch_delta = num_spins * spin_period / 2, with spin_period VALIDMAX = 16 s
# and num_spins max = 16 in the current CoDICE timing model. That yields a
# worst-case delta of 128 s = 128000000000 ns.
EXPECTED_EPOCH_DELTA_VALIDMAX = 128000000000


def assert_l2_epoch_delta_cdf_metadata(cdf_file):
    """Assert L2 epoch delta vars and epoch links are written correctly."""
    with cdflib.CDF(cdf_file) as cdf:
        epoch_attrs = cdf.varattsget("epoch")
        assert epoch_attrs["DELTA_MINUS_VAR"] == "epoch_delta_minus"
        assert epoch_attrs["DELTA_PLUS_VAR"] == "epoch_delta_plus"

        for variable in ("epoch_delta_minus", "epoch_delta_plus"):
            info = cdf.varinq(variable)
            attrs = cdf.varattsget(variable)
            assert info.Data_Type_Description == "CDF_INT8"
            assert attrs["FILLVAL"] == -9223372036854775808
            assert attrs["FORMAT"] == "I19"
            assert attrs["VALIDMIN"] == 0
            assert attrs["VALIDMAX"] == EXPECTED_EPOCH_DELTA_VALIDMAX


EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l2_hi-direct-events",
    "imap_codice_l2_lo-direct-events",
]

DIRECT_EVENT_LUTS = {
    "lo-direct-events": [
        "l2-lo-onboard-energy-table",
        "l2-lo-onboard-energy-bins",
        "l2-lo-onboard-mpq-cal",
        "l2-lo-onboard-mpq-cal",
    ],
    "hi-direct-events": [
        "l2-hi-energy-table",
        "l2-hi-tof-table",
    ],
}

DIRECT_EVENT_DISPLAY_TYPES = {
    "lo-direct-events": {
        "num_events": "time_series",
        "data_quality": "no_plot",
        "gain": "no_plot",
        "multi_flag": "no_plot",
        "spin_angle": "spectrogram",
        "elevation_angle": "spectrogram",
        "tof": "spectrogram",
        "type": "no_plot",
        "apd_energy": "spectrogram",
        "apd_id": "no_plot",
        "energy_per_charge": "spectrogram",
        "energy_step": "no_plot",
        "position": "no_plot",
    },
    "hi-direct-events": {
        "num_events": "time_series",
        "data_quality": "no_plot",
        "gain": "no_plot",
        "multi_flag": "no_plot",
        "spin_angle": "spectrogram",
        "elevation_angle": "spectrogram",
        "tof": "spectrogram",
        "type": "no_plot",
        "ssd_energy": "spectrogram",
        "energy_per_nuc": "spectrogram",
        "ssd_id": "no_plot",
    },
}


@pytest.fixture
def processing_dependencies(codice_lut_path):
    eff_file = "imap_codice_l2-lo-efficiency_20251008_v001.csv"
    gf_file = "imap_codice_l2-lo-gfactor_20251008_v001.csv"
    mpq_file = "imap_codice_lo-mpq-cal_20250101_v001.csv"
    return ProcessingInputCollection(
        AncillaryInput(gf_file), AncillaryInput(eff_file), AncillaryInput(mpq_file)
    )


@pytest.fixture
def mock_get_file_paths(codice_lut_path):
    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        # Ensure the side effect treats science inputs as L1B for these L2 tests
        mock_get_file_paths.side_effect = lambda descriptor, data_type=None: (
            codice_lut_path(descriptor, data_type="l1b")
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


def _generate_direct_events_l2_file(mock_get_file_paths, codice_lut_path, descriptor):
    """Generate a fresh CoDICE direct-events L2 CDF for metadata assertions."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor=descriptor, data_type="l0")
    ]
    l1a_cdf = process_l1a(ProcessingInputCollection())[0]
    processed_l1a_file = write_cdf(l1a_cdf)
    file_path = processed_l1a_file.as_posix()
    mock_get_file_paths.side_effect = [
        [file_path],
        [file_path],
        *[
            codice_lut_path(descriptor=lut_descriptor)
            for lut_descriptor in DIRECT_EVENT_LUTS[descriptor]
        ],
    ]
    processed_l2_ds = process_codice_l2(descriptor, ProcessingInputCollection())
    processed_l2_ds.attrs["Data_version"] = "001"
    return write_cdf(processed_l2_ds)


@pytest.fixture
def mock_half_spin_per_esa_step():
    """
    Mock half_spin_per_esa_step for testing.
    Example:
      ESA steps 0–63 belong to half_spin=2
      ESA steps 64–127 belong to half_spin=3
    """
    half_spin_per_esa = np.repeat([2, 3], 64)
    # repeat along epoch dimension to create shape (2, 128) for testing
    return np.tile(half_spin_per_esa, (2, 1))


def test_compute_geometric_factors_all_full_mode(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 4 means all half_spin values (2 or 3) are < rgfo_half_spin
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([4, 4])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step,
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "full" values everywhere
    expected = np.full((2, 128, 24), 0)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_past_nov_24th(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 1 means all half_spin values (>=2) are >= rgfo_half_spin
    # Although the rgfo_half_spin indicates reduced mode, the date is past Nov 24th,
    # 2025 so we expect full mode to be used.
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([1, 1])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step,
            ),
        },
        # Make sure epoch is past Nov 24th, 2025
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20251125_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "full" values everywhere
    expected = np.full((2, 128, 24), 0)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_all_reduced_mode(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 1 means all half_spin values (>=2) are >= rgfo_half_spin
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([1])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step[0:1],
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "reduced" values everywhere
    expected = np.full((1, 128, 24), 1)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_mixed(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 2
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([2])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step[0:1],
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # ESA steps 0-63 (half_spin=1) -> 2 > 1 → mode=full → 1
    # ESA steps 64-127 (half_spin=2) -> 1 !>1 → mode=reduced → 0
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


def test_get_tof_ns_from_mpq_lut(processing_dependencies, mock_get_file_paths):
    tof_ns = get_mpq_calc_tof_conversion_vals(processing_dependencies)
    assert tof_ns.shape == (1024,)
    mpq_calc_lut_file = processing_dependencies.get_file_paths(
        descriptor="l2-lo-onboard-mpq-cal"
    )[0]
    mpq_df = pd.read_csv(mpq_calc_lut_file, header=None)
    expected_tof_ns = mpq_df.loc[6:, 1].to_numpy().astype(np.float64)
    # Calculated values should be more precise than LUT but should be close
    np.testing.assert_allclose(tof_ns, expected_tof_ns, atol=1e-5)


def test_get_energy_kev_from_mpq_lut(processing_dependencies, mock_get_file_paths):
    energy_kev = get_mpq_calc_energy_conversion_vals(processing_dependencies)
    assert energy_kev.shape == (128,)
    mpq_calc_lut_file = processing_dependencies.get_file_paths(
        descriptor="l2-lo-onboard-mpq-cal"
    )[0]
    mpq_df = pd.read_csv(mpq_calc_lut_file, header=None)
    expected_e_kev = mpq_df.loc[5, 4:].to_numpy().astype(np.float64)
    # Calculated values should be more precise than LUT but should be close
    np.testing.assert_allclose(energy_kev, expected_e_kev, rtol=0.01)


def test_get_hi_de_luts(processing_dependencies, mock_get_file_paths):
    # Mock get_file_paths to return specific files for hi-energy-table and hi-tof-table
    energy_table, tof_table = get_hi_de_luts(processing_dependencies)
    assert energy_table.shape == (2048, 48)
    assert tof_table.shape == (1024, 2)


def test_process_lo_species_intensity(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    l1b_data = process_codice_l1b(processed_l1a_file)
    l1b_val_data_processed = l1b_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_data.epoch), 128, 24)) * 2,
        dims=("epoch", "esa_step", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(np.ones((128, 24)) * 2, dims=("esa_step", "inst_az")),
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
            l1b_data[var]
            / (len_pos * 4 * l1b_data["energy_per_charge"].data)[
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
            "energy_per_charge": (("esa_step",), np.ones(128) * 10),
            "packet_version": ("epoch", np.ones(5)),
            "half_spin_per_esa_step": (("epoch", "esa_step"), np.ones((5, 128)) * 2),
            "rgfo_half_spin": ("epoch", np.ones(5) * 2),
        }
    )

    l1b_val_data_processed = l1b_val_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_val_data.epoch), 128, 24)) * 2,
        dims=("epoch", "energy_per_charge", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(
            np.ones((128, 24)) * 2, dims=("energy_per_charge", "inst_az")
        ),
    ):
        len_pos = 5
        with pytest.raises(ValueError, match="Species hplus not found in dataset"):
            process_lo_species_intensity(
                l1b_val_data_processed,
                LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
                gf,
                None,
                list(np.arange(0, len_pos)),
            )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_codice_l2_sw_species_intensity(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    processed_l1b_file = write_cdf(process_codice_l1b(processed_l1a_file))
    # Mock get_files for l2
    mock_get_file_paths.side_effect = [
        [processed_l1b_file.as_posix()],
        codice_lut_path(descriptor="l2-lo-gfactor"),
        codice_lut_path(descriptor="l2-lo-efficiency"),
    ]
    processed_2_ds = process_codice_l2("lo-sw-species", ProcessingInputCollection())
    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / (
            f"imap_codice_l2_lo-sw-species_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    l2_val_data = load_cdf(l2_val_data)
    for variable in l2_val_data.data_vars:
        processed_val = processed_2_ds[variable].values
        # NOTE: Replace nan with 0 for comparison as the validation data uses 0
        processed_val[np.isnan(processed_val)] = 0.0
        np.testing.assert_allclose(
            processed_val,
            l2_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    processed_2_ds.attrs["Data_version"] = "001"
    assert processed_2_ds.attrs["Logical_source"] == "imap_codice_l2_lo-sw-species"
    cdf_path = write_cdf(processed_2_ds)
    assert_l2_epoch_delta_cdf_metadata(cdf_path)
    with cdflib.CDF(cdf_path) as cdf_file:
        hplus_attrs = cdf_file.varattsget("hplus")
        assert (
            hplus_attrs["CATDESC"] == "Differential intensity for sunward solar-wind H+"
        )
        assert hplus_attrs["FIELDNAM"] == "Sunward Differential Intensity - H+"
        unc_hplus_attrs = cdf_file.varattsget("unc_hplus")
        assert (
            unc_hplus_attrs["CATDESC"]
            == "Uncertainty in differential intensity for sunward solar-wind H+"
        )
        assert unc_hplus_attrs["FIELDNAM"] == "Sunward Uncertainty - H+"
        for var in ["nso_esa_step", "nso_spin_sector"]:
            var_info = cdf_file.varinq(var)
            var_attrs = cdf_file.varattsget(var)
            assert var_info.Data_Type_Description == "CDF_UINT1"
            assert var_attrs["FILLVAL"] == np.uint8(255)


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_codice_l2_lo_de(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-direct-events", data_type="l0")
    ]
    l1a_cdf = process_l1a(ProcessingInputCollection())[0]

    processed_l1a_file = write_cdf(l1a_cdf)
    file_path = processed_l1a_file.as_posix()
    # Mock get_files for l2
    mock_get_file_paths.side_effect = [
        [file_path],
        [file_path],
        codice_lut_path(descriptor="l2-lo-onboard-energy-table"),
        codice_lut_path(descriptor="l2-lo-onboard-energy-bins"),
        codice_lut_path(descriptor="l2-lo-onboard-mpq-cal"),
        codice_lut_path(descriptor="l2-lo-onboard-mpq-cal"),
    ]

    processed_l2_ds = process_codice_l2("lo-direct-events", ProcessingInputCollection())
    l1a_input_ds = load_cdf(processed_l1a_file)
    original_spin_sector = l1a_input_ds["spin_sector"].values
    # Mirror the LO direct-event spin-sector remapping so this test catches any
    # unintended changes to valid sector values while still checking that only
    # invalid sectors are replaced with the uint8 fill value.
    expected_spin_sector = np.where(
        (l1a_input_ds["position"].values >= 13)
        & (l1a_input_ds["position"].values <= 24),
        (original_spin_sector + 12) % 24,
        original_spin_sector,
    )
    invalid_spin_sector = ~np.isfinite(original_spin_sector) | (
        original_spin_sector > 23
    )
    expected_spin_sector = np.where(
        invalid_spin_sector, np.uint8(255), expected_spin_sector
    ).astype(np.uint8)
    assert processed_l2_ds["spin_sector"].dtype == np.uint8
    np.testing.assert_array_equal(
        processed_l2_ds["spin_sector"].values,
        expected_spin_sector,
        err_msg="LO direct-event spin_sector values changed unexpectedly",
    )

    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / (
            f"imap_codice_l2_lo-direct-events_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    l2_val_data = load_cdf(l2_val_data)

    for variable in l2_val_data.data_vars:
        if variable in ["spin_angle", "spin_sector"]:
            # TODO remove this block when joey fixes spin_angle and spin_sector
            #  calculation. Currently they are not setting spin sector and spin angles
            #  to NaNs for invalid positions.
            continue  # skip spin_angle
        if "label" in variable:
            np.testing.assert_array_equal(
                processed_l2_ds[variable].values,
                l2_val_data[variable].values,
                err_msg=f"Mismatch in variable '{variable}'",
            )
        else:
            np.testing.assert_allclose(
                processed_l2_ds[variable].values,
                l2_val_data[variable].values,
                rtol=5e-5,
                err_msg=f"Mismatch in variable '{variable}'",
                equal_nan=True,
            )
    processed_l2_ds.attrs["Data_version"] = "001"
    assert processed_l2_ds.attrs["Logical_source"] == "imap_codice_l2_lo-direct-events"
    file = write_cdf(processed_l2_ds)
    assert_l2_epoch_delta_cdf_metadata(file)
    errors = CDFValidator().validate(file)
    assert not errors
    cdf_file = cdflib.CDF(file)
    spin_sector_info = cdf_file.varinq("spin_sector")
    spin_sector_attrs = cdf_file.varattsget("spin_sector")
    data_quality_info = cdf_file.varinq("data_quality")
    data_quality_attrs = cdf_file.varattsget("data_quality")
    assert spin_sector_info.Data_Type_Description == "CDF_UINT1"
    assert spin_sector_attrs["FILLVAL"] == np.uint8(255)
    assert data_quality_info.Data_Type_Description == "CDF_UINT2"
    assert data_quality_attrs["FILLVAL"] == np.uint16(65535)
    load_cdf(file)


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_codice_l2_hi_de(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-direct-events", data_type="l0")
    ]
    l1a_cdf = process_l1a(ProcessingInputCollection())[0]

    processed_l1a_file = write_cdf(l1a_cdf)
    file_path = processed_l1a_file.as_posix()
    # Mock get_files for l2
    mock_get_file_paths.side_effect = [
        [file_path],
        [file_path],
        codice_lut_path(descriptor="l2-hi-energy-table"),
        codice_lut_path(descriptor="l2-hi-tof-table"),
    ]

    processed_l2_ds = process_codice_l2("hi-direct-events", ProcessingInputCollection())
    l2_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / (
            f"imap_codice_l2_hi-direct-events_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    l2_val_data = load_cdf(l2_val_data)
    for variable in l2_val_data.data_vars:
        if "label" in variable:
            np.testing.assert_array_equal(
                processed_l2_ds[variable].values,
                l2_val_data[variable].values,
                err_msg=f"Mismatch in variable '{variable}'",
            )
        else:
            np.testing.assert_allclose(
                processed_l2_ds[variable].values,
                l2_val_data[variable].values,
                rtol=5e-5,
                err_msg=f"Mismatch in variable '{variable}'",
            )

    processed_l2_ds.attrs["Data_version"] = "001"
    assert processed_l2_ds.attrs["Logical_source"] == "imap_codice_l2_hi-direct-events"
    file = write_cdf(processed_l2_ds)
    assert_l2_epoch_delta_cdf_metadata(file)
    errors = CDFValidator().validate(file)
    assert not errors
    cdf_file = cdflib.CDF(file)
    data_quality_info = cdf_file.varinq("data_quality")
    data_quality_attrs = cdf_file.varattsget("data_quality")
    assert data_quality_info.Data_Type_Description == "CDF_UINT2"
    assert data_quality_attrs["FILLVAL"] == np.uint16(65535)
    load_cdf(file)


@pytest.mark.parametrize("descriptor", ["lo-direct-events", "hi-direct-events"])
@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_codice_l2_direct_events_display_type_cdf_metadata(
    mock_get_file_paths, codice_lut_path, descriptor
):
    file = _generate_direct_events_l2_file(
        mock_get_file_paths, codice_lut_path, descriptor
    )
    with cdflib.CDF(str(file)) as cdf_file:
        for variable, expected_display_type in DIRECT_EVENT_DISPLAY_TYPES[
            descriptor
        ].items():
            attrs = cdf_file.varattsget(variable)
            assert "DISPLAY_TYPE" in attrs, f"{variable} is missing DISPLAY_TYPE"
            assert isinstance(attrs["DISPLAY_TYPE"], str), (
                f"{variable} DISPLAY_TYPE must be stored as a string"
            )
            assert attrs["DISPLAY_TYPE"] == expected_display_type

        if descriptor == "hi-direct-events":
            attrs = cdf_file.varattsget("energy_per_nuc")
            assert attrs["DEPEND_1"] == "priority"
            assert attrs["LABL_PTR_1"] == "priority_label"
