"""Tests the L1b processing for CoDICE L1a data"""

from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access import AncillaryInput, ProcessingInputCollection, ScienceInput

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.codice.codice_new_l1a import process_l1a

pytestmark = pytest.mark.external_test_data

TIME_MISMATCHES = [
    "voltage_table",  # many products
    "epoch_delta_plus",  # many products
    "epoch_delta_minus",  # many products
]


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_sw_species(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-species."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    sci_input = ScienceInput("imap_codice_l0_lo-nsw-angular_20250814_v001.pkts")
    sci_lut_input = AncillaryInput("imap_codice_l1a-sci-lut_20251007_v001.json")
    dependency = ProcessingInputCollection(sci_input, sci_lut_input)
    processed_l1a_file = write_cdf(process_l1a(dependency)[0])
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)
    for variable in l1b_val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    for variable in l1b_val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1b_lo-sw-species_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_nsw_species(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-species."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    sci_input = ScienceInput("imap_codice_l0_lo-nsw-species_20250814_v001.pkts")
    sci_lut_input = AncillaryInput("imap_codice_l1a-sci-lut_20251007_v001.json")
    dependency = ProcessingInputCollection(sci_input, sci_lut_input)
    processed_l1a_file = write_cdf(process_l1a(dependency)[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-nsw-species_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in l1b_val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1b_lo-nsw-species_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_sw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-angular."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    sci_input = ScienceInput("imap_codice_l0_lo-sw-angular_20250814_v001.pkts")
    sci_lut_input = AncillaryInput("imap_codice_l1a-sci-lut_20251007_v001.json")
    dependency = ProcessingInputCollection(sci_input, sci_lut_input)
    processed_l1a_file = write_cdf(process_l1a(dependency)[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-angular_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        assert processed_data[variable].shape == l1b_val_data[variable].shape
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in l1b_val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1b_lo-sw-angular_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_nsw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-angular."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    sci_input = ScienceInput("imap_codice_l0_lo-nsw-angular_20250814_v001.pkts")
    sci_lut_input = AncillaryInput("imap_codice_l1a-sci-lut_20251007_v001.json")
    dependency = ProcessingInputCollection(sci_input, sci_lut_input)
    processed_l1a_file = write_cdf(process_l1a(dependency)[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-nsw-angular_20250814_v007.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in l1b_val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1b_lo-nsw-angular_20250814_v002.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_hi_omni():
    l0_test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-omni_20250814_v001.pkts"
    )
    processed_l1a_file = write_cdf(process_codice_l1a(l0_test_file_path)[0])

    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / "imap_codice_l1b_hi-omni_20250814211100_v0.0.6.cdf"
    )
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=processed_l1a_file)
    # hi-omni has species-specific shapes
    for variable in val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
            continue
        assert processed_data[variable].shape == val_data[variable].shape
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1b_hi-omni_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_hi_sectored():
    l0_test_file_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / "imap_codice_hi-sectored_20250814_v001.pkts"
    )
    processed_l1a_file = write_cdf(process_codice_l1a(l0_test_file_path)[0])
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / "imap_codice_l1b_hi-sectored_20250814211100_v0.0.6.cdf"
    )

    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=processed_l1a_file)
    for variable in val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1b_hi-sectored_20250814_v999.cdf"
