"""Tests the L1b processing for CoDICE L1a data"""

from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)

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

    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_lo-sw-species_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                l1b_val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-sw-species_{VALIDATION_FILE_DATE}_v002.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_nsw_species(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-species."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_lo-nsw-species_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                l1b_val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-nsw-species_{VALIDATION_FILE_DATE}_v002.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_sw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-angular."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_lo-sw-angular_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                l1b_val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-sw-angular_{VALIDATION_FILE_DATE}_v002.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_lo_nsw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-angular."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_lo-nsw-angular_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                l1b_val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # Write to CDF
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-nsw-angular_{VALIDATION_FILE_DATE}_v002.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_hi_omni(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-omni", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    l1a_file_path = write_cdf(process_l1a(dependency=ProcessingInputCollection())[0])
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / f"imap_codice_l1b_hi-omni_{VALIDATION_FILE_DATE}"
        f"_{VALIDATION_FILE_VERSION}.cdf"
    )
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=l1a_file_path)
    # hi-omni has species-specific shapes
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1.5e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == f"imap_codice_l1b_hi-omni_{VALIDATION_FILE_DATE}_v999.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_hi_sectored(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-sectored", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / f"imap_codice_l1b_hi-sectored_{VALIDATION_FILE_DATE}"
        f"_{VALIDATION_FILE_VERSION}.cdf"
    )
    l1a_file_path = write_cdf(process_l1a(dependency=ProcessingInputCollection())[0])
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=l1a_file_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1.2e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert (
        cdf_file.name == f"imap_codice_l1b_hi-sectored_{VALIDATION_FILE_DATE}_v999.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_hi_priorities(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-priorities", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / f"imap_codice_l1b_hi-priority_{VALIDATION_FILE_DATE}"
        f"_{VALIDATION_FILE_VERSION}.cdf"
    )
    l1a_ds = process_l1a(ProcessingInputCollection())[0]
    l1a_file_path = write_cdf(l1a_ds)
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=l1a_file_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1.2e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert (
        cdf_file.name == f"imap_codice_l1b_hi-priority_{VALIDATION_FILE_DATE}_v999.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_nsw_lo_priorities(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / f"imap_codice_l1b_lo-nsw-priority_{VALIDATION_FILE_DATE}"
        f"_{VALIDATION_FILE_VERSION}.cdf"
    )
    l1a_ds = process_l1a(ProcessingInputCollection())[0]
    l1a_file_path = write_cdf(l1a_ds)
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=l1a_file_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-nsw-priority_{VALIDATION_FILE_DATE}_v999.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l1b_sw_lo_priorities(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1b_validation/"
        / f"imap_codice_l1b_lo-sw-priority_{VALIDATION_FILE_DATE}"
        f"_{VALIDATION_FILE_VERSION}.cdf"
    )
    l1a_ds = process_l1a(ProcessingInputCollection())[0]
    l1a_file_path = write_cdf(l1a_ds)
    val_data = load_cdf(val_path)
    processed_data = process_codice_l1b(file_path=l1a_file_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert (
        cdf_file.name
        == f"imap_codice_l1b_lo-sw-priority_{VALIDATION_FILE_DATE}_v999.cdf"
    )
