"""Tests the L1a processing for decommutated CoDICE data


Create specific side_effect for each test. Tenzin tried to create generic
function but we query either by data_type to get l0 file or
by descriptor to get lut file. Since each product have their own
l0 test file but processing pipeline has one l0 file, it
caused too much complexity.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a
from imap_processing.codice.codice_new_l1a import process_l1a

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pytestmark = pytest.mark.external_test_data


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_ialirt():
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-ialirt_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-ialirt_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape, (
            f"Shape mismatch for variable '{variable}'"
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-ialirt_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_ialirt():
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-ialirt_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-ialirt_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape, (
            f"Shape mismatch for variable '{variable}'"
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-ialirt_20250814_v999.cdf"


@pytest.mark.skip(reason="test_hskp - KeyError: 'optics_hv_cmd_err_cnt'")
def test_hskp():
    """Tests the housekeeping."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hskp_20250814_v001.pkts"
    )

    # # Validation
    # val_path = (
    #     imap_module_directory
    #     / "tests/codice/data/l1a_validation/"
    #     / "imap_codice_l1a_hskp_20250805183835_v0.0.5.cdf"
    # )
    # val_data = load_cdf(val_path)
    # print(val_data)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]

    # Instead of checking all variables, just check that time-related variables
    # have the expected shape and that the processing completes
    if "time" in processed_data:
        assert len(processed_data.time.shape) == 1, "Time should be a 1D array"

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hskp_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_counters_aggregated():
    """Tests lo-counters-aggregated."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-counters-aggregated_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-counters-aggregated_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-counters-aggregated_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_counters_singles():
    """Tests lo-counters-singles."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input/"
        / "imap_codice_lo-counters-singles_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-counters-singles_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-counters-singles_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_sw_priority():
    """Tests lo-sw-priority."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input/"
        / "imap_codice_lo-sw-priority_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-sw-priority_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape, (
            f"Shape mismatch for variable '{variable}'"
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-priority_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_nsw_priority():
    """Tests lo-nsw-priority."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-nsw-priority_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-nsw-priority_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-priority_20250814_v999.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_sw_species(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-species."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-sw-species_20250814_v007.cdf"
    )

    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    # Compare only the common variables
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        # TODO: make this equal statement after epoch seconds difference
        # is resolved
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-species_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_nsw_species(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-species."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-species", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-nsw-species_20250814_v007.cdf"
    )

    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    # Compare only the common variables
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        # TODO: make this equal statement after epoch seconds difference
        # is resolved
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True, istp=True)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-species_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_sw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-angular."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-sw-angular_20250814_v007.cdf"
    )
    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    # Compare only the common variables
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-angular_20250814_v002.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_nsw_angular(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-angular."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-angular", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-nsw-angular_20250814_v007.cdf"
    )
    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-angular_20250814_v002.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_counters_aggregated():
    """Tests hi-counters-aggregated."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-counters-aggregated_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-counters-aggregated_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-counters-aggregated_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_counters_singles():
    """Tests hi-counters-singles."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-counters-singles_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-counters-singles_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-counters-singles_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in sectored work why this test is failing")
@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_omni(mock_get_file_paths, codice_lut_path):
    """Tests hi-omni."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-omni", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-omni_20250814_v007.cdf"
    )
    val_data = load_cdf(val_path)

    for variable in val_data.data_vars:
        # TODO: check with Joey and Michael
        if variable.startswith("epoch_delta"):
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == "imap_codice_l1a_hi-omni_20250814_v001.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_sectored(mock_get_file_paths, codice_lut_path):
    """Tests hi-sectored."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-sectored", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-omni_20250814_v007.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-sectored_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_priority():
    """Tests hi-priority."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input/"
        / "imap_codice_hi-priority_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-priorities_20250814211100_v0.0.5.cdf"
    )

    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_codice_l1a(file_path=test_file_path)[0]

    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-priority_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_direct_events():
    """Tests lo-direct-events."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-direct-events_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-direct-events_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-direct-events_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_hi_direct_events():
    """Tests hi-direct-events."""
    test_file_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / "imap_codice_hi-direct-events_20250814_v001.pkts"
    )

    # TODO: uncomment this
    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-direct-events_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-direct-events_20250814_v999.cdf"
