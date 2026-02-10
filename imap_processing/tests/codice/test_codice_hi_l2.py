from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access.processing_input import (
    ProcessingInputCollection,
)

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.codice.codice_l2 import (
    process_codice_l2,
)
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)

pytestmark = pytest.mark.external_test_data


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


def test_l2_hi_omni(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-omni", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    processed_l1b_file = write_cdf(process_codice_l1b(processed_l1a_file))
    # Mock get_files for l2
    mock_get_file_paths.side_effect = [
        [processed_l1b_file.as_posix()],
        [processed_l1b_file.as_posix()],
        codice_lut_path(descriptor="l2-hi-omni-efficiency"),
    ]

    processed_l2 = process_codice_l2("hi-omni", ProcessingInputCollection())

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf"
    )

    val_data = load_cdf(val_data)
    for variable in val_data.data_vars:
        if variable.startswith("unc_"):
            continue
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1.2e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    # Check coordinates
    for variable in val_data.coords:
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
        # Tests that dimensions match
        assert processed_l2[variable].dims == val_data[variable].dims, (
            f"Dimension mismatch in coordinate '{variable}'"
        )

    processed_l2.attrs["Data_version"] = "001"
    omni_cdf_file = write_cdf(processed_l2)
    assert (
        omni_cdf_file.name == f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"
    )


def test_l2_hi_sectored(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-sectored", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    processed_l1b_file = write_cdf(process_codice_l1b(processed_l1a_file))
    # Mock get_files for l2
    mock_get_file_paths.side_effect = [
        [processed_l1b_file.as_posix()],
        [processed_l1b_file.as_posix()],
        codice_lut_path(descriptor="l2-hi-sectored-efficiency"),
    ]

    processed_l2 = process_codice_l2("hi-sectored", ProcessingInputCollection())

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / (
            f"imap_codice_l2_hi-sectored_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    val_data = load_cdf(val_data)
    # TODO fix validation data to have correct array name. Spin_angles -> spin_angle
    val_data = val_data.rename({"spin_angles": "spin_angle"})
    # Check data variables
    for variable in val_data.data_vars:
        if variable.startswith("unc_"):
            continue
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1.2e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
        # Tests that dimensions match
        if variable in ["epoch_delta_plus", "epoch_delta_minus"]:
            continue
        assert processed_l2[variable].dims == val_data[variable].dims, (
            f"Dimension mismatch in variable '{variable}'"
        )

    # Check coordinates
    for variable in val_data.coords:
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_l2[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
        # Tests that dimensions match
        assert processed_l2[variable].dims == val_data[variable].dims, (
            f"Dimension mismatch in coordinate '{variable}'"
        )

    processed_l2.attrs["Data_version"] = "001"
    sectored_cdf_file = write_cdf(processed_l2)
    assert (
        sectored_cdf_file.name
        == f"imap_codice_l2_hi-sectored_{VALIDATION_FILE_DATE}_v001.cdf"
    )
