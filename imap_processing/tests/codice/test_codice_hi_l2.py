from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
)

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l2 import (
    process_codice_l2,
)

pytestmark = pytest.mark.external_test_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l2_hi_omni(mock_get_file_paths, codice_lut_path):
    mock_get_file_paths.side_effect = codice_lut_path
    sci_input = ScienceInput("imap_codice_l1b_hi-omni_20250814_v006.cdf")
    anc_input = AncillaryInput("imap_codice_l2-hi-omni-efficiency_20251008_v001.csv")
    dependencies = ProcessingInputCollection(anc_input, sci_input)

    processed_l2 = process_codice_l2("hi-omni", dependencies)

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / "imap_codice_l2_hi-omni_20250814_v006.cdf"
    )

    val_data = load_cdf(val_data)
    for variable in val_data.data_vars:
        if variable.startswith("unc_"):
            continue
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1e-5,
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
    assert omni_cdf_file.name == "imap_codice_l2_hi-omni_20250814_v001.cdf"


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l2_hi_sectored(mock_get_file_paths, codice_lut_path):
    # Ensure mocked ProcessingInputCollection.get_file_paths returns LUT paths
    mock_get_file_paths.side_effect = codice_lut_path

    anc_input = AncillaryInput(
        "imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv"
    )
    sci_input = ScienceInput("imap_codice_l1b_hi-sectored_20250814_v006.cdf")
    dependencies = ProcessingInputCollection(anc_input, sci_input)

    processed_l2 = process_codice_l2("hi-sectored", dependencies)

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / "imap_codice_l2_hi-sectored_20250814_v006.cdf"
    )

    val_data = load_cdf(val_data)

    # Check data variables
    for variable in val_data.data_vars:
        if variable.startswith("unc_"):
            continue
        np.testing.assert_allclose(
            processed_l2[variable].values,
            val_data[variable].values,
            rtol=1e-5,
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
    assert sectored_cdf_file.name == "imap_codice_l2_hi-sectored_20250814_v001.cdf"
