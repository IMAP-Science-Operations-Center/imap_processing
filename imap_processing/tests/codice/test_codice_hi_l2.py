from unittest.mock import patch

import numpy as np
import pytest
from imap_data_access.processing_input import AncillaryInput, ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l2 import (
    process_codice_l2,
)

pytestmark = pytest.mark.external_test_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l2_hi_omni(mock_get_file_paths, codice_lut_path):
    # Ensure mocked ProcessingInputCollection.get_file_paths returns LUT paths
    mock_get_file_paths.side_effect = codice_lut_path
    input_data = (
        imap_module_directory
        / "tests/codice/data/l1b_validation"
        / "imap_codice_l1b_hi-omni_20250814211100_v0.0.6.cdf"
    )

    anc_input = AncillaryInput("imap_codice_l2-hi-omni-efficiency_20251008_v001.csv")
    dependencies = ProcessingInputCollection(anc_input)

    processed_l2 = process_codice_l2(input_data, dependencies)

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / "imap_codice_l2_hi-omni_20250814211100_v0.0.6.cdf"
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


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_l2_hi_sectored(mock_get_file_paths, codice_lut_path):
    # Ensure mocked ProcessingInputCollection.get_file_paths returns LUT paths
    mock_get_file_paths.side_effect = codice_lut_path
    input_data = (
        imap_module_directory
        / "tests/codice/data/l1b_validation"
        / "imap_codice_l1b_hi-sectored_20250814211100_v0.0.6.cdf"
    )

    anc_input = AncillaryInput(
        "imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv"
    )
    dependencies = ProcessingInputCollection(anc_input)

    processed_l2 = process_codice_l2(input_data, dependencies)

    val_data = (
        imap_module_directory
        / "tests/codice/data/l2_validation"
        / "imap_codice_l2_hi-sectored_20250814211100_v0.0.6.cdf"
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
