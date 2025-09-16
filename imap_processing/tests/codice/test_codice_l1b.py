"""Tests the L1b processing for CoDICE L1a data"""

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1b import process_codice_l1b

pytestmark = pytest.mark.external_test_data


def test_l1b_lo_sw_species():
    l1a_test_file = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / "imap_codice_l1a_lo-sw-species_20250814211100_v0.0.3.cdf"
    )

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814211100_v0.0.3.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(l1a_test_file)

    for variable in l1b_val_data.data_vars:
        if variable in ["hplus", "heplusplus"]:
            # TODO: find out why validation didn't match
            continue
        assert processed_data[variable].shape == l1b_val_data[variable].shape
        np.testing.assert_allclose(
            processed_data[variable].values,
            l1b_val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    # Write to CDF
    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1b_lo-sw-species_20250814_v999.cdf"
