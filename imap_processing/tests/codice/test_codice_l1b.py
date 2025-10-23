"""Tests the L1b processing for CoDICE L1a data"""

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b

pytestmark = pytest.mark.external_test_data

TIME_MISMATCHES = [
    "voltage_table",  # many products
    "epoch_delta_plus",  # many products
    "epoch_delta_minus",  # many products
]


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_lo_sw_species():
    l0_test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-sw-species_20250814_v001.pkts"
    )

    processed_l1a = process_codice_l1a(l0_test_file_path)
    processed_l1a_file = write_cdf(processed_l1a[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814_v006.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
            continue
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


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_lo_nsw_species():
    l0_test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-nsw-species_20250814_v001.pkts"
    )

    processed_l1a = process_codice_l1a(l0_test_file_path)
    processed_l1a_file = write_cdf(processed_l1a[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-nsw-species_20250814211100_v0.0.5.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
            continue
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
    assert cdf_file.name == "imap_codice_l1b_lo-nsw-species_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_lo_sw_angular():
    l0_test_file = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / "imap_codice_lo-sw-angular_20250814_v001.pkts"
    )
    processed_l1a = process_codice_l1a(l0_test_file)
    processed_l1a_file = write_cdf(processed_l1a[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-angular_20250814_v005.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
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
    assert cdf_file.name == "imap_codice_l1b_lo-sw-angular_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_l1b_lo_nsw_angular():
    l0_test_file = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / "imap_codice_lo-nsw-angular_20250814_v001.pkts"
    )

    processed_l1a = process_codice_l1a(l0_test_file)
    processed_l1a_file = write_cdf(processed_l1a[0])

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-nsw-angular_20250814_v005.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(processed_l1a_file)

    for variable in l1b_val_data.data_vars:
        if variable.startswith("unc_") or variable in TIME_MISMATCHES:
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
    assert cdf_file.name == "imap_codice_l1b_lo-nsw-angular_20250814_v999.cdf"


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
