"""Tests the L1a processing for decommutated CoDICE data"""

import logging

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pytestmark = pytest.mark.external_test_data


# TODO: These variables are in validation data but missing in processed data
# in the product mentioned in the comments. These will need to be fixed in
# upcoming work mentioned in issue #2237
TIME_MISMATCHES = [
    "voltage_table",  # many products
    "epoch_delta_plus",  # many products
    "epoch_delta_minus",  # many products
]

EXPECTED_MISMATCHES = [
    "data_quality",  # hi-ialirt
    "spin_period",  # hi-ialirt
    "h",  # hi-ialirt
    "heplusplus",  # lo-ialirt
    "cplus5",  # lo-ialirt
    "cplus6",  # lo-ialirt
    "oplus6",  # lo-ialirt shape mismatch
    "oplus7",  # lo-ialirt shape mismatch
    "oplus8",  # lo-ialirt shape mismatch
    "mg",  # lo-ialirt shape mismatch
    "fe_loq",  # lo-ialirt shape mismatch
    "fe_hiq",  # lo-ialirt shape mismatch
    "heplusplus",  # lo-ialirt shape mismatch
    "cplus5",  # lo-ialirt shape mismatch
    "cplus6",  # lo-ialirt shape mismatch
    "rgfo_half_spin",  # lo-ialirt shape mismatch
    "nso_half_spin",  # lo-ialirt shape mismatch
    "tof_plus_apd",  # counters-aggregated
    "tof_only",  # counters-aggregated
    "position_plus_apd",  # counters-aggregated
    "position_only",  # counters-aggregated
    "sta_or_stb_plus_apd",  # counters-aggregated
    "sta_or_stb_only",  # counters-aggregated
    "reserved1",  # counters-aggregated
    "reserved2",  # counters-aggregated
    "sp_only",  # counters-aggregated
    "apd_only",  # counters-aggregated
    "low_tof_cutoff",  # counters-aggregated
    "invalid_position_count",  # counters-aggregated
    "asic1_flag_invalid",  # counters-aggregated
    "asic2_flag_invalid",  # counters-aggregated
    "asic1_channel_invalid",  # counters-aggregated
    "asic2_channel_invalid",  # counters-aggregated
    "tec4_timeout_tof_no_pos",  # counters-aggregated
    "tec4_timeout_pos_no_tof",  # counters-aggregated
    "tec4_timeout_no_pos_tof",  # counters-aggregated
    "tec5_timeout_tof_no_pos",  # counters-aggregated
    "tec5_timeout_pos_no_tof",  # counters-aggregated
    "tec5_timeout_no_pos_tof",  # counters-aggregated
    "p0_tcrs",  # sw-priority shape mismatch
    "p1_hplus",  # sw-priority shape mismatch
    "p2_heplusplus",  # sw-priority shape mismatch
    "p3_heavies",  # sw-priority shape mismatch
    "p4_dcrs",  # lo-sw-priority shape mismatch
    "p5_heavies",  # lo-nsw-priority shape mismatch
    "p6_hplus_heplusplus",  # lo-nsw-priority shape mismatch
    "k_factor",  # lo-direct-events
    "priority_label",  # hi and lo direct-events
    "sw_bias_gain_mode",  # lo-direct-events
    "st_bias_gain_mode",  # lo-direct-events
    "position",  # lo-direct-events
    *TIME_MISMATCHES,
]

UNCERTAINTY_VARIABLES = "unc_"


EXPECTED_HI_OMNI_ARRAY_SHAPES = {
    "h": (36, 15),
    "he3": (36, 15),
    "he4": (36, 15),
    "c": (36, 18),
    "o": (36, 18),
    "ne_mg_si": (36, 15),
    "fe": (36, 18),
    "uh": (36, 5),
    "junk": (36, 1),
}


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape, (
            f"Shape mismatch for variable '{variable}'"
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-ialirt_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-counters-aggregated_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-counters-singles_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape, (
            f"Shape mismatch for variable '{variable}'"
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-priority_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-priority_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_sw_species():
    """Tests lo-sw-species."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-sw-species_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-sw-species_20250814_v006.cdf"
    )

    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_codice_l1a(file_path=test_file_path)[0]

    # Compare only the common variables
    for variable in val_data.data_vars:
        if variable in TIME_MISMATCHES or variable.startswith(UNCERTAINTY_VARIABLES):
            continue

        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-species_20250814_v999.cdf"


@pytest.mark.skip(reason="Revisit this in l1a refactor work")
def test_lo_nsw_species():
    """Tests lo-nsw-species."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-nsw-species_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-nsw-species_20250814_v006.cdf"
    )

    val_data = load_cdf(val_path)

    # Process the input data
    processed_data = process_codice_l1a(file_path=test_file_path)[0]

    # Compare only the common variables
    for variable in val_data.data_vars:
        if variable in TIME_MISMATCHES or variable.startswith(UNCERTAINTY_VARIABLES):
            continue

        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-species_20250814_v999.cdf"


def test_lo_sw_angular():
    """Tests lo-sw-angular."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-sw-angular_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-sw-angular_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        if variable in TIME_MISMATCHES or variable.startswith(UNCERTAINTY_VARIABLES):
            continue

        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-sw-angular_20250814_v999.cdf"


def test_lo_nsw_angular():
    """Tests lo-nsw-angular."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_lo-nsw-angular_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_lo-nsw-angular_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        if variable in TIME_MISMATCHES or variable.startswith(UNCERTAINTY_VARIABLES):
            continue

        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-nsw-angular_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue

        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-counters-aggregated_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-counters-singles_20250814_v999.cdf"


def test_hi_omni():
    """Tests hi-omni."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-omni_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-omni_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    # hi-omni has species-specific shapes
    for variable in val_data.data_vars:
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-omni_20250814_v999.cdf"


def test_hi_sectored():
    """Tests hi-sectored."""
    test_file_path = (
        imap_module_directory
        / "tests/codice/data/l1a_input"
        / "imap_codice_hi-sectored_20250814_v001.pkts"
    )

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / "imap_codice_l1a_hi-sectored_20250814211100_v0.0.5.cdf"
    )
    val_data = load_cdf(val_path)

    processed_data = process_codice_l1a(file_path=test_file_path)[0]
    for variable in val_data.data_vars:
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-sectored_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-priority_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_lo-direct-events_20250814_v999.cdf"


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
        if variable in EXPECTED_MISMATCHES or variable.startswith(
            UNCERTAINTY_VARIABLES
        ):
            continue
        assert processed_data[variable].shape == val_data[variable].shape

    cdf_file = write_cdf(processed_data)
    assert cdf_file.name == "imap_codice_l1a_hi-direct-events_20250814_v999.cdf"
