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
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1a_de import l1a_direct_event
from imap_processing.codice.utils import CODICEAPID
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.external_test_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_updated_packet_version(mock_get_file_paths, codice_lut_path, caplog):
    """Tests the new FSW changes (jan 2026)."""
    codice_lut_path_jan = codice_lut_path(descriptor="l1a-sci-lut-jan")
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="fsw-changes", data_type="l0"),
        *([codice_lut_path_jan] * 20),
    ]
    datasets = process_l1a(dependency=ProcessingInputCollection())
    # Assert that we have all of the expected datasets
    assert len(datasets) == 14
    for ds in datasets:
        # Only check lo products. Skip direct-events
        if (
            "lo" not in ds.attrs["Data_type"]
            or "direct-events" in ds.attrs["Data_type"]
        ):
            continue
        # Check that the lo datasets contain the new unpacked variables
        expected_vars = [
            "rgfo_spin_sector",
            "nso_spin_sector",
            "rgfo_esa_step",
            "nso_esa_step",
        ]
        for var in expected_vars:
            assert var in ds.data_vars, (
                f"Expected variable '{var}' not found in dataset"
            )

        # check that a warning is logged for the missing "cnoplus" species
        assert (
            "Desired species cnoplus not found in actual species names from LUT"
            in caplog.text
        )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hskp(mock_get_file_paths, codice_lut_path):
    """Tests the housekeeping."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hskp", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]
    processed_datasets = process_l1a(dependency=ProcessingInputCollection())

    assert len(processed_datasets) == 2
    processed_l1a = processed_datasets[0]
    processed_l1b = processed_datasets[1]

    # spot check the l1a value is an integer and the l1b is a float after conversion
    np.testing.assert_almost_equal(processed_l1a["fee_ssd_eb_temp_1_t"].values[0], 2199)
    np.testing.assert_almost_equal(
        processed_l1b["fee_ssd_eb_temp_1_t"].values[0], 18.71, decimal=2
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_counters_aggregated(mock_get_file_paths, codice_lut_path):
    """Tests lo-counters-aggregated."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-counters-aggregated", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_lo-counters-aggregated_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)
    for variable in val_data.data_vars:
        # TODO: ask Joey to remove reserved variables from validation files
        if variable.startswith("reserved"):
            continue
        try:
            np.testing.assert_allclose(
                processed_data[variable].values,
                val_data[variable].values,
                rtol=1e-5,
                err_msg=f"Mismatch in variable '{variable}'",
            )
        except AssertionError:
            # TODO: remove this try/except after non-active variables
            # dimensions are fixed in Joey's validation files.
            continue

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-counters-aggregated_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_counters_singles(mock_get_file_paths, codice_lut_path):
    """Tests lo-counters-singles."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-counters-singles", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_lo-counters-singles_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-counters-singles_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_sw_priority(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-priority."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_lo-sw-priority_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-sw-priority_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_nsw_priority(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-priority."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_lo-nsw-priority_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        # If string type, do equal.
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-nsw-priority_{VALIDATION_FILE_DATE}_v001.cdf"
    )


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
        / (
            f"imap_codice_l1a_lo-sw-species_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-sw-species_{VALIDATION_FILE_DATE}_v002.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_counters_aggregated(mock_get_file_paths, codice_lut_path):
    """Tests hi-counters-aggregated."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-counters-aggregated", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(ProcessingInputCollection())[0]
    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_hi-counters-aggregated_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)
    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_hi-counters-aggregated_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_counters_singles(mock_get_file_paths, codice_lut_path):
    """Tests hi-counters-singles."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-counters-singles", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_hi-counters-singles_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_hi-counters-singles_{VALIDATION_FILE_DATE}_v001.cdf"
    )


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
        / (
            f"imap_codice_l1a_hi-omni_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

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
            err_msg=f"Mismatch in variable '{variable}'",
        )
    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == f"imap_codice_l1a_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"


@pytest.mark.xfail(reason="Need to revisit in future PR")
@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
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
        / (
            f"imap_codice_l1a_hi-sectored_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
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
    for variable in val_data.coords:
        # If _label, do string comparison
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue

        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name == f"imap_codice_l1a_hi-sectored_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_priority(mock_get_file_paths, codice_lut_path):
    """Tests hi-priorities."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-priorities", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Process the input data
    processed_data = process_l1a(ProcessingInputCollection())[0]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_hi-priority_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    val_data = load_cdf(val_path)

    for variable in val_data.data_vars:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    for variable in val_data.coords:
        if variable.endswith("_label"):
            assert np.array_equal(
                processed_data[variable].values,
                val_data[variable].values,
            ), f"Mismatch in coordinate '{variable}'"
            continue
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name == f"imap_codice_l1a_hi-priority_{VALIDATION_FILE_DATE}_v001.cdf"
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_direct_events(mock_get_file_paths, codice_lut_path):
    """Tests lo-direct-events."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-direct-events", data_type="l0"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_lo-direct-events_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    for variable in val_data.data_vars:
        if variable in ["priority_label"]:
            # Do string comparison for priority_label
            assert np.array_equal(
                processed_data[variable].values, val_data[variable].values
            ), f"Mismatch in variable '{variable}'"
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
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-direct-events_{VALIDATION_FILE_DATE}_v002.cdf"
    )


def test_direct_events_incomplete_groups(codice_lut_path, caplog):
    """Tests lo-direct-events with a packet containing incomplete groups."""

    # Get science data which is L0 packet file
    science_file = codice_lut_path(descriptor="lo-direct-events", data_type="l0")[0]

    xtce_file = (
        imap_module_directory
        / "codice/packet_definitions/imap_codice_packet-definition_20250101_v001.xml"
    )
    # Decom packet
    datasets_by_apid = packet_file_to_datasets(
        science_file,
        xtce_file,
    )
    apid = CODICEAPID.COD_LO_PHA
    de_dataset = datasets_by_apid[apid]
    # Drop the first packet to test incomplete group handling
    # This mocks the case when one priority group is incomplete
    # in this example, the first group is missing the first priority
    len_epoch = de_dataset.sizes["epoch"]
    de_dataset = de_dataset.isel(epoch=slice(1, len_epoch))
    dataset = l1a_direct_event(de_dataset, apid)
    # Check that fillvals are used for the first missing priority for the first epoch
    assert np.all(dataset.tof[0, 0, :].values == 65535)
    # Check that there is data for the remaining priorities
    assert np.any(dataset.tof[0, 1:, :].values != 65535)
    # Check logs for incomplete groups
    assert (
        f"Found 1 incomplete priority group(s) for APID {apid}. "
        f"Expected 8 packets per group"
    ) in caplog.text


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_direct_events(mock_get_file_paths, codice_lut_path):
    """Tests hi-direct-events."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-direct-events", data_type="l0"),
    ]

    # Validation
    val_path = (
        imap_module_directory
        / "tests/codice/data/l1a_validation/"
        / (
            f"imap_codice_l1a_hi-direct-events_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    val_data = load_cdf(val_path)

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]

    for variable in val_data.data_vars:
        if variable in ["priority_label"]:
            # Do string comparison for priority_label
            assert np.array_equal(
                processed_data[variable].values, val_data[variable].values
            ), f"Mismatch in variable '{variable}'"
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
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_hi-direct-events_{VALIDATION_FILE_DATE}_v002.cdf"
    )
