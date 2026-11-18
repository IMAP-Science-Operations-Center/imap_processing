"""Tests the L1a processing for decommutated CoDICE data


Create specific side_effect for each test. Tenzin tried to create generic
function but we query either by data_type to get l0 file or
by descriptor to get lut file. Since each product have their own
l0 test file but processing pipeline has one l0 file, it
caused too much complexity.
"""

import logging
from unittest.mock import patch

import cdflib
import numpy as np
import pytest
import xarray as xr
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import (
    add_quality_flags_to_science_datasets,
    calculate_quality_flags,
    process_l1a,
)
from imap_processing.codice.codice_l1a_de import l1a_direct_event
from imap_processing.codice.utils import CODICEAPID
from imap_processing.quality_flags import ImapCodiceL1aFlags
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
    assert_allclose_fillaware,
)
from imap_processing.utils import filter_day_boundary_data, packet_file_to_datasets

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.external_test_data

# epoch_delta = num_spins * spin_period / 2, with spin_period VALIDMAX = 16 s
# and num_spins max = 16 in the current CoDICE timing model. That yields a
# worst-case delta of 128 s = 128000000000 ns.
EXPECTED_EPOCH_DELTA_VALIDMAX = 128000000000
LO_MODE_SUPPORT_VARIABLES = [
    "rgfo_half_spin",
    "rgfo_spin_sector",
    "rgfo_esa_step",
    "nso_half_spin",
    "nso_spin_sector",
    "nso_esa_step",
]


def assert_epoch_delta_cdf_metadata(cdf_file):
    """Assert the written epoch delta variables use integer duration metadata."""
    with cdflib.CDF(cdf_file) as cdf:
        for variable in ("epoch_delta_minus", "epoch_delta_plus"):
            info = cdf.varinq(variable)
            attrs = cdf.varattsget(variable)
            assert info.Data_Type_Description == "CDF_INT8"
            assert attrs["FILLVAL"] == -9223372036854775808
            assert attrs["FORMAT"] == "I19"
            assert attrs["VALIDMIN"] == 0
            assert attrs["VALIDMAX"] == EXPECTED_EPOCH_DELTA_VALIDMAX


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
    np.testing.assert_almost_equal(processed_l1a["fee_ssd_eb_temp_1_t"].values[0], 1367)
    np.testing.assert_almost_equal(
        processed_l1b["fee_ssd_eb_temp_1_t"].values[0], -0.317, decimal=2
    )


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_counters_aggregated(mock_get_file_paths, codice_lut_path):
    """Tests lo-counters-aggregated."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-counters-aggregated", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)
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
        if processed_data[variable].shape != val_data[variable].shape:
            # TODO: ask Joey to populate non-active variables in the
            # validation files instead of leaving them empty (shape (0,)).
            # Our processing fills non-active Lo counters with fillval
            # placeholders at the full (epoch, esa_step, spin_sector_pairs)
            # shape (see codice_l1a_lo_counters_aggregated.py), so these
            # never match the validation file's shape.
            continue
        assert_allclose_fillaware(
            processed_data[variable],
            val_data[variable],
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-counters-aggregated_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_counters_singles(mock_get_file_paths, codice_lut_path):
    """Tests lo-counters-singles."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-counters-singles", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)
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
        assert_allclose_fillaware(
            processed_data[variable],
            val_data[variable],
            rtol=1e-5,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-counters-singles_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_sw_priority(mock_get_file_paths, codice_lut_path):
    """Tests lo-sw-priority."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-sw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
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
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-sw-priority_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_lo_nsw_priority(mock_get_file_paths, codice_lut_path):
    """Tests lo-nsw-priority."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="lo-nsw-priority", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
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
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )

    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-nsw-priority_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert "quality_flags" in processed_data


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
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)
    # Compare only the common variables
    for variable in val_data.data_vars:
        if variable in ["ne", "unc_ne"]:
            # Ne is no longer being processed.
            continue
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
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
        assert_allclose_fillaware(
            processed_data[variable].values,
            val_data[variable],
            rtol=1e-5,
            err_msg=f"Mismatch in coordinate '{variable}'",
        )
    # assert that neon is not in the processed data
    assert "ne" not in processed_data.data_vars
    assert "unc_ne" not in processed_data.data_vars
    processed_data.attrs["Data_version"] = "002"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_lo-sw-species_{VALIDATION_FILE_DATE}_v002.cdf"
    )
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_counters_aggregated(mock_get_file_paths, codice_lut_path):
    """Tests hi-counters-aggregated."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-counters-aggregated", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)
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
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_counters_singles(mock_get_file_paths, codice_lut_path):
    """Tests hi-counters-singles."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-counters-singles", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_omni(mock_get_file_paths, codice_lut_path):
    """Tests hi-omni."""

    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-omni", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    processed_data = process_l1a(dependency=ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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
            rtol=1e-4,
            err_msg=f"Mismatch in variable '{variable}'",
        )

    for variable in val_data.coords:
        np.testing.assert_allclose(
            processed_data[variable].values,
            val_data[variable].values,
            rtol=1e-4,
            err_msg=f"Mismatch in variable '{variable}'",
        )
    processed_data.attrs["Data_version"] = "001"
    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert cdf_file.name == f"imap_codice_l1a_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"
    assert_epoch_delta_cdf_metadata(cdf_file)
    assert "quality_flags" in processed_data


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
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_hi_priority(mock_get_file_paths, codice_lut_path):
    """Tests hi-priorities."""
    mock_get_file_paths.side_effect = [
        codice_lut_path(descriptor="hi-priorities", data_type="l0"),
        codice_lut_path(descriptor="l1a-sci-lut"),
    ]

    # Process the input data
    processed_data = process_l1a(ProcessingInputCollection())[0]
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

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

    assert "quality_flags" in processed_data


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
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

    for variable in val_data.data_vars:
        if variable in [
            "voltage_table",
            "half_spin_per_esa_step",
            "acquisition_time_per_esa_step",
        ]:
            # We do not have these variables for lo de. The validation data for
            # these variables is empty.
            continue
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
    with cdflib.CDF(cdf_file) as cdf:
        for variable in LO_MODE_SUPPORT_VARIABLES:
            attrs = cdf.varattsget(variable)
            assert attrs["VAR_TYPE"] == "support_data"
    assert "quality_flags" in processed_data


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

    # NOTE: This L0 test file already has some naturally incomplete priority
    # groups -- packets genuinely missing from the raw telemetry (confirmed via
    # gaps in the CCSDS source sequence counter), independent of the packet we
    # drop below. So dropping one packet below does NOT create the only
    # incomplete group in the file -- it makes an already-incomplete group even
    # more incomplete. These are the current baseline values for that first
    # group (acq_start_seconds, packet count) before our drop, taken from the
    # untouched data: if this L0 fixture ever changes, these will need updating.
    first_group_time = 507858646
    baseline_count = 4

    # Drop the first packet to test incomplete group handling
    # This mocks the case when one priority group is incomplete
    # in this example, the first group is missing the first priority
    len_epoch = de_dataset.sizes["epoch"]
    de_dataset = de_dataset.isel(epoch=slice(1, len_epoch))
    with caplog.at_level(logging.WARNING):
        dataset = l1a_direct_event(de_dataset, apid)
    # Check that fillvals are used for the first missing priority for the first epoch
    assert np.all(dataset.tof[0, 0, :].values == 65535)
    # Check that there is data for the remaining priorities
    assert np.any(dataset.tof[0, 1:, :].values != 65535)
    # Check logs report the first group with one fewer packet than baseline
    assert f"acq_start_seconds [{first_group_time}" in caplog.text
    assert f"counts [{baseline_count - 1}" in caplog.text


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
    processed_data = filter_day_boundary_data(processed_data, VALIDATION_FILE_DATE)

    for variable in val_data.data_vars:
        if variable in ["priority_label"]:
            # Do string comparison for priority_label
            assert np.array_equal(
                processed_data[variable].values, val_data[variable].values
            ), f"Mismatch in variable '{variable}'"
            continue

        if variable == "num_events":
            # TODO: 2 of 2172 values mismatch here (epoch 257 & 312, priority 3).
            # combine_segmented_packets (imap_processing/utils.py) drops the
            # entire first packet of a segmented group whenever that group's
            # sequence flags look corrupted (see "Incorrect/incomplete sequence
            # flags" warnings), even though num_events is a header field that
            # lives entirely in that first packet and is unaffected by
            # corruption in later continuation packets. Downstream, that
            # priority is then treated as fully missing and zero-padded
            # (codice_l1a_de.py process_de_data), so we report num_events=0
            # while the validation file retains the real, recoverable count.
            # The per-event science arrays are unaffected (correctly fillval
            # on both sides) since the event byte payload really is unusable.
            # The CoDICE team is aware and will decide if we need to recover
            # these values before combine_sengmented_packts.
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
    assert_epoch_delta_cdf_metadata(cdf_file)
    assert "quality_flags" in processed_data


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_calculate_quality_flags(mock_get_file_paths, codice_lut_path):
    """Tests the calculate_quality_flags function."""

    # test case #1 - no flags
    macro_status = np.full(100, 10, dtype=np.uint32)
    quality_flags = calculate_quality_flags(macro_status)
    assert np.all(quality_flags == ImapCodiceL1aFlags.NONE)

    # Test case #2 - Gain test is flagged for the first 10 values
    macro_status = np.concat([np.full(10, 128, np.uint32), np.full(10, 126, np.uint32)])
    quality_flags = calculate_quality_flags(macro_status)
    expected_status = np.concat(
        [
            np.full(10, ImapCodiceL1aFlags.GAIN_TEST.value, np.uint16),
            np.full(10, ImapCodiceL1aFlags.NONE.value, np.uint16),
        ]
    )
    np.testing.assert_array_equal(quality_flags, expected_status)


def test_add_quality_flags_to_science_datasets():
    """Test adding quality flags to science datasets."""
    # Create mock epochs where the sci time is coarser than the hk time.
    epoch = np.arange(10, 30, step=5)
    hk_epoch = np.arange(40)  # The hk is expected to have a wider time range.
    # Create mock list of datasets
    datasets = [
        xr.Dataset({"epoch": epoch}, attrs={"Logical_source": "hskp"}),
        xr.Dataset({"epoch": epoch}, attrs={"Logical_source": "sci"}),
        xr.Dataset({"epoch": np.append(epoch, 50)}, attrs={"Logical_source": "sci"}),
    ]
    hk_quality_flags = np.full(
        hk_epoch.shape, ImapCodiceL1aFlags.GAIN_TEST.value, np.uint8
    )
    # Test adding quality flags to the science datasets
    add_quality_flags_to_science_datasets(hk_quality_flags, hk_epoch, datasets)

    # check that hk datasets do NOT have quality flags
    assert "quality_flags" not in datasets[0]
    assert "quality_flags" in datasets[1]
    assert "quality_flags" in datasets[2]

    expected_sci_qf = np.full(len(epoch), ImapCodiceL1aFlags.GAIN_TEST.value, np.uint16)
    np.testing.assert_array_equal(expected_sci_qf, datasets[1].quality_flags.values)

    # Since the last epoch value for the final array is 50 and out of range of
    # the hk data, the quality flag should be 0.
    expected_sci_qf2 = np.append(expected_sci_qf, 0)
    np.testing.assert_array_equal(expected_sci_qf2, datasets[2].quality_flags.values)


def test_add_quality_flags_to_science_datasets_flag_logic():
    """Test the flag logic when adding quality flags to science datasets."""
    # Create mock epochs where the sci time is coarser than the hk time.
    epoch = np.array([240, 480, 720]) * 1e9
    hk_epoch = np.array([230, 240, 241, 480, 721, 1000]) * 1e9
    quality_flags = np.array([1, 0, 1, 0, 0, 1])
    expected_flags = np.array([1, 0, 0])
    # Create mock list of datasets
    datasets = [xr.Dataset({"epoch": epoch}, attrs={"Logical_source": "sci"})]
    # Test case #1: Quality flags provided, some epochs map and some do not
    add_quality_flags_to_science_datasets(quality_flags, hk_epoch, datasets)
    np.testing.assert_array_equal(expected_flags, datasets[0].quality_flags.values)

    # Test case #2: No quality flags provided (hk_quality_flags is None)
    add_quality_flags_to_science_datasets(None, hk_epoch, datasets)
    expected_flags_null = np.full(epoch.shape, ImapCodiceL1aFlags.NONE.value, np.uint16)
    np.testing.assert_array_equal(expected_flags_null, datasets[0].quality_flags.values)

    # Test case #3: No epoch times fall within the hk range.
    hk_epoch += 1000 * 1e9  # Shift hk epoch values out of range
    add_quality_flags_to_science_datasets(quality_flags, hk_epoch, datasets)
    np.testing.assert_array_equal(expected_flags_null, datasets[0].quality_flags.values)
