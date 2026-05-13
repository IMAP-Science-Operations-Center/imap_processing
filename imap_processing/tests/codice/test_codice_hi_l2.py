from unittest.mock import patch

import cdflib
import numpy as np
import pytest
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
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


def _generate_hi_l1b_file(descriptor: str, codice_lut_path):
    """Generate a fresh Hi L1B CDF for metadata regression tests.

    We need this helper because the checked-in ``tests/codice/data/l1b_validation``
    artifacts predate the epoch-delta dtype fix and still serialize
    ``epoch_delta_plus`` / ``epoch_delta_minus`` as integer CDF variables.
    The L2 metadata tests below are specifically trying to verify the current
    regenerated pipeline behavior, so they must consume an L1B file produced by
    the current L1A -> L1B code path instead of the historical validation
    artifact.

    If we later refresh the Hi L1B validation CDFs to include this dtype fix,
    this helper can be removed and the L2 metadata tests can go back to using
    the checked-in ``l1b_validation`` files directly.
    """

    def _lookup_l1_inputs(request_descriptor=None, data_type=None, **kwargs):
        # ``process_l1a()`` asks for two different inputs through the same file
        # lookup hook:
        # 1. the raw science packet via ``data_type='l0'`` with no descriptor
        # 2. the science LUT via ``descriptor='l1a-sci-lut'``
        #
        # Patch the lookup so the real production code can run unchanged while
        # the test routes those requests to the correct local test artifacts.
        request_descriptor = kwargs.get("descriptor", request_descriptor)
        if request_descriptor is None and data_type == "l0":
            return codice_lut_path(descriptor, data_type="l0")
        return codice_lut_path(request_descriptor, data_type)

    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        mock_get_file_paths.side_effect = _lookup_l1_inputs
        processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
        processed_l1b = process_codice_l1b(processed_l1a_file)
        processed_l1b.attrs["Data_version"] = "001"
        return write_cdf(processed_l1b)


def _mock_l2_file_paths(descriptor: str, l1b_file, codice_lut_path):
    """Return a side effect that points L2 processing at a generated L1B file."""

    def _side_effect(request_descriptor=None, data_type=None, **kwargs):
        request_descriptor = kwargs.get("descriptor", request_descriptor)
        if request_descriptor == descriptor:
            return [l1b_file]
        return codice_lut_path(request_descriptor, data_type)

    return _side_effect


@pytest.fixture
def mock_get_file_paths(codice_lut_path):
    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        # Ensure the side effect treats science inputs as L1B for these L2 tests
        mock_get_file_paths.side_effect = lambda descriptor, data_type=None: (
            codice_lut_path(descriptor, data_type="l1b")
        )
        yield mock_get_file_paths


def test_l2_hi_omni(mock_get_file_paths):
    sci_input = ScienceInput(
        f"imap_codice_l1b_hi-omni_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf"
    )
    anc_input = AncillaryInput("imap_codice_l2-hi-omni-efficiency_20251212_v003.csv")
    dependencies = ProcessingInputCollection(anc_input, sci_input)

    processed_l2 = process_codice_l2("hi-omni", dependencies)

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
    assert (
        omni_cdf_file.name == f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"
    )


def test_l2_hi_sectored(mock_get_file_paths):
    anc_input = AncillaryInput(
        "imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv"
    )
    sci_input = ScienceInput(
        f"imap_codice_l1b_hi-sectored_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf"
    )
    dependencies = ProcessingInputCollection(anc_input, sci_input)

    processed_l2 = process_codice_l2("hi-sectored", dependencies)

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
        # Spin angle bug is fixed but the old validation data is outdated.
        # Verified with new 20260201 L2 validation file from Joey.
        if variable.startswith("unc_"):
            continue
        if variable == "spin_angle":
            # The external validation file has outdated spin_angle values, but we
            # still verify structure and basic numeric sanity to guard against
            # regressions in the spin angle computation.
            assert processed_l2[variable].dims == val_data[variable].dims, (
                f"Dimension mismatch in variable '{variable}'"
            )
            spin_vals = processed_l2[variable].values
            # All values should be finite and lie within a reasonable angular range.
            assert np.all(np.isfinite(spin_vals)), (
                "spin_angle contains non-finite values"
            )
            assert np.min(spin_vals) >= 0.0, "spin_angle has values below 0 degrees"
            assert np.max(spin_vals) <= 360.0, "spin_angle has values above 360 degrees"
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


@pytest.mark.parametrize(
    ("descriptor", "efficiency_file"),
    [
        ("hi-omni", "imap_codice_l2-hi-omni-efficiency_20251212_v003.csv"),
        ("hi-sectored", "imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv"),
    ],
)
def test_l2_hi_epoch_delta_cdf_metadata(descriptor, efficiency_file, codice_lut_path):
    l1b_file = _generate_hi_l1b_file(descriptor, codice_lut_path)
    dependencies = ProcessingInputCollection(
        AncillaryInput(efficiency_file),
        ScienceInput(l1b_file.name),
    )

    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        mock_get_file_paths.side_effect = _mock_l2_file_paths(
            descriptor, l1b_file, codice_lut_path
        )
        processed_l2 = process_codice_l2(descriptor, dependencies)

    processed_l2.attrs["Data_version"] = "001"
    cdf_file = cdflib.CDF(write_cdf(processed_l2))
    for var in ["epoch_delta_minus", "epoch_delta_plus"]:
        var_info = cdf_file.varinq(var)
        var_attrs = cdf_file.varattsget(var)
        assert var_info.Data_Type_Description == "CDF_DOUBLE"
        assert np.isclose(var_attrs["FILLVAL"], np.float64(-1.0e31))
