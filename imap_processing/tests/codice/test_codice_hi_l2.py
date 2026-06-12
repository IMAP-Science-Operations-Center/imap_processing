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
from imap_processing.codice.codice_l2 import (
    HI_SPECIES_DISPLAY_NAMES,
    process_codice_l2,
)
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)

pytestmark = pytest.mark.external_test_data

# epoch_delta = num_spins * spin_period / 2, with spin_period VALIDMAX = 16 s
# and num_spins max = 16 in the current CoDICE timing model. That yields a
# worst-case delta of 128 s = 128000000000 ns.
EXPECTED_EPOCH_DELTA_VALIDMAX = 128000000000


def assert_l2_epoch_delta_cdf_metadata(cdf_file):
    """Assert L2 epoch delta vars and epoch links are written correctly."""
    with cdflib.CDF(cdf_file) as cdf:
        epoch_attrs = cdf.varattsget("epoch")
        assert epoch_attrs["DELTA_MINUS_VAR"] == "epoch_delta_minus"
        assert epoch_attrs["DELTA_PLUS_VAR"] == "epoch_delta_plus"

        for variable in ("epoch_delta_minus", "epoch_delta_plus"):
            info = cdf.varinq(variable)
            attrs = cdf.varattsget(variable)
            assert info.Data_Type_Description == "CDF_INT8"
            assert attrs["FILLVAL"] == -9223372036854775808
            assert attrs["FORMAT"] == "I19"
            assert attrs["VALIDMIN"] == 0
            assert attrs["VALIDMAX"] == EXPECTED_EPOCH_DELTA_VALIDMAX


def _expected_hi_energy_labels(species: str, energies: np.ndarray) -> np.ndarray:
    species_display = HI_SPECIES_DISPLAY_NAMES[species]
    return np.array(
        [f"{species_display} int @{energy:.3f} MeV/nuc" for energy in energies]
    )


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
        if variable.startswith("energy_") and variable.endswith("_label"):
            species = variable.removeprefix("energy_").removesuffix("_label")
            np.testing.assert_array_equal(
                processed_l2[variable].values,
                _expected_hi_energy_labels(
                    species,
                    processed_l2[f"energy_{species}"].values,
                ),
                err_msg=f"Mismatch in coordinate '{variable}'",
            )
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
    omni_cdf_file = write_cdf(processed_l2)
    assert (
        omni_cdf_file.name == f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert_l2_epoch_delta_cdf_metadata(omni_cdf_file)
    with cdflib.CDF(omni_cdf_file) as cdf_file:
        data_quality_attrs = cdf_file.varattsget("data_quality")
        assert data_quality_attrs["VAR_TYPE"] == "data"
        assert cdf_file.varattsget("energy_h")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_minus")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_plus")["FORMAT"] == "F12.6"
        energy_h_label_attrs = cdf_file.varattsget("energy_h_label")
        assert energy_h_label_attrs["FORMAT"] == "A32"
        assert (
            energy_h_label_attrs["CATDESC"]
            == "Energy-channel labels for H differential intensity"
        )
        assert energy_h_label_attrs["FIELDNAM"] == "H Energy Channel Labels"
        energy_h_attrs = cdf_file.varattsget("energy_h")
        assert energy_h_attrs["CATDESC"] == "Geometric mean energy per nucleon for H"
        assert energy_h_attrs["FIELDNAM"] == "H Energy"
        energy_junk_label_attrs = cdf_file.varattsget("energy_junk_label")
        assert (
            energy_junk_label_attrs["CATDESC"]
            == "Energy-channel labels for Junk differential intensity"
        )
        assert energy_junk_label_attrs["FIELDNAM"] == "Junk Energy Channel Labels"
        h_attrs = cdf_file.varattsget("h")
        assert h_attrs["CATDESC"] == (
            "Differential intensity for H at root-2-spaced energy-per-nucleon channels"
        )
        assert h_attrs["DELTA_MINUS_VAR"] == "unc_h"
        assert h_attrs["DELTA_PLUS_VAR"] == "unc_h"
        assert h_attrs["FIELDNAM"] == "Differential Intensity - H"
        unc_h_attrs = cdf_file.varattsget("unc_h")
        assert unc_h_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for H at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert unc_h_attrs["FIELDNAM"] == "Uncertainty - H"
        junk_attrs = cdf_file.varattsget("junk")
        assert junk_attrs["CATDESC"] == (
            "Differential intensity for Junk (unclassified counts) at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert junk_attrs["FIELDNAM"] == "Differential Intensity - Junk"
        assert junk_attrs["VAR_NOTES"].strip() == (
            "Catch-all bin for counts that do not fall into any CoDICE Hi "
            "species classification bin."
        )
        unc_junk_attrs = cdf_file.varattsget("unc_junk")
        assert unc_junk_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for Junk (unclassified counts) "
            "at root-2-spaced energy-per-nucleon channels"
        )
        assert unc_junk_attrs["VAR_NOTES"].strip() == (
            "Catch-all uncertainty bin for counts that do not fall into any "
            "CoDICE Hi species classification bin."
        )
        uh_attrs = cdf_file.varattsget("uh")
        assert uh_attrs["CATDESC"] == (
            "Differential intensity for Ultra-Heavy ions at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert uh_attrs["FIELDNAM"] == "Differential Intensity - Ultra-Heavy"
        np.testing.assert_array_equal(
            cdf_file.varget("energy_h_label"),
            _expected_hi_energy_labels("h", processed_l2["energy_h"].values),
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
            assert processed_l2[variable].dims == (
                "spin_sector",
                "elevation_angle",
            )
            spin_vals = processed_l2[variable].values
            assert np.all(np.isfinite(spin_vals)), (
                "spin_angle contains non-finite values"
            )
            assert np.min(spin_vals) >= 0.0, "spin_angle has values below 0 degrees"
            assert np.max(spin_vals) <= 360.0, "spin_angle has values above 360 degrees"
        else:
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
        if variable.startswith("energy_") and variable.endswith("_label"):
            species = variable.removeprefix("energy_").removesuffix("_label")
            np.testing.assert_array_equal(
                processed_l2[variable].values,
                _expected_hi_energy_labels(
                    species,
                    processed_l2[f"energy_{species}"].values,
                ),
                err_msg=f"Mismatch in coordinate '{variable}'",
            )
            continue
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
    assert_l2_epoch_delta_cdf_metadata(sectored_cdf_file)
    with cdflib.CDF(sectored_cdf_file) as cdf_file:
        data_quality_attrs = cdf_file.varattsget("data_quality")
        assert data_quality_attrs["VAR_TYPE"] == "data"
        assert data_quality_attrs["FORMAT"] == "I3"
        spin_sector_attrs = cdf_file.varattsget("spin_sector")
        assert spin_sector_attrs["FORMAT"] == "I3"
        spin_angle_attrs = cdf_file.varattsget("spin_angle")
        assert spin_angle_attrs["VAR_TYPE"] == "support_data"
        assert cdf_file.varattsget("energy_h")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_minus")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_plus")["FORMAT"] == "F12.6"
        energy_h_label_attrs = cdf_file.varattsget("energy_h_label")
        assert energy_h_label_attrs["FORMAT"] == "A32"
        assert (
            energy_h_label_attrs["CATDESC"]
            == "Energy-channel labels for H differential intensity"
        )
        assert energy_h_label_attrs["FIELDNAM"] == "H Energy Channel Labels"
        energy_h_attrs = cdf_file.varattsget("energy_h")
        assert energy_h_attrs["CATDESC"] == "Geometric mean energy per nucleon for H"
        assert energy_h_attrs["FIELDNAM"] == "H Energy"
        h_attrs = cdf_file.varattsget("h")
        assert h_attrs["CATDESC"] == (
            "Differential intensity for H by energy, spin sector, and "
            "elevation at x2-spaced energy-per-nucleon channels"
        )
        assert h_attrs["FIELDNAM"] == "Differential Intensity - H"
        unc_h_attrs = cdf_file.varattsget("unc_h")
        assert unc_h_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for H by energy, spin "
            "sector, and elevation at x2-spaced energy-per-nucleon channels"
        )
        assert unc_h_attrs["FIELDNAM"] == "Uncertainty - H"
        np.testing.assert_array_equal(
            cdf_file.varget("energy_h_label"),
            _expected_hi_energy_labels("h", processed_l2["energy_h"].values),
        )
