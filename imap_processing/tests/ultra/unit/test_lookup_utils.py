from unittest import mock

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.quality_flags import ImapDEOutliersUltraFlags
from imap_processing.ultra.l1b.lookup_utils import (
    get_angular_profiles,
    get_back_position,
    get_de_product_name,
    get_ebins,
    get_energy_efficiencies,
    get_energy_norm,
    get_geometric_factor,
    get_image_params,
    get_norm,
    get_ph_corrected,
    get_scattering_coefficients,
    get_scattering_thresholds,
    get_y_adjust,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"
TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.mark.external_test_data
def test_get_y_adjust(ancillary_files):
    """Tests function get_y_adjust."""

    yadjust_path = TEST_PATH / "imap_ultra_l1b-yadjust-lookup_20250101_v001.csv"
    yadjust_df = pd.read_csv(yadjust_path).set_index("dYLUT")

    array = np.array([8])
    res = get_y_adjust(array, ancillary_files)

    assert res == yadjust_df["dYAdj"][8]


@pytest.mark.external_test_data
def test_get_stop_norm(ancillary_files):
    """Tests function get_stop_norm."""

    tdc_norm_path = (
        TEST_PATH / "imap_ultra_l1b-45sensor-tdc-norm-lookup_20250101_v000.csv"
    )
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=1)

    array = np.array([378])
    stop_norm = get_norm(array, "SpE", "ultra45", ancillary_files)

    assert stop_norm == tdc_norm_df["SpE"][378]


@pytest.mark.external_test_data
def test_get_back_position(ancillary_files):
    """Tests function get_back_position."""

    back_pos_path = (
        TEST_PATH / "imap_ultra_l1b-45sensor-back-pos-lookup_20250101_v000.csv"
    )
    back_pos_df = pd.read_csv(back_pos_path, index_col="Index_offset")

    array = np.array([-2000])
    dn_converted = get_back_position(array, "XBkBt", "ultra45", ancillary_files)

    assert dn_converted == back_pos_df["XBkBt"].iloc[-2000]


@pytest.mark.external_test_data
def test_get_egy_norm(ancillary_files):
    """Tests function get_egy_norm."""

    egy_norm_path = TEST_PATH / "imap_ultra_l1b-egynorm-lookup_20250101_v000.csv"
    egy_norm_df = pd.read_csv(egy_norm_path)

    norm_composite_energy = get_energy_norm(
        np.array([2]), np.array([2]), ancillary_files
    )

    expected_energy = egy_norm_df.iloc[2 * 4096 + 2]["NormEnergy"]
    np.testing.assert_array_equal(norm_composite_energy, [expected_energy])


@pytest.mark.external_test_data
def test_get_image_params(ancillary_files):
    """Tests function get_image_params."""
    image_params = get_image_params("XFTLTOFF", "ultra45", ancillary_files)

    assert image_params == 49.3


def test_get_angular_profiles():
    """Tests function get_image_params."""

    ancillary_files = {
        "l1b-45sensor-leftslit-lookup": "test1.csv",
        "l1b-45sensor-rightslit-lookup": "test2.csv",
        "l1b-90sensor-leftslit-lookup": "test3.csv",
        "l1b-90sensor-rightslit-lookup": "test4.csv",
    }
    with mock.patch(
        "imap_processing.ultra.l1b.lookup_utils.pd.read_csv"
    ) as mock_read_csv:
        get_angular_profiles("left", "ultra45", ancillary_files)
        mock_read_csv.assert_called_with("test1.csv")

        get_angular_profiles("right", "ultra45", ancillary_files)
        mock_read_csv.assert_called_with("test2.csv")


@pytest.mark.external_test_data
def test_get_energy_efficiencies(ancillary_files):
    """Tests function get_get_energy_efficiencies."""

    u45_efficiencies = get_energy_efficiencies(ancillary_files, "ultra45")

    assert u45_efficiencies.shape == (58081, 157)

    # Test that the function can also read the ultra90 efficiencies
    u90_efficiencies = get_energy_efficiencies(ancillary_files, "ultra90")
    assert u90_efficiencies.shape == (58081, 157)


@pytest.mark.external_test_data
def test_get_geometric_function(ancillary_files):
    """Tests function get_get_energy_efficiencies."""

    phi = np.array([-65, -64, -39, -1.3, 0, 1.3, 39, 64, 65])
    theta = np.array([-65, -64, -39, -1.3, 0, 1.3, 39, 64, 65])
    quality_flags = np.full(
        phi.shape, ImapDEOutliersUltraFlags.NONE.value, dtype=np.uint16
    )
    gf = get_geometric_factor(
        phi,
        theta,
        quality_flags,
        ancillary_files,
        "l1b-sensor-gf-noblades",
    )

    np.testing.assert_array_equal(
        gf, np.array([0, 0, 0.13713, 0.1792, 0.35507, 0.1792, 0.13713, 0, 0])
    )
    np.testing.assert_array_equal(quality_flags, np.array([1, 1, 0, 0, 0, 0, 0, 1, 1]))


@pytest.mark.external_test_data
def test_get_ph_corrected(ancillary_files):
    """Tests function get_ph_corrected."""

    # Should be between 1 and 32 (0 and 31)
    xlut = np.array([0, 10, 31, 32])
    # Should be between 1 and 20 (0 and 19)
    ylut = np.array([3, 10, 19, 32])
    quality_flags = np.full(
        xlut.shape, ImapDEOutliersUltraFlags.NONE.value, dtype=np.uint16
    )
    ph_correct_top, quality_flags = get_ph_corrected(
        "ultra45", "tp", ancillary_files, xlut, ylut, quality_flags
    )

    np.testing.assert_array_equal(
        ph_correct_top, np.array([1429.143693, 1001.839137, 2667.220492, 3214.786627])
    )
    np.testing.assert_array_equal(
        quality_flags,
        np.array([0, 0, 2, 2]),
    )


@pytest.mark.external_test_data
def test_get_ebins(ancillary_files):
    """Tests function get_ph_corrected."""

    energy = np.array([618, 4])
    ctof = np.array([73, 24])
    fillval_uint8 = 255
    ebins = np.full(energy.shape, fillval_uint8, dtype=np.uint8)
    ebins = get_ebins("l1b-tofxph", energy, ctof, ebins, ancillary_files)

    np.testing.assert_array_equal(ebins, np.array([15, 19]))


@pytest.mark.external_test_data
def test_get_scattering_coefficients(ancillary_files):
    """Tests function get_scattering_data."""

    theta_coeffs, phi_coeffs = get_scattering_coefficients(
        np.array([47, 43]),
        np.array([43, 42]),
        lookup_tables=None,
        ancillary_files=ancillary_files,
        instrument_id=45,
    )
    # Test a theta coefficients
    np.testing.assert_array_equal(theta_coeffs[:, 0], np.array([np.nan, 35.23100]))
    # Test b theta coefficients
    np.testing.assert_array_equal(theta_coeffs[:, 1], np.array([np.nan, -0.72148]))
    # Test a phi coefficients
    np.testing.assert_array_equal(phi_coeffs[:, 0], np.array([np.nan, 168.3100]))
    # Test b phi coefficients
    np.testing.assert_array_equal(phi_coeffs[:, 1], np.array([np.nan, -1.0752]))


@pytest.mark.external_test_data
def test_get_scattering_thresholds(ancillary_files):
    """Tests function get_scattering_thresholds."""

    thresholds = get_scattering_thresholds(
        ancillary_files=ancillary_files,
    )
    assert thresholds[(1.0, 5.0)] == 12.0
    assert thresholds[(5.0, 8.0)] == 10.0
    assert thresholds[(8.0, 10.0)] == 8.0
    assert thresholds[(10.0, 20.0)] == 6.0
    assert thresholds[(20.0, np.inf)] == 4.0


def test_get_de_product_name_no_repoint():
    """Tests function get_de_product_name when the lookup is missing the repoint."""
    ancillary_files = {
        "l1b-45sensor-de-product-lookup": TEST_PATH
        / "imap_ultra_l1b-45sensor-de-product-lookup_20251001_v001.csv"
    }
    with mock.patch(
        "imap_processing.ultra.l1b.lookup_utils.pd.read_csv"
    ) as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            {
                "repointing_id_start": [1, 2],
                "repointing_id_end": [3, 4],
                "de_product": [
                    "imap_ultra_l1b_45sensor-de",
                    "imap_ultra_l1b_45sensor-priority-1-de",
                ],
            }
        )
        with pytest.raises(ValueError, match="No DE product found for repoint ID 0"):
            get_de_product_name("repoint00000", 45, "l1b", ancillary_files)


def test_get_de_product_name_multiple_products():
    """Tests function get_de_product_name when the lookup is ambiguous."""
    ancillary_files = {
        "l1b-45sensor-de-product-lookup": TEST_PATH
        / "imap_ultra_l1b-45sensor-de-product-lookup_20251001_v001.csv"
    }
    with mock.patch(
        "imap_processing.ultra.l1b.lookup_utils.pd.read_csv"
    ) as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            {
                "repointing_id_start": [2, 2],
                "repointing_id_end": [3, 4],
                "de_product": [
                    "imap_ultra_l1b_45sensor-de",
                    "imap_ultra_l1b_45sensor-priority-1-de",
                ],
            }
        )
        with pytest.raises(ValueError, match="Multiple DE products found"):
            get_de_product_name("repoint00002", 45, "l1b", ancillary_files)


def test_get_de_product_name():
    """Tests function get_de_product_name."""
    ancillary_files = {
        "l1b-45sensor-de-product-lookup": TEST_PATH
        / "imap_ultra_l1b-45sensor-de-product-lookup_20251001_v001.csv"
    }
    with mock.patch(
        "imap_processing.ultra.l1b.lookup_utils.pd.read_csv"
    ) as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            {
                "repointing_id_start": [0, 2, 4],
                "repointing_id_end": [1, 4, np.nan],
                "de_product": [
                    "imap_ultra_l1b_45sensor-de",
                    "imap_ultra_l1b_45sensor-priority-1-de",
                    "imap_ultra_l1b_45sensor-priority-2-de",
                ],
            }
        )
        # Test with a repoint in the future. Should return the priority 2 de product
        # since the last repoint range does not have an end and should be assumed to
        # cover all future repoints.
        de_product = get_de_product_name("repoint00100", 45, "l1b", ancillary_files)
        assert de_product == "imap_ultra_l1b_45sensor-priority-2-de"

        # Test with valid repoint that falls in the second range.
        de_product = get_de_product_name("repoint00003", 45, "l1b", ancillary_files)
        assert de_product == "imap_ultra_l1b_45sensor-priority-1-de"
