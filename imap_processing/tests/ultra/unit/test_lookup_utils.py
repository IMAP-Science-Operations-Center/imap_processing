import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.quality_flags import ImapDEUltraFlags
from imap_processing.ultra.l1b.lookup_utils import (
    get_angular_profiles,
    get_back_position,
    get_energy_efficiencies,
    get_energy_norm,
    get_geometric_factor,
    get_image_params,
    get_norm,
    get_y_adjust,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"


def test_get_y_adjust():
    """Tests function get_y_adjust."""

    yadjust_path = BASE_PATH / "yadjust.csv"
    yadjust_df = pd.read_csv(yadjust_path).set_index("dYLUT")

    array = np.array([8])
    res = get_y_adjust(array)

    assert res == yadjust_df["dYAdj"][8]


def test_get_stop_norm():
    """Tests function get_stop_norm."""

    tdc_norm_path = BASE_PATH / "ultra45_tdc_norm.csv"
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=1)

    array = np.array([378])
    stop_norm = get_norm(array, "SpE", "ultra45")

    assert stop_norm == tdc_norm_df["SpE"][378]


def test_get_back_position():
    """Tests function get_back_position."""

    back_pos_path = BASE_PATH / "ultra45_back-pos-luts.csv"
    back_pos_df = pd.read_csv(back_pos_path, index_col="Index_offset")

    array = np.array([-2000])
    dn_converted = get_back_position(array, "XBkBt", "ultra45")

    assert dn_converted == back_pos_df["XBkBt"].iloc[-2000]


def test_get_egy_norm():
    """Tests function get_egy_norm."""

    egy_norm_path = BASE_PATH / "EgyNorm.mem.csv"
    egy_norm_df = pd.read_csv(egy_norm_path)

    norm_composite_energy = get_energy_norm(np.array([2]), np.array([2]))

    assert int(norm_composite_energy) == egy_norm_df.iloc[2 * 4096 + 2]["NormEnergy"]


def test_get_image_params():
    """Tests function get_image_params."""

    image_params = get_image_params("XFTLTOFF", "ultra45")

    assert image_params == 49.3


def test_get_angular_profiles():
    """Tests function get_image_params."""

    u45_left = get_angular_profiles("left", "ultra45")
    u45_right = get_angular_profiles("right", "ultra45")

    assert u45_left.shape == (525, 7)
    assert u45_right.shape == (525, 7)


@pytest.mark.external_test_data
def test_get_energy_efficiencies():
    """Tests function get_get_energy_efficiencies."""

    path = imap_module_directory / "tests" / "ultra" / "data" / "l1"
    ancillary_files = {
        "l1b-45sensor-logistic-interpolation": path
        / "imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv"
    }
    u45_efficiencies = get_energy_efficiencies(ancillary_files)

    assert u45_efficiencies.shape == (58081, 157)


@pytest.mark.external_test_data
def test_get_geometric_function():
    """Tests function get_get_energy_efficiencies."""

    path = imap_module_directory / "tests" / "ultra" / "data" / "l1"
    ancillary_files = {
        "l1b-sensor-gf-noblades": path
        / "imap_ultra_l1b-sensor-gf-noblades_20250101_v000.csv"
    }
    phi = np.array([-65, -64, -39, -1.3, 0, 1.3, 39, 64, 65])
    theta = np.array([-65, -64, -39, -1.3, 0, 1.3, 39, 64, 65])
    quality_flags = np.full(phi.shape, ImapDEUltraFlags.NONE.value, dtype=np.uint16)
    gf = get_geometric_factor(
        ancillary_files, "l1b-sensor-gf-noblades", phi, theta, quality_flags
    )

    np.testing.assert_array_equal(
        gf, np.array([0, 0, 0.13713, 0.1792, 0.35507, 0.1792, 0.13713, 0, 0])
    )
    np.testing.assert_array_equal(quality_flags, np.array([2, 2, 0, 0, 0, 0, 0, 2, 2]))
