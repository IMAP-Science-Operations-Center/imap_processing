import numpy as np
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.ultra.l1b.lookup_utils import (
    get_back_position,
    get_energy_norm,
    get_image_params,
    get_norm,
    get_y_adjust,
)

base_path = f"{imap_module_directory}/ultra/lookup_tables"


def test_get_y_adjust():
    """Tests function get_y_adjust."""
    # TODO: Add more test cases and data

    yadjust_path = f"{base_path}/yadjust.csv"
    yadjust_df = pd.read_csv(yadjust_path).set_index("dYLUT")

    array = np.array([8])
    res = get_y_adjust(array)

    assert res == yadjust_df["dYAdj"][8]


def test_get_stop_norm():
    """Tests function get_stop_norm."""
    # TODO: Add more test cases and data

    tdc_norm_path = f"{base_path}/ultra45_tdc_norm.csv"
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=1)

    array = np.array([378])
    stop_norm = get_norm(array, "SpE", "ultra45")

    assert stop_norm == tdc_norm_df["SpE"][378]


def test_get_back_position():
    """Tests function get_back_position."""
    # TODO: Add more test cases and data

    back_pos_path = f"{base_path}/ultra45_back-pos-luts.csv"
    back_pos_df = pd.read_csv(back_pos_path, index_col="Index_offset")

    array = np.array([-2000])
    dn_converted = get_back_position(array, "XBkBt", "ultra45")

    assert dn_converted == back_pos_df["XBkBt"].iloc[-2000]


def test_get_egy_norm():
    """Tests function get_egy_norm."""
    # TODO: Add more test cases and data

    egy_norm_path = f"{base_path}/EgyNorm.mem.csv"
    egy_norm_df = pd.read_csv(egy_norm_path)

    norm_composite_energy = get_energy_norm(np.array([2]), np.array([2]))

    assert norm_composite_energy == egy_norm_df.iloc[2 * 4096 + 2]["NormEnergy"]


def test_get_image_params():
    """Tests function get_image_params."""
    # TODO: Add more test cases and data

    image_params = get_image_params("XFTLTOFF")

    assert image_params == 47.5
