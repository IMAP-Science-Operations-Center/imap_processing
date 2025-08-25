import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import (
    SWAPI_LIVETIME,
    solve_full_sweep_energy,
    swapi_l2,
)
from imap_processing.swapi.swapi_utils import read_swapi_lut_table


@pytest.fixture(scope="session")
def esa_unit_conversion_table() -> pd.DataFrame:
    """
    Read the ESA unit conversion table.

    Returns
    -------
    esa_unit_conversion_table : pandas.DataFrame
        The ESA unit conversion table.
    """
    esa_file_path = (
        imap_module_directory
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv"
    )
    df = read_swapi_lut_table(esa_file_path)
    return df


@pytest.fixture(scope="session")
def lut_notes_table() -> pd.DataFrame:
    """
    Read the LUT notes table.

    Returns
    -------
    lut_notes_table : pandas.DataFrame
        The LUT notes table.
    """
    lut_notes_file_path = (
        imap_module_directory / "tests/swapi/lut/imap_swapi_lut-notes_20250626_v006.csv"
    )
    df = read_swapi_lut_table(lut_notes_file_path)

    return df


@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
def test_swapi_l2_cdf(
    mock_get_file_paths,
    swapi_l0_test_data_path,
    esa_unit_conversion_table,
    lut_notes_table,
):
    """Test housekeeping processing and CDF file creation"""
    test_packet_file = swapi_l0_test_data_path / "imap_swapi_l0_raw_20240924_v001.pkts"

    # Mock paths of files to be processed
    def first_get_file_paths_side_effect(descriptor):
        if descriptor == "raw":
            return [test_packet_file]
        elif descriptor == "hk":
            return []
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = first_get_file_paths_side_effect
    # Processing inputs
    processing_input = [
        {"type": "science", "files": ["imap_swapi_l0_raw_20240924_v001.pkts"]}
    ]
    collection_obj = ProcessingInputCollection()
    collection_obj.deserialize(
        json.dumps(processing_input),
    )
    # Create HK CDF File
    processed_hk_data = swapi_l1(collection_obj)
    hk_cdf_filename = "imap_swapi_l1a_hk_20240924_v999.cdf"
    hk_cdf_path = write_cdf(processed_hk_data[0])
    assert hk_cdf_path.name == hk_cdf_filename

    # Mock paths of files to be processed
    def second_get_file_paths_side_effect(descriptor):
        if descriptor == "raw":
            return [test_packet_file]
        elif descriptor == "hk":
            return [hk_cdf_path]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = second_get_file_paths_side_effect
    processing_input = [
        {"type": "science", "files": ["imap_swapi_l0_raw_20240924_v001.pkts"]},
        {"type": "science", "files": ["imap_swapi_l1a_hk_20240924_v999.cdf"]},
    ]
    collection_obj = ProcessingInputCollection()
    collection_obj.deserialize(
        json.dumps(processing_input),
    )
    # Create L1 CDF File
    processed_sci_data = swapi_l1(collection_obj)
    cdf_filename = "imap_swapi_l1_sci_20240924_v999.cdf"
    cdf_path = write_cdf(processed_sci_data[0])
    assert cdf_path.name == cdf_filename

    l1_dataset = load_cdf(cdf_path)
    l2_dataset = swapi_l2(
        l1_dataset,
        esa_table_df=esa_unit_conversion_table,
        lut_notes_df=lut_notes_table,
    )
    l2_cdf = write_cdf(l2_dataset)
    assert l2_cdf.name == "imap_swapi_l2_sci_20240924_v999.cdf"

    # Test uncertainty variables are as expected
    np.testing.assert_array_equal(
        l2_dataset["swp_pcem_rate_stat_uncert_plus"],
        l1_dataset["swp_pcem_counts_stat_uncert_plus"] / SWAPI_LIVETIME,
    )
    # Since L2 data's date is before any date in ESA unit conversion table,
    # check that it returns nan in first 63 energy steps
    assert np.isnan(l2_dataset["swp_esa_energy"].values[0, :63]).all()
    # Check fine steps
    fine_energies = [
        4290.0,
        4199.0,
        4109.0,
        4020.0,
        3934.0,
        3850.0,
        3767.0,
        3687.0,
        3608.0,
    ]
    assert np.all(l2_dataset["swp_esa_energy"].values[0, -9:] == fine_energies)


def test_solve_full_sweep_energy(esa_unit_conversion_table, lut_notes_table):
    """Test the solve_full_sweep_energy function"""
    # Find 9 fine energies for unique ESA_LVL5 values
    esa_lvl5_arr = [4663]
    sweep_table = [0]
    data_time = [np.datetime64("2025-02-24T00:00:00", "ns")]
    esa_lvl5_hex = np.vectorize(lambda x: format(x, "X"))(esa_lvl5_arr)
    sweeps_energy_value = solve_full_sweep_energy(
        esa_lvl5_hex, sweep_table, esa_unit_conversion_table, lut_notes_table, data_time
    )
    assert sweeps_energy_value.shape == (1, 72)

    # First check that first 63 values are same as the fixed energy values.
    fixed_energy_values = np.array(
        [
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1163,
            1068,
            981,
            901,
            828,
            760,
            698,
            641,
            589,
            544,
            497,
            459,
            421,
            389,
            355,
            326,
            298,
            275,
            252,
            234,
            214,
            195,
            181,
            167,
            153,
            139,
            129,
            120,
            107,
        ]
    )
    assert np.all(sweeps_energy_value[:, :63] == fixed_energy_values)

    # Now, test that the last 9 fine energy values are as expected for first sweep.
    # I manually picked those values from LUT table.
    expected_fine_energies = np.array(
        [3220.0, 3151.0, 3083.0, 3017.0, 2953.0, 2889.0, 2827.0, 2767.0, 2707.0]
    )
    assert np.all(sweeps_energy_value[0, -9:] == expected_fine_energies)

    # Test that we get different values for date later than 2025-05-19
    data_time = [np.datetime64("2025-05-20T00:00:00", "ns")]
    sweeps_energy_value = solve_full_sweep_energy(
        esa_lvl5_hex, sweep_table, esa_unit_conversion_table, lut_notes_table, data_time
    )
    new_fixed_energy_values = np.array(
        [
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1201.466213,
            1131.97314,
            1038.004216,
            951.8359709,
            872.8208434,
            800.3650292,
            733.9240176,
            672.9984993,
            617.1306146,
            565.9005123,
            518.9231943,
            475.8456225,
            436.3440658,
            400.1216672,
            366.9062125,
            336.4480852,
            308.5183902,
            282.9072338,
            259.4221462,
            237.8866352,
            218.13886,
            200.0304145,
            183.4252123,
            168.1984643,
            154.2357401,
            141.432109,
            129.6913506,
            118.9252324,
            109.0528461,
            100,
        ]
    )
    assert np.all(sweeps_energy_value[:, :63] == new_fixed_energy_values)

    # Test mismatch values for 9 fine steps x 4 steps.
    mismatch_value = [1]
    with pytest.raises(
        ValueError, match="These ESA_LVL5 values not found in lut-notes table"
    ):
        solve_full_sweep_energy(
            np.array(mismatch_value),
            [0],
            esa_unit_conversion_table,
            lut_notes_table,
            data_time,
        )
