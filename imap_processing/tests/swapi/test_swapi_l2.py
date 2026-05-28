import json
from unittest.mock import patch

import cdflib
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

SWAPI_RATE_VALIDMAX = 65535 / SWAPI_LIVETIME


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
    processed_hk_data = swapi_l1(collection_obj, descriptor="hk")
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
    processed_sci_data = swapi_l1(collection_obj, descriptor="sci")
    cdf_filename = "imap_swapi_l1_sci_20240924_v999.cdf"
    cdf_path = write_cdf(processed_sci_data[0])
    assert cdf_path.name == cdf_filename

    l1_dataset = load_cdf(cdf_path)
    # Since L2 data's date is before any date supported by ESA unit conversion LUT
    # and earliest date doesn't have energy data besides 'Sweep #' 0,
    # we need to update sweep_id to be 0 instead of 1 to get valid energy values.
    l1_dataset["sweep_table"].values[:] = 0

    # Create L2 CDF File
    l2_dataset = swapi_l2(
        l1_dataset,
        esa_table_df=esa_unit_conversion_table,
        lut_notes_df=lut_notes_table,
    )
    l2_cdf = write_cdf(l2_dataset)
    assert l2_cdf.name == "imap_swapi_l2_sci_20240924_v999.cdf"
    cdf_file = cdflib.CDF(l2_cdf)
    esa_energy_info = cdf_file.varinq("esa_energy")
    esa_energy_attrs = cdf_file.varattsget("esa_energy")
    esa_step_attrs = cdf_file.varattsget("esa_step")
    sci_start_time_attrs = cdf_file.varattsget("sci_start_time")
    swp_l1a_flags_attrs = cdf_file.varattsget("swp_l1a_flags")
    global_attrs = cdf_file.globalattsget()
    assert esa_energy_info.Data_Type_Description == "CDF_DOUBLE"
    assert np.isclose(esa_energy_attrs["FILLVAL"], np.float64(-1.0e31))
    assert esa_energy_attrs["VALIDMAX"] == np.float64(21000.0)
    assert esa_energy_attrs["VALIDMIN"] == np.float64(0.0)
    assert esa_energy_attrs["VAR_TYPE"] == "data"
    assert esa_energy_attrs["DEPEND_1"] == "esa_step"
    assert esa_step_attrs["SCALETYP"] == "linear"
    assert "SCALE_TYP" not in esa_step_attrs
    assert esa_energy_attrs["CATDESC"] == (
        "ESA energy in eV/q corresponding to each step id for each sweep"
    )
    assert esa_energy_attrs["LABLAXIS"] == "Energy (eV/q)"
    assert (
        "corresponding energy in eV/q is provided by esa_energy"
        in esa_step_attrs["CATDESC"]
    )
    assert sci_start_time_attrs["FORMAT"] == "A23"
    assert swp_l1a_flags_attrs["VALIDMAX"] == np.uint16(32767)
    assert "SWP_PCEM_COMP" in swp_l1a_flags_attrs["VAR_NOTES"]
    assert "SCEM_INT_ST" in swp_l1a_flags_attrs["VAR_NOTES"]
    assert (
        "top-hat electrostatic analyzer designed to measure energy-per-charge "
        "distributions" in global_attrs["TEXT"][0]
    )
    assert (
        "https://imap.princeton.edu/spacecraft/instruments/"
        "solar-wind-and-pickup-ions-swapi" in global_attrs["TEXT"][0]
    )
    assert "constant livetime of 145ms" in global_attrs["TEXT"][0]
    assert (
        "includes the ESA energy associated with each voltage step"
        in (global_attrs["TEXT"][0])
    )

    rate_variables = [
        "swp_pcem_rate",
        "swp_scem_rate",
        "swp_coin_rate",
        "swp_pcem_rate_stat_uncert_plus",
        "swp_pcem_rate_stat_uncert_minus",
        "swp_scem_rate_stat_uncert_plus",
        "swp_scem_rate_stat_uncert_minus",
        "swp_coin_rate_stat_uncert_plus",
        "swp_coin_rate_stat_uncert_minus",
    ]
    for variable in rate_variables:
        variable_attrs = cdf_file.varattsget(variable)
        assert np.isclose(variable_attrs["VALIDMAX"], SWAPI_RATE_VALIDMAX)

    pcem_rate_attrs = cdf_file.varattsget("swp_pcem_rate")
    pcem_uncert_plus_attrs = cdf_file.varattsget("swp_pcem_rate_stat_uncert_plus")
    pcem_uncert_minus_attrs = cdf_file.varattsget("swp_pcem_rate_stat_uncert_minus")
    assert pcem_rate_attrs["DELTA_PLUS_VAR"] == "swp_pcem_rate_stat_uncert_plus"
    assert pcem_rate_attrs["DELTA_MINUS_VAR"] == "swp_pcem_rate_stat_uncert_minus"
    assert pcem_uncert_plus_attrs["VAR_TYPE"] == "support_data"
    assert pcem_uncert_minus_attrs["VAR_TYPE"] == "support_data"
    assert pcem_uncert_plus_attrs["FIELDNAM"] != pcem_uncert_minus_attrs["FIELDNAM"]
    assert pcem_uncert_plus_attrs["LABLAXIS"] != pcem_uncert_minus_attrs["LABLAXIS"]
    assert pcem_uncert_plus_attrs["CATDESC"] != pcem_uncert_minus_attrs["CATDESC"]

    # Test uncertainty variables are as expected
    np.testing.assert_array_equal(
        l2_dataset["swp_pcem_rate_stat_uncert_plus"],
        l1_dataset["swp_pcem_counts_stat_uncert_plus"] / SWAPI_LIVETIME,
    )
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
    assert np.all(l2_dataset["esa_energy"].values[0, -9:] == fine_energies)


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
    np.testing.assert_array_equal(sweeps_energy_value[0, :63], fixed_energy_values)

    # Now, test that the last 9 fine energy values are as expected for first sweep.
    # I manually picked those values from LUT table.
    expected_fine_energies = np.array(
        [3220.0, 3151.0, 3083.0, 3017.0, 2953.0, 2889.0, 2827.0, 2767.0, 2707.0]
    )
    np.testing.assert_array_equal(sweeps_energy_value[0, -9:], expected_fine_energies)

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
    np.testing.assert_array_equal(sweeps_energy_value[0, :63], new_fixed_energy_values)

    # Test mismatch values for 9 fine steps x 4 steps.
    mismatch_value = [1]
    with pytest.raises(
        ValueError, match="ESA DAC value '1' not found in LUT notes table"
    ):
        solve_full_sweep_energy(
            np.array(mismatch_value),
            [0],
            esa_unit_conversion_table,
            lut_notes_table,
            data_time,
        )


def test_solve_full_sweep_energy_id3(esa_unit_conversion_table, lut_notes_table):
    """Test the solve_full_sweep_energy function for sweep id 3 case"""
    # Modify the current conversion table we have to have entries for sweep id 3
    esa_unit_conversion_table = esa_unit_conversion_table.copy(deep=True)
    esa_unit_conversion_table.loc[
        esa_unit_conversion_table["Sweep #"] == 2, "Sweep #"
    ] = 3
    # Update the first 3 fine energies to be 0 rather than "solve" / negative
    # This makes 6 fine energy steps for sweep id 3
    # Get sweep 3 rows and set Energy values at indices 63-66 (inclusive) to 0
    sweep_3_mask = esa_unit_conversion_table["Sweep #"] == 3
    sweep_3_indices = esa_unit_conversion_table[sweep_3_mask].index
    esa_unit_conversion_table.loc[sweep_3_indices[63:66], "Energy"] = 0

    # Update the ESA Index Number in lut_notes_table for sweep id 3
    # to match the 6 fine energy steps
    esa_unit_conversion_table.loc[sweep_3_indices[66:], "ESA Index Number"] = np.array(
        [-40, -24, -8, 8, 24, 40]
    )

    esa_lvl5_arr = [4663]
    sweep_table = [3]
    data_time = [np.datetime64("2025-12-24T00:00:00", "ns")]
    esa_lvl5_hex = np.vectorize(lambda x: format(x, "X"))(esa_lvl5_arr)
    sweeps_energy_value = solve_full_sweep_energy(
        esa_lvl5_hex, sweep_table, esa_unit_conversion_table, lut_notes_table, data_time
    )
    assert sweeps_energy_value.shape == (1, 72)

    # The first 3 fine step values should be 0 for sweep id 3
    np.testing.assert_array_equal(
        sweeps_energy_value[0, 63:66], np.array([0.0, 0.0, 0.0])
    )
    # The -6th value (first fine step) should be the same as the last index-80
    # 4663 corresponds to 383rd index in lut_notes_table
    assert sweeps_energy_value[0, -6] == lut_notes_table["Energy"].values[383 - 80]
    assert sweeps_energy_value[0, -1] == lut_notes_table["Energy"].values[383]
