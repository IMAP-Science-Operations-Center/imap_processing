import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import (
    TIME_PER_BIN,
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
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20250211_v000.csv"
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
        imap_module_directory / "tests/swapi/lut/imap_swapi_lut-notes_20250211_v000.csv"
    )
    df = read_swapi_lut_table(lut_notes_file_path)

    return df


def test_swapi_l2_cdf(
    swapi_l0_test_data_path, esa_unit_conversion_table, lut_notes_table
):
    """Test housekeeping processing and CDF file creation"""
    test_packet_file = swapi_l0_test_data_path / "imap_swapi_l0_raw_20240924_v001.pkts"
    # Create HK CDF File
    processed_hk_data = swapi_l1([test_packet_file])
    hk_cdf_filename = "imap_swapi_l1_hk_20240924_v999.cdf"
    hk_cdf_path = write_cdf(processed_hk_data[0])
    assert hk_cdf_path.name == hk_cdf_filename

    # Create L1 CDF File
    processed_sci_data = swapi_l1([test_packet_file, hk_cdf_path])
    cdf_filename = "imap_swapi_l1_sci_20240924_v999.cdf"
    cdf_path = write_cdf(processed_sci_data[0])
    assert cdf_path.name == cdf_filename

    l1_dataset = processed_sci_data[0]
    l2_dataset = swapi_l2(
        l1_dataset,
        esa_table_df=esa_unit_conversion_table,
        lut_notes_df=lut_notes_table,
    )
    l2_cdf = write_cdf(l2_dataset)
    assert l2_cdf.name == "imap_swapi_l2_sci_20240924_v999.cdf"

    # Test uncertainty variables are as expected
    np.testing.assert_array_equal(
        l2_dataset["swp_pcem_rate_err_plus"],
        l1_dataset["swp_pcem_counts_err_plus"] / TIME_PER_BIN,
    )


def test_solve_full_sweep_energy(esa_unit_conversion_table, lut_notes_table):
    """Test the solve_full_sweep_energy function"""
    # Find 9 fine energies for unique ESA_LVL5 values
    esa_lvl5_arr = [7778, 5673, 4973, 4311]
    esa_lvl5_hex = np.vectorize(lambda x: format(x, "X"))(esa_lvl5_arr)
    sweeps_energy_value = solve_full_sweep_energy(
        esa_lvl5_hex, esa_unit_conversion_table, lut_notes_table
    )
    assert sweeps_energy_value.shape == (4, 72)

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
    expected_energy_values = np.array(
        [17310, 17682, 18062, 18450, 18846, 19251, 19251, 19251, 19251]
    )

    assert np.all(sweeps_energy_value[0, -9:] == expected_energy_values)

    # Test mismatch values for 9 fine steps x 4 steps.
    mismatch_value = [1]
    with pytest.raises(
        ValueError, match="These ESA_LVL5 values not found in lut-notes table"
    ):
        solve_full_sweep_energy(
            np.array(mismatch_value), esa_unit_conversion_table, lut_notes_table
        )

    # Check for value that should return 0 index's energy value.
    # Same as before, I picked values from lut notes table that would
    # result in 0 index.
    esa_lvl5_arr = np.array([format(8168, "X")])
    sweeps_energy_value = solve_full_sweep_energy(
        esa_lvl5_arr, esa_unit_conversion_table, lut_notes_table
    )
    assert sweeps_energy_value.shape == (1, 72)
    assert sweeps_energy_value[0][63] == 19149
    expected_energy_values = np.array(
        [
            19149,
            19251,
            19251,
            19251,
            19251,
            19251,
            19251,
            19251,
            19251,
        ]
    )
    assert np.all(sweeps_energy_value[0, -9:] == expected_energy_values)
