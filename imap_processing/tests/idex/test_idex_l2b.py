"""Tests the L2b processing for IDEX data"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex.idex_l2b import (
    bin_spin_phases,
    get_science_acquisition_timestamps,
    idex_l2b,
)
from imap_processing.tests.idex.conftest import L1B_EVT_CDF


@pytest.fixture
def l2b_dataset(l2a_dataset: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    l1b_evt_dataset = load_cdf(L1B_EVT_CDF)
    dataset = idex_l2b(
        [l2a_dataset, l2a_dataset], [l1b_evt_dataset, l1b_evt_dataset]
    )  # decom_test_data_evt[1]])
    return dataset


def test_l2b_logical_source_and_cdf(l2b_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2b_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l2b_sci-1mo"
    assert l2b_dataset.attrs["Logical_source"] == expected_src
    # Verify the CDF file can be created with no errors.
    l2b_dataset.attrs["Data_version"] = "999"
    file_name = write_cdf(l2b_dataset)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l2b_sci-1mo_20251017_v999.cdf"


def test_l2a_cdf_variables(l2b_dataset: xr.Dataset):
    """Tests that the ``idex_l2a`` function generates datasets
    with the expected variables.

    Parameters
    ----------
    l2b_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_vars = [
        "epoch",
        "impact_day_of_year",
        "counts_by_charge",
        "counts_by_mass",
        "rate_by_charge",
        "rate_by_mass",
    ]

    cdf_vars = l2b_dataset.variables
    for var in expected_vars:
        assert var in cdf_vars
    for var in l2b_dataset.data_vars:
        assert "DICT_KEY" in l2b_dataset[var].attrs, (
            f"Variable {var} is missing the DICT_KEY attribute for SPASE metadata."
        )


def test_bin_spin_phases():
    """Tests that bin_spin_phases() produces expected results."""
    # Spin Phase -> 4 bins [315°-45°,45°-135°,135°-225°, 225°-315°]
    spin_phase_angles = xr.DataArray([314, 315, 316, 90, 1, 10, 200, 359, 179, 100])
    expected_bins = [4, 1, 1, 2, 1, 1, 3, 1, 3, 2]

    spin_quadrants = bin_spin_phases(spin_phase_angles)
    assert_array_equal(spin_quadrants, expected_bins)

    # Test with a larger number of random values
    spin_phase_angles = np.random.randint(0, 360, 1000)
    spin_quadrants = bin_spin_phases(spin_phase_angles)
    unique_quadrants = np.unique(spin_quadrants)
    assert set(unique_quadrants) == {1, 2, 3, 4}

    # Test values that are exactly on bin edges
    spin_quadrants = bin_spin_phases(np.array([315, 45, 135, 225]))
    assert_array_equal(spin_quadrants, [1, 2, 3, 4])


def test_bin_spin_phases_warning(caplog):
    """Tests that bin_spin_phases() logs expected out of range warning."""
    # The last value in the array should trigger a warning since it is >=360.
    spin_phase_angles = xr.DataArray([90, 1, 10, 200, 360])

    with caplog.at_level("WARNING"):
        bin_spin_phases(spin_phase_angles)

    assert (
        f"Spin phase angles, {spin_phase_angles.data} "
        f"are outside of the expected spin phase angle range, [0, 360)."
    ) in caplog.text


def test_science_acquisition_times(decom_test_data_evt: list[xr.Dataset]):
    """Tests that the expected science acquisition times and messages are present.

    Parameters
    ----------
    decom_test_data_evt : list[xr.Dataset]
        A ``xarray`` dataset containing the test data
    """
    logs, times, vals = get_science_acquisition_timestamps(decom_test_data_evt[1])
    # For this example event message dataset we expect science acquisition events.
    assert len(logs) == 2
    assert len(times) == 2
    assert len(vals) == 2
    # The first event message is the start of the science acquisition.
    assert logs[0] == "SCI state change: ACQSETUP to ACQ"
    # The second event message is the end of the science acquisition.
    assert logs[1] == "SCI state change: ACQ to CHILL"

    # assert the values are correct
    np.testing.assert_array_equal(vals, [1, 0])
