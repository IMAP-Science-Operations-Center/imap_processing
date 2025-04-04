"""Tests the L2b processing for IDEX data"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from imap_processing.idex.idex_l2b import epoch_to_doy, idex_l2b, round_spin_phases
from imap_processing.spice.time import et_to_utc


@pytest.fixture(scope="module")
def l2b_dataset(l2a_dataset: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    dataset = idex_l2b(l2a_dataset, data_version="001")
    return dataset


def test_l2b_logical_source(l2b_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2b_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l2b_sci"
    assert l2b_dataset.attrs["Logical_source"] == expected_src


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
        "spin_phase_quadrants",
        "target_low_fit_impact_charge",
        "target_low_fit_impact_mass_estimate",
        "target_high_fit_impact_charge",
        "target_high_fit_impact_mass_estimate",
        "ion_grid_fit_impact_charge",
        "ion_grid_fit_impact_mass_estimate",
    ]

    cdf_vars = l2b_dataset.variables
    for var in expected_vars:
        assert var in cdf_vars


def test_round_spin_phases():
    """Tests that round_spin_phases() produces expected results."""
    spin_phase_angles = xr.DataArray([90, 1, 10, 200, 359, 179, 100])
    expected_quadrants = [90, 0, 0, 180, 0, 180, 90]

    spin_quadrants = round_spin_phases(spin_phase_angles)
    assert_array_equal(spin_quadrants, expected_quadrants)

    # Test with a larger number of random values
    spin_phase_angles = np.random.randint(0, 360, 1000)
    spin_quadrants = round_spin_phases(spin_phase_angles)
    unique_quadrants = np.unique(spin_quadrants)
    assert set(unique_quadrants) == {0, 90, 180, 270}


def test_round_spin_phases_warning(caplog):
    """Tests that round_spin_phases() logs expected out of range warning."""
    # The last value in the array should trigger a warning since it is >=360.
    spin_phase_angles = xr.DataArray([90, 1, 10, 200, 360])

    with caplog.at_level("WARNING"):
        round_spin_phases(spin_phase_angles)

    assert (
        f"Spin phase angles, {spin_phase_angles.data} "
        f"are outside of the expected spin phase angle range, [0, 360)."
    ) in caplog.text


def test_epoch_to_doy():
    """Tests that epoch_to_doy() produces expected doys."""

    epoch = 756196488384840064
    expected_doy = int(et_to_utc(epoch * 1e-9, "D").split("//")[0].split("-")[1])
    example_epoch = np.full(10, epoch)
    doy = epoch_to_doy(example_epoch)
    assert np.all(doy == expected_doy)
