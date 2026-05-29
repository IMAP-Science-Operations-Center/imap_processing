"""Tests the L2b processing for IDEX data"""

import cdflib
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from imap_processing.cdf.utils import write_cdf
from imap_processing.idex.idex_constants import (
    FG_TO_KG,
    IDEX_SPACING_DEG,
    NANOSECONDS_IN_DAY,
    SECONDS_IN_DAY,
)
from imap_processing.idex.idex_l2b import (
    CHARGE_BIN_EDGES,
    MASS_BIN_EDGES,
    SKY_GRID,
    SPIN_PHASE_BIN_EDGES,
    bin_spin_phases,
    compute_counts_by_charge_and_mass,
    compute_rates_by_charge_and_mass,
    get_science_acquisition_on_percentage,
    idex_l2b,
)

INT_FILLVAL = np.iinfo(np.int64).min


@pytest.fixture
def l2b_and_l2c_datasets(l2a_dataset: xr.Dataset, test_l1b_msg) -> list[xr.Dataset]:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    datasets : list[xr.Dataset]
        A list of ``xarray`` datasets containing the test data for L2B and L2C.
    """
    l1b_msg_dataset2 = (
        test_l1b_msg.copy()
    )  # Add a second dataset with different epoch values for testing
    l2a_dataset2 = (
        l2a_dataset.copy()
    )  # Add a second dataset with different epoch values for testing
    l1b_msg_dataset2["epoch"] = l1b_msg_dataset2["epoch"] + NANOSECONDS_IN_DAY
    l2a_dataset2["epoch"] = l2a_dataset2["epoch"] + NANOSECONDS_IN_DAY
    datasets = idex_l2b(
        [l2a_dataset, l2a_dataset2], [test_l1b_msg.copy(), l1b_msg_dataset2]
    )
    return datasets


@pytest.mark.external_test_data
def test_l2b_logical_source_and_cdf(l2b_and_l2c_datasets: list[xr.Dataset]):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2b_and_l2c_datasets : list[xr.Dataset]
        A ``xarray`` dataset containing the test data
    """
    l2b_dataset = l2b_and_l2c_datasets[0]
    expected_src = "imap_idex_l2b_sci-1mo"
    assert l2b_dataset.attrs["Logical_source"] == expected_src
    # Verify the CDF file can be created with no errors.
    l2b_dataset.attrs["Data_version"] = "999"
    file_name = write_cdf(l2b_dataset)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l2b_sci-1mo_20231218_v999.cdf"
    with cdflib.CDF(file_name) as cdf_file:
        assert cdf_file.varattsget("impact_charge")["LABL_PTR_1"] == "charge_labels"
        for variable_name in ("counts_by_charge", "counts_by_mass"):
            var_info = cdf_file.varinq(variable_name)
            var_attrs = cdf_file.varattsget(variable_name)
            assert var_info.Data_Type_Description == "CDF_INT8"
            assert int(var_attrs["FILLVAL"]) == INT_FILLVAL


@pytest.mark.external_test_data
def test_l2c_attrs_and_vars(
    l2b_and_l2c_datasets: list[xr.Dataset], l2a_dataset: xr.Dataset
):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected variables and attributes.

    Parameters
    ----------
    l2b_and_l2c_datasets : list[xr.Dataset]
        A ``xarray`` dataset containing the l2c test data.
    l2a_dataset
        A ``xarray`` dataset containing the l1b test data.
    """
    l2c_dataset = l2b_and_l2c_datasets[1]
    assert l2c_dataset.attrs["Logical_source"] == "imap_idex_l2c_rectangular-map-1mo"
    # TODO: Uncomment when NAN block fixed
    # The total counts in the map should be equal to the number of dust events
    # in the l2a_dataset (*2 because the l2b fixture counts are doubled)
    # np.testing.assert_allclose(
    #     l2c_dataset["counts_by_charge_map"].sum(), len(l2a_dataset.epoch) * 2
    # )
    # np.testing.assert_allclose(
    #     l2c_dataset["counts_by_mass_map"].sum(), len(l2a_dataset.epoch) * 2
    # )
    assert l2c_dataset.sizes == {
        "on_off_times": 4,
        "epoch": 2,
        "impact_charge": 10,
        "mass": 10,
        "rectangular_lon_pixel": int(360 / IDEX_SPACING_DEG),
        "rectangular_lat_pixel": int(180 / IDEX_SPACING_DEG),
    }
    l2c_dataset.attrs["Data_version"] = "999"
    # Check the attributes of the dataset by writing to a CDF file
    rect_file_name = write_cdf(l2c_dataset)
    assert rect_file_name.exists()
    assert rect_file_name.name == "imap_idex_l2c_rectangular-map-1mo_20231218_v999.cdf"
    with cdflib.CDF(rect_file_name) as cdf_file:
        assert cdf_file.varattsget("impact_charge")["LABL_PTR_1"] == "charge_labels"
        for variable_name in ("counts_by_charge_map", "counts_by_mass_map"):
            var_info = cdf_file.varinq(variable_name)
            var_attrs = cdf_file.varattsget(variable_name)
            assert var_info.Data_Type_Description == "CDF_INT8"
            assert int(var_attrs["FILLVAL"]) == INT_FILLVAL

    for var in l2c_dataset.data_vars:
        assert "DICT_KEY" in l2c_dataset[var].attrs, (
            f"Variable {var} is missing the DICT_KEY attribute for SPASE metadata."
        )

    expected_fill_vars = [
        "counts_by_charge_map",
        "counts_by_mass_map",
    ]
    for var in expected_fill_vars:
        expected_fill = np.full(l2c_dataset[var].shape, INT_FILLVAL, dtype=np.int64)
        assert np.array_equal(l2c_dataset[var].data, expected_fill), (
            f"Variable {var} should be fully set to the integer fill value "
            "for the temporary L2B/L2C patch."
        )

    expected_nan_vars = [
        "rate_by_charge_map",
        "rate_by_mass_map",
    ]
    for var in expected_nan_vars:
        assert np.isnan(l2c_dataset[var].data).all(), (
            f"Variable {var} should be fully NaN for the temporary L2B/L2C patch."
        )


@pytest.mark.external_test_data
def test_l2b_cdf_variables(l2b_and_l2c_datasets: list[xr.Dataset]):
    """Tests that the ``idex_l2a`` function generates datasets
    with the expected variables.

    Parameters
    ----------
    l2b_and_l2c_datasets : list[xr.Dataset]
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
    l2b_dataset = l2b_and_l2c_datasets[0]
    cdf_vars = l2b_dataset.variables
    for var in expected_vars:
        assert var in cdf_vars
    for var in l2b_dataset.data_vars:
        assert "DICT_KEY" in l2b_dataset[var].attrs, (
            f"Variable {var} is missing the DICT_KEY attribute for SPASE metadata."
        )

    expected_fill_vars = [
        "counts_by_charge",
        "counts_by_mass",
    ]
    for var in expected_fill_vars:
        expected_fill = np.full(l2b_dataset[var].shape, INT_FILLVAL, dtype=np.int64)
        assert np.array_equal(l2b_dataset[var].data, expected_fill), (
            f"Variable {var} should be fully set to the integer fill value "
            "for the temporary L2B patch."
        )

    expected_nan_vars = [
        "rate_by_charge",
        "rate_by_mass",
    ]
    for var in expected_nan_vars:
        assert np.isnan(l2b_dataset[var].data).all(), (
            f"Variable {var} should be fully NaN for the temporary L2B patch."
        )


def test_bin_spin_phases():
    """Tests that bin_spin_phases() produces expected results."""
    # Spin Phase -> 4 bins [315°-45°,45°-135°,135°-225°, 225°-315°]
    spin_phase_angles = xr.DataArray([314, 315, 316, 90, 1, 10, 200, 359, 179, 100])
    expected_bins = [3, 0, 0, 1, 0, 0, 2, 0, 2, 1]

    spin_quadrants = bin_spin_phases(spin_phase_angles)
    assert_array_equal(spin_quadrants, expected_bins)

    # Test with a larger number of random values
    spin_phase_angles = np.random.randint(0, 360, 1000)
    spin_quadrants = bin_spin_phases(spin_phase_angles)
    unique_quadrants = np.unique(spin_quadrants)
    assert set(unique_quadrants) == {0, 1, 2, 3}

    # Test values that are exactly on bin edges
    spin_quadrants = bin_spin_phases(np.array([315, 45, 135, 225]))
    assert_array_equal(spin_quadrants, [0, 1, 2, 3])


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


def test_get_science_acquisition_on_percentage(test_l1b_msg: xr.Dataset):
    """Test the function that calculates the percentage of uptime."""
    test_l1b_msg = test_l1b_msg.isel(epoch=np.isin(test_l1b_msg.science_on, [0, 1]))
    msg_time = test_l1b_msg.epoch.data
    msg_event = test_l1b_msg.science_on.data
    on_percentages = get_science_acquisition_on_percentage(msg_time, msg_event)
    # We expect 1 DOY with less than 1% uptime for the science acquisition.
    assert len(on_percentages) == 1
    # The DOY should be 8 for this test dataset.
    assert on_percentages[8] < 1

    msg_ds = test_l1b_msg.copy()
    msg_ds_shifted = msg_ds.copy()
    msg_ds_shifted["epoch"] = msg_ds["epoch"] + NANOSECONDS_IN_DAY
    combined_ds = xr.concat([msg_ds, msg_ds_shifted], dim="epoch")
    # expect a second DOY.
    msg_time = combined_ds.epoch.data
    msg_event = combined_ds.science_on.data
    on_percentages = get_science_acquisition_on_percentage(msg_time, msg_event)
    # We expect 2 DOYs
    assert len(on_percentages) == 2
    # The uptime should be less than 1% for both
    assert on_percentages[8] < 1
    assert on_percentages[9] < 1  # The uptime should be less than 1%


def test_get_science_acquisition_on_percentage_no_acquisition(caplog):
    """Test the function returns an empty dict when there is no science acquisition."""
    on_percentages = get_science_acquisition_on_percentage(np.array([]), np.array([]))
    assert not on_percentages
    assert "No science acquisition events found" in caplog.text


def test_compute_counts_by_charge_and_mass():
    """Test the compute_counts_by_charge_and_mass function."""

    # Create a mock l2a_dataset
    epochs = np.array([1, 1, 2, 2, 3, 4])
    epochs = epochs * NANOSECONDS_IN_DAY

    # Create a test dataset. There should be 1 in the first 5 impact charge bins
    # and mass bins all in the first spin phase bin. The test should be zero. This
    # should be the same for each epoch except the second epoch which has 2 counts in
    # the first 5 mass and impact charge bins.

    l2a_dataset = xr.Dataset(
        {
            "epoch": epochs,
            "target_low_dust_mass_estimate": ((MASS_BIN_EDGES / FG_TO_KG)[:6] + 1e-5),
            "target_low_impact_charge": CHARGE_BIN_EDGES[:6],
            "spin_phase": np.full((6,), 0),
            "longitude": np.full(6, 5),
            "latitude": np.full(6, 0),
        }
    )

    # Unique days of year
    epoch_doy_unique = np.unique(epochs / NANOSECONDS_IN_DAY).astype(int) + 1

    counts_by_charge, counts_by_mass, charge_map, mass_map, daily_epoch = (
        compute_counts_by_charge_and_mass(l2a_dataset, epoch_doy_unique)
    )

    expected_shape = (
        len(epoch_doy_unique),
        len(CHARGE_BIN_EDGES) - 1,
        len(SPIN_PHASE_BIN_EDGES) - 1,
    )
    expected_map_shape = (
        len(epoch_doy_unique),
        len(CHARGE_BIN_EDGES) - 1,
        len(SKY_GRID.az_bin_edges) - 1,
        len(SKY_GRID.el_bin_edges) - 1,
    )
    # Check shapes
    assert counts_by_charge.shape == expected_shape
    assert counts_by_mass.shape == expected_shape
    assert charge_map.shape == expected_map_shape
    assert mass_map.shape == expected_map_shape

    # Check that the counts are correctly binned
    expected_array = np.zeros(expected_shape)
    expected_map_array = np.zeros(expected_map_shape)
    # Add ones where we expect counts
    expected_array[0, 0:2, 0] = 1
    expected_array[1, 2:4, 0] = 1
    expected_array[2, 4, 0] = 1
    expected_array[3, 5, 0] = 1
    # Add ones where we expect counts for the map
    expected_map_array[0, 0:2, 0, 15] = 1
    expected_map_array[1, 2:4, 0, 15] = 1
    expected_map_array[2, 4, 0, 15] = 1
    expected_map_array[3, 5, 0, 15] = 1
    # assert that the counts are as expected
    np.testing.assert_array_equal(counts_by_charge, expected_array)
    np.testing.assert_array_equal(counts_by_mass, expected_array)
    # assert that the counts are as expected for the map
    np.testing.assert_array_equal(charge_map, expected_map_array)
    np.testing.assert_array_equal(mass_map, expected_map_array)


def test_compute_counts_by_charge_and_mass_out_of_bounds():
    """Test the compute_counts_by_charge_and_mass function.

    Test when there are mass and charge values out of the expected bin edges"""

    # Create a mock l2a_dataset
    epochs = np.array([1, 2])
    epochs = epochs * NANOSECONDS_IN_DAY

    # Create a test dataset with values that are out of the expected bin edges.
    l2a_dataset = xr.Dataset(
        {
            "epoch": epochs,
            "target_low_dust_mass_estimate": np.array(
                [MASS_BIN_EDGES[0] - 1e-05, MASS_BIN_EDGES[-1] + 1e-05]
            )
            / FG_TO_KG,
            "target_low_impact_charge": np.array(
                [CHARGE_BIN_EDGES[0] - 1e-05, CHARGE_BIN_EDGES[-1] + 1e-05]
            ),
            "spin_phase": np.full((6,), 0),
            "longitude": np.array([0, 365]),
            "latitude": np.array([-90, 90]),
        }
    )

    # Unique days of year
    epoch_doy_unique = np.unique(epochs / NANOSECONDS_IN_DAY).astype(int) + 1

    counts_by_charge, counts_by_mass, charge_map, mass_map, daily_epoch = (
        compute_counts_by_charge_and_mass(l2a_dataset, epoch_doy_unique)
    )

    expected_shape = (
        len(epoch_doy_unique),
        len(CHARGE_BIN_EDGES) - 1,
        len(SPIN_PHASE_BIN_EDGES) - 1,
    )
    expected_map_shape = (
        len(epoch_doy_unique),
        len(CHARGE_BIN_EDGES) - 1,
        len(SKY_GRID.az_bin_edges) - 1,
        len(SKY_GRID.el_bin_edges) - 1,
    )
    # Check shapes
    assert counts_by_charge.shape == expected_shape
    assert counts_by_mass.shape == expected_shape
    assert charge_map.shape == expected_map_shape
    assert mass_map.shape == expected_map_shape

    # Check that the counts are correctly binned
    expected_array = np.zeros(expected_shape)
    expected_map_array = np.zeros(expected_map_shape)
    # Add ones where we expect counts
    expected_array[0, 0, 0] = 1
    expected_array[1, len(CHARGE_BIN_EDGES) - 2, 0] = 1
    # Add ones where we expect counts for the map
    expected_map_array[0, 0, 0, 0] = 1
    expected_map_array[1, len(CHARGE_BIN_EDGES) - 2, 0, 29] = 1
    # assert that the counts are as expected
    np.testing.assert_array_equal(counts_by_charge, expected_array)
    np.testing.assert_array_equal(counts_by_mass, expected_array)
    np.testing.assert_array_equal(charge_map, expected_map_array)
    np.testing.assert_array_equal(mass_map, expected_map_array)


def test_compute_rates_by_charge_and_mass():
    """Test the compute_rates_by_charge_and_mass function."""
    # Mock example inputs
    day_counts = np.full((len(CHARGE_BIN_EDGES), len(SPIN_PHASE_BIN_EDGES) - 1), 1.0)
    day_counts_map = np.full(
        (
            len(CHARGE_BIN_EDGES),
            len(SKY_GRID.az_bin_edges) - 1,
            len(SKY_GRID.el_bin_edges) - 1,
        ),
        1.0,
    )
    counts_by_charge = np.stack(
        [day_counts, day_counts + 1, day_counts + 2, day_counts + 2]
    )
    counts_by_mass = counts_by_charge
    counts_by_mass_map = np.stack(
        [day_counts_map, day_counts_map + 1, day_counts_map + 2, day_counts_map + 2]
    )
    counts_by_charge_map = counts_by_mass_map
    # Mock DOY values for the epochs
    epoch_doy = np.array([1, 2, 3, 4])
    # Mock daily idex uptime percentages
    daily_on_percentage = {1: 50.0, 2: 25.0, 3: 0.05, 4: 0.0}
    # Compute the rates by charge and mass
    rate_by_charge, rate_by_mass, charge_map, mass_map, quality_flags = (
        compute_rates_by_charge_and_mass(
            counts_by_charge,
            counts_by_mass,
            counts_by_charge_map,
            counts_by_mass_map,
            epoch_doy,
            daily_on_percentage,
        )
    )

    # Check shapes
    expected_shape = counts_by_mass.shape
    expected_map_shape = counts_by_mass_map.shape
    np.testing.assert_equal(rate_by_charge.shape, expected_shape)
    np.testing.assert_equal(rate_by_mass.shape, expected_shape)
    np.testing.assert_equal(charge_map.shape, expected_map_shape)
    np.testing.assert_equal(mass_map.shape, expected_map_shape)

    # Assert all quality flags are 1.
    np.testing.assert_array_equal(quality_flags, np.ones_like(quality_flags))
    # assert day 1 rates are as expected
    np.testing.assert_equal(rate_by_charge[0], 1 / (SECONDS_IN_DAY / 2))
    np.testing.assert_equal(rate_by_mass[0], 1 / (SECONDS_IN_DAY / 2))
    np.testing.assert_equal(charge_map[0], 1 / (SECONDS_IN_DAY / 2))
    np.testing.assert_equal(mass_map[0], 1 / (SECONDS_IN_DAY / 2))
    # assert day 2 rates are as expected
    np.testing.assert_equal(rate_by_charge[1], 2 / (SECONDS_IN_DAY / 4))
    np.testing.assert_equal(rate_by_mass[1], 2 / (SECONDS_IN_DAY / 4))
    np.testing.assert_equal(charge_map[1], 2 / (SECONDS_IN_DAY / 4))
    np.testing.assert_equal(mass_map[1], 2 / (SECONDS_IN_DAY / 4))
    # assert day 3 rates are as expected
    np.testing.assert_equal(rate_by_charge[2], 3 / (SECONDS_IN_DAY / 2000))
    np.testing.assert_equal(rate_by_mass[2], 3 / (SECONDS_IN_DAY / 2000))
    np.testing.assert_equal(charge_map[2], 3 / (SECONDS_IN_DAY / 2000))
    np.testing.assert_equal(mass_map[2], 3 / (SECONDS_IN_DAY / 2000))
    # assert day 4 rates are as expected
    np.testing.assert_equal(rate_by_charge[3], -1.0)
    np.testing.assert_equal(rate_by_mass[3], -1.0)
    np.testing.assert_equal(charge_map[3], -1.0)
    np.testing.assert_equal(mass_map[3], -1.0)


def test_compute_rates_by_charge_and_mass_missing_acquisition_time(caplog):
    """Test that the function throws an error for missing data."""
    caplog.at_level("WARNING")
    # Mock example inputs
    counts_by_charge = np.ones(
        (2, len(CHARGE_BIN_EDGES), len(SPIN_PHASE_BIN_EDGES) - 1)
    )
    counts_by_mass = counts_by_charge
    counts_by_charge_map = np.ones(
        (
            2,
            len(CHARGE_BIN_EDGES),
            len(SKY_GRID.az_bin_edges) - 1,
            len(SKY_GRID.el_bin_edges) - 1,
        )
    )
    counts_by_mass_map = counts_by_charge_map
    # Mock DOY values for the epochs
    epoch_doy = np.array([1, 2])
    # Mock daily idex uptime percentages. Purposefully leave out day 2 to simulate
    # missing acquisition times
    daily_on_percentage = {1: 100.0}
    # Compute the rates by charge and mass and assert there is a warning in the logs.
    rate_by_charge, rate_by_mass, rate_mass_map, rate_charge_map, quality_flags = (
        compute_rates_by_charge_and_mass(
            counts_by_charge,
            counts_by_mass,
            counts_by_charge_map,
            counts_by_mass_map,
            epoch_doy,
            daily_on_percentage,
        )
    )
    assert (
        "Missing science acquisition uptime percentages for day(s) of year: [2]."
        in caplog.text
    )

    # All rates by charge and mass should be -1.0 at epoch 2
    np.testing.assert_array_equal(
        rate_by_charge[1], np.full(rate_by_charge[1].shape, -1.0)
    )
    np.testing.assert_array_equal(rate_by_mass[1], np.full(rate_by_mass[1].shape, -1.0))

    # Assert that quality flags are 0 for the missing acquisition time
    assert quality_flags[0] == 1
    assert quality_flags[1] == 0


def test_compute_rates_no_acquisition_data(caplog):
    """Test that the function produces -1 rates when hk data isn't available."""
    caplog.at_level("WARNING")
    # Mock example inputs
    counts_by_charge = np.ones(
        (2, len(CHARGE_BIN_EDGES), len(SPIN_PHASE_BIN_EDGES) - 1)
    )
    counts_by_mass = counts_by_charge
    counts_by_charge_map = np.ones(
        (
            2,
            len(CHARGE_BIN_EDGES),
            len(SKY_GRID.az_bin_edges) - 1,
            len(SKY_GRID.el_bin_edges) - 1,
        )
    )
    counts_by_mass_map = counts_by_charge_map
    # Mock DOY values for the epochs
    epoch_doy = np.array([1, 2])
    # Mock daily idex uptime percentages. Purposefully leave out day 2 to simulate
    # missing acquisition times
    daily_on_percentage = {}
    # Compute the rates by charge and mass and assert there is a warning in the logs.
    rate_by_charge, rate_by_mass, rate_mass_map, rate_charge_map, quality_flags = (
        compute_rates_by_charge_and_mass(
            counts_by_charge,
            counts_by_mass,
            counts_by_charge_map,
            counts_by_mass_map,
            epoch_doy,
            daily_on_percentage,
        )
    )

    # All rates by charge and mass should be -1.0
    np.testing.assert_array_equal(rate_by_charge, np.full(rate_by_charge.shape, -1.0))
    np.testing.assert_array_equal(rate_by_mass, np.full(rate_by_mass.shape, -1.0))

    # Assert that quality flags are all 0 for missing acquisition times
    np.testing.assert_array_equal(quality_flags, np.full(quality_flags.shape, 0))
