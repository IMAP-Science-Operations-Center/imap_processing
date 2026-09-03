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
    compute_counts_agnostic,
    compute_counts_by_charge_and_mass,
    compute_rates_agnostic,
    compute_rates_by_charge_and_mass,
    get_science_acquisition_on_time,
    idex_l2b,
)
from imap_processing.spice.time import TTJ2000_EPOCH

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
    # idex_l2b takes a single L2A dataset and a single L1B msg dataset, each spanning
    # one 10-day window. Concat the two simulated days together here to exercise a
    # multi-day window.
    combined_l2a_dataset = xr.concat([l2a_dataset, l2a_dataset2], dim="epoch")
    combined_msg_dataset = xr.concat(
        [test_l1b_msg.copy(), l1b_msg_dataset2], dim="epoch"
    )
    datasets = idex_l2b(combined_l2a_dataset, combined_msg_dataset)
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
    expected_src = "imap_idex_l2b_sci-10days"
    assert l2b_dataset.attrs["Logical_source"] == expected_src
    # Verify the CDF file can be created with no errors.
    l2b_dataset.attrs["Data_version"] = "999"
    file_name = write_cdf(l2b_dataset)

    expected_dt64 = TTJ2000_EPOCH + l2b_dataset["epoch"].values[0].astype(
        "timedelta64[ns]"
    )
    expected_date = np.datetime_as_string(expected_dt64, unit="D").replace("-", "")

    assert file_name.exists()
    assert file_name.name == f"imap_idex_l2b_sci-10days_{expected_date}_v999.cdf"
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
    assert l2c_dataset.attrs["Logical_source"] == "imap_idex_l2c_rectangular-map-10days"
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
        "epoch": 1,
        "impact_charge": 10,
        "mass": 10,
        "rectangular_lon_pixel": int(360 / IDEX_SPACING_DEG),
        "rectangular_lat_pixel": int(180 / IDEX_SPACING_DEG),
    }
    l2c_dataset.attrs["Data_version"] = "999"
    # Check the attributes of the dataset by writing to a CDF file
    rect_file_name = write_cdf(l2c_dataset)
    expected_dt64 = TTJ2000_EPOCH + l2c_dataset["epoch"].values[0].astype(
        "timedelta64[ns]"
    )
    expected_date = np.datetime_as_string(expected_dt64, unit="D").replace("-", "")
    assert rect_file_name.exists()
    assert (
        rect_file_name.name
        == f"imap_idex_l2c_rectangular-map-10days_{expected_date}_v999.cdf"
    )
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
        "counts_by_charge",
        "counts_by_mass",
        "rate_by_charge",
        "rate_by_mass",
        "counts",
        "rate",
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

    # The agnostic products are independently computed and remain publishable. The
    # fixture has valid science acquisition uptime tracked over the window, so the
    # rate is a real (non-NaN) value derived from the window's on_seconds.
    assert l2b_dataset["counts"].data.sum() > 0
    assert not np.isnan(l2b_dataset["rate"].data).any()


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


def test_get_science_acquisition_on_time(test_l1b_msg: xr.Dataset):
    """Test the function that calculates the science acquisition on-time."""
    test_l1b_msg = test_l1b_msg.isel(epoch=np.isin(test_l1b_msg.science_on, [0, 1]))
    msg_time = test_l1b_msg.epoch.data
    msg_event = test_l1b_msg.science_on.data
    on_seconds, total_seconds = get_science_acquisition_on_time(msg_time, msg_event)
    # The test data spans 1 day, so the total tracked time should be one day, with
    # less than 1% uptime for the science acquisition.
    assert total_seconds == pytest.approx(SECONDS_IN_DAY)
    assert (on_seconds / total_seconds) * 100 < 1

    msg_ds = test_l1b_msg.copy()
    msg_ds_shifted = msg_ds.copy()
    msg_ds_shifted["epoch"] = msg_ds["epoch"] + NANOSECONDS_IN_DAY
    combined_ds = xr.concat([msg_ds, msg_ds_shifted], dim="epoch")
    # Now spanning 2 days.
    msg_time = combined_ds.epoch.data
    msg_event = combined_ds.science_on.data
    on_seconds, total_seconds = get_science_acquisition_on_time(msg_time, msg_event)
    assert total_seconds == pytest.approx(2 * SECONDS_IN_DAY)
    assert (on_seconds / total_seconds) * 100 < 1


def test_get_science_acquisition_on_time_no_acquisition(caplog):
    """Test the function returns zeros when there is no science acquisition."""
    on_seconds, total_seconds = get_science_acquisition_on_time(
        np.array([]), np.array([])
    )
    assert on_seconds == 0.0
    assert total_seconds == 0.0
    assert "No science acquisition events found" in caplog.text


def test_compute_counts_agnostic_filters_non_dust():
    """Agnostic products count only records classified as dust hits."""
    dataset = xr.Dataset(
        {
            "epoch": ("epoch", np.zeros(4, dtype=np.int64)),
            "dust_hit_flag": ("epoch", [1, 0, 1, 0]),
            "spin_phase": ("epoch", [0, 90, 180, 270]),
            "longitude": ("epoch", [0.0, 1.0, 2.0, 3.0]),
            "latitude": ("epoch", [0.0, 1.0, 2.0, 3.0]),
        }
    )

    counts, counts_map = compute_counts_agnostic(dataset)

    assert counts.sum() == 2
    assert counts_map.sum() == 2


def test_compute_counts_agnostic_excludes_events_without_dust_flag():
    """A missing dust-hit flag must not classify events as dust impacts."""
    dataset = xr.Dataset(
        {
            "epoch": ("epoch", np.zeros(2, dtype=np.int64)),
            "spin_phase": ("epoch", [0, 90]),
            "longitude": ("epoch", [0.0, 1.0]),
            "latitude": ("epoch", [0.0, 1.0]),
        }
    )

    counts, counts_map = compute_counts_agnostic(dataset)

    assert counts.sum() == 0
    assert counts_map.sum() == 0


def test_compute_counts_by_charge_and_mass():
    """Test the compute_counts_by_charge_and_mass function."""

    # Create a mock l2a_dataset with 6 events, each landing in a different one of the
    # first 6 impact charge/mass bins, all in the same spin phase quadrant and sky
    # pixel. Counts should aggregate into a single window record.
    n_events = 6
    l2a_dataset = xr.Dataset(
        {
            "epoch": np.arange(n_events, dtype=np.int64),
            "target_low_dust_mass_estimate": (
                (MASS_BIN_EDGES / FG_TO_KG)[:n_events] + 1e-5
            ),
            "target_low_impact_charge": CHARGE_BIN_EDGES[:n_events],
            "spin_phase": np.full(n_events, 0),
            "longitude": np.full(n_events, 5),
            "latitude": np.full(n_events, 0),
            "dust_hit_flag": np.ones(n_events, dtype=np.int8),
        }
    )

    counts_by_charge, counts_by_mass, charge_map, mass_map, window_epoch = (
        compute_counts_by_charge_and_mass(l2a_dataset)
    )

    expected_shape = (1, len(CHARGE_BIN_EDGES) - 1, len(SPIN_PHASE_BIN_EDGES) - 1)
    expected_map_shape = (
        1,
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
    expected_array[0, 0:n_events, 0] = 1
    # Add ones where we expect counts for the map
    expected_map_array[0, 0:n_events, 0, 15] = 1
    # assert that the counts are as expected
    np.testing.assert_array_equal(counts_by_charge, expected_array)
    np.testing.assert_array_equal(counts_by_mass, expected_array)
    # assert that the counts are as expected for the map
    np.testing.assert_array_equal(charge_map, expected_map_array)
    np.testing.assert_array_equal(mass_map, expected_map_array)

    # The window epoch is the center of the accumulation period: the midpoint
    # between the first and last input epochs.
    epoch_data = l2a_dataset["epoch"].data
    np.testing.assert_allclose(
        window_epoch, [(epoch_data.min() + epoch_data.max()) / 2]
    )


def test_compute_counts_by_charge_and_mass_out_of_bounds():
    """Test the compute_counts_by_charge_and_mass function.

    Test when there are mass and charge values out of the expected bin edges"""

    # Create a test dataset with values that are out of the expected bin edges.
    n_events = 2
    l2a_dataset = xr.Dataset(
        {
            "epoch": np.arange(n_events, dtype=np.int64),
            "target_low_dust_mass_estimate": np.array(
                [MASS_BIN_EDGES[0] - 1e-05, MASS_BIN_EDGES[-1] + 1e-05]
            )
            / FG_TO_KG,
            "target_low_impact_charge": np.array(
                [CHARGE_BIN_EDGES[0] - 1e-05, CHARGE_BIN_EDGES[-1] + 1e-05]
            ),
            "spin_phase": np.full(n_events, 0),
            "longitude": np.array([0, 365]),
            "latitude": np.array([-90, 90]),
            "dust_hit_flag": np.ones(2, dtype=np.int8),
        }
    )

    counts_by_charge, counts_by_mass, charge_map, mass_map, window_epoch = (
        compute_counts_by_charge_and_mass(l2a_dataset)
    )

    expected_shape = (1, len(CHARGE_BIN_EDGES) - 1, len(SPIN_PHASE_BIN_EDGES) - 1)
    expected_map_shape = (
        1,
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
    expected_array[0, len(CHARGE_BIN_EDGES) - 2, 0] = 1
    # Add ones where we expect counts for the map
    expected_map_array[0, 0, 0, 0] = 1
    expected_map_array[0, len(CHARGE_BIN_EDGES) - 2, 0, 29] = 1
    # assert that the counts are as expected
    np.testing.assert_array_equal(counts_by_charge, expected_array)
    np.testing.assert_array_equal(counts_by_mass, expected_array)
    np.testing.assert_array_equal(charge_map, expected_map_array)
    np.testing.assert_array_equal(mass_map, expected_map_array)


def test_compute_rates_by_charge_and_mass():
    """Test the compute_rates_by_charge_and_mass function."""
    # Mock example inputs. A single window record with counts of 5 everywhere.
    counts_by_charge = np.full(
        (1, len(CHARGE_BIN_EDGES) - 1, len(SPIN_PHASE_BIN_EDGES) - 1), 5.0
    )
    counts_by_mass = counts_by_charge.copy()
    counts_by_charge_map = np.full(
        (
            1,
            len(CHARGE_BIN_EDGES) - 1,
            len(SKY_GRID.az_bin_edges) - 1,
            len(SKY_GRID.el_bin_edges) - 1,
        ),
        5.0,
    )
    counts_by_mass_map = counts_by_charge_map.copy()
    on_seconds = 100.0
    total_seconds = 200.0
    # Compute the rates by charge and mass
    rate_by_charge, rate_by_mass, charge_map, mass_map, quality_flags = (
        compute_rates_by_charge_and_mass(
            counts_by_charge,
            counts_by_mass,
            counts_by_charge_map,
            counts_by_mass_map,
            on_seconds,
            total_seconds,
        )
    )

    # Check shapes
    np.testing.assert_equal(rate_by_charge.shape, counts_by_charge.shape)
    np.testing.assert_equal(rate_by_mass.shape, counts_by_mass.shape)
    np.testing.assert_equal(charge_map.shape, counts_by_charge_map.shape)
    np.testing.assert_equal(mass_map.shape, counts_by_mass_map.shape)

    # The quality flag should be 1 (valid) and rates should be counts/on_seconds.
    np.testing.assert_array_equal(quality_flags, [1])
    np.testing.assert_allclose(rate_by_charge, 5.0 / on_seconds)
    np.testing.assert_allclose(rate_by_mass, 5.0 / on_seconds)
    np.testing.assert_allclose(charge_map, 5.0 / on_seconds)
    np.testing.assert_allclose(mass_map, 5.0 / on_seconds)


def test_compute_rates_agnostic():
    """Test the compute_rates_agnostic function for a single window record."""
    counts = np.full((1, 4), 5.0)
    counts_map = np.full((1, 2, 2), 5.0)
    on_seconds = 100.0
    total_seconds = 200.0

    rate, rate_map = compute_rates_agnostic(
        counts, counts_map, on_seconds, total_seconds
    )

    np.testing.assert_allclose(rate, 5.0 / on_seconds)
    np.testing.assert_allclose(rate_map, 5.0 / on_seconds)


def test_compute_rates_agnostic_uses_nan_for_invalid_uptime():
    """Test that agnostic rates use NaN when uptime is missing or zero."""
    counts = np.ones((1, 4))
    counts_map = np.ones((1, 2, 2))

    rate, rate_map = compute_rates_agnostic(counts, counts_map, 0.0, 0.0)

    assert np.all(np.isnan(rate))
    assert np.all(np.isnan(rate_map))


def test_compute_rates_by_charge_and_mass_no_acquisition_data(caplog):
    """Test that the function produces -1 rates when there is no uptime data."""
    caplog.at_level("WARNING")
    # Mock example inputs
    counts_by_charge = np.ones(
        (1, len(CHARGE_BIN_EDGES) - 1, len(SPIN_PHASE_BIN_EDGES) - 1)
    )
    counts_by_mass = counts_by_charge.copy()
    counts_by_charge_map = np.ones(
        (
            1,
            len(CHARGE_BIN_EDGES) - 1,
            len(SKY_GRID.az_bin_edges) - 1,
            len(SKY_GRID.el_bin_edges) - 1,
        )
    )
    counts_by_mass_map = counts_by_charge_map.copy()
    # Compute the rates by charge and mass and assert there is a warning in the logs.
    rate_by_charge, rate_by_mass, charge_map, mass_map, quality_flags = (
        compute_rates_by_charge_and_mass(
            counts_by_charge,
            counts_by_mass,
            counts_by_charge_map,
            counts_by_mass_map,
            0.0,
            0.0,
        )
    )
    assert "Missing or zero science acquisition uptime for this window." in caplog.text

    # All rates by charge and mass should be -1.0
    np.testing.assert_array_equal(rate_by_charge, np.full(rate_by_charge.shape, -1.0))
    np.testing.assert_array_equal(rate_by_mass, np.full(rate_by_mass.shape, -1.0))
    np.testing.assert_array_equal(charge_map, np.full(charge_map.shape, -1.0))
    np.testing.assert_array_equal(mass_map, np.full(mass_map.shape, -1.0))

    # The quality flag should be 0 (invalid) for the missing uptime data.
    np.testing.assert_array_equal(quality_flags, [0])
