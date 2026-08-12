"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

import io
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi import hi_l1c, utils
from imap_processing.hi.utils import HIAPID, GainConfigLookupTable, HiConstants
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_et


@pytest.fixture(scope="module")
def hi_l1b_de_dataset(hi_l1_test_data_path):
    """Load the Hi L1B DE test dataset."""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    return load_cdf(l1b_de_path)


@pytest.fixture(scope="module")
def hi_goodtimes_dataset(hi_l1_test_data_path):
    """Load the Hi goodtimes test dataset."""
    goodtimes_path = (
        hi_l1_test_data_path / "imap_hi_l1b_45sensor-goodtimes_20250415_v999.cdf"
    )
    return load_cdf(goodtimes_path)


@mock.patch("imap_processing.hi.hi_l1c.generate_pset_dataset")
def test_hi_l1c(
    mock_generate_pset_dataset,
    hi_test_cal_prod_config_path,
    hi_test_background_config_path,
    hi_test_gain_configuration_path,
):
    """Test coverage for hi_l1c function"""
    mock_generate_pset_dataset.return_value = xr.Dataset()
    pset = hi_l1c.hi_l1c(
        xr.Dataset(),
        hi_test_cal_prod_config_path,
        xr.Dataset(),
        hi_test_background_config_path,
        hi_test_gain_configuration_path,
    )[0]
    # Empty attributes, global values get added in post-processing
    assert pset.attrs == {}


@pytest.mark.external_kernel
@pytest.mark.external_test_data
def test_generate_pset_dataset(
    hi_l1b_de_dataset,
    hi_goodtimes_dataset,
    hi_test_cal_prod_config_path,
    hi_test_background_config_path,
    hi_test_gain_configuration_path,
    use_fake_spin_data_for_time,
    use_fake_repoint_data_for_time,
    imap_ena_sim_metakernel,
):
    """Test coverage for generate_pset_dataset function"""
    use_fake_spin_data_for_time(482372987.999)
    l1b_dataset = hi_l1b_de_dataset.copy()
    # The real fixture CDF predates the gain_configuration_id L1B global
    # attribute; add a placeholder so pset_geometric_factor() has something
    # to look up.
    l1b_dataset.attrs["gain_configuration_id"] = 0
    l1b_met = l1b_dataset["ccsds_met"].values[0]
    # Set repoint start and end times.
    seconds_per_day = 24 * 60 * 60
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + seconds_per_day]),
        np.asarray([l1b_met, l1b_met + seconds_per_day + 1]),
    )
    goodtimes = hi_goodtimes_dataset

    l1c_dataset = hi_l1c.generate_pset_dataset(
        l1b_dataset,
        hi_test_cal_prod_config_path,
        goodtimes,
        hi_test_background_config_path,
        hi_test_gain_configuration_path,
    )

    assert l1c_dataset.epoch.data[0] == l1b_dataset.epoch.data[0].astype(np.int64)
    assert l1c_dataset.epoch_delta.data[0] == seconds_per_day * 1e9

    np.testing.assert_array_equal(l1c_dataset.despun_z.data.shape, (1, 3))
    np.testing.assert_array_equal(l1c_dataset.hae_latitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(l1c_dataset.hae_longitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(l1c_dataset.exposure_times.data.shape, (1, 9, 3600))
    for var in [
        "counts",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        np.testing.assert_array_equal(l1c_dataset[var].data.shape, (1, 9, 2, 3600))

    # Test ISTP compliance by writing CDF
    write_cdf(l1c_dataset)


@mock.patch("imap_processing.hi.hi_l1c.pset_backgrounds")
@mock.patch("imap_processing.hi.hi_l1c.pset_exposure")
@mock.patch("imap_processing.hi.hi_l1c.pset_counts")
@mock.patch("imap_processing.hi.hi_l1c.pset_geometry")
@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times")
def test_generate_pset_dataset_uses_midpoint_time(
    mock_get_pointing_times,
    mock_pset_geometry,
    mock_pset_counts,
    mock_pset_exposure,
    mock_pset_backgrounds,
    hi_test_cal_prod_config_path,
    hi_test_background_config_path,
    hi_test_gain_configuration_path,
):
    """Test that generate_pset_dataset uses midpoint ET for pset_geometry."""
    # Create a mock L1B dataset
    l1b_met = 482373065.0
    n_energy_steps = 2
    mock_l1b_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(np.arange(10), dims=["epoch"]),
        },
        data_vars={
            "ccsds_met": xr.DataArray(np.full(10, l1b_met), dims=["epoch"]),
            "esa_energy_step": xr.DataArray(
                np.concat(
                    (np.arange(n_energy_steps + 1).repeat(2), np.array([255, 255]))
                ),
                attrs={"FILLVAL": 255},
            ),
        },
        attrs={
            "Logical_file_id": "imap_hi_l1b_45sensor-de_20250415_v999",
            "Logical_source": "imap_hi_l1b_45sensor-de",
            "gain_configuration_id": 0,
        },
    )

    # Mock get_pointing_times to return known start and end times
    pointing_start_met = l1b_met - 1000.0
    pointing_end_met = l1b_met + 1000.0
    mock_get_pointing_times.return_value = (pointing_start_met, pointing_end_met)

    # Mock the return values for the sub-functions
    mock_pset_geometry.return_value = {}
    mock_pset_counts.return_value = {}
    # pset_exposure must return exposure_times for pset_backgrounds to use
    mock_exposure_times = xr.DataArray(
        np.ones((1, n_energy_steps, 3600), dtype=np.float32),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )
    mock_pset_exposure.return_value = {"exposure_times": mock_exposure_times}
    mock_pset_backgrounds.return_value = {}

    # Call generate_pset_dataset
    _ = hi_l1c.generate_pset_dataset(
        mock_l1b_dataset,
        hi_test_cal_prod_config_path,
        xr.Dataset(),
        hi_test_background_config_path,
        hi_test_gain_configuration_path,
    )

    # Calculate expected midpoint ET
    # The PSET dataset should have epoch and epoch_delta based on pointing times
    expected_epoch = met_to_ttj2000ns(np.array([pointing_start_met]))[0]
    expected_epoch_delta = (
        met_to_ttj2000ns(np.array([pointing_end_met]))[0]
        - met_to_ttj2000ns(np.array([pointing_start_met]))[0]
    )
    expected_midpoint_ttj2000 = expected_epoch + expected_epoch_delta / 2
    expected_midpoint_et = ttj2000ns_to_et(expected_midpoint_ttj2000)

    # Verify that pset_geometry was called with the midpoint ET time
    mock_pset_geometry.assert_called_once()
    actual_et_arg = mock_pset_geometry.call_args[0][0]
    actual_sensor_arg = mock_pset_geometry.call_args[0][1]

    # Use approximate comparison for the ET time (floating point)
    np.testing.assert_allclose(actual_et_arg, expected_midpoint_et, rtol=1e-10)
    assert actual_sensor_arg == "45sensor"


@mock.patch("imap_processing.hi.hi_l1c.load_gain_configuration")
def test_pset_geometric_factor_looks_up_by_config_id(mock_load_gain_config):
    """Test coverage for pset_geometric_factor when the pointing was classified."""
    gain_config_df = pd.DataFrame(
        {"geometric_factor": [0.001, 0.002, 0.003]},
        index=pd.MultiIndex.from_tuples(
            [(0, 1), (0, 2), (0, 3)], names=["config_id", "esa_energy_step"]
        ),
    )
    mock_load_gain_config.return_value = gain_config_df

    pset_coords = {
        "epoch": xr.DataArray([0], dims=["epoch"]),
        "esa_energy_step": xr.DataArray([1, 2, 3], dims=["esa_energy_step"]),
    }
    l1b_de_dataset = xr.Dataset(attrs={"gain_configuration_id": 0})

    result = hi_l1c.pset_geometric_factor(
        pset_coords, l1b_de_dataset, "Fake gain config path"
    )

    np.testing.assert_allclose(
        result["geometric_factor"].values[0], [0.001, 0.002, 0.003], rtol=1e-6
    )
    mock_load_gain_config.assert_called_once_with("Fake gain config path")


def test_pset_geometric_factor_no_match_returns_fillval():
    """Test coverage for pset_geometric_factor when the pointing was not classified."""
    pset_coords = {
        "epoch": xr.DataArray([0], dims=["epoch"]),
        "esa_energy_step": xr.DataArray([1, 2, 3], dims=["esa_energy_step"]),
    }
    l1b_de_dataset = xr.Dataset(
        attrs={"gain_configuration_id": GainConfigLookupTable.NO_MATCH}
    )

    result = hi_l1c.pset_geometric_factor(
        pset_coords, l1b_de_dataset, "Fake gain config path"
    )

    fillval = np.float32(result["geometric_factor"].attrs["FILLVAL"])
    np.testing.assert_array_equal(
        result["geometric_factor"].values[0], [fillval, fillval, fillval]
    )


def test_empty_pset_dataset(use_fake_repoint_data_for_time):
    """Test coverage for empty_pset_dataset function"""
    n_energy_steps = 8
    l1b_esa_energy_steps = xr.DataArray(
        data=np.concat((np.arange(n_energy_steps + 1).repeat(2), np.array([255, 255]))),
        attrs={"FILLVAL": 255},
    )
    # Create calibration product numbers array (0, 1, 2, 3, 4)
    cal_prod_numbers = np.arange(5)
    sensor_str = HIAPID.H90_SCI_DE.sensor
    l1b_met = 482373065
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + 24 * 60 * 60])
    )

    dataset = hi_l1c.empty_pset_dataset(
        l1b_met, l1b_esa_energy_steps, cal_prod_numbers, sensor_str
    )

    assert dataset.epoch.size == 1
    assert dataset.epoch_delta.size == 1
    assert dataset.spin_angle_bin.size == 3600
    assert dataset.esa_energy_step.size == n_energy_steps
    np.testing.assert_array_equal(
        dataset.esa_energy_step.data, np.arange(n_energy_steps) + 1
    )
    assert dataset.calibration_prod.size == len(cal_prod_numbers)
    np.testing.assert_array_equal(dataset.calibration_prod.data, cal_prod_numbers)

    # verify that attrs defined in hi_pset_epoch have overwritten default
    # epoch attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
    for var_name in ["epoch", "epoch_delta"]:
        expected_attrs = attr_mgr.get_variable_attributes(
            f"hi_pset_{var_name}", check_schema=False
        )
        for k, v in expected_attrs.items():
            assert k in dataset[var_name].attrs
            assert dataset[var_name].attrs[k] == v


@pytest.mark.parametrize("sensor_str", ["90sensor", "45sensor"])
@mock.patch("imap_processing.spice.geometry.frame_transform")
@mock.patch("imap_processing.hi.hi_l1c.frame_transform")
def test_pset_geometry(mock_frame_transform, mock_geom_frame_transform, sensor_str):
    """Test coverage for pset_geometry function"""
    # pset_geometry uses both frame_transform and frame_transform_az_el. By mocking
    # the frame_transform imported into hi_l1c as well as the geometry.frame_transform
    # the underlying need for SPICE kernels is remove. Mock them both to just return
    # the input position vectors.
    mock_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos
    mock_geom_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos

    geometry_vars = hi_l1c.pset_geometry(0, sensor_str)

    assert "despun_z" in geometry_vars
    np.testing.assert_array_equal(geometry_vars["despun_z"].data, [[0, 0, 1]])

    assert "hae_latitude" in geometry_vars
    assert "hae_longitude" in geometry_vars
    # frame_transform is mocked to return the input vectors. For Hi-90, we
    # expect hae_latitude to be 0, and for Hi-45 we expect -45. Both sensors
    # have an expected longitude to be 0.1 degree steps starting at 0.05
    expected_latitude = 0 if sensor_str == "90sensor" else -45
    np.testing.assert_array_equal(
        geometry_vars["hae_latitude"].data, np.full((1, 3600), expected_latitude)
    )
    np.testing.assert_allclose(
        geometry_vars["hae_longitude"].data,
        np.arange(0.05, 360, 0.1, dtype=np.float32).reshape((1, 3600)),
        atol=4e-05,
    )


@pytest.mark.external_test_data
@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
def test_pset_counts(
    mock_pointing_times,
    hi_l1b_de_dataset,
    hi_goodtimes_dataset,
    hi_test_cal_prod_config_path,
    hi_test_background_config_path,
):
    """Test coverage for pset_counts function."""
    cal_config_df = utils.CalibrationProductConfig.from_csv(
        hi_test_cal_prod_config_path
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100,
        hi_l1b_de_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )
    counts_var = hi_l1c.pset_counts(
        empty_pset.coords, cal_config_df, hi_l1b_de_dataset, hi_goodtimes_dataset
    )
    assert "counts" in counts_var


@pytest.mark.external_test_data
@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
def test_pset_counts_empty_l1b(
    mock_pointing_times,
    hi_l1b_de_dataset,
    hi_goodtimes_dataset,
    hi_test_cal_prod_config_path,
    hi_test_background_config_path,
):
    """Test coverage for pset_counts function when the input L1b contains no counts."""
    # Make a copy and modify it -
    # remove all but one event and set its trigger_id to zero
    l1b_dataset = hi_l1b_de_dataset.isel(event_met=[0]).copy(deep=True)
    l1b_dataset["trigger_id"].data[0] = 0
    cal_config_df = utils.CalibrationProductConfig.from_csv(
        hi_test_cal_prod_config_path
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100,
        l1b_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )
    counts_var = hi_l1c.pset_counts(
        empty_pset.coords, cal_config_df, l1b_dataset, hi_goodtimes_dataset
    )
    assert counts_var["counts"].data.sum() == 0


def test_get_tof_window_mask():
    """Test coverage for get_tof_window_mask function."""
    # Create a synthetic dataframe with required columns containing data
    # intended to test all aspects of the function.
    fill_vals = {
        "tof_ab": -11,
        "tof_ac1": -12,
        "tof_bc1": -13,
        "tof_c1c2": -14,
    }
    # Use dict-based tof_windows instead of named tuple
    tof_windows = {
        "tof_ab": (0, 1),
        "tof_ac1": (-1, 2),
        "tof_bc1": (1, 5),
        "tof_c1c2": (4, 6),
    }
    synth_df = xr.Dataset(
        coords={
            "event_met": xr.DataArray(
                np.arange(7), name="event_met", dims=["event_met"]
            )
        },
        data_vars={
            "tof_ab": xr.DataArray(
                np.array(
                    [0, 2, 1, 0, -1, -5, -11], dtype=np.int32
                ),  # T, F, T, T, F, F, FILL
                dims=["event_met"],
            ),
            "tof_ac1": xr.DataArray(
                np.array(
                    [-1, 2, -2, 0, 3, 0, -12], dtype=np.int32
                ),  # T, T, F, T, F, T, FILL
                dims=["event_met"],
            ),
            "tof_bc1": xr.DataArray(
                np.array(
                    [1, 5, 3, 0, 6, 2, -13], dtype=np.int32
                ),  # T, T, T, F, F, T, FILL
                dims=["event_met"],
            ),
            "tof_c1c2": xr.DataArray(
                np.array(
                    [4, 6, 5, 3, 7, -9, -14], dtype=np.int32
                ),  # T, T, T, F, F, F, FILL
                dims=["event_met"],
            ),
        },
    )
    expected_mask = np.array([True, False, False, False, False, False, True])
    window_mask = utils.get_tof_window_mask(synth_df, tof_windows, fill_vals)
    np.testing.assert_array_equal(expected_mask, window_mask)


def test_empty_pset_dataset_arbitrary_cal_prod_numbers(use_fake_repoint_data_for_time):
    """Test empty_pset_dataset with non-sequential calibration product numbers."""
    n_energy_steps = 3
    l1b_esa_energy_steps = xr.DataArray(
        data=np.concat((np.arange(n_energy_steps + 1).repeat(2), np.array([255, 255]))),
        attrs={"FILLVAL": 255},
    )
    # Use non-sequential calibration product numbers
    cal_prod_numbers = np.array([5, 10, 100])
    sensor_str = HIAPID.H45_SCI_DE.sensor
    l1b_met = 482373065
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + 24 * 60 * 60])
    )

    dataset = hi_l1c.empty_pset_dataset(
        l1b_met, l1b_esa_energy_steps, cal_prod_numbers, sensor_str
    )

    # Verify calibration_prod coordinate has the correct non-sequential values
    assert dataset.calibration_prod.size == len(cal_prod_numbers)
    np.testing.assert_array_equal(dataset.calibration_prod.data, cal_prod_numbers)
    # Verify the calibration_prod_label reflects the actual numbers
    expected_labels = np.array(["5", "10", "100"])
    np.testing.assert_array_equal(dataset.calibration_prod_label.data, expected_labels)


@pytest.mark.external_test_data
def test_pset_counts_arbitrary_cal_prod_numbers(
    hi_l1b_de_dataset, hi_goodtimes_dataset, use_fake_repoint_data_for_time
):
    """Test pset_counts with non-sequential calibration product numbers."""
    # Create a test calibration product config with non-sequential numbers
    csv_content = """\
calibration_prod,esa_energy_step,geometric_factor,coincidence_type_list,tof_ab_low,tof_ab_high,tof_ac1_low,tof_ac1_high,tof_bc1_low,tof_bc1_high,tof_c1c2_low,tof_c1c2_high
5,1,0.00055,ABC1C2,0,1023,-1023,1023,-1023,1023,0,1023
5,2,0.00085,ABC1C2,0,1023,-1023,1023,-1023,1023,0,1023
10,1,0.00055,BC1C2,0,1023,-1023,1023,-1023,1023,0,1023
10,2,0.00085,BC1C2,0,1023,-1023,1023,-1023,1023,0,1023
    """

    cal_config_df = utils.CalibrationProductConfig.from_csv(io.StringIO(csv_content))

    # Create PSET with non-sequential calibration product numbers
    l1b_met = 482373065
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + 24 * 60 * 60])
    )

    empty_pset = hi_l1c.empty_pset_dataset(
        l1b_met,
        hi_l1b_de_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )

    # Verify the calibration_prod coordinate has non-sequential values
    np.testing.assert_array_equal(empty_pset.calibration_prod.data, np.array([5, 10]))

    # Mock get_pointing_times to avoid SPICE kernel requirements
    with mock.patch(
        "imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200)
    ):
        counts_var = hi_l1c.pset_counts(
            empty_pset.coords, cal_config_df, hi_l1b_de_dataset, hi_goodtimes_dataset
        )

    # Verify counts array has correct shape based on coordinates
    assert "counts" in counts_var
    # Shape should be (n_epoch, n_esa_energy, n_cal_prod, n_spin_bins)
    # where n_cal_prod is 2 (for products 5 and 10)
    expected_shape = (
        1,
        empty_pset.esa_energy_step.size,
        2,  # Two calibration products: 5 and 10
        3600,
    )
    assert counts_var["counts"].data.shape == expected_shape
    # Check that total number of expected counts is correct
    # ABC1C2 is coincidence type 15
    esa_1_2_mask = (
        hi_l1b_de_dataset["esa_step"][hi_l1b_de_dataset["ccsds_index"]] < 3
    ).values
    coincidence_15_mask = (hi_l1b_de_dataset["coincidence_type"] == 15).values
    np.testing.assert_equal(
        np.sum(counts_var["counts"].data[:, :, 0]),
        np.sum(coincidence_15_mask & esa_1_2_mask),
    )
    # BC1C2 is coincidence type 7
    coincidence_7_mask = (hi_l1b_de_dataset["coincidence_type"] == 7).values
    np.testing.assert_equal(
        np.sum(counts_var["counts"].data[:, :, 1]),
        np.sum(coincidence_7_mask & esa_1_2_mask),
    )


@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
@mock.patch("imap_processing.hi.hi_l1c.iter_qualified_events_by_config")
def test_pset_counts_goodtimes_filtering(
    mock_iter_qualified,
    mock_pointing_times,
):
    """Test that pset_counts properly filters events based on goodtimes."""
    # Create 10 events: METs 100-109, nominal_bins 0-9, all at spin_phase=0.5
    # (spin_phase 0.5 -> spin_angle_bin 1800)
    n_events = 10
    event_mets = np.arange(100.0, 100.0 + n_events)
    nominal_bins = np.arange(n_events, dtype=np.uint8)

    l1b_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(np.arange(2), dims=["epoch"]),
            "event_met": xr.DataArray(event_mets, dims=["event_met"]),
        },
        data_vars={
            "trigger_id": xr.DataArray(
                np.ones(n_events, dtype=np.uint16),
                dims=["event_met"],
                attrs={"FILLVAL": 65535},
            ),
            "nominal_bin": xr.DataArray(nominal_bins, dims=["event_met"]),
            "spin_phase": xr.DataArray(np.full(n_events, 0.5), dims=["event_met"]),
            "ccsds_index": xr.DataArray(
                np.zeros(n_events, dtype=np.int32), dims=["event_met"]
            ),
            "esa_energy_step": xr.DataArray(
                np.array([1, 1], dtype=np.uint8),
                dims=["epoch"],
                attrs={"FILLVAL": 255},
            ),
        },
        attrs={"Logical_source": "imap_hi_l1b_90sensor-de"},
    )

    # Goodtimes: METs 100-104 good, METs 105-109 bad
    goodtimes_ds = xr.Dataset(
        {
            "cull_flags": xr.DataArray(
                np.zeros((2, 90), dtype=np.uint8),
                dims=["met", "spin_bin"],
            ),
        },
        coords={"met": [100.0, 105.0], "spin_bin": np.arange(90)},
    )
    goodtimes_ds["cull_flags"].values[1, :] = 1  # All bins bad for MET >= 105

    # Create empty pset with single ESA step and single calibration product
    empty_pset = hi_l1c.empty_pset_dataset(
        100,
        l1b_dataset.esa_energy_step,
        np.array([0]),
        HIAPID.H90_SCI_DE.sensor,
    )

    # Mock iter_qualified_events_by_config to mark all events as qualified
    # and return a single (esa_energy, config_row, mask) tuple
    mock_config_row = MagicMock()
    mock_config_row.Index = (0, 1)  # (calibration_prod, esa_energy_step)

    def mock_iter(de_ds, config_df, esa_energy_steps):
        n_remaining = len(de_ds["event_met"])
        yield 1, mock_config_row, np.ones(n_remaining, dtype=bool)

    mock_iter_qualified.side_effect = mock_iter

    # Use MagicMock for cal_config since it's not used with our mock
    mock_cal_config = MagicMock()

    counts_var = hi_l1c.pset_counts(
        empty_pset.coords, mock_cal_config, l1b_dataset, goodtimes_ds
    )

    # Only 5 events (METs 100-104) should pass goodtimes filtering
    # All 5 events have spin_phase=0.5 -> spin_angle_bin 1800
    total_counts = counts_var["counts"].data.sum()
    assert total_counts == 5, f"Expected 5 counts, got {total_counts}"
    # Verify all counts are in the expected spin bin (1800)
    assert counts_var["counts"].data[0, 0, 0, 1800] == 5


@pytest.mark.external_test_data
def test_pset_backgrounds(
    hi_test_background_config_path,
    hi_test_cal_prod_config_path,
    hi_l1b_de_dataset,
    hi_goodtimes_dataset,
    use_fake_spin_data_for_time,
    use_fake_repoint_data_for_time,
):
    """Test coverage for pset_backgrounds function."""
    # Setup required SPICE data
    use_fake_spin_data_for_time(482372987.999)
    l1b_met = hi_l1b_de_dataset["ccsds_met"].values[0]
    seconds_per_day = 24 * 60 * 60
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + seconds_per_day]),
        np.asarray([l1b_met, l1b_met + seconds_per_day + 1]),
    )

    # Load the background config
    background_df = utils.BackgroundConfig.from_csv(hi_test_background_config_path)

    # Create empty pset dataset to get coordinates
    cal_config_df = utils.CalibrationProductConfig.from_csv(
        hi_test_cal_prod_config_path
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        l1b_met,
        hi_l1b_de_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )

    # Create exposure_times for the test
    exposure_times_data = np.full(
        (
            len(empty_pset.coords["epoch"]),
            len(empty_pset.coords["esa_energy_step"]),
            len(empty_pset.coords["spin_angle_bin"]),
        ),
        1.0,
        dtype=np.float32,
    )
    exposure_times = xr.DataArray(
        exposure_times_data,
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
        coords={
            "epoch": empty_pset.coords["epoch"],
            "esa_energy_step": empty_pset.coords["esa_energy_step"],
            "spin_angle_bin": empty_pset.coords["spin_angle_bin"],
        },
    )

    # Call pset_backgrounds with the new signature
    backgrounds_vars = hi_l1c.pset_backgrounds(
        empty_pset.coords,
        background_df,
        hi_l1b_de_dataset,
        hi_goodtimes_dataset,
        exposure_times,
    )

    assert "background_rates" in backgrounds_vars
    assert backgrounds_vars["background_rates"].data.shape == (
        len(empty_pset.coords["epoch"]),
        len(empty_pset.coords["esa_energy_step"]),
        len(empty_pset.coords["calibration_prod"]),
        len(empty_pset.coords["spin_angle_bin"]),
    )

    assert "background_rates_uncertainty" in backgrounds_vars
    assert backgrounds_vars["background_rates_uncertainty"].data.shape == (
        len(empty_pset.coords["epoch"]),
        len(empty_pset.coords["esa_energy_step"]),
        len(empty_pset.coords["calibration_prod"]),
        len(empty_pset.coords["spin_angle_bin"]),
    )

    # Verify ESA-dependent backgrounds: different ESA steps should have different
    # background rates (since scaling factors vary by ESA in the test config).
    # Check that not all ESA steps have identical background rates for each cal_prod.
    bg_rates = backgrounds_vars["background_rates"].data
    for i_cal_prod in range(len(empty_pset.coords["calibration_prod"])):
        # Get background rates for this cal_prod across all ESA steps
        # (take first spin bin)
        rates_by_esa = bg_rates[0, :, i_cal_prod, 0]
        # If there are any non-zero background counts, rates should vary by ESA
        if np.any(rates_by_esa > 0):
            # Verify not all ESA steps have identical rates
            assert not np.allclose(rates_by_esa, rates_by_esa[0]), (
                f"Background rates should vary by ESA for cal_prod {i_cal_prod}"
            )


@mock.patch("imap_processing.hi.hi_l1c.good_time_and_phase_mask")
def test_compute_background_counts_missing_cal_prod_raises_error(
    mock_good_time_and_phase_mask,
    hi_test_background_config_path,
):
    """Test _compute_background_counts raises ValueError with invalid bkgnd config."""
    # Mock good_time_and_phase_mask to return all True
    mock_good_time_and_phase_mask.side_effect = lambda a, b, c: np.ones(
        a.shape, dtype=bool
    )
    # Load the background config (has cal prods 0 and 1)
    background_df = utils.BackgroundConfig.from_csv(hi_test_background_config_path)

    # Create minimal pset_coords with a calibration product (999) that's
    # NOT in the background config
    missing_cal_prod = 999
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "calibration_prod": xr.DataArray(
            np.array([0, 1, missing_cal_prod], dtype=np.int32),
            dims=["calibration_prod"],
        ),
    }

    hi_l1b_de_dataset = xr.Dataset(
        {
            "coincidence_type": xr.DataArray(
                np.array([15], dtype=np.uint8), dims=["event_met"]
            ),
            "trigger_id": xr.DataArray(
                np.array([0], dtype=np.float64),
                dims=["event_met"],
                attrs={"FILLVAL": 65535},
            ),
            "nominal_bin": xr.DataArray(
                np.array([0], dtype=np.uint8), dims=["event_met"]
            ),
            "tof_ab": xr.DataArray(
                np.array([50], dtype=np.float32), dims=["event_met"]
            ),
            "tof_ac1": xr.DataArray(
                np.array([50], dtype=np.float32), dims=["event_met"]
            ),
            "tof_bc1": xr.DataArray(
                np.array([50], dtype=np.float32), dims=["event_met"]
            ),
            "tof_c1c2": xr.DataArray(
                np.array([50], dtype=np.float32), dims=["event_met"]
            ),
        },
        coords={
            "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
            "event_met": xr.DataArray(
                np.array([0], dtype=np.float64), dims=["event_met"]
            ),
        },
    )

    # Verify that calling _compute_background_counts raises ValueError
    # with expected message
    with pytest.raises(
        ValueError,
        match=f"Calibration product {missing_cal_prod} not found "
        f"in background configuration",
    ):
        hi_l1c._compute_background_counts(
            pset_coords,
            background_df,
            hi_l1b_de_dataset,
            xr.Dataset(),
        )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_cal_prod_mismatch_raises_error(
    mock_compute_background_counts,
):
    """Test pset_backgrounds raises ValueError when cal prods don't match.

    This tests the validation in pset_backgrounds that checks
    if calibration products in pset_coords match those in background_config_df.
    """
    # Create pset_coords with calibration products [0, 1]
    n_epoch = 1
    n_energy = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(
            np.arange(n_energy) + 1, dims=["esa_energy_step"]
        ),
        "calibration_prod": xr.DataArray(
            np.array([0, 1], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create a background config DataFrame with DIFFERENT calibration products [5, 6]
    # This simulates a mismatch between pset_coords and background_config_df
    # Now includes esa_energy_step in the multi-index
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",), ("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,), (15,), (15,)],
        "tof_ab_low": [0, 0, 0, 0],
        "tof_ab_high": [100, 100, 100, 100],
        "tof_ac1_low": [0, 0, 0, 0],
        "tof_ac1_high": [100, 100, 100, 100],
        "tof_bc1_low": [0, 0, 0, 0],
        "tof_bc1_high": [100, 100, 100, 100],
        "tof_c1c2_low": [0, 0, 0, 0],
        "tof_c1c2_high": [100, 100, 100, 100],
        "scaling_factor": [1.0, 1.0, 1.0, 1.0],
        "uncertainty": [0.1, 0.1, 0.1, 0.1],
    }
    # Use calibration products [5, 6] which don't match pset_coords [0, 1]
    mismatched_cal_prods = [5, 5, 6, 6]
    background_indices = [0, 0, 0, 0]
    esa_energy_steps = [1, 2, 1, 2]
    multi_index = pd.MultiIndex.from_arrays(
        [mismatched_cal_prods, background_indices, esa_energy_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create mock exposure_times
    exposure_times = xr.DataArray(
        np.ones((n_epoch, n_energy, n_spin_bins), dtype=np.float32),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return a DataArray with the mismatched
    # calibration products (simulating what would happen if the earlier check
    # didn't catch the mismatch)
    mock_background_counts = xr.DataArray(
        np.zeros((n_epoch, 2, 1)),
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [5, 6],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    # Create minimal l1b dataset and goodtimes (not used due to mock)
    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    # Verify that pset_backgrounds raises ValueError with expected message
    with pytest.raises(
        ValueError,
        match="Calibration products in pset_coords and "
        "background_config_df do not match",
    ):
        hi_l1c.pset_backgrounds(
            pset_coords,
            background_df,
            l1b_de_dataset,
            goodtimes_ds,
            exposure_times,
        )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_esa_energy_step_mismatch_raises_error(
    mock_compute_background_counts,
):
    """Test pset_backgrounds raises ValueError when esa_energy_steps don't match.

    This tests the validation in pset_backgrounds that checks
    if ESA energy steps in pset_coords match those in background_config_df.
    """
    # Create pset_coords with ESA energy steps [1, 2]
    n_epoch = 1
    n_energy = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(np.array([1, 2]), dims=["esa_energy_step"]),
        "calibration_prod": xr.DataArray(
            np.array([0], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create a background config DataFrame with DIFFERENT ESA energy steps [3, 4]
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,)],
        "tof_ab_low": [0, 0],
        "tof_ab_high": [100, 100],
        "tof_ac1_low": [0, 0],
        "tof_ac1_high": [100, 100],
        "tof_bc1_low": [0, 0],
        "tof_bc1_high": [100, 100],
        "tof_c1c2_low": [0, 0],
        "tof_c1c2_high": [100, 100],
        "scaling_factor": [1.0, 1.0],
        "uncertainty": [0.1, 0.1],
    }
    # Use ESA energy steps [3, 4] which don't match pset_coords [1, 2]
    cal_prods = [0, 0]
    background_indices = [0, 0]
    mismatched_esa_steps = [3, 4]
    multi_index = pd.MultiIndex.from_arrays(
        [cal_prods, background_indices, mismatched_esa_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create mock exposure_times
    exposure_times = xr.DataArray(
        np.ones((n_epoch, n_energy, n_spin_bins), dtype=np.float32),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return a valid DataArray
    mock_background_counts = xr.DataArray(
        np.zeros((n_epoch, 1, 1)),
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [0],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    # Create minimal l1b dataset and goodtimes (not used due to mock)
    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    # Verify that pset_backgrounds raises ValueError with expected message
    with pytest.raises(
        ValueError,
        match="ESA energy steps in pset_coords and background_config_df do not match",
    ):
        hi_l1c.pset_backgrounds(
            pset_coords,
            background_df,
            l1b_de_dataset,
            goodtimes_ds,
            exposure_times,
        )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_applies_offset_correction(mock_compute_background_counts):
    """Test that pset_backgrounds subtracts EXCESS_BACKGROUND_COUNT_RATE from rates.

    The function should subtract HiConstants.EXCESS_BACKGROUND_COUNT_RATE (0.003/s)
    from the combined background rates after computing them.
    """
    # Create minimal pset_coords
    n_epoch = 1
    n_energy = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(np.array([1, 2]), dims=["esa_energy_step"]),
        "calibration_prod": xr.DataArray(
            np.array([0], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create background config with scaling_factor=1 and uncertainty=0 for simplicity
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,)],
        "tof_ab_low": [0, 0],
        "tof_ab_high": [100, 100],
        "tof_ac1_low": [0, 0],
        "tof_ac1_high": [100, 100],
        "tof_bc1_low": [0, 0],
        "tof_bc1_high": [100, 100],
        "tof_c1c2_low": [0, 0],
        "tof_c1c2_high": [100, 100],
        "scaling_factor": [1.0, 1.0],
        "uncertainty": [0.0, 0.0],
    }
    cal_prods = [0, 0]
    background_indices = [0, 0]
    esa_steps = [1, 2]
    multi_index = pd.MultiIndex.from_arrays(
        [cal_prods, background_indices, esa_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create exposure times that sum to 1.0 second for easy rate calculation
    exposure_times = xr.DataArray(
        np.full((n_epoch, n_energy, n_spin_bins), 1.0 / (n_energy * n_spin_bins)),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return counts that give a known rate
    # With 100 counts and total_exposure_time=1.0s, rate = 100/s before offset
    mock_background_counts = xr.DataArray(
        np.array([[[100]]]),  # shape: (epoch=1, calibration_prod=1, background_index=1)
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [0],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    # Create minimal l1b dataset and goodtimes (not used due to mock)
    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    # Call pset_backgrounds
    result = hi_l1c.pset_backgrounds(
        pset_coords,
        background_df,
        l1b_de_dataset,
        goodtimes_ds,
        exposure_times,
    )

    # Expected rate: 100/s (count rate) * 1.0 (scaling) - 0.003 (offset) = 99.997
    expected_rate = 100.0 - HiConstants.EXCESS_BACKGROUND_COUNT_RATE
    # All values should be the same (broadcast across all dimensions)
    np.testing.assert_allclose(
        result["background_rates"].values,
        expected_rate,
        rtol=1e-6,
        err_msg="Background rate offset correction not applied correctly",
    )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_offset_does_not_go_negative(mock_compute_background_counts):
    """Test that pset_backgrounds clips rates to 0 after offset subtraction.

    When the background rate is less than the offset (0.003/s), the result
    should be clipped to 0 rather than going negative.
    """

    # Create minimal pset_coords
    n_epoch = 1
    n_energy = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(np.array([1, 2]), dims=["esa_energy_step"]),
        "calibration_prod": xr.DataArray(
            np.array([0], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create background config
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,)],
        "tof_ab_low": [0, 0],
        "tof_ab_high": [100, 100],
        "tof_ac1_low": [0, 0],
        "tof_ac1_high": [100, 100],
        "tof_bc1_low": [0, 0],
        "tof_bc1_high": [100, 100],
        "tof_c1c2_low": [0, 0],
        "tof_c1c2_high": [100, 100],
        "scaling_factor": [1.0, 1.0],
        "uncertainty": [0.0, 0.0],
    }
    cal_prods = [0, 0]
    background_indices = [0, 0]
    esa_steps = [1, 2]
    multi_index = pd.MultiIndex.from_arrays(
        [cal_prods, background_indices, esa_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create exposure times that sum to 1.0 second
    exposure_times = xr.DataArray(
        np.full((n_epoch, n_energy, n_spin_bins), 1.0 / (n_energy * n_spin_bins)),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return very small counts
    # With 0.001 counts and total_exposure_time=1.0s, rate = 0.001/s before offset
    # After subtracting 0.003 offset, would be -0.002, but should be clipped to 0
    mock_background_counts = xr.DataArray(
        np.array([[[0.001]]]),
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [0],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    # Create minimal l1b dataset and goodtimes (not used due to mock)
    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    # Call pset_backgrounds
    result = hi_l1c.pset_backgrounds(
        pset_coords,
        background_df,
        l1b_de_dataset,
        goodtimes_ds,
        exposure_times,
    )

    # Verify rate is 0 (clipped, not negative)
    assert np.all(result["background_rates"].values >= 0), (
        "Background rates should not be negative after offset subtraction"
    )
    # Since 0.001 - 0.003 = -0.002, should be clipped to 0
    np.testing.assert_allclose(
        result["background_rates"].values,
        0.0,
        atol=1e-10,
        err_msg="Background rates should be clipped to 0 when offset exceeds rate",
    )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_uncertainty_includes_constant_terms(
    mock_compute_background_counts,
):
    """Test that background uncertainty includes EXCESS_BACKGROUND_COUNT_RATE_UNC.

    The function should add EXCESS_BACKGROUND_COUNT_RATE_UNC (0.001/s) in
    quadrature to the background rate uncertainty.
    """
    # Create minimal pset_coords with ESAs 1 and 2 (which do NOT get the extra
    # UPPER_ESA_EXTRA_BACKGROUND_UNC)
    n_epoch = 1
    n_energy = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(np.array([1, 2]), dims=["esa_energy_step"]),
        "calibration_prod": xr.DataArray(
            np.array([0], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create background config with scaling_factor=1 and uncertainty=0 for simplicity
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,)],
        "tof_ab_low": [0, 0],
        "tof_ab_high": [100, 100],
        "tof_ac1_low": [0, 0],
        "tof_ac1_high": [100, 100],
        "tof_bc1_low": [0, 0],
        "tof_bc1_high": [100, 100],
        "tof_c1c2_low": [0, 0],
        "tof_c1c2_high": [100, 100],
        "scaling_factor": [1.0, 1.0],
        "uncertainty": [0.0, 0.0],
    }
    cal_prods = [0, 0]
    background_indices = [0, 0]
    esa_steps = [1, 2]
    multi_index = pd.MultiIndex.from_arrays(
        [cal_prods, background_indices, esa_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create exposure times that sum to 1.0 second for easy rate calculation
    exposure_times = xr.DataArray(
        np.full((n_epoch, n_energy, n_spin_bins), 1.0 / (n_energy * n_spin_bins)),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return 100 counts
    # With 100 counts and total_exposure_time=1.0s, Poisson uncertainty
    # = sqrt(100)/1 = 10
    mock_background_counts = xr.DataArray(
        np.array([[[100]]]),
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [0],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    result = hi_l1c.pset_backgrounds(
        pset_coords,
        background_df,
        l1b_de_dataset,
        goodtimes_ds,
        exposure_times,
    )

    # Expected uncertainty calculation:
    # Poisson = sqrt(100)/1 * 1 = 10
    # Scaling = 100 * 0 = 0
    # Combined = sqrt(10^2 + 0) = 10
    # After adding constant terms: sqrt(10^2 + 0.001^2 + 0^2) for ESAs 1,2
    # (UPPER_ESA_EXTRA_BACKGROUND_UNC is 0 for ESAs 1-6)
    expected_unc = np.sqrt(
        10.0**2
        + HiConstants.EXCESS_BACKGROUND_COUNT_RATE_UNC**2
        + 0**2  # UPPER_ESA_EXTRA_BACKGROUND_UNC=0 for ESAs 1,2
    )
    np.testing.assert_allclose(
        result["background_rates_uncertainty"].values,
        expected_unc,
        rtol=1e-6,
        err_msg="Bg rate uncertainty should include EXCESS_BACKGROUND_COUNT_RATE_UNC",
    )


@mock.patch("imap_processing.hi.hi_l1c._compute_background_counts")
def test_pset_backgrounds_esa_7_8_9_extra_uncertainty(mock_compute_background_counts):
    """Test that ESAs 7, 8, 9 get extra uncertainty (UPPER_ESA_EXTRA_BACKGROUND_UNC).

    The function should add 0.0025/s extra uncertainty in quadrature ONLY for
    ESAs 7, 8, and 9 to account for possible unidentified additional background
    in these ESA steps.
    """
    # Create pset_coords with ESAs 1, 7, and 9 to compare low vs high ESAs
    n_epoch = 1
    n_energy = 3
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.array([0], dtype=np.int64), dims=["epoch"]),
        "esa_energy_step": xr.DataArray(np.array([1, 7, 9]), dims=["esa_energy_step"]),
        "calibration_prod": xr.DataArray(
            np.array([0], dtype=np.int64),
            dims=["calibration_prod"],
        ),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins), dims=["spin_angle_bin"]),
    }

    # Create background config with scaling_factor=1 and uncertainty=0
    background_config_data = {
        "coincidence_type_list": [("ABC1C2",), ("ABC1C2",), ("ABC1C2",)],
        "coincidence_type_values": [(15,), (15,), (15,)],
        "tof_ab_low": [0, 0, 0],
        "tof_ab_high": [100, 100, 100],
        "tof_ac1_low": [0, 0, 0],
        "tof_ac1_high": [100, 100, 100],
        "tof_bc1_low": [0, 0, 0],
        "tof_bc1_high": [100, 100, 100],
        "tof_c1c2_low": [0, 0, 0],
        "tof_c1c2_high": [100, 100, 100],
        "scaling_factor": [1.0, 1.0, 1.0],
        "uncertainty": [0.0, 0.0, 0.0],
    }
    cal_prods = [0, 0, 0]
    background_indices = [0, 0, 0]
    esa_steps = [1, 7, 9]
    multi_index = pd.MultiIndex.from_arrays(
        [cal_prods, background_indices, esa_steps],
        names=["calibration_prod", "background_index", "esa_energy_step"],
    )
    background_df = pd.DataFrame(background_config_data, index=multi_index)

    # Create exposure times that sum to 1.0 second
    exposure_times = xr.DataArray(
        np.full((n_epoch, n_energy, n_spin_bins), 1.0 / (n_energy * n_spin_bins)),
        dims=["epoch", "esa_energy_step", "spin_angle_bin"],
    )

    # Mock _compute_background_counts to return 0 counts so that the only
    # uncertainty is from the constant terms (making the test sensitive to
    # the ESA-dependent uncertainty difference)
    mock_background_counts = xr.DataArray(
        np.array([[[0.0]]]),
        dims=["epoch", "calibration_prod", "background_index"],
        coords={
            "epoch": pset_coords["epoch"],
            "calibration_prod": [0],
            "background_index": [0],
        },
    )
    mock_compute_background_counts.return_value = mock_background_counts

    l1b_de_dataset = xr.Dataset()
    goodtimes_ds = xr.Dataset()

    result = hi_l1c.pset_backgrounds(
        pset_coords,
        background_df,
        l1b_de_dataset,
        goodtimes_ds,
        exposure_times,
    )

    # Get uncertainties for each ESA step from the result
    # Shape is (epoch, esa_energy_step, calibration_prod, spin_angle_bin)
    # Use isel for positional indexing since the output doesn't have labeled coords
    unc_result = result["background_rates_uncertainty"]

    # With 0 counts, Poisson uncertainty is 0, so the only uncertainties are:
    # - EXCESS_BACKGROUND_COUNT_RATE_UNC = 0.001 (for all ESAs)
    # - UPPER_ESA_EXTRA_BACKGROUND_UNC = 0.0025 for ESAs 7 and 8, 0.0055 for
    #   ESA 9

    # Expected uncertainty for ESA 1 (low ESA, no extra uncertainty):
    # sqrt(0 + 0.001^2 + 0) = 0.001
    expected_unc_esa1 = np.sqrt(
        0**2 + HiConstants.EXCESS_BACKGROUND_COUNT_RATE_UNC**2 + 0**2
    )

    # Expected uncertainty for ESAs 7 and 8 (high ESAs, with extra uncertainty):
    # sqrt(0 + 0.001^2 + 0.0025^2) = sqrt(0.000001 + 0.00000625) ≈ 0.002693
    expected_unc_esa7 = np.sqrt(
        0**2
        + HiConstants.EXCESS_BACKGROUND_COUNT_RATE_UNC**2
        + 0.0025**2  # UPPER_ESA_EXTRA_BACKGROUND_UNC
    )
    # Expected uncertainty for ESA 9:
    # sqrt(0 + 0.001^2 + 0.0055^2) = sqrt(0.000001 + 0.00000625) ≈ 0.002693
    expected_unc_esa9 = np.sqrt(
        0**2
        + HiConstants.EXCESS_BACKGROUND_COUNT_RATE_UNC**2
        + 0.0055**2  # UPPER_ESA_EXTRA_BACKGROUND_UNC
    )

    # ESA 1 is at index 0, ESA 7 at index 1, ESA 9 at index 2 in the output
    # Use isel to select by position
    unc_esa1 = unc_result.isel(esa_energy_step=0).values
    np.testing.assert_allclose(
        unc_esa1,
        expected_unc_esa1,
        rtol=1e-6,
        err_msg="ESA 1 uncertainty should NOT include UPPER_ESA_EXTRA_BACKGROUND_UNC",
    )

    # ESAs 7 and 9 should have larger uncertainty (with extra term)
    unc_esa7 = unc_result.isel(esa_energy_step=1).values
    np.testing.assert_allclose(
        unc_esa7,
        expected_unc_esa7,
        rtol=1e-6,
        err_msg="ESA 7 uncertainty should include UPPER_ESA_EXTRA_BACKGROUND_UNC",
    )

    unc_esa9 = unc_result.isel(esa_energy_step=2).values
    np.testing.assert_allclose(
        unc_esa9,
        expected_unc_esa9,
        rtol=1e-6,
        err_msg="ESA 9 uncertainty should include UPPER_ESA_EXTRA_BACKGROUND_UNC",
    )

    # Verify that ESAs 7,9 have higher uncertainty than ESA 1
    assert np.all(unc_esa7 > unc_esa1), (
        "ESA 7 should have higher uncertainty than ESA 1 due to extra term"
    )
    assert np.all(unc_esa9 > unc_esa1), (
        "ESA 9 should have higher uncertainty than ESA 1 due to extra term"
    )


@mock.patch("imap_processing.hi.hi_l1c.good_time_and_phase_mask")
@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
@mock.patch("imap_processing.hi.hi_l1c.get_spin_data", return_value=None)
@mock.patch(
    "imap_processing.hi.hi_l1c.get_spacecraft_to_instrument_spin_phase_offset",
    return_value=0.0,
)
@mock.patch("imap_processing.hi.hi_l1c.get_spacecraft_spin_phase")
@mock.patch("imap_processing.hi.hi_l1c.get_de_clock_ticks_for_esa_step")
@mock.patch("imap_processing.hi.hi_l1c.find_last_de_packet_data")
def test_pset_exposure(
    mock_find_last_de_packet_data,
    mock_de_clock_ticks,
    mock_sc_spin_phase,
    mock_phase_offset,
    mock_spin_data,
    mock_pointing_times,
    mock_good_time_and_phase_mask,
):
    """Test coverage for pset_exposure function"""
    l1b_energy_steps = xr.DataArray(
        np.arange(2) + 1,
        attrs={"FILLVAL": 255},
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100, l1b_energy_steps, np.array([0, 1]), HIAPID.H90_SCI_DE.sensor
    )
    # Set the mock of find_last_de_packet_data to return a xr.Dataset
    # with some dummy data. ESA 1 will get binned data once, ESA 2 will get
    # binned data twice.
    mock_find_last_de_packet_data.return_value = xr.Dataset(
        coords={"epoch": xr.DataArray(np.arange(3), dims=["epoch"])},
        data_vars={
            "ccsds_met": xr.DataArray(np.arange(3), dims=["epoch"]),
            "esa_energy_step": xr.DataArray(np.array([1, 2, 2]), dims=["epoch"]),
        },
    )
    # Set mock of get_de_clock_ticks_for_esa_step and spin phase to generate
    # deterministic histogram values.
    # ESA step 1 should have repeating values of 3, 1.
    # ESA step 2 should have repeating values of 6, 2
    mock_sc_spin_phase.return_value = np.concat(
        [hi_l1c.SPIN_PHASE_BIN_CENTERS, hi_l1c.SPIN_PHASE_BIN_CENTERS[::2]]
    )
    mock_de_clock_ticks.return_value = (
        np.zeros(hi_l1c.N_SPIN_BINS + hi_l1c.N_SPIN_BINS // 2),
        np.concat([np.ones(hi_l1c.N_SPIN_BINS), np.ones(hi_l1c.N_SPIN_BINS // 2) * 2]),
    )

    # The above mocks mean no data needs to be in the l1b_dataset. It
    # only needs to provide a logical source that contains "90sensor".
    l1b_dataset = MagicMock()
    l1b_dataset.attrs = {"Logical_source": "90sensor"}

    # Mock goodtime to return all true
    mock_good_time_and_phase_mask.side_effect = lambda x, y, z: np.ones(
        x.shape, dtype=bool
    )

    # All the setup is done, call the pset_exposure function
    exposure_dict = hi_l1c.pset_exposure(empty_pset.coords, l1b_dataset, xr.Dataset())

    # Based on the spin phase and clock_tick mocks, the expected clock ticks are:
    # - Repeated values of 3, 1 for the first half of the spin bins
    # - Repeated values of 3, 2 for the second half of the spin bins
    expected_values = np.stack(
        [
            np.tile([3, 1], hi_l1c.N_SPIN_BINS // 2),
            np.tile([6, 2], hi_l1c.N_SPIN_BINS // 2),
        ]
    ).astype(float)[None, :, :]
    # Convert expected clock ticks to seconds
    expected_values *= HiConstants.DE_CLOCK_TICK_S
    np.testing.assert_allclose(
        exposure_dict["exposure_times"].data,
        expected_values,
        atol=HiConstants.DE_CLOCK_TICK_S / 100,
    )


@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
@mock.patch("imap_processing.hi.hi_l1c.get_spin_data", return_value=None)
@mock.patch(
    "imap_processing.hi.hi_l1c.get_spacecraft_to_instrument_spin_phase_offset",
    return_value=0.0,
)
@mock.patch("imap_processing.hi.hi_l1c.get_spacecraft_spin_phase")
@mock.patch("imap_processing.hi.hi_l1c.get_de_clock_ticks_for_esa_step")
@mock.patch("imap_processing.hi.hi_l1c.find_last_de_packet_data")
def test_pset_exposure_goodtimes_filtering(
    mock_find_last_de_packet_data,
    mock_de_clock_ticks,
    mock_sc_spin_phase,
    mock_phase_offset,
    mock_spin_data,
    mock_pointing_times,
):
    """Test that pset_exposure properly filters clock ticks based on goodtimes."""
    l1b_energy_steps = xr.DataArray(
        np.arange(1) + 1,  # Single ESA step for simplicity
        attrs={"FILLVAL": 255},
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100, l1b_energy_steps, np.array([0]), HIAPID.H90_SCI_DE.sensor
    )

    # Mock find_last_de_packet_data to return a single ESA step
    mock_find_last_de_packet_data.return_value = xr.Dataset(
        coords={"epoch": xr.DataArray(np.arange(1), dims=["epoch"])},
        data_vars={
            "ccsds_met": xr.DataArray(np.array([150.0]), dims=["epoch"]),
            "esa_energy_step": xr.DataArray(np.array([1]), dims=["epoch"]),
        },
    )

    # Create 10 clock ticks at METs 100-109 with uniform spin phases
    n_ticks = 10
    clock_tick_mets = np.arange(100.0, 100.0 + n_ticks)
    mock_de_clock_ticks.return_value = (clock_tick_mets, np.ones(n_ticks))

    # Mock spacecraft spin phase - each tick maps to a different spin bin
    # Spin phases 0.0, 0.1, 0.2, ... -> nominal_bins 0, 9, 18, ...
    spin_phases = np.arange(n_ticks) / n_ticks
    mock_sc_spin_phase.return_value = spin_phases

    # Create a goodtimes dataset that marks half the clock ticks as bad
    # METs 100-104 are good (cull_flags=0), METs 105-109 are bad (cull_flags=1)
    goodtimes_ds = xr.Dataset(
        {
            "cull_flags": xr.DataArray(
                np.zeros((2, 90), dtype=np.uint8),
                dims=["met", "spin_bin"],
            ),
        },
        coords={"met": [100.0, 105.0], "spin_bin": np.arange(90)},
    )
    # Mark all spin bins as bad for METs >= 105
    goodtimes_ds["cull_flags"].values[1, :] = 1

    # Mock l1b_dataset
    l1b_dataset = MagicMock()
    l1b_dataset.attrs = {"Logical_source": "90sensor"}

    # Call pset_exposure with the goodtimes dataset
    exposure_dict = hi_l1c.pset_exposure(empty_pset.coords, l1b_dataset, goodtimes_ds)

    # Only the first 5 clock ticks (METs 100-104) should contribute
    # Their spin phases are 0.0, 0.1, 0.2, 0.3, 0.4 -> spin_angle_bins 0, 360, 720, ...
    total_exposure_ticks = exposure_dict["exposure_times"].data.sum()
    expected_ticks = 5.0 * HiConstants.DE_CLOCK_TICK_S
    np.testing.assert_allclose(total_exposure_ticks, expected_ticks, rtol=0.01)


def test_find_second_de_packet_data():
    """Test coverage for find_second_de_packet_data function"""
    # Create a test l1b_dataset
    # Indices represent CCSDS packets at various ESA steps
    # Index:      0  1  2  3  4  5  6  7  8  9 10 11 12 13
    # esa_step:   1  2  2  2  2  4  5  5  6  6  0  0  7  7
    # esa_energy: 1  2  2  3  3  4  5  5  6  6  0  0  7  7
    #
    # Expected last packet indices from diff logic: [0, 2, 4, 5, 7, 9, 11, 13]
    # Remove index 11: esa_energy_step is 0 (calibration)
    # Expected final indices: [0, 2, 4, 5, 7, 9, 13]
    esa_steps = np.array([1, 2, 2, 2, 2, 4, 5, 5, 6, 6, 0, 0, 7, 7])
    esa_energy_steps = np.array([1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 0, 0, 7, 7])
    l1b_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                np.arange(esa_steps.size),
                dims=["epoch"],
            ),
            "event_met": xr.DataArray(
                np.arange(10),
                dims=["event_met"],
            ),
        },
        data_vars={
            "esa_step": xr.DataArray(
                esa_steps,
                dims=["epoch"],
            ),
            "esa_energy_step": xr.DataArray(
                esa_energy_steps,
                dims=["epoch"],
                attrs={"FILLVAL": 255},
            ),
            "coincidence_type": xr.DataArray(
                np.ones(10),
                dims=["event_met"],
            ),
        },
    )
    subset = hi_l1c.find_last_de_packet_data(l1b_dataset)
    np.testing.assert_array_equal(subset.epoch.data, np.array([0, 2, 4, 5, 7, 9, 13]))


@pytest.fixture(scope="module")
def fake_spin_df():
    """Generate a synthetic spin dataframe"""
    # Generate some spin periods that vary by a random fraction of a second
    spin_period = np.full(10, 15) + np.random.randn(10) / 10
    d = {
        "spin_start_met": np.add.accumulate(spin_period),
        "spin_period_sec": spin_period,
    }
    spin_df = pd.DataFrame.from_dict(d)
    return spin_df


def test_get_de_clock_ticks_for_esa_step(fake_spin_df):
    """Test coverage for get_de_clock_ticks_for_esa_step function."""

    # Test nominal cases where CCSDS met falls after 8th spin start and before
    # the end spin in the table + 1/2 spin period
    for _, spin_row in fake_spin_df.iloc[8:].iterrows():
        for ccsds_met in np.linspace(
            spin_row.spin_start_met,
            spin_row.spin_start_met + np.floor(spin_row.spin_period_sec / 2),
            10,
        ):
            clock_tick_mets, clock_tick_weights = (
                hi_l1c.get_de_clock_ticks_for_esa_step(ccsds_met, fake_spin_df)
            )
            np.testing.assert_array_equal(clock_tick_mets.shape, clock_tick_mets.shape)
            # Verify last weight entry
            exp_final_weight = (
                np.absolute(
                    fake_spin_df.spin_start_met.to_numpy() - clock_tick_mets[-1]
                ).min()
                / HiConstants.DE_CLOCK_TICK_S
            )
            assert clock_tick_weights[-1] == exp_final_weight
            assert np.all(clock_tick_weights[:-1] == 1)


def test_get_de_clock_ticks_for_esa_step_exceptions(fake_spin_df):
    """Test the exception logic in the get_de_clock_ticks_for_esa_step function."""
    # Test the ccsds_met being > 1/2 spin period past the spin start
    bad_ccsds_met = (
        fake_spin_df.iloc[8].spin_start_met
        + fake_spin_df.iloc[8].spin_period_sec / 2
        + 0.1
    )
    with pytest.raises(
        ValueError, match="The difference between ccsds_met and spin_start_met"
    ):
        hi_l1c.get_de_clock_ticks_for_esa_step(bad_ccsds_met, fake_spin_df)

    # Test the ccsds_met being too close to the start of the spin table
    bad_ccsds_met = fake_spin_df.iloc[7].spin_start_met
    with pytest.raises(
        ValueError, match="Error determining start/end time for exposure time"
    ):
        hi_l1c.get_de_clock_ticks_for_esa_step(bad_ccsds_met, fake_spin_df)


class TestGoodTimeAndPhaseMask:
    """Tests for good_time_and_phase_mask function."""

    def test_filters_bad_times_with_nominal_bins(self):
        """Events in bad times are filtered out using nominal_bins."""
        # Create mock goodtimes with some bad times
        gt_ds = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((3, 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                )
            },
            coords={"met": [100.0, 200.0, 300.0], "spin_bin": np.arange(90)},
        )
        # Mark spin_bin 10 as bad at MET 200
        gt_ds["cull_flags"].values[1, 10] = 1

        mets = np.array([150.0, 250.0, 250.0])
        nominal_bins = np.array([10, 10, 20])

        mask = hi_l1c.good_time_and_phase_mask(mets, nominal_bins, gt_ds)
        # Event at 150 maps to MET index 0, bin 10 → good (cull_flags[0,10]=0)
        # Event at 250 maps to MET index 1, bin 10 → bad (cull_flags[1,10]=1)
        # Event at 250 maps to MET index 1, bin 20 → good (cull_flags[1,20]=0)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_met_before_goodtimes_range(self):
        """Events before goodtimes range are clipped to first interval."""
        gt_ds = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.zeros((2, 90), dtype=np.uint8),
                    dims=["met", "spin_bin"],
                )
            },
            coords={"met": [100.0, 200.0], "spin_bin": np.arange(90)},
        )
        # Mark spin_bin 0 as bad at first MET
        gt_ds["cull_flags"].values[0, 0] = 1

        # Event at MET 50 (before goodtimes range) should use index 0
        mets = np.array([50.0])
        nominal_bins = np.array([0])

        mask = hi_l1c.good_time_and_phase_mask(mets, nominal_bins, gt_ds)
        # Clipped to index 0, bin 0 is bad
        assert not mask[0]

    def test_all_bins_bad_for_interval(self):
        """When all bins are bad for an interval, all events are filtered."""
        gt_ds = xr.Dataset(
            {
                "cull_flags": xr.DataArray(
                    np.ones((1, 90), dtype=np.uint8),  # All bad
                    dims=["met", "spin_bin"],
                )
            },
            coords={"met": [100.0], "spin_bin": np.arange(90)},
        )

        mets = np.array([100.0, 150.0, 200.0])
        nominal_bins = np.array([0, 45, 89])

        mask = hi_l1c.good_time_and_phase_mask(mets, nominal_bins, gt_ds)
        assert not np.any(mask)
