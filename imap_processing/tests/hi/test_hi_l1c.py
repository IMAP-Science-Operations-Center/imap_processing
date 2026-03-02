"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

import io
from collections import namedtuple
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imap_processing.hi.utils
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi import hi_l1c
from imap_processing.hi.utils import HIAPID, HiConstants
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_et


@mock.patch("imap_processing.hi.hi_l1c.generate_pset_dataset")
def test_hi_l1c(mock_generate_pset_dataset, hi_test_cal_prod_config_path):
    """Test coverage for hi_l1c function"""
    mock_generate_pset_dataset.return_value = xr.Dataset()
    pset = hi_l1c.hi_l1c(xr.Dataset(), hi_test_cal_prod_config_path)[0]
    # Empty attributes, global values get added in post-processing
    assert pset.attrs == {}


@pytest.mark.external_kernel
@pytest.mark.external_test_data
def test_generate_pset_dataset(
    hi_l1_test_data_path,
    hi_test_cal_prod_config_path,
    use_fake_spin_data_for_time,
    use_fake_repoint_data_for_time,
    imap_ena_sim_metakernel,
):
    """Test coverage for generate_pset_dataset function"""
    use_fake_spin_data_for_time(482372987.999)
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    l1b_met = l1b_dataset["ccsds_met"].values[0]
    # Set repoint start and end times.
    seconds_per_day = 24 * 60 * 60
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + seconds_per_day]),
        np.asarray([l1b_met, l1b_met + seconds_per_day + 1]),
    )
    l1c_dataset = hi_l1c.generate_pset_dataset(
        l1b_dataset, hi_test_cal_prod_config_path
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
        },
    )

    # Mock get_pointing_times to return known start and end times
    pointing_start_met = l1b_met - 1000.0
    pointing_end_met = l1b_met + 1000.0
    mock_get_pointing_times.return_value = (pointing_start_met, pointing_end_met)

    # Mock the return values for the sub-functions
    mock_pset_geometry.return_value = {}
    mock_pset_counts.return_value = {}
    mock_pset_exposure.return_value = {}
    mock_pset_backgrounds.return_value = {}

    # Call generate_pset_dataset
    _ = hi_l1c.generate_pset_dataset(mock_l1b_dataset, hi_test_cal_prod_config_path)

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
    hi_l1_test_data_path,
    hi_test_cal_prod_config_path,
):
    """Test coverage for pset_counts function."""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    cal_config_df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
        hi_test_cal_prod_config_path
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100,
        l1b_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )
    counts_var = hi_l1c.pset_counts(empty_pset.coords, cal_config_df, l1b_dataset)
    assert "counts" in counts_var


@pytest.mark.external_test_data
@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
def test_pset_counts_empty_l1b(
    mock_pointing_times,
    hi_l1_test_data_path,
    hi_test_cal_prod_config_path,
):
    """Test coverage for pset_counts function when the input L1b contains no counts."""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    # remove all but one event and set its trigger_id to zero
    l1b_dataset = l1b_dataset.isel(event_met=[0])
    l1b_dataset["trigger_id"].data[0] = 0
    cal_config_df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
        hi_test_cal_prod_config_path
    )
    empty_pset = hi_l1c.empty_pset_dataset(
        100,
        l1b_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )
    counts_var = hi_l1c.pset_counts(empty_pset.coords, cal_config_df, l1b_dataset)
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
    Row = namedtuple(
        "Row",
        [
            "Index",
            "tof_ab_low",
            "tof_ab_high",
            "tof_ac1_low",
            "tof_ac1_high",
            "tof_bc1_low",
            "tof_bc1_high",
            "tof_c1c2_low",
            "tof_c1c2_high",
        ],
    )
    prod_config_row = Row((1, 0), 0, 1, -1, 2, 1, 5, 4, 6)
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
    window_mask = hi_l1c.get_tof_window_mask(synth_df, prod_config_row, fill_vals)
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
    hi_l1_test_data_path, use_fake_repoint_data_for_time
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

    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)

    cal_config_df = imap_processing.hi.utils.CalibrationProductConfig.from_csv(
        io.StringIO(csv_content)
    )

    # Create PSET with non-sequential calibration product numbers
    l1b_met = 482373065
    use_fake_repoint_data_for_time(
        np.asarray([l1b_met - 15 * 60, l1b_met + 24 * 60 * 60])
    )

    empty_pset = hi_l1c.empty_pset_dataset(
        l1b_met,
        l1b_dataset.esa_energy_step,
        cal_config_df.cal_prod_config.calibration_product_numbers,
        HIAPID.H90_SCI_DE.sensor,
    )

    # Verify the calibration_prod coordinate has non-sequential values
    np.testing.assert_array_equal(empty_pset.calibration_prod.data, np.array([5, 10]))

    # Mock get_pointing_times to avoid SPICE kernel requirements
    with mock.patch(
        "imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200)
    ):
        counts_var = hi_l1c.pset_counts(empty_pset.coords, cal_config_df, l1b_dataset)

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
    esa_1_2_mask = (l1b_dataset["esa_step"][l1b_dataset["ccsds_index"]] < 3).values
    coincidence_15_mask = (l1b_dataset["coincidence_type"] == 15).values
    np.testing.assert_equal(
        np.sum(counts_var["counts"].data[:, :, 0]),
        np.sum(coincidence_15_mask & esa_1_2_mask),
    )
    # BC1C2 is coincidence type 7
    coincidence_7_mask = (l1b_dataset["coincidence_type"] == 7).values
    np.testing.assert_equal(
        np.sum(counts_var["counts"].data[:, :, 1]),
        np.sum(coincidence_7_mask & esa_1_2_mask),
    )


def test_pset_backgrounds():
    """Test coverage for pset_backgrounds function."""
    # Create some fake coordinates to use
    n_epoch = 1
    n_energy = 9
    n_cal_prod = 2
    n_spin_bins = 3600
    pset_coords = {
        "epoch": xr.DataArray(np.arange(n_epoch)),
        "esa_energy_step": xr.DataArray(np.arange(n_energy) + 1),
        "calibration_prod": xr.DataArray(np.arange(n_cal_prod)),
        "spin_angle_bin": xr.DataArray(np.arange(n_spin_bins)),
    }
    backgrounds_vars = hi_l1c.pset_backgrounds(pset_coords)
    assert "background_rates" in backgrounds_vars
    np.testing.assert_array_equal(
        backgrounds_vars["background_rates"].data,
        np.zeros((n_epoch, n_energy, n_cal_prod, n_spin_bins)),
    )
    assert "background_rates_uncertainty" in backgrounds_vars
    np.testing.assert_array_equal(
        backgrounds_vars["background_rates_uncertainty"].data,
        np.ones((n_epoch, n_energy, n_cal_prod, n_spin_bins)),
    )


@mock.patch("imap_processing.hi.hi_l1c.get_pointing_times", return_value=(100, 200))
@mock.patch("imap_processing.hi.hi_l1c.get_spin_data", return_value=None)
@mock.patch("imap_processing.hi.hi_l1c.get_instrument_spin_phase")
@mock.patch("imap_processing.hi.hi_l1c.get_de_clock_ticks_for_esa_step")
@mock.patch("imap_processing.hi.hi_l1c.find_last_de_packet_data")
def test_pset_exposure(
    mock_find_last_de_packet_data,
    mock_de_clock_ticks,
    mock_spin_phase,
    mock_spin_data,
    mock_pointing_times,
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
    mock_spin_phase.return_value = np.concat(
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

    # All the setup is done, call the pset_exposure function
    exposure_dict = hi_l1c.pset_exposure(empty_pset.coords, l1b_dataset)

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
