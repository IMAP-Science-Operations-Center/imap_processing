"""Test coverage for imap_processing.hi.l1b.hi_l1b.py"""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1b.hi_l1b import (
    CoincidenceBitmap,
    compute_coincidence_type_and_time_deltas,
    compute_hae_coordinates,
    de_esa_energy_step,
    de_nominal_bin_and_spin_phase,
    hi_l1b,
)
from imap_processing.hi.utils import HiConstants
from imap_processing.spice.geometry import SpiceFrame


def test_hi_l1b_hk(hi_l0_test_data_path):
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    housekeeping L1A as input"""
    # TODO: once things are more stable, check in an L1A HK file as test data
    bin_data_path = hi_l0_test_data_path / "20231030_H45_APP_NHK.bin"
    data_version = "001"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version=data_version)

    l1b_dataset = hi_l1b(processed_data[0], data_version=data_version)
    assert l1b_dataset.attrs["Logical_source"] == "imap_hi_l1b_45sensor-hk"


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_hi_l1b_de(
    hi_l1a_test_data_path, spice_test_data_path, use_fake_spin_data_for_time
):
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    direct events L1A as input"""
    # Start MET time of spin for simulated input data is 482372988
    use_fake_spin_data_for_time(482372988)
    l1a_test_file_path = (
        hi_l1a_test_data_path / "imap_hi_l1a_45sensor-de_20250415_v000.cdf"
    )
    # Process using test data
    data_version = "001"
    l1a_dataset = load_cdf(l1a_test_file_path)

    l1b_dataset = hi_l1b(l1a_dataset, data_version=data_version)
    assert l1b_dataset.attrs["Logical_source"] == "imap_hi_l1b_45sensor-de"
    assert len(l1b_dataset.data_vars) == 14


@pytest.fixture()
def synthetic_trigger_id_and_tof_data():
    """Create synthetic minimum dataset for testing the
    coincidence_type_and_time_deltas algorithm."""
    # The following coincidence type table shows possible values to consider
    # Value| # Exp | Requirements to get this value
    # -----|-------|-------------------------------
    #   0  |   0   | Non-event not recorded
    #   1  |   0   | Can't trigger c2 only
    #   2  |   2   | trigger_id = 3, tof_3 invalid
    #   3  |   2   | trigger_id = 3, tof_3 valid
    #   4  |   2   | trigger_id = 2, no valid tofs
    #   5  |   0   | B and C2 not possible?
    #   6  |   4   | trigger_id = 2 OR 3, tof_2 valid
    #   7  |   4   | trigger_id = 2 OR 3, tof_2/3 valid
    #   8  |   2   | trigger_id = 3, no valid tofs
    #   9  |   0   | A and C2 not possible?
    #  10  |   3   | trigger_id = 1, tof_2 OR trigger_id = 3, tof_1
    #  11  |   3   | trigger_id = 1, tof_2/3, OR trigger_id = 3, tof_1/3
    #  12  |   2   | trigger_id = 1 OR 2, tof_1
    #  13  |   0   | A/B and C2 not possible?
    #  14  |   3   | trigger_id = 1 OR 2 OR 3, tof_1/2
    #  15  |   3   | trigger_id = 1, 2, 3, tof_1/2/3

    # Use meshgrid to get all combinations of trigger_id and tof valid/invalid
    # Note: this generates 6 impossible occurrences where C1 is not triggered
    #    but C2 is. Those are manually removed below.
    ids = np.arange(3) + 1
    tof1s = np.array(np.concatenate((HiConstants.TOF1_BAD_VALUES, [1])))
    tof2s = np.array(np.concatenate((HiConstants.TOF2_BAD_VALUES, [2])))
    tof3s = np.array(np.concatenate((HiConstants.TOF3_BAD_VALUES, [3])))
    var_names = ["trigger_id", "tof_1", "tof_2", "tof_3"]
    data = np.meshgrid(ids, tof1s, tof2s, tof3s)
    data = [arr.flatten() for arr in data]
    # Remove impossible combinations
    good_inds = np.nonzero(
        np.logical_not(
            np.logical_and(data[0] != 3, ((data[2] >= 511) & (data[3] < 511)))
        )
    )
    data = [arr[good_inds] for arr in data]
    data_vars = {
        n: xr.DataArray(arr, dims=["epoch"]) for n, arr in zip(var_names, data)
    }
    synthetic_l1a_ds = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                np.arange(data_vars["trigger_id"].size), name="epoch", dims=["epoch"]
            )
        },
        data_vars=data_vars,
    )
    expected_histogram = np.array([0, 0, 2, 2, 2, 0, 4, 4, 2, 0, 3, 3, 2, 0, 3, 3])
    return synthetic_l1a_ds, expected_histogram


def test_compute_coincidence_type_and_time_deltas(synthetic_trigger_id_and_tof_data):
    """Test coverage for
    `imap_processing.hi.hi_l1b.compute_coincidence_type_and_time_deltas`."""
    new_vars = compute_coincidence_type_and_time_deltas(
        synthetic_trigger_id_and_tof_data[0]
    )
    for var_name in [
        "coincidence_type",
        "delta_t_ab",
        "delta_t_ac1",
        "delta_t_bc1",
        "delta_t_c1c2",
    ]:
        assert var_name in new_vars
    # verify coincidence type values
    coincidence_hist, bins = np.histogram(
        new_vars["coincidence_type"], bins=np.arange(17)
    )
    np.testing.assert_array_equal(
        coincidence_hist, synthetic_trigger_id_and_tof_data[1]
    )
    # verify delta_t values are valid in the correct locations
    np.testing.assert_array_equal(
        new_vars["delta_t_ab"] != new_vars["delta_t_ab"].FILLVAL,
        new_vars["coincidence_type"] >= 12,
    )
    np.testing.assert_array_equal(
        new_vars["delta_t_ac1"] != new_vars["delta_t_ac1"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.A.value),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
        ),
    )
    np.testing.assert_array_equal(
        new_vars["delta_t_bc1"] != new_vars["delta_t_bc1"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.B.value),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
        ),
    )
    np.testing.assert_array_equal(
        new_vars["delta_t_c1c2"] != new_vars["delta_t_c1c2"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C2),
        ),
    )


@mock.patch("imap_processing.hi.l1b.hi_l1b.parse_sensor_number", return_value=90)
@mock.patch("imap_processing.hi.l1b.hi_l1b.get_instrument_spin_phase")
@mock.patch("imap_processing.hi.l1b.hi_l1b.get_spacecraft_spin_phase")
def test_de_nominal_bin_and_spin_phase(
    spacecraft_phase_moc, instrument_phase_mock, parse_sensor_number_mock
):
    """Test coverage for de_nominal_bin_and_spin_phase."""
    # set the spacecraft_phase_mock to return an array of values between 0 and 1
    # that is rolled 30 places for easy testing
    spacecraft_phase_roll = 30
    spacecraft_phase_moc.side_effect = lambda x: np.roll(
        np.arange(0, 1, 1 / len(x)), spacecraft_phase_roll
    )
    # set the get_instrument_spin_phase mock to return an array of values between
    # 0 and 1
    instrument_phase_mock.side_effect = lambda x, y: np.arange(0, 1, 1 / len(x))
    # generate a fake dataset with epoch coordinate and event_met variable
    de_list_length = 720
    synthetic_ds = xr.Dataset(
        coords={
            "epoch": xr.DataArray(
                np.arange(de_list_length), name="epoch", dims=["epoch"]
            )
        },
        data_vars={
            "event_met": xr.DataArray(np.arange(de_list_length), dims=["epoch"])
        },
        attrs={"Logical_source": "foo_source"},
    )

    new_vars = de_nominal_bin_and_spin_phase(synthetic_ds)
    # Check spin_phase
    assert "spin_phase" in new_vars
    assert new_vars["spin_phase"].shape == (de_list_length,)
    np.testing.assert_array_equal(
        new_vars["spin_phase"].values,
        np.linspace(0, 1, de_list_length + 1, dtype=np.float32)[:-1],
    )
    # Check nominal_bin
    assert "nominal_bin" in new_vars
    expected_nominal_bin = np.roll(
        np.digitize(np.arange(0, 360, 360 / de_list_length), np.arange(90) * 4) - 1,
        spacecraft_phase_roll,
    )
    np.testing.assert_array_equal(new_vars["nominal_bin"].values, expected_nominal_bin)


@pytest.mark.parametrize("sensor_number", [45, 90])
@mock.patch("imap_processing.hi.l1b.hi_l1b.instrument_pointing")
def test_compute_hae_coordinates(mock_instrument_pointing, sensor_number):
    """Test coverage for compute_hae_coordinates function."""

    # Mock out the instrument_pointing function to avoid needing kernels
    def side_effect_func(et, inst_frame: SpiceFrame, to_frame):
        """
        Side effect function to replace `geometry.instrument_pointing`.

        The function signature matches the signature of `instrument_pointing`.
        The output is the same type and shape as what `instrument_pointing` returns
        but is filled with values that match the instrument sensor number.
        """
        return np.full((et.size, 2), 45 if "45" in inst_frame.name else 90)

    mock_instrument_pointing.side_effect = side_effect_func

    # Make a fake dataset with epoch and Logical_source
    fake_dataset = xr.Dataset(
        attrs={"Logical_source": f"imap_hi_l1a_{sensor_number}sensor-de"},
        coords={"epoch": xr.DataArray(np.arange(200), name="epoch", dims=["epoch"])},
    )

    new_vars = compute_hae_coordinates(fake_dataset)
    assert "hae_latitude" in new_vars
    assert new_vars["hae_latitude"].shape == fake_dataset.epoch.shape
    np.testing.assert_allclose(new_vars["hae_latitude"].values, sensor_number)
    assert "hae_longitude" in new_vars
    assert new_vars["hae_longitude"].shape == fake_dataset.epoch.shape
    np.testing.assert_allclose(new_vars["hae_longitude"].values, sensor_number)


def test_de_esa_energy_step():
    """Test coverage for de_esa_energy_step function."""
    n_epoch = 20
    fake_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(np.arange(n_epoch), name="epoch", dims=["epoch"])
        },
        data_vars={"esa_step": xr.DataArray(np.arange(n_epoch) % 9, dims=["epoch"])},
    )
    esa_energy_step_var = de_esa_energy_step(fake_dataset)
    # TODO: The below check is for the temporary implementation and should be
    #    removed when the function is update.
    np.testing.assert_array_equal(
        esa_energy_step_var["esa_energy_step"].values, fake_dataset.esa_step.values
    )
