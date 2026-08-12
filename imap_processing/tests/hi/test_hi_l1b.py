"""Test coverage for imap_processing.hi.l1b.hi_l1b.py"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.hi.hi_l1b import (
    annotate_direct_events,
    any_good_direct_events,
    classify_gain_configuration,
    compute_coincidence_type_and_tofs,
    compute_hae_coordinates,
    de_ccsds_qf,
    de_esa_energy_step,
    de_esa_step_met,
    de_gain_configuration,
    de_nominal_bin_and_spin_phase,
    get_esa_to_esa_energy_step_lut,
    get_gain_configuration_lut,
    housekeeping,
)
from imap_processing.hi.utils import (
    CoincidenceBitmap,
    EsaEnergyStepLookupTable,
    GainConfigLookupTable,
    HiConstants,
)
from imap_processing.quality_flags import ImapHiL1bDeFlags
from imap_processing.spice.geometry import SpiceFrame

# Nominal detector voltage config (config_id=0) matching the real
# gain-configuration ancillary file, for use as a permissive default in tests
# that don't care about gain-configuration classification.
NOMINAL_HV_VALUES = {
    "pos_defl": 6300.0,
    "neg_defl": -6300.0,
    "tof": -8000.0,
    "mcp_f": -3000.0,
    "mcp_b": -2125.0,
    "cem_f": -4500.0,
    "cem_bk_a": -2350.0,
    "cem_bk_b": -2350.0,
}
# Tolerances matching the real gain-configuration ancillary file
NOMINAL_HV_TOLERANCES = {
    "pos_defl": 700.0,
    "neg_defl": 700.0,
    "tof": 50.0,
    "mcp_f": 25.0,
    "mcp_b": 50.0,
    "cem_f": 50.0,
    "cem_bk_a": 100.0,
    "cem_bk_b": 100.0,
}


def _make_gain_config_df(hv_tolerance=None, n_esa_energy_steps=9, config_id=0):
    """Build a gain configuration DataFrame for a single config_id.

    Indexed by (config_id, esa_energy_step), matching the real
    gain-configuration ancillary file's long format -- one row per
    esa_energy_step, with {field}_v/{field}_delta_v repeated across every row
    for the config_id.

    If hv_tolerance is None, realistic per-field tolerances are used
    (matching the real ancillary file). Otherwise, hv_tolerance is applied
    as a flat tolerance for every field (useful for permissive matching).
    """
    if hv_tolerance is None:
        tolerances = NOMINAL_HV_TOLERANCES
    else:
        tolerances = dict.fromkeys(NOMINAL_HV_VALUES, hv_tolerance)

    hv_row = {}
    for field, mean_value in NOMINAL_HV_VALUES.items():
        hv_row[f"{field}_v"] = mean_value
        hv_row[f"{field}_delta_v"] = tolerances[field]
    rows = [
        {"geometric_factor": 0.001 * step, **hv_row}
        for step in range(1, n_esa_energy_steps + 1)
    ]
    index = pd.MultiIndex.from_product(
        [[config_id], range(1, n_esa_energy_steps + 1)],
        names=["config_id", "esa_energy_step"],
    )
    return pd.DataFrame(rows, index=index)


def test_hi_l1b_hk(hi_l0_test_data_path):
    """Test coverage for imap_processing.hi.hi_l1b() with
    housekeeping L1A as input"""
    bin_data_path = hi_l0_test_data_path / "H90_NHK_20241104.bin"

    l1b_datasets = housekeeping(bin_data_path)
    assert len(l1b_datasets) == 1
    assert l1b_datasets[0].attrs["Logical_source"] == "imap_hi_l1b_90sensor-hk"


@pytest.mark.external_kernel
@pytest.mark.external_test_data
@mock.patch("imap_processing.hi.hi_l1b.get_gain_configuration_lut")
@mock.patch("imap_processing.hi.hi_l1b.get_esa_to_esa_energy_step_lut")
def test_hi_annotate_direct_events(
    mock_get_esa_lut,
    mock_get_gain_config_lut,
    hi_l1_test_data_path,
    use_fake_spin_data_for_time,
    imap_ena_sim_metakernel,
):
    """Test coverage for imap_processing.hi_l1b.annotate_direct_events() with
    direct events L1A as input and spice kernel coverage."""
    # Mock the esa LUT object to map esa_step to the same esa_energy_step value
    mock_esa_lut = mock.MagicMock(spec=EsaEnergyStepLookupTable())
    mock_esa_lut.query.side_effect = lambda a, b: b
    mock_get_esa_lut.return_value = mock_esa_lut

    # Mock the gain config LUT to always classify config_id=0 with no bad
    # detector voltage segments.
    mock_gain_lut = mock.MagicMock(spec=GainConfigLookupTable())
    mock_gain_lut.query.side_effect = lambda mets: np.zeros(
        len(np.atleast_1d(mets)), dtype=np.int64
    )
    mock_get_gain_config_lut.return_value = (mock_gain_lut, 0)

    # Start MET time of spin for simulated input data is 482372988
    use_fake_spin_data_for_time(482372987.999)
    l1a_test_file_path = (
        hi_l1_test_data_path / "imap_hi_l1a_45sensor-de_20250415_v999.cdf"
    )
    esa_energies_csv = (
        hi_l1_test_data_path / "imap_hi_90sensor-esa-energies_20240101_v001.csv"
    )
    gain_config_csv = (
        hi_l1_test_data_path / "imap_hi_90sensor-gain-configuration_20240101_v001.csv"
    )
    # Process using test data
    l1a_dataset = load_cdf(l1a_test_file_path)

    l1b_datasets = annotate_direct_events(
        l1a_dataset, xr.Dataset(), esa_energies_csv, gain_config_csv
    )
    assert len(l1b_datasets) == 1
    assert l1b_datasets[0].attrs["Logical_source"] == "imap_hi_l1b_45sensor-de"
    assert l1b_datasets[0].attrs["gain_configuration_id"] == 0
    assert len(l1b_datasets[0].data_vars) == 18


@pytest.mark.parametrize(
    "trigger_id_data, fillval, expected_result",
    [([0], 0, False), ([15, 15], 15, False), ([1], 0, True), ([1, 2, 3], 65536, True)],
)
def test_any_good_direct_events(trigger_id_data, fillval, expected_result):
    """Test coverage for any_good_direct_events()"""
    ds = xr.Dataset(
        data_vars={
            "trigger_id": xr.DataArray(
                trigger_id_data, name="trigger_id", attrs={"FILLVAL": fillval}
            )
        }
    )
    assert any_good_direct_events(ds) == expected_result


@pytest.mark.external_test_data
@mock.patch(
    "imap_processing.spice.spin.get_spacecraft_to_instrument_spin_phase_offset",
    return_value=0,
)
@mock.patch("imap_processing.hi.hi_l1b.instrument_pointing")
def test_annotate_direct_events_with_hk(
    mock_instrument_pointing,
    mock_inst_phase_offset,
    hi_l1_test_data_path,
    use_fake_spin_data_for_time,
):
    """Test imap_processing.hi_l1b.annotate_direct_events() with a
    coincident de and hk dataset but mocked spice."""
    # Mock instrument pointing to return zeros since we don't have spice kernels
    # for this time.
    mock_instrument_pointing.side_effect = lambda et, frame_a, frame_b: np.zeros(
        (len(et), 3)
    )

    l1a_de_file_path = (
        hi_l1_test_data_path / "imap_hi_l1a_90sensor-de_20241105-repoint00099_v001.cdf"
    )
    l1b_hk_file_path = (
        hi_l1_test_data_path / "imap_hi_l1b_90sensor-hk_20241105-repoint00099_v001.cdf"
    )
    esa_energies_csv = (
        hi_l1_test_data_path / "imap_hi_90sensor-esa-energies_20240101_v001.csv"
    )
    gain_config_csv = (
        hi_l1_test_data_path / "imap_hi_90sensor-gain-configuration_20240101_v001.csv"
    )
    # Process using test data
    l1a_dataset = load_cdf(l1a_de_file_path)
    hk_dataset = load_cdf(l1b_hk_file_path)
    # Cross-cal data used a 15 second spin period. Calculate start time of first
    # spin
    spin_start_met = (
        l1a_dataset["esa_step_seconds"].data[0].astype(float)
        + l1a_dataset["esa_step_milliseconds"].data[0].astype(float) / 1000
    )
    use_fake_spin_data_for_time(spin_start_met)

    l1b_datasets = annotate_direct_events(
        l1a_dataset, hk_dataset, esa_energies_csv, gain_config_csv
    )
    assert len(l1b_datasets) == 1
    assert l1b_datasets[0].attrs["Logical_source"] == "imap_hi_l1b_90sensor-de"
    assert len(l1b_datasets[0].data_vars) == 18
    # Verify new L1B variables exist
    assert "esa_step_met" in l1b_datasets[0].data_vars
    assert "ccsds_qf" in l1b_datasets[0].data_vars


@pytest.fixture
def synthetic_trigger_id_and_tof_data():
    """Create synthetic minimum dataset for testing the
    coincidence_type_and_time_deltas algorithm."""
    # The following coincidence type table shows possible values to consider
    # Value| # Exp | Requirements to get this value
    # -----|-------|-------------------------------
    #   0  |   0   | Non-event not recorded
    #   1  |   0   | Can't trigger c2 only
    #   2  |   1   | trigger_id = 3, tof_3 invalid
    #   3  |   1   | trigger_id = 3, tof_3 valid
    #   4  |   1   | trigger_id = 2, no valid tofs
    #   5  |   0   | B and C2 not possible?
    #   6  |   2   | trigger_id = 2 OR 3, tof_2 valid
    #   7  |   2   | trigger_id = 2 OR 3, tof_2/3 valid
    #   8  |   1   | trigger_id = 3, no valid tofs
    #   9  |   0   | A and C2 not possible?
    #  10  |   2   | trigger_id = 1, tof_2 OR trigger_id = 3, tof_1
    #  11  |   2   | trigger_id = 1, tof_2/3, OR trigger_id = 3, tof_1/3
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
        n: xr.DataArray(arr, dims=["event_met"])
        for n, arr in zip(var_names, data, strict=False)
    }
    synthetic_l1a_ds = xr.Dataset(
        coords={
            "event_met": xr.DataArray(
                np.arange(data_vars["trigger_id"].size), name="epoch", dims=["epoch"]
            )
        },
        data_vars=data_vars,
    )
    expected_histogram = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 2, 0, 3, 3])
    return synthetic_l1a_ds, expected_histogram


@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=True)
def test_compute_coincidence_type_and_time_deltas(
    mock_any_good_de, synthetic_trigger_id_and_tof_data
):
    """Test coverage for
    `imap_processing.hi.hi_l1b.compute_coincidence_type_and_time_deltas`."""
    new_vars = compute_coincidence_type_and_tofs(synthetic_trigger_id_and_tof_data[0])
    for var_name in [
        "coincidence_type",
        "tof_ab",
        "tof_ac1",
        "tof_bc1",
        "tof_c1c2",
    ]:
        assert var_name in new_vars
    # verify coincidence type values
    coincidence_hist, bins = np.histogram(
        new_vars["coincidence_type"], bins=np.arange(17)
    )
    np.testing.assert_array_equal(
        coincidence_hist, synthetic_trigger_id_and_tof_data[1]
    )
    # verify tof values are valid in the correct locations
    np.testing.assert_array_equal(
        new_vars["tof_ab"] != new_vars["tof_ab"].FILLVAL,
        new_vars["coincidence_type"] >= 12,
    )
    np.testing.assert_array_equal(
        new_vars["tof_ac1"] != new_vars["tof_ac1"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.A.value),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
        ),
    )
    np.testing.assert_array_equal(
        new_vars["tof_bc1"] != new_vars["tof_bc1"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.B.value),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
        ),
    )
    np.testing.assert_array_equal(
        new_vars["tof_c1c2"] != new_vars["tof_c1c2"].FILLVAL,
        np.logical_and(
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C1),
            np.bitwise_and(new_vars["coincidence_type"], CoincidenceBitmap.C2),
        ),
    )


@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=True)
@mock.patch("imap_processing.hi.hi_l1b.parse_sensor_number", return_value=90)
@mock.patch("imap_processing.hi.hi_l1b.get_instrument_spin_phase")
@mock.patch("imap_processing.hi.hi_l1b.get_spacecraft_spin_phase")
def test_de_nominal_bin_and_spin_phase(
    spacecraft_phase_moc,
    instrument_phase_mock,
    parse_sensor_number_mock,
    any_good_de_mock,
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
@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=True)
@mock.patch("imap_processing.hi.hi_l1b.instrument_pointing")
def test_compute_hae_coordinates(
    mock_instrument_pointing, mock_any_good_de, sensor_number
):
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
        coords={
            "event_met": xr.DataArray(
                np.arange(200), name="event_met", dims=["event_met"]
            )
        },
    )

    new_vars = compute_hae_coordinates(fake_dataset)
    assert "hae_latitude" in new_vars
    assert new_vars["hae_latitude"].shape == fake_dataset.event_met.shape
    np.testing.assert_allclose(new_vars["hae_latitude"].values, sensor_number)
    assert "hae_longitude" in new_vars
    assert new_vars["hae_longitude"].shape == fake_dataset.event_met.shape
    np.testing.assert_allclose(new_vars["hae_longitude"].values, sensor_number)


@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=True)
@mock.patch("imap_processing.hi.hi_l1b.pd.read_csv")
@mock.patch("imap_processing.hi.hi_l1b.get_esa_to_esa_energy_step_lut")
def test_de_esa_energy_step(mock_get_esa_lut, mock_read_csv, mock_any_good_de):
    """Test coverage for de_esa_energy_step function."""
    esa_energy_step_fillval = 255
    # Packet at index 5 fails to find a matching esa_energy_step (FILLVAL).
    mock_esa_lut = mock.MagicMock(spec=EsaEnergyStepLookupTable())
    mock_esa_lut.query.side_effect = lambda a, b: np.where(
        np.arange(len(a)) == 5, esa_energy_step_fillval, np.arange(len(a))[::-1] % 9
    )
    mock_get_esa_lut.return_value = mock_esa_lut

    n_epoch = 20
    fake_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(np.arange(n_epoch), name="epoch", dims=["epoch"])
        },
        data_vars={
            "ccsds_met": xr.DataArray(np.arange(n_epoch) % 9, dims=["epoch"]),
            "esa_step": xr.DataArray(np.arange(n_epoch), dims=["epoch"]),
            # Pre-existing "ccsds_qf", as created by de_ccsds_qf().
            "ccsds_qf": xr.DataArray(np.zeros(n_epoch, dtype=np.uint8), dims=["epoch"]),
        },
    )
    new_vars = de_esa_energy_step(fake_dataset, xr.Dataset(), "Fake path")

    expected_esa_energy_step = np.arange(n_epoch)[::-1] % 9
    expected_esa_energy_step[5] = esa_energy_step_fillval
    np.testing.assert_array_equal(
        new_vars["esa_energy_step"].values, expected_esa_energy_step
    )
    # de_esa_energy_step modifies "ccsds_qf" on fake_dataset in place.
    expected_qf_bits = np.zeros(n_epoch, dtype=np.uint8)
    expected_qf_bits[5] = ImapHiL1bDeFlags.BAD_ESA_VOLTAGE
    np.testing.assert_array_equal(fake_dataset["ccsds_qf"].values, expected_qf_bits)


@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=True)
@mock.patch("imap_processing.hi.hi_l1b.load_gain_configuration")
@mock.patch("imap_processing.hi.hi_l1b.get_gain_configuration_lut")
def test_de_gain_configuration(
    mock_get_gain_config_lut, mock_load_gain_config, mock_any_good_de
):
    """Test coverage for de_gain_configuration function."""
    # Mock the gain config LUT to classify config_id=0, with the packet at
    # ccsds_met == 5 failing to match (bad detector voltage).
    mock_gain_lut = mock.MagicMock(spec=GainConfigLookupTable())
    mock_gain_lut.query.side_effect = lambda mets: np.where(
        np.atleast_1d(mets) == 5, GainConfigLookupTable.NO_MATCH, 0
    )
    mock_get_gain_config_lut.return_value = (mock_gain_lut, 0)
    mock_load_gain_config.return_value = _make_gain_config_df()

    n_epoch = 10
    esa_energy_step_fillval = 255
    fake_dataset = xr.Dataset(
        coords={
            "epoch": xr.DataArray(np.arange(n_epoch), name="epoch", dims=["epoch"])
        },
        data_vars={
            "ccsds_met": xr.DataArray(np.arange(n_epoch), dims=["epoch"]),
            "esa_energy_step": xr.DataArray(
                np.full(n_epoch, 3, dtype=np.uint8),
                dims=["epoch"],
                attrs={"FILLVAL": esa_energy_step_fillval},
            ),
            "ccsds_qf": xr.DataArray(np.zeros(n_epoch, dtype=np.uint8), dims=["epoch"]),
        },
    )

    result = de_gain_configuration(fake_dataset, xr.Dataset(), "Fake gain config path")

    # de_gain_configuration modifies fake_dataset in place and returns it.
    assert result is fake_dataset
    assert result.attrs["gain_configuration_id"] == 0
    expected_esa_energy_step = np.full(n_epoch, 3)
    expected_esa_energy_step[5] = esa_energy_step_fillval
    np.testing.assert_array_equal(
        result["esa_energy_step"].values, expected_esa_energy_step
    )
    expected_qf_bits = np.zeros(n_epoch, dtype=np.uint8)
    expected_qf_bits[5] = ImapHiL1bDeFlags.BAD_DETECTOR_VOLTAGE
    np.testing.assert_array_equal(result["ccsds_qf"].values, expected_qf_bits)


@mock.patch("imap_processing.hi.hi_l1b.any_good_direct_events", return_value=False)
def test_de_gain_configuration_no_good_events(mock_any_good_de):
    """Test that gain_configuration_id is NO_MATCH when there are no good DEs."""
    fake_dataset = xr.Dataset(attrs={})

    result = de_gain_configuration(fake_dataset, xr.Dataset(), "Fake gain config path")

    assert result is fake_dataset
    assert result.attrs["gain_configuration_id"] == GainConfigLookupTable.NO_MATCH


class TestGetEsaToEsaEnergyStepLut:
    """Test suite for get_esa_to_esa_energy_step_lut function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the EsaEnergyStepLookupTable class
        self.mock_lut = mock.Mock(spec=EsaEnergyStepLookupTable())

        # Sample ESA energies lookup table
        self.esa_energies_lut = pd.DataFrame(
            {
                "inner_esa_voltage": [-100.0, -200.0, -300.0],
                "outer_esa_voltage": [50.0, 100.0, 150.0],
                "inner_esa_delta_v": [5.0, 5.0, 5.0],
                "outer_esa_delta_v": [2.5, 2.5, 2.5],
                "esa_energy_step": [1, 2, 3],
            }
        )

    def create_mock_dataset(
        self,
        op_modes,
        esa_steps,
        inner_esa_state,
        inner_esa_hi,
        inner_esa_lo,
        outer_esa_values,
        shcoarse_values,
        **hv_overrides,
    ):
        """Helper method to create mock L1B housekeeping dataset."""
        n = len(op_modes)
        hv_values = dict(NOMINAL_HV_VALUES)
        hv_values.update(hv_overrides)
        data_vars = {
            "op_mode": (["epoch"], op_modes),
            "sci_esa_step": (["epoch"], esa_steps),
            "inner_esa_state": (["epoch"], inner_esa_state),
            "inner_esa_hi": (["epoch"], inner_esa_hi),
            "inner_esa_lo": (["epoch"], inner_esa_lo),
            "outer_esa": (["epoch"], outer_esa_values),
            "shcoarse": (["epoch"], shcoarse_values),
        }
        for field, value in hv_values.items():
            data_vars[field] = (["epoch"], np.full(n, value))
        return xr.Dataset(data_vars)

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    def test_basic_functionality_single_hvsci_segment(self, mock_lut_class):
        """Test basic functionality with a single HVSCI segment."""
        mock_lut_class.return_value = self.mock_lut

        # Create test data with single HVSCI segment
        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["HVSCI", "HVSCI", "HVSCI", "HVSCI"],
            esa_steps=[1, 1, 2, 2],
            inner_esa_state=["HI", "HI", "HI", "HI"],
            inner_esa_hi=[
                -98.0,
                -102.0,
                -198.0,
                -202.0,
            ],  # Should match steps 1 and 2
            inner_esa_lo=[0, 0, 0, 0],
            outer_esa_values=[49.0, 51.0, 99.0, 101.0],
            shcoarse_values=[1000, 1001, 1002, 1003],
        )

        lut = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # Verify LUT was instantiated
        mock_lut_class.assert_called_once()

        # Verify add_entry was called for each ESA step
        assert self.mock_lut.add_entry.call_count == 2

        # Check the calls made to add_entry
        calls = self.mock_lut.add_entry.call_args_list

        # First call should be for esa_step 1
        assert calls[0][0] == (
            1000,
            1003,
            1,
            1,
        )  # start_time, end_time, esa_step, esa_energy_step

        # Second call should be for esa_step 2
        assert calls[1][0] == (1000, 1003, 2, 2)

        assert lut == self.mock_lut

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    def test_multiple_hvsci_segments(self, mock_lut_class):
        """Test with multiple separate HVSCI segments."""
        mock_lut_class.return_value = self.mock_lut

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["OTHER", "HVSCI", "HVSCI", "OTHER", "HVSCI", "HVSCI"],
            esa_steps=[1, 1, 1, 2, 2, 2],
            inner_esa_state=["LO", "LO", "LO", "LO", "LO", "LO"],
            inner_esa_hi=[0, 0, 0, 0, 0, 0],
            inner_esa_lo=[-100.0, -98.0, -102.0, -200.0, -198.0, -202.0],
            outer_esa_values=[50.0, 49.0, 51.0, 100.0, 99.0, 101.0],
            shcoarse_values=[1000, 1001, 1002, 1003, 1004, 1005],
        )

        _ = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # Should have 2 calls to add_entry (one for each segment)
        assert self.mock_lut.add_entry.call_count == 2

        calls = self.mock_lut.add_entry.call_args_list
        # First segment: indices 1-2, esa_step 1
        assert calls[0][0] == (1001, 1002, 1, 1)
        # Second segment: indices 4-5, esa_step 2
        assert calls[1][0] == (1004, 1005, 2, 2)

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    def test_no_hvsci_segments(self, mock_lut_class):
        """Test with no HVSCI segments."""
        mock_lut_class.return_value = self.mock_lut

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["OTHER", "LVSCI", "LVSCI"],
            esa_steps=[1, 2, 3],
            inner_esa_state=["HI", "HI", "HI"],
            inner_esa_hi=[-100.0, -200.0, -300.0],
            inner_esa_lo=[-100.0, -200.0, -300.0],
            outer_esa_values=[50.0, 100.0, 150.0],
            shcoarse_values=[1000, 1001, 1002],
        )

        lut = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # No add_entry calls should be made
        self.mock_lut.add_entry.assert_not_called()
        assert lut == self.mock_lut

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    @mock.patch("imap_processing.hi.hi_l1b.logger")
    def test_no_voltage_match_found(self, mock_logger, mock_lut_class):
        """Test when no matching ESA energy is found."""
        mock_lut_class.return_value = self.mock_lut

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["HVSCI", "HVSCI"],
            esa_steps=[1, 1],
            inner_esa_state=["HI", "LO"],
            inner_esa_hi=[-500.0, -500.0],  # No match in lookup table
            inner_esa_lo=[-500.0, -500.0],
            outer_esa_values=[500.0, 500.0],
            shcoarse_values=[1000, 1001],
        )

        _ = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # Should log critical error
        mock_logger.critical.assert_called_once()
        assert (
            "No esa_energy_step matches found" in mock_logger.critical.call_args[0][0]
        )

        # No add_entry should be called
        self.mock_lut.add_entry.assert_not_called()

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    @mock.patch("imap_processing.hi.hi_l1b.logger")
    def test_multiple_voltage_matches_found(self, mock_logger, mock_lut_class):
        """Test when multiple matching ESA energies are found."""
        mock_lut_class.return_value = self.mock_lut

        # Create lookup table with overlapping voltage ranges
        overlapping_lut = pd.DataFrame(
            {
                "inner_esa_voltage": [-100.0, -102.0],  # Overlapping ranges
                "outer_esa_voltage": [50.0, 52.0],
                "inner_esa_delta_v": [10.0, 10.0],  # Large deltas create overlap
                "outer_esa_delta_v": [10.0, 10.0],
                "esa_energy_step": [1, 2],
            }
        )

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["HVSCI", "HVSCI"],
            esa_steps=[1, 1],
            inner_esa_state=["HI", "LO"],
            inner_esa_hi=[-101.0, 0],  # Matches both rows
            inner_esa_lo=[0, -101.0],  # Matches both rows
            outer_esa_values=[51.0, 51.0],
            shcoarse_values=[1000, 1001],
        )

        _ = get_esa_to_esa_energy_step_lut(l1b_hk_ds, overlapping_lut)

        # Should log critical error for multiple matches
        mock_logger.critical.assert_called_once()
        assert (
            "Multiple esa_energy_step matches found"
            in mock_logger.critical.call_args[0][0]
        )

        # No add_entry should be called
        self.mock_lut.add_entry.assert_not_called()

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    def test_single_data_point_segment(self, mock_lut_class):
        """Test with HVSCI segment containing only one data point."""
        mock_lut_class.return_value = self.mock_lut

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["HVSCI"],
            esa_steps=[1],
            inner_esa_state=["HI"],
            inner_esa_hi=[-100.0],
            inner_esa_lo=[-100.0],
            outer_esa_values=[50.0],
            shcoarse_values=[1000],
        )

        _ = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # Should still work with single data point
        self.mock_lut.add_entry.assert_called_once_with(1000, 1000, 1, 1)

    @mock.patch("imap_processing.hi.hi_l1b.EsaEnergyStepLookupTable")
    def test_esa_step_not_in_segment(self, mock_lut_class):
        """Test when an ESA step doesn't appear in a particular HVSCI segment."""
        mock_lut_class.return_value = self.mock_lut

        l1b_hk_ds = self.create_mock_dataset(
            op_modes=["OTHER", "HVSCI", "HVSCI", "OTHER"],
            esa_steps=[1, 2, 2, 1],  # ESA step 1 not in HVSCI segment
            inner_esa_state=["HI", "HI", "HI", "HI"],
            inner_esa_hi=[-100.0, -198.0, -202.0, -100.0],
            inner_esa_lo=[0, 0, 0, 0],
            outer_esa_values=[50.0, 99.0, 101.0, 50.0],
            shcoarse_values=[1000, 1001, 1002, 1003],
        )

        _ = get_esa_to_esa_energy_step_lut(l1b_hk_ds, self.esa_energies_lut)

        # Only ESA step 2 should be processed (it's the only one in HVSCI segment)
        self.mock_lut.add_entry.assert_called_once_with(1001, 1002, 2, 2)

    @pytest.mark.external_test_data
    def test_cal_data(self, hi_l1_test_data_path):
        """Test with calibration data."""
        l1b_hk_ds = load_cdf(
            hi_l1_test_data_path
            / "imap_hi_l1b_90sensor-hk_20241105-repoint00099_v001.cdf"
        )
        # Create a esa energies pandas DataFrame
        esa_energies_lut_data = {
            "esa_energy_step": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "nominal_central_energy": [
                np.nan,
                0.50,
                0.75,
                1.10,
                1.65,
                2.50,
                3.75,
                5.70,
                8.52,
                12.8,
            ],
            "inner_esa_voltage": [
                0,
                -472,
                -713,
                -1010,
                -1524,
                -2060,
                -2870,
                -4107,
                -5908,
                -8625,
            ],
            "inner_esa_delta_v": [25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
            "outer_esa_voltage": [0, 122, 164, 213, 270, 718, 1232, 2034, 3185, 4911],
            "outer_esa_delta_v": [25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
        }
        esa_energies_lut = pd.DataFrame(esa_energies_lut_data)

        lut = get_esa_to_esa_energy_step_lut(l1b_hk_ds, esa_energies_lut)

        # Check the generated lookup table
        # We expect 1 dataframe entry per esa step in the range [1, 9]
        np.testing.assert_array_equal(lut.df["esa_step"].values, np.arange(9) + 1)


class TestGetGainConfigurationLut:
    """Test suite for get_gain_configuration_lut function."""

    def _make_hk_ds(self, op_modes, shcoarse_values=None, **hv_overrides):
        n = len(op_modes)
        hv_values = dict(NOMINAL_HV_VALUES)
        hv_values.update(hv_overrides)
        if shcoarse_values is None:
            shcoarse_values = np.arange(n, dtype=float) + 1000.0
        data_vars = {
            "op_mode": (["epoch"], op_modes),
            "shcoarse": (["epoch"], np.asarray(shcoarse_values, dtype=float)),
        }
        for field, value in hv_values.items():
            data_vars[field] = (["epoch"], np.full(n, value))
        return xr.Dataset(data_vars)

    def test_matching_segment_is_added_to_lut(self):
        """Test that a segment matching the classified config gets a LUT entry."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI", "HVSCI"])

        lut, config_id = get_gain_configuration_lut(hk_ds, gain_config_df)

        assert config_id == 0
        assert lut.query(1000.0) == 0
        assert lut.query(1002.0) == 0
        assert lut.query(999.0) == GainConfigLookupTable.NO_MATCH

    def test_mismatched_segment_excluded_from_lut(self):
        """Test that a segment not matching the classified config is excluded."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(
            ["HVSCI", "HVSCI", "OTHER", "HVSCI", "HVSCI"],
            shcoarse_values=[1000, 1001, 1002, 1003, 1004],
        )
        # Second HVSCI segment (indices 3-4) has a mismatched CEM front voltage,
        # simulating a mid-pointing gain test excursion.
        hk_ds["cem_f"].values[3:] = -4200.0

        lut, config_id = get_gain_configuration_lut(hk_ds, gain_config_df)

        # First segment classifies the pointing's gain configuration (config 0)
        assert config_id == 0
        # Only the first (matching) segment should have a LUT entry
        assert lut.query(1000.0) == 0
        assert lut.query(1001.0) == 0
        assert lut.query(1003.0) == GainConfigLookupTable.NO_MATCH
        assert lut.query(1004.0) == GainConfigLookupTable.NO_MATCH

    def test_no_hvsci_segments_returns_no_match_everywhere(self):
        """Test that no HVSCI segments results in an empty LUT and config_id=None."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["OTHER", "LVSCI"])

        lut, config_id = get_gain_configuration_lut(hk_ds, gain_config_df)

        assert config_id is None
        assert lut.query(1000.0) == GainConfigLookupTable.NO_MATCH

    def test_classification_failure_returns_no_match_everywhere(self):
        """Test that an unclassifiable pointing produces an empty LUT."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI"], cem_f=0.0)

        lut, config_id = get_gain_configuration_lut(hk_ds, gain_config_df)

        assert config_id is None
        assert lut.query(1000.0) == GainConfigLookupTable.NO_MATCH


class TestClassifyGainConfiguration:
    """Test suite for classify_gain_configuration function."""

    def _make_hk_ds(self, op_modes, **hv_overrides):
        n = len(op_modes)
        hv_values = dict(NOMINAL_HV_VALUES)
        hv_values.update(hv_overrides)
        data_vars = {
            "op_mode": (["epoch"], op_modes),
            "shcoarse": (["epoch"], np.arange(n, dtype=float) + 1000.0),
        }
        for field, value in hv_values.items():
            data_vars[field] = (["epoch"], np.full(n, value))
        return xr.Dataset(data_vars)

    def test_classifies_nominal_config(self):
        """Test that nominal voltages classify as config_id 0."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI", "HVSCI"])
        assert classify_gain_configuration(hk_ds, gain_config_df) == 0

    def test_classifies_second_config(self):
        """Test that a second configuration's voltages are classified correctly."""
        gain_config_df = pd.concat(
            [_make_gain_config_df(config_id=0), _make_gain_config_df(config_id=1)]
        )
        # Give config 1 a distinctly different CEM front voltage
        gain_config_df.loc[1, "cem_f_v"] = -4200.0
        gain_config_df.loc[1, "cem_f_delta_v"] = 25.0

        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI"], cem_f=-4200.0)
        assert classify_gain_configuration(hk_ds, gain_config_df) == 1

    def test_no_hvsci_segments_returns_none(self):
        """Test that no HVSCI segments results in None."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["OTHER", "LVSCI"])
        assert classify_gain_configuration(hk_ds, gain_config_df) is None

    def test_no_match_returns_none(self):
        """Test that voltages matching no configuration return None."""
        gain_config_df = _make_gain_config_df()
        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI"], cem_f=0.0)
        assert classify_gain_configuration(hk_ds, gain_config_df) is None

    def test_only_uses_first_hvsci_segment(self):
        """Test that only the first HVSCI segment is used for classification."""
        gain_config_df = _make_gain_config_df()
        # First segment matches config 0; a later segment (e.g. a mid-pointing
        # gain test) does not match anything -- this should not affect the
        # classification result, which is based only on the first segment.
        hk_ds = self._make_hk_ds(["HVSCI", "HVSCI", "OTHER", "HVSCI", "HVSCI"])
        hk_ds["cem_f"].values[3:] = 0.0
        assert classify_gain_configuration(hk_ds, gain_config_df) == 0


class TestDeEsaStepMet:
    """Tests for de_esa_step_met function."""

    def test_computes_esa_step_met(self):
        """Test that esa_step_met calculation from seconds and milliseconds."""
        ds = xr.Dataset(
            coords={"epoch": [0, 1, 2], "event_met": [0.0, 1.0]},
            data_vars={
                "esa_step_seconds": (
                    ["epoch"],
                    np.array([100, 200, 300], dtype=np.uint32),
                ),
                "esa_step_milliseconds": (
                    ["epoch"],
                    np.array([500, 250, 750], dtype=np.uint16),
                ),
                "trigger_id": xr.DataArray(
                    [1, 2], dims=["event_met"], attrs={"FILLVAL": 0}
                ),
            },
        )
        result = de_esa_step_met(ds)
        expected = np.array([100.5, 200.25, 300.75])
        np.testing.assert_array_almost_equal(result["esa_step_met"].values, expected)


class TestDeCcsdsQf:
    """Tests for de_ccsds_qf function."""

    def test_packet_full_flag_set(self):
        """Test that PACKET_FULL flag is set for packets with 664 events."""
        n_packets = 3
        # Create events: packet 0 has 664 events, packet 1 has 100, packet 2 has 664
        ccsds_indices = np.concatenate(
            [
                np.zeros(664, dtype=np.uint16),  # 664 events for packet 0
                np.ones(100, dtype=np.uint16),  # 100 events for packet 1
                np.full(664, 2, dtype=np.uint16),  # 664 events for packet 2
            ]
        )
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(len(ccsds_indices), dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.ones(len(ccsds_indices), dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": 0},
                ),
                "spin_invalids": (["epoch"], np.zeros(n_packets, dtype=np.uint8)),
            },
        )
        result = de_ccsds_qf(ds)
        # Packet 0 and 2 should have PACKET_FULL flag (1), packet 1 should be 0
        assert result["ccsds_qf"].values[0] == ImapHiL1bDeFlags.PACKET_FULL
        assert result["ccsds_qf"].values[1] == 0
        assert result["ccsds_qf"].values[2] == ImapHiL1bDeFlags.PACKET_FULL

    def test_no_full_packets(self):
        """Test that no flags are set when no packets are full."""
        n_packets = 2
        ccsds_indices = np.concatenate(
            [
                np.zeros(100, dtype=np.uint16),
                np.ones(200, dtype=np.uint16),
            ]
        )
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(len(ccsds_indices), dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.ones(len(ccsds_indices), dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": 0},
                ),
                "spin_invalids": (["epoch"], np.zeros(n_packets, dtype=np.uint8)),
            },
        )
        result = de_ccsds_qf(ds)
        assert result["ccsds_qf"].values[0] == 0
        assert result["ccsds_qf"].values[1] == 0

    def test_spin_invalid_flag_set(self):
        """Test that BADSPIN flag is set for packets with nonzero spin_invalids."""
        n_packets = 3
        ccsds_indices = np.concatenate(
            [
                np.zeros(10, dtype=np.uint16),
                np.ones(10, dtype=np.uint16),
                np.full(10, 2, dtype=np.uint16),
            ]
        )
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(len(ccsds_indices), dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.ones(len(ccsds_indices), dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": 0},
                ),
                # Packet 1 has an invalid spin, packets 0 and 2 do not
                "spin_invalids": (
                    ["epoch"],
                    np.array([0, 1, 0], dtype=np.uint8),
                ),
            },
        )
        result = de_ccsds_qf(ds)
        np.testing.assert_array_equal(
            result["ccsds_qf"].values, [0, ImapHiL1bDeFlags.BADSPIN, 0]
        )

    def test_spin_invalid_and_packet_full_flags_combined(self):
        """Test that BADSPIN and PACKET_FULL flags can be set together."""
        n_packets = 2
        ccsds_indices = np.concatenate(
            [
                np.zeros(664, dtype=np.uint16),  # 664 events for packet 0
                np.ones(10, dtype=np.uint16),
            ]
        )
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(len(ccsds_indices), dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.ones(len(ccsds_indices), dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": 0},
                ),
                # Packet 0 is both full and has an invalid spin
                "spin_invalids": (
                    ["epoch"],
                    np.array([1, 0], dtype=np.uint8),
                ),
            },
        )
        result = de_ccsds_qf(ds)
        np.testing.assert_array_equal(
            result["ccsds_qf"].values,
            [ImapHiL1bDeFlags.PACKET_FULL | ImapHiL1bDeFlags.BADSPIN, 0],
        )

    def test_no_valid_direct_events_all_fill_trigger_id(self):
        """de_ccsds_qf returns all zeros when trigger_id is entirely FILLVAL."""
        n_packets = 3
        # Some arbitrary, in-range CCSDS indices that would normally map to packets
        ccsds_indices = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.uint16)
        n_events = len(ccsds_indices)
        # All trigger_id values are set to the FILLVAL (0),
        # meaning no valid direct events
        trigger_fillval = 0
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(n_events, dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.full(n_events, trigger_fillval, dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": trigger_fillval},
                ),
                "spin_invalids": (["epoch"], np.zeros(n_packets, dtype=np.uint8)),
            },
        )
        result = de_ccsds_qf(ds)
        # With no valid direct events, all CCSDS quality flags should be zero
        assert "ccsds_qf" in result
        assert result["ccsds_qf"].shape[0] == n_packets
        assert np.all(result["ccsds_qf"].values == 0)

    def test_ccsds_index_fillvals_ignored(self):
        """de_ccsds_qf returns all zeros when ccsds_index includes FILLVALs (65535)."""
        n_packets = 2
        fillval = np.uint16(65535)
        # Include some events with CCSDS index FILLVAL that should be ignored
        ccsds_indices = np.array([fillval, fillval, 0, 0, 1, 1], dtype=np.uint16)
        n_events = len(ccsds_indices)
        ds = xr.Dataset(
            coords={
                "epoch": np.arange(n_packets),
                "event_met": np.arange(n_events, dtype=np.float64),
            },
            data_vars={
                "ccsds_index": (["event_met"], ccsds_indices),
                "trigger_id": xr.DataArray(
                    np.ones(n_events, dtype=np.uint8),
                    dims=["event_met"],
                    attrs={"FILLVAL": 0},
                ),
                "spin_invalids": (["epoch"], np.zeros(n_packets, dtype=np.uint8)),
            },
        )
        result = de_ccsds_qf(ds)
        # No packet reaches the full-packet threshold;
        # FILLVAL indices must not cause errors
        assert "ccsds_qf" in result
        assert result["ccsds_qf"].shape[0] == n_packets
        assert np.all(result["ccsds_qf"].values == 0)
