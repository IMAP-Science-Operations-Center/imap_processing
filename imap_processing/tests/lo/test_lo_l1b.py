import logging
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.lo.constants import LoConstants
from imap_processing.lo.l1b.lo_l1b import (
    DE_CLOCK_TICK_S,
    calculate_de_rates,
    calculate_histogram_rates,
    calculate_star_sensor_profile_for_group,
    calculate_star_sensor_profiles_by_group,
    calculate_tof1_for_golden_triples,
    convert_start_end_acq_times,
    convert_tofs_to_eu,
    create_badtimes_dataset,
    create_datasets,
    filter_valid_star_records,
    get_avg_spin_durations_per_cycle,
    get_pivot_angle_from_nhk,
    get_sampling_cadence_from_nhk,
    get_spin_start_times,
    identify_species,
    initialize_l1b_de,
    l1b_bgrates_and_goodtimes,
    l1b_star,
    lo_l1b,
    resweep_histogram_data,
    set_avg_spin_durations_per_event,
    set_coincidence_type,
    set_each_event_epoch,
    set_esa_mode,
    set_event_met,
    set_pointing_bin,
    set_pointing_direction,
    set_spin_cycle,
    set_spin_cycle_from_spin_data,
    split_backgrounds_and_goodtimes_dataset,
)
from imap_processing.spice.spin import get_spin_data
from imap_processing.spice.time import (
    et_to_met,
    et_to_ttj2000ns,
    met_to_ttj2000ns,
    str_to_et,
    ttj2000ns_to_met,
)

SPIN_BIN_6_FIELDS = [
    "h_counts",
    "o_counts",
    "tof0_tof1_counts",
    "tof0_tof2_counts",
    "tof1_tof2_counts",
    "silver_triple_counts",
]

SPIN_BIN_60_FIELDS = [
    "start_a_counts",
    "start_c_counts",
    "stop_b0_counts",
    "stop_b3_counts",
    "tof0_counts",
    "tof1_counts",
    "tof2_counts",
    "tof3_counts",
    "disc_tof0_counts",
    "disc_tof1_counts",
    "disc_tof2_counts",
    "disc_tof3_counts",
    "pos0_counts",
    "pos1_counts",
    "pos2_counts",
    "pos3_counts",
]


@pytest.fixture
def dependencies():
    data = {
        "imap_lo_l1a_de": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_de_20241022_v002.cdf"
        ),
        "imap_lo_l1a_spin": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_spin_20241022_v002.cdf"
        ),
    }

    # We have 0 for num_completed which causes issues downstream
    # when calculating the average spin durations and cascading
    # failures. Set to 28 for testing.
    data["imap_lo_l1a_spin"]["num_completed"] = 28

    # There are 3 shcoarse values for some reason, this is a bad
    # set of test data, so modify in-place here rather than updating
    data["imap_lo_l1a_de"]["shcoarse"] = data["imap_lo_l1a_de"]["shcoarse"].values[0]
    return data


@pytest.fixture
def anc_dependencies():
    return [
        str(
            imap_module_directory
            / "tests/lo/test_anc/imap_lo_sweep-table-small_20250101_20260301_v001.csv",
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/imap_lo_bad-times-small_20250101_20270101_v001.csv",
        ),
        str(
            imap_module_directory / "tests/lo/test_anc/imap_lo_esa-mode-lut_v001.csv",
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/imap_lo_bg-rates-anti-ram-overrides_20250901_v001.csv",
        ),
    ]


@pytest.fixture
def attr_mgr_l1b():
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1b")
    return attr_mgr_l1b


@pytest.fixture
def attr_mgr_l1a():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    return attr_mgr


@pytest.fixture
def l1b_histrates():
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-15T02:00:00", "2025-04-15T03:00:00"])
    )

    # Build dataset with all expected fields
    data_vars = {}
    for f in SPIN_BIN_6_FIELDS:
        data_vars[f] = (("epoch", "esa_step", "spin_bin_6"), np.zeros((2, 7, 60)))
    for f in SPIN_BIN_60_FIELDS:
        data_vars[f] = (("epoch", "esa_step", "spin_bin_60"), np.zeros((2, 7, 6)))

    l1b_histrates = xr.Dataset(
        data_vars,
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
            "spin_bin_60": np.arange(6),
        },
    )

    return l1b_histrates


@pytest.fixture
def l1a_hist():
    epoch_date = et_to_ttj2000ns(str_to_et(["2025-04-15T02:00:00"]))
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "oxygen": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "tof0_tof1": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "tof0_tof2": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "tof1_tof2": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "silver": (("epoch", "esa_step", "azimuth_6"), np.zeros((1, 7, 60))),
            "start_a": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "start_c": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "stop_b0": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "stop_b3": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "tof0_count": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "tof1_count": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "tof2_count": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "tof3_count": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "disc_tof0": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "disc_tof1": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "disc_tof2": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "disc_tof3": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "pos0": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "pos1": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "pos2": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
            "pos3": (("epoch", "esa_step", "azimuth_60"), np.zeros((1, 7, 6))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "azimuth_6": np.arange(60),
            "azimuth_60": np.arange(6),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )
    return l1a_hist


@patch(
    "imap_processing.lo.l1b.lo_l1b.frame_transform",
    return_value=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
)
@patch(
    "imap_processing.lo.l1b.lo_l1b.lo_instrument_pointing",
    return_value=np.zeros((2000, 3)),
)
@patch(
    "imap_processing.lo.l1b.lo_l1b.get_pointing_times",
    return_value=(473389199, 473472001),
)
@patch("imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=0)
@patch(
    "imap_processing.lo.l1b.lo_l1b.cartesian_to_latitudinal",
    return_value=np.zeros((2000, 3)),
)
@patch("imap_processing.lo.l1b.lo_l1b.interpolate_spin_data")
def test_lo_l1b_de(
    mock_interpolate_spin_data,
    mock_frame_transform,
    mock_lo_instrument_pointing,
    mocked_get_pointing_times,
    mock_spin_number,
    mock_cartesian_to_latitudinal,
    dependencies,
    anc_dependencies,
):
    # Arrange
    # Mock the spin data to provide spin start times
    # Create a DataFrame covering the time range of the test data
    mock_spin_df = pd.DataFrame(
        {
            "spin_start_met": np.ones([1]),
        }
    )
    mock_interpolate_spin_data.return_value = mock_spin_df

    # Add l1b_nhk dependency with pivot angle information
    l1b_nhk = xr.Dataset(
        {"pcc_cumulative_cnt_pri": ("epoch", [45.0])},
        coords={"epoch": [met_to_ttj2000ns(473389200)]},
    )
    dependencies["imap_lo_l1b_nhk"] = l1b_nhk

    expected_logical_source_de = "imap_lo_l1b_de"

    # Act
    output_files = lo_l1b(dependencies, anc_dependencies, descriptor="de")

    # Assert
    assert expected_logical_source_de == output_files[-1].attrs["Logical_source"]
    # Verify that pivot_angle is present in the output
    assert "pivot_angle" in output_files[-1]
    assert output_files[-1]["pivot_angle"].values[0] == 45.0


@patch("imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=0)
@patch(
    "imap_processing.lo.l1b.lo_l1b.get_pointing_times",
    return_value=(473389199, 473472001),
)
def test_lo_l1b_histogram_rates(
    mock_repoint_times, mock_spin_number, l1a_hist, anc_dependencies
):
    # Arrange
    met = et_to_met(str_to_et(["2025-04-15T02:00:00"]))
    l1a_spin = xr.Dataset(
        {
            "shcoarse": ("epoch", [0]),
            "num_completed": ("epoch", [28]),
            "acq_start_sec": ("epoch", met),
            "acq_start_subsec": ("epoch", [0]),
            "acq_end_sec": ("epoch", met + 420),
            "acq_end_subsec": ("epoch", [0]),
        },
        coords={
            "epoch": et_to_ttj2000ns(str_to_et(["2025-04-15T02:00:00"])),
        },
        attrs={"Logical_source": "imap_lo_l1a_spin"},
    )
    sci_dependencies = {
        "imap_lo_l1a_histogram": l1a_hist,
        "imap_lo_l1a_spin": l1a_spin,
    }

    # Act
    l1b_datasets = lo_l1b(sci_dependencies, anc_dependencies, descriptor="all-rates")

    # Assert
    assert "h_rates" in l1b_datasets[-2].data_vars
    assert "o_rates" in l1b_datasets[-2].data_vars
    assert "exposure_time_6deg" in l1b_datasets[-2].data_vars
    assert "h_counts" in l1b_datasets[-2].data_vars
    assert "o_counts" in l1b_datasets[-2].data_vars
    assert l1b_datasets[-2]["exposure_time_6deg"].values[0, 0, 0] == 2
    # Should be 10x as large
    assert l1b_datasets[-1]["exposure_time_60deg"].values[0, 0, 0] == 20


def test_create_datasets():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1b")

    logical_source = "imap_lo_l1b_de"

    data_field_tup = namedtuple("data_field_tup", ["name"])
    data_fields = [
        data_field_tup("ESA_STEP"),
        data_field_tup("MODE"),
        data_field_tup("TOF0"),
        data_field_tup("TOF1"),
        data_field_tup("TOF2"),
        data_field_tup("TOF3"),
        data_field_tup("COINCIDENCE_TYPE"),
        data_field_tup("POS"),
        data_field_tup("COINCIDENCE"),
        data_field_tup("BADTIME"),
        data_field_tup("DIRECTION"),
    ]

    dataset = create_datasets(attr_mgr, logical_source, data_fields)

    # verify that epoch does not have a DEPEND_0 attribute
    assert "DEPEND_0" not in dataset["epoch"].attrs

    assert len(dataset.tof0.shape) == 1
    assert dataset.tof0.shape[0] == 3
    assert len(dataset.tof1.shape) == 1
    assert dataset.tof1.shape[0] == 3
    assert len(dataset.tof2.shape) == 1
    assert dataset.tof2.shape[0] == 3
    assert len(dataset.tof3.shape) == 1
    assert dataset.tof3.shape[0] == 3
    assert len(dataset.mode.shape) == 1
    assert dataset.mode.shape[0] == 3
    assert len(dataset.coincidence_type.shape) == 1
    assert dataset.coincidence_type.shape[0] == 3
    assert len(dataset.pos.shape) == 1
    assert dataset.pos.shape[0] == 3
    assert len(dataset.direction.shape) == 2
    assert dataset.direction.shape[0] == 3
    assert dataset.direction.shape[1] == 3
    assert len(dataset.badtime.shape) == 1
    assert dataset.badtime.shape[0] == 3
    assert len(dataset.esa_step.shape) == 1
    assert dataset.esa_step.shape[0] == 3


def test_initialize_dataset(dependencies, attr_mgr_l1b):
    # Arrange
    l1a_de = dependencies["imap_lo_l1a_de"]
    logical_source = "imap_lo_l1b_de"

    # Act
    l1b_de = initialize_l1b_de(l1a_de, attr_mgr_l1b, logical_source)

    # Assert
    assert l1b_de.attrs["Logical_source"] == logical_source
    assert list(l1b_de.coords.keys()) == []
    assert len(l1b_de.data_vars) == 5
    assert len(l1b_de.coords) == 0
    for l1b_name, l1a_name in {
        "pos": "pos",
        "mode_bit": "mode",
        "absent": "coincidence_type",
        "esa_step": "esa_step",
    }.items():
        assert l1b_name in l1b_de.data_vars
        np.testing.assert_array_equal(l1b_de[l1b_name], l1a_de[l1a_name])

    expected_l1b_shcoarse = np.repeat(
        l1a_de["shcoarse"].values, l1a_de["de_count"].values
    )
    np.testing.assert_array_equal(l1b_de["shcoarse"], expected_l1b_shcoarse)


def test_set_esa_mode(anc_dependencies, attr_mgr_l1b):
    # Arrange
    l1b_de = xr.Dataset(
        {},
        coords={"epoch": [0, 1, 2, 3, 4]},
    )
    pointing_start_met = 473389199
    pointing_end_met = 473472001

    expected_esa_mode = np.array([0, 0, 0, 0, 0])

    # Act
    l1b_de = set_esa_mode(
        pointing_start_met, pointing_end_met, anc_dependencies, l1b_de
    )

    # Assert
    np.testing.assert_array_equal(l1b_de["esa_mode"].values, expected_esa_mode)


def test_set_esa_mode_error(anc_dependencies, attr_mgr_l1b):
    # Arrange
    l1b_de = xr.Dataset(
        {},
        coords={"epoch": [0, 1, 2, 3, 4]},
    )
    pointing_start_met = 473389199
    pointing_end_met = 509369021

    # Act / Assert
    with pytest.raises(
        ValueError, match="Multiple ESA modes found in sweep table for pointing."
    ):
        l1b_de = set_esa_mode(
            pointing_start_met, pointing_end_met, anc_dependencies, l1b_de
        )


def test_convert_start_end_acq_times():
    # Arrange
    spin = xr.Dataset(
        {
            "acq_start_sec": ("epoch", [1, 2, 3]),
            "acq_start_subsec": ("epoch", [4, 5, 6]),
            "acq_end_sec": ("epoch", [7, 8, 9]),
            "acq_end_subsec": ("epoch", [10, 11, 12]),
        },
        coords={"epoch": [0, 1, 2]},
    )

    acq_start_expected = xr.DataArray(
        [
            spin["acq_start_sec"][0] + spin["acq_start_subsec"][0] * 1e-6,
            spin["acq_start_sec"][1] + spin["acq_start_subsec"][1] * 1e-6,
            spin["acq_start_sec"][2] + spin["acq_start_subsec"][2] * 1e-6,
        ],
        dims="epoch",
    )
    acq_end_expected = xr.DataArray(
        [
            spin["acq_end_sec"][0] + spin["acq_end_subsec"][0] * 1e-6,
            spin["acq_end_sec"][1] + spin["acq_end_subsec"][1] * 1e-6,
            spin["acq_end_sec"][2] + spin["acq_end_subsec"][2] * 1e-6,
        ],
        dims="epoch",
    )

    # Act
    acq_start, acq_end = convert_start_end_acq_times(spin)

    # Assert
    np.testing.assert_array_equal(acq_start.values, acq_start_expected.values)
    np.testing.assert_array_equal(acq_end.values, acq_end_expected.values)


def test_get_avg_spin_durations():
    # Arrange
    spin_ds = xr.Dataset(
        {
            "acq_start_sec": ("epoch", [1, 2, 3]),
            "acq_start_subsec": ("epoch", [1e6, 2e6, 3e6]),
            "acq_end_sec": ("epoch", [100, 200, 300]),
            "acq_end_subsec": ("epoch", [1e6, 2e6, 3e6]),
            "num_completed": ("epoch", [28, 14, 28]),
        },
        coords={"epoch": [0, 1, 2]},
    )
    expected_avg_spin_durations = np.array(
        [(101 - 2) / 28, (202 - 4) / 14, (303 - 6) / 28]
    )

    # Act
    avg_spin_durations = get_avg_spin_durations_per_cycle(spin_ds)
    # Assert
    np.testing.assert_array_equal(avg_spin_durations, expected_avg_spin_durations)


@patch("imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=0)
def test_spin_cycle(mock_get_spin_number):
    # Arrange
    de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "esa_step": ("direct_event", [1, 2, 3, 4, 5]),
            "met": ("epoch", [0, 7]),
        },
        coords={"epoch": [0, 1], "direct_event": [1, 2, 3, 4, 5]},
    )
    pointing_start_met = 0

    # spin_cycle = spin_start + 7 + (esa_step - 1) * 2
    # where spin start is the spin number for the first spin
    # in an Aggregated Science Cycle (first spin number of an epoch)
    # and esa_step is the esa_step for a direct event
    spin_cycle_expected = np.array([7, 9, 39, 41, 43])
    spin_cycle_data = xr.Dataset()

    # Act
    spin_cycle_data = set_spin_cycle(pointing_start_met, de, spin_cycle_data)

    # Assert
    np.testing.assert_array_equal(spin_cycle_data["spin_cycle"], spin_cycle_expected)


@patch("imap_processing.lo.l1b.lo_l1b.interpolate_spin_data")
def test_get_spin_start_times(mock_interpolate_spin_data):
    # Arrange
    # Mock the spin data to return specific spin start times
    mock_spin_df = pd.DataFrame(
        {
            "spin_start_met": [10.5, 30.1],
        }
    )
    mock_interpolate_spin_data.return_value = mock_spin_df

    l1a_de = xr.Dataset(
        {
            "met": ("epoch", [15, 35]),
            "de_count": ("epoch", [2, 3]),
            "de_time": ("direct_event", [0, 1000, 2000, 3000, 4000]),
        },
        coords={"epoch": [0, 1], "direct_event": [0, 1, 2, 3, 4]},
    )

    # Expected: met 15 should match spin at index 0 (10 < 15 < 20)
    # met 35 should match spin at index 2 (30 < 35 < 40)
    # Repeated by de_count: [2, 3] -> [index0, index0, index2, index2, index2]
    spin_start_times_expected = np.array(
        [10.5, 10.5, 30.1, 30.1, 30.1]  # 10 + 0.5e6*1e-6  # 30 + 0.1e6*1e-6
    )

    # Act
    spin_start_times = get_spin_start_times(l1a_de)

    # Assert
    np.testing.assert_allclose(
        spin_start_times,
        spin_start_times_expected,
        atol=1e-4,
    )


@patch("imap_processing.lo.l1b.lo_l1b.interpolate_spin_data")
def test_set_event_met(mock_interpolate_spin_data):
    # Arrange
    # Mock the spin data
    mock_spin_df = pd.DataFrame(
        {
            "spin_start_met": [10, 30],
        }
    )
    mock_interpolate_spin_data.return_value = mock_spin_df

    l1b_de = xr.Dataset()
    l1a_de = xr.Dataset(
        {
            "met": ("epoch", [15, 35]),
            "de_count": ("epoch", [2, 3]),
            "de_time": ("direct_event", [0, 1000, 2000, 3000, 4000]),
        },
        coords={
            "epoch": [0, 1],
            "direct_event": [0, 1, 2, 3, 4],
        },
    )

    # met 15 -> spin_start 10, met 35 -> spin_start 30
    # event_met = spin_start + de_time * DE_CLOCK_TICK_S
    expected_event_met = np.array(
        [
            10 + 0 * DE_CLOCK_TICK_S,  # 10.0
            10 + 1000 * DE_CLOCK_TICK_S,  # 14.096
            30 + 2000 * DE_CLOCK_TICK_S,  # 38.192
            30 + 3000 * DE_CLOCK_TICK_S,  # 42.288
            30 + 4000 * DE_CLOCK_TICK_S,  # 46.384
        ]
    )

    # Act
    l1b_de = set_event_met(l1a_de, l1b_de)

    # Assert
    np.testing.assert_allclose(
        l1b_de["event_met"].values,
        expected_event_met,
        atol=1e-4,
    )


def test_set_each_event_epoch():
    l1b_de = xr.Dataset(
        {
            "event_met": ("epoch", [10, 20, 30, 40, 50]),
        },
        coords={
            "epoch": [0, 1, 2, 3, 4],
        },
    )
    epoch_expected = met_to_ttj2000ns(np.array([10, 20, 30, 40, 50]))

    l1b_de = set_each_event_epoch(l1b_de)

    np.testing.assert_allclose(
        l1b_de["epoch"].values,
        epoch_expected,
        atol=1e-4,
    )


def test_set_avg_spin_durations_per_event():
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
        }
    )
    l1b_de = xr.Dataset(coords={"epoch": [0, 1, 2, 3, 4]})

    avg_spin_durations = xr.DataArray([5, 10])

    # Act
    l1b_de = set_avg_spin_durations_per_event(l1a_de, l1b_de, avg_spin_durations)

    # Assert
    np.testing.assert_array_equal(
        l1b_de["avg_spin_durations"].values, np.array([5, 5, 10, 10, 10])
    )


def test_calculate_tof1_for_golden_triples():
    # Arrange
    l1a_de = xr.Dataset(
        {
            "coincidence_type": ("epoch", [0, 0, 0]),
            "mode": ("epoch", [0, 0, 1]),
            "tof0": ("epoch", [2, 4, 2]),
            "tof1": ("epoch", [0, 0, 0]),
            "tof2": ("epoch", [2, 6, 2]),
            "tof3": ("epoch", [2, 8, 2]),
            "cksm": ("epoch", [2, 12, 2]),
        }
    )

    l1a_de_expected = xr.Dataset(
        {
            "coincidence_type": ("epoch", [0, 0, 0]),
            "mode": ("epoch", [0, 0, 1]),
            "tof0": ("epoch", [2, 4, 2]),
            "tof1": ("epoch", [0, 0, 42]),
            "tof2": ("epoch", [2, 6, 2]),
            "tof3": ("epoch", [2, 8, 2]),
            "cksm": ("epoch", [2, 12, 2]),
        }
    )

    # Act
    l1a_de = calculate_tof1_for_golden_triples(l1a_de)

    # Assert
    xr.testing.assert_equal(l1a_de, l1a_de_expected)


def test_set_coincidence_type(attr_mgr_l1a):
    # Arrange
    l1b_de = xr.Dataset()
    tof_fill = attr_mgr_l1a.get_variable_attributes("tof0")["FILLVAL"]
    ckm_fill = attr_mgr_l1a.get_variable_attributes("cksm")["FILLVAL"]
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [3]),
            "coincidence_type": ("direct_events", [0, 0, 4]),
            "mode": ("direct_events", [1, 0, 1]),
            "tof0": ("direct_events", [5, 2, 10]),
            "tof1": ("direct_events", [10, 4, tof_fill]),
            "tof2": ("direct_events", [15, 6, 20]),
            "tof3": ("direct_events", [20, 8, 30]),
            "cksm": ("direct_events", [25, ckm_fill, ckm_fill]),
        },
        coords={
            "epoch": [0],
            "direct_events": [0, 1, 2],
        },
    )

    coincidence_type_expected = np.array(["111111", "111100", "101101"])

    # Act
    l1b_de = set_coincidence_type(l1a_de, l1b_de, attr_mgr_l1a)

    # Assert
    np.testing.assert_array_equal(
        l1b_de["coincidence_type"].values,
        coincidence_type_expected,
    )


def test_convert_tofs_to_eu(attr_mgr_l1b, attr_mgr_l1a):
    l1b_de = xr.Dataset()
    tof_fill_l1a = attr_mgr_l1a.get_variable_attributes("tof0")["FILLVAL"]
    tof_fill_l1b = attr_mgr_l1b.get_variable_attributes("tof1")["FILLVAL"]
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2]),
            "coincidence_type": ("direct_events", [0, 4]),
            "mode": ("direct_events", [1, 0]),
            "tof0": ("direct_events", [5, 2]),
            "tof1": ("direct_events", [10, tof_fill_l1a]),
            "tof2": ("direct_events", [15, 6]),
            "tof3": ("direct_events", [20, 8]),
        },
        coords={
            "epoch": [0],
            "direct_events": [0, 1],
        },
    )

    tof0_expected = np.array([1.394394, 0.889272])
    tof1_expected = np.array([0.931059, tof_fill_l1b])
    tof2_expected = np.array([2.870557, 1.372876])
    tof3_expected = np.array([3.89606, 1.83878])

    # Act
    l1b_de = convert_tofs_to_eu(l1a_de, l1b_de, attr_mgr_l1a, attr_mgr_l1b)

    tof_checks = [
        ("tof0", tof0_expected),
        ("tof1", tof1_expected),
        ("tof2", tof2_expected),
        ("tof3", tof3_expected),
    ]
    # Assert
    for tof, expected_tof in tof_checks:
        np.testing.assert_allclose(
            l1b_de[tof].values,
            expected_tof,
            atol=1e-6,
        )


def test_identify_species(attr_mgr_l1b):
    # Arrange
    fill_val = attr_mgr_l1b.get_variable_attributes("tof2")["FILLVAL"]
    l1b_de = xr.Dataset(
        {
            "tof2": ("epoch", [1, 14, 50, 80, 500, fill_val]),
        }
    )

    expected_species = np.array(["U", "H", "U", "O", "U", "U"])

    # Act
    l1b_de = identify_species(l1b_de)

    # Assert
    np.testing.assert_array_equal(l1b_de["species"], expected_species)


@patch(
    "imap_processing.lo.l1b.lo_l1b.lo_instrument_pointing",
    return_value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
)
def test_set_direction(mock_lo_instrument_pointing, imap_ena_sim_metakernel):
    # Arrange
    l1b_de = xr.Dataset(
        {
            "pivot_angle": ("epoch", [0, 0, 0, 0]),
        },
        coords={
            "epoch": [0, 1, 2, 3],
        },
    )
    # latitudes are -90 to 90
    expected_hae_x = np.array([1, 4, 7, 10])
    expected_hae_y = np.array([2, 5, 8, 11])
    expected_hae_z = np.array([3, 6, 9, 12])

    # Act
    l1b_de = set_pointing_direction(l1b_de)

    # Assert
    np.testing.assert_allclose(
        l1b_de["hae_x"].values,
        expected_hae_x,
        atol=1e-1,
    )
    np.testing.assert_allclose(
        l1b_de["hae_y"].values,
        expected_hae_y,
        atol=1e-1,
    )
    np.testing.assert_allclose(
        l1b_de["hae_z"].values,
        expected_hae_z,
        atol=1e-1,
    )


@pytest.mark.parametrize("pivot_angle", [75, 90, 105])
def test_pointing_bins(pivot_angle):
    # Arrange - Mock returns depend on pivot_angle
    # Calculate offset based on pivot angle: lats = lats - (90 - pivot_angle)
    offset = 90 - pivot_angle

    with (
        patch(
            "imap_processing.lo.l1b.lo_l1b.frame_transform",
            return_value=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        patch(
            "imap_processing.lo.l1b.lo_l1b.cartesian_to_latitudinal",
            # Adjust latitude values based on pivot angle offset
            # Longitudes: -180 -> 180, 0 -> 0, 90 -> 90, 180 -> 180
            # After shift to 0-360: 180, 0, 90, 180
            return_value=np.array(
                [
                    [0, -180, -2 + offset],
                    [0, 0, 0 + offset],
                    [0, 90, 1 + offset],
                    [0, 180, 2 + offset],
                ]
            ),
        ),
    ):
        l1b_de = xr.Dataset(
            {
                "hae_x": ("epoch", [1, 1, 1, 1]),
                "hae_y": ("epoch", [0, 0, 0, 0]),
                "hae_z": ("epoch", [0, 0, 0, 0]),
            },
            coords={
                "epoch": [
                    7.9794907049e17,
                    7.9794907153e17,
                    7.9794907254e17,
                    7.9794907354e17,
                ],
                "pivot_angle": [pivot_angle],
            },
        )

        expected_pointing_lats = np.array([0, 20, 30, 40])
        # Longitude bins are now in 0-360 range after the shift
        expected_pointing_lons = np.array([1800, 0, 900, 1800])

        # Act
        l1b_de = set_pointing_bin(l1b_de)

        # Assert
        np.testing.assert_array_equal(l1b_de["off_angle_bin"], expected_pointing_lats)
        np.testing.assert_array_equal(l1b_de["spin_bin"], expected_pointing_lons)


def test_badtimes_no_spin():
    """An empty dataset should still be returned when no spin data is found."""
    badtimes_ds = create_badtimes_dataset()

    assert len(badtimes_ds["epoch"]) == 0
    # We should have put empty variables into the dataset
    assert "BadTime_start" in badtimes_ds.data_vars


def test_badtimes_with_spin(spice_test_data_path, use_test_spin_data_csv):
    """Verify some actual badtimes are created from thruster firings."""
    # Initialize the spin data
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    use_test_spin_data_csv([fake_spin_path])

    badtimes_ds = create_badtimes_dataset()
    spin_df = get_spin_data()

    thruster_df = spin_df[spin_df["thruster_firing"]]
    n_thruster_firings = len(thruster_df)
    # We should have some thruster firings
    assert n_thruster_firings > 0

    # Check the thruster firings we created match those in the spin data
    assert len(badtimes_ds["epoch"]) == n_thruster_firings
    np.testing.assert_array_equal(
        badtimes_ds["BadTime_start"], thruster_df["spin_start_sec_sclk"]
    )
    np.testing.assert_array_equal(badtimes_ds["badtime_flag"], 1)

    # There should be a dataset returned from the main code in this case
    datasets = lo_l1b({}, [], descriptor="badtimes")
    assert len(datasets) == 1


def test_l1b_badtimes_skipped_if_empty():
    datasets = lo_l1b({}, [], descriptor="badtimes")
    assert len(datasets) == 0


def test_lo_l1b_unexpected_descriptor(caplog):
    """Test that an unexpected descriptor logs a warning and returns empty list."""
    datasets = lo_l1b({}, [], descriptor="unknown")
    assert len(datasets) == 0
    assert "Unexpected descriptor: 'unknown'" in caplog.text


def test_resweep_histogram_success(l1b_histrates, anc_dependencies):
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-15T02:00:00", "2025-04-15T03:00:00"])
    )
    l1b_histrates["epoch"] = epoch_date
    exposure_factor_6deg = np.full((2, 7, 60), 4)
    exposure_factor_60deg = np.full((2, 7, 6), 4)
    exposure_factor_6deg[:, 0, :] = 8
    exposure_factor_60deg[:, 0, :] = 8
    exposure_factor_6deg[:, 1, :] = 0
    exposure_factor_60deg[:, 1, :] = 0

    l1b_histrates.h_counts[0, 0, 0] = 5
    l1b_histrates.h_counts[0, 1, 0] = 10
    l1b_histrates.h_counts[0, 2, 0] = 2
    l1b_histrates.o_counts[1, 0, 0] = 2
    l1b_histrates.o_counts[1, 1, 0] = 3
    l1b_histrates.o_counts[1, 2, 0] = 4

    l1b_histrates, exposure_factor = resweep_histogram_data(
        l1b_histrates, anc_dependencies
    )

    assert l1b_histrates.h_counts[0, 0, 0] == 15
    assert l1b_histrates.h_counts[0, 1, 0] == 0
    assert l1b_histrates.h_counts[0, 2, 0] == 2

    assert l1b_histrates.o_counts[1, 0, 0] == 5
    assert l1b_histrates.o_counts[1, 1, 0] == 0
    assert l1b_histrates.o_counts[1, 2, 0] == 4

    for field in SPIN_BIN_6_FIELDS + SPIN_BIN_60_FIELDS:
        np.testing.assert_array_equal(l1b_histrates[field], l1b_histrates[field])
    np.testing.assert_array_equal(exposure_factor["6deg"], exposure_factor_6deg)
    np.testing.assert_array_equal(exposure_factor["60deg"], exposure_factor_60deg)


def test_resweep_histogram_no_date_in_sweep(l1b_histrates, anc_dependencies, caplog):
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-25T02:00:00", "2025-04-25T03:00:00"])
    )
    l1b_histrates["epoch"] = epoch_date

    l1b_histrates.h_counts[0, 0, 0] = 5
    l1b_histrates.h_counts[0, 1, 0] = 10
    l1b_histrates.h_counts[0, 2, 0] = 2

    pytest.raises(ValueError, resweep_histogram_data, l1b_histrates, anc_dependencies)


def test_resweep_histogram_no_table_in_lut(l1b_histrates, anc_dependencies, caplog):
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2024-01-01T02:00:00", "2024-01-01T03:00:00"])
    )
    l1b_histrates["epoch"] = epoch_date

    l1b_histrates.h_counts[0, 0, 0] = 5
    l1b_histrates.h_counts[0, 1, 0] = 10
    l1b_histrates.h_counts[0, 2, 0] = 2

    with caplog.at_level(logging.WARNING):
        result, _ = resweep_histogram_data(l1b_histrates, anc_dependencies)

        resweep_histogram_data(l1b_histrates, anc_dependencies)
    # Check that warning was logged
    assert any(
        "No LUT entries for epoch" in record.message for record in caplog.records
    )


def test_resweep_histogram_multiple_lut(l1b_histrates, anc_dependencies, caplog):
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-16T02:00:00", "2025-04-16T03:00:00"])
    )

    l1b_histrates["epoch"] = epoch_date

    with caplog.at_level(logging.WARNING):
        result, _ = resweep_histogram_data(l1b_histrates, anc_dependencies)

    # Check that warning was logged
    assert any(
        "Multiple LUT tables found for epoch" in record.message
        for record in caplog.records
    )
    assert any("but found tables" in record.message for record in caplog.records)


def test_calculate_histogram_rates(l1b_histrates):
    acq_start = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-15T01:55:00")),
            et_to_met(str_to_et("2025-04-15T02:55:00")),
        ]
    )
    acq_end = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-15T02:02:00")),
            et_to_met(str_to_et("2025-04-15T03:02:00")),
        ]
    )
    avg_spin_durations_per_cycle = xr.DataArray([30, 15])
    # default zeros then set a sample exposure as in original test intent
    exposure_factors_6deg = np.zeros((2, 7, 60))
    exposure_factors_60deg = np.zeros((2, 7, 6))
    exposure_factors_6deg[0, 0, 0] = 1
    exposure_factors_60deg[0, 0, 0] = 1
    exposure_factors_6deg[0, 1, 0] = 0
    exposure_factors_60deg[0, 1, 0] = 0

    exposure_factors = {}
    exposure_factors["6deg"] = exposure_factors_6deg
    exposure_factors["60deg"] = exposure_factors_60deg

    # Populate counts used by assertions
    l1b_histrates.h_counts[0, 0, 0] = 30
    l1b_histrates.h_counts[0, 1, 0] = 10
    l1b_histrates.h_counts[0, 2, 0] = 2
    l1b_histrates.h_counts[1, 0, 0] = 15
    l1b_histrates.h_counts[1, 1, 0] = 30
    l1b_histrates.h_counts[1, 2, 0] = 45

    l1b_histrates.o_counts[0, 0, 0] = 100
    l1b_histrates.o_counts[0, 1, 0] = 50
    l1b_histrates.o_counts[0, 2, 0] = 25
    l1b_histrates.o_counts[1, 0, 0] = 2
    l1b_histrates.o_counts[1, 1, 0] = 3
    l1b_histrates.o_counts[1, 2, 0] = 4

    l1b_histrate = calculate_histogram_rates(
        l1b_histrates,
        acq_start,
        acq_end,
        avg_spin_durations_per_cycle,
        exposure_factors,
    )

    hist_rates_h_epoch_0 = l1b_histrate["h_rates"]
    hist_rates_h_epoch_0[0, :, :] = hist_rates_h_epoch_0[0, :, :] / 2
    hist_rates_h_epoch_0[0, :, 0] = hist_rates_h_epoch_0[0, :, 0] / 2
    hist_rates_o_epoch_0 = l1b_histrate["o_rates"]
    hist_rates_o_epoch_0[0, :, :] = hist_rates_o_epoch_0[0, :, :] / 2
    hist_rates_o_epoch_0[0, :, 0] = hist_rates_o_epoch_0[0, :, 0] / 2

    np.testing.assert_array_equal(
        l1b_histrate["h_rates"][0, :, :], hist_rates_h_epoch_0[0, :, :]
    )
    np.testing.assert_array_equal(
        l1b_histrate["h_rates"][1, :, :], hist_rates_h_epoch_0[1, :, :]
    )
    np.testing.assert_array_equal(
        l1b_histrate["o_rates"][0, :, :], hist_rates_o_epoch_0[0, :, :]
    )
    np.testing.assert_array_equal(
        l1b_histrate["o_rates"][1, :, :], hist_rates_o_epoch_0[1, :, :]
    )


def test_calculate_histogram_rates_no_interval_found(l1b_histrates):
    acq_start = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-30T01:55:00")),
            et_to_met(str_to_et("2025-04-30T02:55:00")),
        ]
    )
    acq_end = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-30T02:02:00")),
            et_to_met(str_to_et("2025-04-30T03:02:00")),
        ]
    )
    avg_spin_durations_per_cycle = xr.DataArray([30, 15])

    exposure_factors_6deg = np.zeros((2, 7, 60))
    exposure_factors_60deg = np.zeros((2, 7, 6))
    exposure_factors = {}
    exposure_factors["6deg"] = exposure_factors_6deg
    exposure_factors["60deg"] = exposure_factors_60deg

    l1b_histrate = calculate_histogram_rates(
        l1b_histrates,
        acq_start,
        acq_end,
        avg_spin_durations_per_cycle,
        exposure_factors,
    )

    np.testing.assert_array_equal(l1b_histrate["h_rates"], np.zeros((2, 7, 60)))
    np.testing.assert_array_equal(l1b_histrate["o_rates"], np.zeros((2, 7, 60)))


def test_calculate_histogram_rates_zero_exposure_time(l1b_histrates):
    acq_start = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-15T01:55:00")),
            et_to_met(str_to_et("2025-04-15T02:55:00")),
        ]
    )
    acq_end = xr.DataArray(
        [
            et_to_met(str_to_et("2025-04-15T02:02:00")),
            et_to_met(str_to_et("2025-04-15T03:02:00")),
        ]
    )
    avg_spin_durations_per_cycle = xr.DataArray([0, 15])

    exposure_factors_6deg = np.zeros((2, 7, 60))
    exposure_factors_60deg = np.zeros((2, 7, 6))
    exposure_factors = {}
    exposure_factors["6deg"] = exposure_factors_6deg
    exposure_factors["60deg"] = exposure_factors_60deg

    l1b_histrate = calculate_histogram_rates(
        l1b_histrates,
        acq_start,
        acq_end,
        avg_spin_durations_per_cycle,
        exposure_factors,
    )

    np.testing.assert_array_equal(l1b_histrate["h_rates"], np.zeros((2, 7, 60)))
    np.testing.assert_array_equal(l1b_histrate["o_rates"], np.zeros((2, 7, 60)))


def test_set_spin_cycle_from_spin_data_histogram():
    """Test spin cycle calculation for histogram data."""
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-15T02:00:00", "2025-04-15T03:00:00"])
    )
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "azimuth_6": np.arange(60),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )

    l1b_hist = xr.Dataset(
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
        }
    )

    met_times = ttj2000ns_to_met(epoch_date)
    spin_data = xr.Dataset(
        {
            "shcoarse": ("epoch", met_times),
            "num_completed": ("epoch", [28, 28]),
            "acq_start_sec": ("epoch", met_times),
            "acq_start_subsec": ("epoch", [0, 0]),
            "acq_end_sec": ("epoch", met_times),
            "acq_end_subsec": ("epoch", [0, 0]),
        },
        coords={
            "epoch": epoch_date,
        },
    )

    # Mock get_spin_number to return predictable values
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=np.array([0, 28])
    ):
        # Act
        l1b_hist = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    # Expected: spin_cycle = spin_start + 7 + (esa_step - 1) * 2
    # For epoch 0: 0 + 7 + (1-1)*2 = 7, (2-1)*2 = 9, ..., (7-1)*2 = 19
    # For epoch 1: 28 + 7 + (1-1)*2 = 35, ..., 28 + 7 + (7-1)*2 = 47
    expected_spin_cycles = np.array(
        [[7, 9, 11, 13, 15, 17, 19], [35, 37, 39, 41, 43, 45, 47]]
    )

    # Assert
    assert "spin_cycle" in l1b_hist.data_vars
    np.testing.assert_array_equal(l1b_hist["spin_cycle"].values, expected_spin_cycles)


def test_set_spin_cycle_from_spin_data_matching_ascs():
    """Test that science ASCs correctly match to spin ASCs."""
    # Arrange - Science ASCs that should match different spin ASCs
    science_met = [100, 200, 300]
    spin_met = [50, 150, 250]  # Science times fall after these spin times

    epoch_date = met_to_ttj2000ns(science_met)
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (("epoch", "esa_step"), np.zeros((3, 7))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )

    l1b_hist = xr.Dataset(coords={"epoch": epoch_date, "esa_step": np.arange(1, 8)})

    spin_data = xr.Dataset(
        {
            "shcoarse": ("epoch", spin_met),
            "num_completed": ("epoch", [28, 28, 28]),
            "acq_start_sec": ("epoch", spin_met),
            "acq_start_subsec": ("epoch", [0, 0, 0]),
            "acq_end_sec": ("epoch", spin_met),
            "acq_end_subsec": ("epoch", [0, 0, 0]),
        },
        coords={"epoch": met_to_ttj2000ns(spin_met)},
    )

    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_spin_number",
        return_value=np.array([0, 28, 56]),
    ):
        # Act
        l1b_hist = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    # Assert - Each epoch should use the correct spin start number
    assert l1b_hist["spin_cycle"][0, 0] == 7  # 0 + 7 + 0
    assert l1b_hist["spin_cycle"][1, 0] == 35  # 28 + 7 + 0
    assert l1b_hist["spin_cycle"][2, 2] == 67  # 56 + 7 + 2*2


def test_set_spin_cycle_from_spin_data_repeated_closest():
    """Test when multiple science ASCs map to the same spin ASC."""
    # Arrange - Multiple science ASCs close to the same spin ASC
    science_met = [100, 101, 200, 201]
    spin_met = [50, 150]  # First two science ASCs map to spin[0], last two to spin[1]

    epoch_date = met_to_ttj2000ns(science_met)
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (("epoch", "esa_step"), np.zeros((4, 7))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )

    l1b_hist = xr.Dataset(coords={"epoch": epoch_date, "esa_step": np.arange(1, 8)})

    spin_data = xr.Dataset(
        {
            "shcoarse": ("epoch", spin_met),
            "num_completed": ("epoch", [28, 28]),
            "acq_start_sec": ("epoch", spin_met),
            "acq_start_subsec": ("epoch", [0, 0]),
            "acq_end_sec": ("epoch", spin_met),
            "acq_end_subsec": ("epoch", [0, 0]),
        },
        coords={"epoch": met_to_ttj2000ns(spin_met)},
    )

    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_spin_number",
        return_value=np.array([10, 10, 38, 38]),
    ):
        # Act
        l1b_hist = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    # Assert - First two should use spin 10, last two should use spin 38
    assert l1b_hist["spin_cycle"][0, 0] == 17  # 10 + 7 + 0
    assert l1b_hist["spin_cycle"][1, 0] == 17  # 10 + 7 + 0 (same spin)
    assert l1b_hist["spin_cycle"][2, 0] == 45  # 38 + 7 + 0
    assert l1b_hist["spin_cycle"][3, 0] == 45  # 38 + 7 + 0 (same spin)


def test_set_spin_cycle_from_spin_data_all_esa_steps():
    """Test that all ESA steps get correct spin cycles."""
    # Arrange
    epoch_date = et_to_ttj2000ns(str_to_et(["2025-04-15T02:00:00"]))
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (("epoch", "esa_step"), np.zeros((1, 7))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )

    l1b_hist = xr.Dataset(coords={"epoch": epoch_date, "esa_step": np.arange(1, 8)})

    met_time = ttj2000ns_to_met(epoch_date)
    spin_data = xr.Dataset(
        {
            "shcoarse": ("epoch", [met_time[0]]),
            "num_completed": ("epoch", [28]),
            "acq_start_sec": ("epoch", [met_time[0]]),
            "acq_start_subsec": ("epoch", [0]),
            "acq_end_sec": ("epoch", [met_time[0]]),
            "acq_end_subsec": ("epoch", [0]),
        },
        coords={"epoch": epoch_date},
    )

    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=np.array([0])
    ):
        # Act
        l1b_hist = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    # Assert - Verify the formula: spin_start + 7 + (esa_step - 1) * 2
    expected = np.array([7, 9, 11, 13, 15, 17, 19])
    np.testing.assert_array_equal(l1b_hist["spin_cycle"].values[0], expected)


def test_set_spin_cycle_from_spin_data_insufficient_spins():
    """Test that ASCs with fewer than 28 spins are filtered out."""
    # Arrange - Mix of valid and invalid spin counts
    science_met = [100, 200, 300]
    spin_met = [50, 150, 250]

    epoch_date = met_to_ttj2000ns(science_met)
    l1a_hist = xr.Dataset(
        {
            "hydrogen": (["epoch", "esa_step", "azimuth"], np.ones((3, 7, 60))),
            "oxygen": (["epoch", "esa_step", "azimuth"], np.ones((3, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "azimuth": np.arange(60),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )

    l1b_hist = xr.Dataset(coords={"epoch": epoch_date, "esa_step": np.arange(1, 8)})

    # Spin data with mixed valid/invalid counts
    spin_data = xr.Dataset(
        {
            "shcoarse": ("epoch", spin_met),
            "num_completed": ("epoch", [20, 28, 15]),  # 20 and 15 are < 28
            "acq_start_sec": ("epoch", np.array([50, 150, 250])),
            "acq_start_subsec": ("epoch", np.zeros(3)),
            "acq_end_sec": ("epoch", np.array([78, 178, 278])),
            "acq_end_subsec": ("epoch", np.zeros(3)),
        },
        coords={"epoch": np.arange(3)},
    )

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_spin_number",
        return_value=np.array([28, 26, 24]),
    ):
        result = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    assert len(result["epoch"]) == 3
    np.testing.assert_array_equal(result["epoch"].values, epoch_date)

    # Verify spin_cycle shape has all valid ESA steps and all epochs
    assert result["spin_cycle"].shape == (3, 7)
    # We should have added a flag about an incomplete ASC
    np.testing.assert_array_equal(result["incomplete_asc"], [True, False, True])


@patch(
    "imap_processing.lo.l1b.lo_l1b.get_pointing_times",
    return_value=(473389199, 473472001),
)
@patch(
    "imap_processing.lo.l1b.lo_l1b._get_esa_level_indices",
    return_value=np.arange(7),
)
def test_calculate_de_rates(
    mock_get_esa_level_indices, mock_get_pointing_times, attr_mgr_l1b, anc_dependencies
):
    """Test the calculate_de_rates function."""
    # Use MET times from the test sweep table (2025-01-01)
    met_start = 473389200
    epoch_time = met_to_ttj2000ns([met_start, met_start + 15 * 28])

    # Create individual epochs for each direct event in TTJ2000ns
    de_epochs = met_to_ttj2000ns(
        [
            met_start + 10,
            met_start + 20,
            met_start + 30,
            met_start + 15 * 28 + 10,
            met_start + 15 * 28 + 20,
        ]
    )

    # Create a simple l1b_de dataset with a few direct events
    l1b_de = xr.Dataset(
        {
            "spin_cycle": ("epoch", [7, 9, 11, 35, 37]),
            "esa_step": ("epoch", [1, 2, 3, 1, 2]),
            "spin_bin": ("epoch", [0, 120, 240, 60, 180]),
            "species": ("epoch", ["H", "O", "H", "H", "O"]),
            "coincidence_type": (
                "epoch",
                ["111111", "110100", "111000", "101000", "100100"],
            ),
            "avg_spin_durations": ("epoch", [15.0, 15.0, 15.0, 15.0, 15.0]),
        },
        coords={"epoch": de_epochs},
    )

    # Create l1a_spin dataset
    l1a_spin = xr.Dataset(
        {
            "shcoarse": ("epoch", [met_start, met_start + 15 * 28]),
            "num_completed": ("epoch", [28, 28]),
            "acq_start_sec": ("epoch", [met_start, met_start + 15 * 28]),
            "acq_start_subsec": ("epoch", [0, 0]),
            "acq_end_sec": ("epoch", [met_start + 15 * 28, met_start + 2 * 15 * 28]),
            "acq_end_subsec": ("epoch", [0, 0]),
        },
        coords={"epoch": epoch_time},
    )

    # Create l1b_nhk dataset with pivot angle information
    l1b_nhk = xr.Dataset(
        {"pcc_cumulative_cnt_pri": ("epoch", [45.0])},
        coords={"epoch": epoch_time[:1]},
    )

    # Add pivot_angle to l1b_de (normally set from l1b_nhk in l1b_de function)
    l1b_de["pivot_angle"] = xr.DataArray([45.0], dims=["pivot_angle"])

    sci_dependencies = {
        "imap_lo_l1b_de": l1b_de,
        "imap_lo_l1a_spin": l1a_spin,
        "imap_lo_l1b_nhk": l1b_nhk,
    }

    result = calculate_de_rates(sci_dependencies, anc_dependencies, attr_mgr_l1b)

    # Test that result can be written to CDF - this verifies that
    # attributes are ok with cdflib
    _ = write_cdf(result)

    assert result.attrs["Logical_source"] == "imap_lo_l1b_derates"
    assert "epoch" in result.coords
    assert "esa_step" in result.coords
    assert "spin_bin" in result.coords

    # Check that all expected data variables are present
    expected_vars = [
        "h_counts",
        "o_counts",
        "triple_counts",
        "double_counts",
        "h_rates",
        "o_rates",
        "triple_rates",
        "double_rates",
        "exposure_time",
        "spin_cycle",
        "esa_mode",
    ]
    for var in expected_vars:
        assert var in result.data_vars

    # Check shapes
    assert result["h_counts"].shape == (2, 7, 60)  # (num_asc, num_esa_steps, num_bins)
    assert result["o_counts"].shape == (2, 7, 60)
    assert result["exposure_time"].shape == (2, 7)

    # Verify some counts are correct based on our test data
    # First ASC (spin_cycle 0) has 3 events at esa_step 1, 2, 3
    # Second ASC (spin_cycle 28) has 2 events at esa_step 1, 2
    # H species: indices 0, 2, 3
    # ASC 0 has 2 H (esa_step 1, 3), ASC 1 has 1 H (esa_step 1)
    # O species: indices 1, 4
    # ASC 0 has 1 O (esa_step 2), ASC 1 has 1 O (esa_step 2)
    # First ASC, esa_step 1, spin_bin 0
    assert result["h_counts"][0, 0, 0] == 1
    # First ASC, esa_step 3, spin_bin 4 (240//60)
    assert result["h_counts"][0, 2, 4] == 1
    # First ASC, esa_step 2, spin_bin 2 (120//60)
    assert result["o_counts"][0, 1, 2] == 1

    # Check that pivot angle was set
    assert result["pivot_angle"].values[0] == 45.0

    # Test that lo_l1b() with descriptor="derates" produces the correct output
    output_datasets = lo_l1b(sci_dependencies, anc_dependencies, descriptor="derates")
    assert len(output_datasets) == 1
    assert output_datasets[0].attrs["Logical_source"] == "imap_lo_l1b_derates"


# ============================================================================
# Star Sensor L1B Tests
# ============================================================================
class TestGetSamplingCadenceFromNhk:
    """Tests for get_sampling_cadence_from_nhk function."""

    def test_extracts_mean_cadence(self):
        """Test extracting sampling cadence from NHK dataset."""
        # Arrange
        l1b_nhk = xr.Dataset(
            {
                "ifb_data_interval": ("epoch", [20.0, 20.5, 21.0]),
            },
            coords={"epoch": [0, 1, 2]},
        )
        expected_cadence = 20.5  # Mean of [20.0, 20.5, 21.0]

        # Act
        sampling_cadence = get_sampling_cadence_from_nhk(l1b_nhk)

        # Assert
        assert sampling_cadence == expected_cadence

    def test_raises_error_when_field_missing(self):
        """Test error when ifb_data_interval field is missing."""
        # Arrange
        l1b_nhk = xr.Dataset(
            {
                "other_field": ("epoch", [1, 2, 3]),
            },
            coords={"epoch": [0, 1, 2]},
        )

        # Act / Assert
        with pytest.raises(
            KeyError,
            match="ifb_data_interval field not found in L1B NHK dataset",
        ):
            get_sampling_cadence_from_nhk(l1b_nhk)


class TestFilterValidStarRecords:
    """Tests for filter_valid_star_records function."""

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_filters_by_count_threshold(self, mock_repoint):
        """Test filtering star records by COUNT >= 700."""
        # Arrange - Mock repoint data (no repoints in progress)
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, False, False, False, False]}
        )

        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [650, 700, 720, 699, 715]),
                "shcoarse": (
                    "epoch",
                    np.arange(5, dtype=np.float64),
                ),  # Already in seconds
            },
            coords={"epoch": [0, 1, 2, 3, 4]},
        )
        expected_mask = np.array([False, True, True, False, True])

        # Act
        valid_mask = filter_valid_star_records(l1a_star, min_count=700)

        # Assert
        np.testing.assert_array_equal(valid_mask, expected_mask)

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_filters_by_count_and_time_window(self, mock_repoint):
        """Test filtering star records by both COUNT and time window."""
        # Arrange - Mock repoint data (no repoints in progress)
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, False, False, False, False]}
        )

        # Create times: 0s, 10s, 20s, 30s, 40s (already in seconds)
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [700, 710, 720, 715, 720]),
                "shcoarse": ("epoch", np.array([0, 10, 20, 30, 40], dtype=np.float64)),
            },
            coords={"epoch": [0, 1, 2, 3, 4]},
        )
        # # Time window: [5s, 25s] - should include epochs 1 and 2
        expected_mask = np.array([False, True, True, False, False])

        # Act
        valid_mask = filter_valid_star_records(
            l1a_star,
            min_count=700,
            time_window_offset=5.0,
            time_window_duration=20.0,
        )

        # Assert
        np.testing.assert_array_equal(valid_mask, expected_mask)

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_processes_all_data_without_time_window(self, mock_repoint):
        """Test filtering without time window (process all data)."""
        # Arrange - Mock repoint data (no repoints in progress)
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, False, False]}
        )

        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [700, 710, 720]),
                "shcoarse": ("epoch", np.array([0, 10, 20], dtype=np.float64)),
            },
            coords={"epoch": [0, 1, 2]},
        )
        expected_mask = np.array([True, True, True])

        # Act
        valid_mask = filter_valid_star_records(
            l1a_star, min_count=700, time_window_duration=None
        )

        # Assert
        np.testing.assert_array_equal(valid_mask, expected_mask)

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_excludes_records_during_repoint(self, mock_repoint):
        """Test filtering records during repoint maneuvers."""
        # Arrange - Mock repoint data with some repoints in progress
        # Epochs 1 and 3 are during repoint maneuvers
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, True, False, True, False]}
        )

        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [700, 710, 720, 715, 720]),
                "shcoarse": ("epoch", np.arange(5, dtype=np.float64)),
            },
            coords={"epoch": [0, 1, 2, 3, 4]},
        )
        # Expected: epochs 0, 2, 4 pass (COUNT >= 700 AND not during repoint)
        # Epochs 1 and 3 fail because they are during repoint
        expected_mask = np.array([True, False, True, False, True])

        # Act
        valid_mask = filter_valid_star_records(l1a_star, min_count=700)

        # Assert
        np.testing.assert_array_equal(valid_mask, expected_mask)


class TestCalculateStarSensorProfile:
    """Tests for star sensor profile calculation functions."""

    def test_profile_for_group_basic(self):
        """Test basic star sensor profile calculation for a group."""
        # Arrange - 3 records with uniform data
        np.random.seed(42)
        data = np.random.randint(100, 200, size=(3, 720)).astype(np.uint16)
        counts = np.array([720, 720, 720])

        # Act
        avg_amplitude, count_per_bin = calculate_star_sensor_profile_for_group(
            data, counts, end_bins_to_exclude=0
        )

        # Assert
        assert len(avg_amplitude) == 720
        assert len(count_per_bin) == 720
        # All bins should have 3 samples
        np.testing.assert_array_equal(count_per_bin, np.full(720, 3))
        # Averages should be between 100 and 200
        assert np.all(avg_amplitude >= 100)
        assert np.all(avg_amplitude <= 200)

    def test_profile_for_group_end_bins_excluded(self):
        """Test that edge bins are properly excluded."""
        # Arrange - 2 records with uniform data
        data = np.ones((2, 720), dtype=np.uint16) * 100
        counts = np.array([720, 720])

        # Act
        avg_amplitude, count_per_bin = calculate_star_sensor_profile_for_group(
            data, counts, end_bins_to_exclude=2
        )

        # Assert
        # Last 2 bins should have count=0
        assert count_per_bin[718] == 0
        assert count_per_bin[719] == 0
        # All other bins should have count=2
        assert np.all(count_per_bin[:718] == 2)
        # Averages should be 100 for all bins except the excluded ones
        assert np.all(avg_amplitude[:718] == 100.0)
        # Excluded bins should be NaN
        assert np.isnan(avg_amplitude[718])
        assert np.isnan(avg_amplitude[719])

    def test_profile_for_group_empty_data(self):
        """Test handling of empty data array."""
        # Arrange
        data = np.empty((0, 720), dtype=np.uint16)
        counts = np.array([], dtype=np.int32)

        # Act
        avg_amplitude, count_per_bin = calculate_star_sensor_profile_for_group(
            data, counts
        )

        # Assert
        np.testing.assert_array_equal(count_per_bin, np.zeros(720))
        # Empty data returns NaN for all bins (consistent with bins having no samples)
        assert np.all(np.isnan(avg_amplitude))

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_profiles_by_group_creates_correct_groups(self, mock_repoint):
        """Test that profiles are grouped correctly into 64-record groups."""
        # Arrange - Create 150 records (should produce 3 groups: 64, 64, 22)
        n_records = 150
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False] * n_records}
        )
        met_times = np.arange(n_records, dtype=np.float64) * 15.0
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720] * n_records),
                "shcoarse": (
                    "epoch",
                    np.arange(n_records, dtype=np.float64) * 15.0,
                ),
                "data": (
                    ("epoch", "samples"),
                    np.ones((n_records, 720), dtype=np.uint16) * 100,
                ),
            },
            coords={
                "epoch": met_to_ttj2000ns(met_times),
                "samples": np.arange(720),
            },
        )

        # Act
        (
            spin_angle,
            group_epochs,
            avg_amplitudes,
            counts_per_bin,
        ) = calculate_star_sensor_profiles_by_group(
            l1a_star,
            sampling_cadence=21.0,
            spin_period=15.0,
            group_size=64,
        )

        # Assert
        assert len(spin_angle) == 720
        assert len(group_epochs) == 3  # 150 records -> 3 groups
        assert avg_amplitudes.shape == (3, 720)
        assert counts_per_bin.shape == (3, 720)
        # First two groups should have 64 samples per bin, last group 22
        assert np.all(counts_per_bin[0, 2:718] == 64)
        assert np.all(counts_per_bin[1, 2:718] == 64)
        assert np.all(counts_per_bin[2, 2:718] == 22)

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_profiles_by_group_handles_no_valid_records(self, mock_repoint):
        """Test handling when no records pass the COUNT threshold."""
        # Arrange
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, False, False]}
        )
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [650, 600, 699]),  # All below 700
                "shcoarse": ("epoch", np.array([0.0, 15.0, 30.0], dtype=np.float64)),
                "data": (
                    ("epoch", "samples"),
                    np.ones((3, 720), dtype=np.uint16) * 100,
                ),
            },
            coords={
                "epoch": met_to_ttj2000ns([0.0, 15.0, 30.0]),
                "samples": np.arange(720),
            },
        )

        # Act
        (
            spin_angle,
            group_epochs,
            avg_amplitudes,
            counts_per_bin,
        ) = calculate_star_sensor_profiles_by_group(
            l1a_star,
            sampling_cadence=21.0,
            spin_period=15.0,
            min_count_threshold=700,
        )

        # Assert
        assert len(spin_angle) == 720
        assert len(group_epochs) == 0  # No valid records
        assert avg_amplitudes.shape == (0, 720)

    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_profiles_by_group_angle_wrapping(self, mock_repoint):
        """Test that spin angles wrap correctly to [0, 360) range."""
        # Arrange
        mock_repoint.return_value = pd.DataFrame({"repoint_in_progress": [False]})
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720]),
                "shcoarse": ("epoch", np.array([0.0], dtype=np.float64)),
                "data": (
                    ("epoch", "samples"),
                    np.ones((1, 720), dtype=np.uint16) * 100,
                ),
            },
            coords={"epoch": met_to_ttj2000ns([0.0]), "samples": np.arange(720)},
        )

        # Act
        spin_angle, _, _, _ = calculate_star_sensor_profiles_by_group(
            l1a_star,
            sampling_cadence=21.0,
            spin_period=15.0,
            start_angle_offset=350.0,  # Large offset to test wrapping
        )

        # Assert
        assert np.all(spin_angle >= 0)
        assert np.all(spin_angle < 360)
        # With offset=350°, first bin should be around 350°
        assert 350.0 < spin_angle[0] < 351.0
        # Some bins will wrap to the lower range
        assert np.any(spin_angle > 300)
        assert np.any(spin_angle < 100)  # Some angles wrapped to lower range


class TestL1bStar:
    """Tests for l1b_star function."""

    @patch("imap_processing.lo.l1b.lo_l1b.get_pointing_mid_time")
    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_initializes_with_spin_data(
        self, mock_repoint, mock_pointing_mid, attr_mgr_l1b
    ):
        """Test successful initialization of L1B star dataset with spin data."""
        # Arrange - Create 150 records to produce multiple groups
        n_records = 150
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False] * n_records}
        )
        mock_pointing_mid.return_value = 1000.0  # Mock pointing mid time in MET
        np.random.seed(42)
        met_times = np.arange(n_records, dtype=np.float64) * 15.0
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720] * n_records),
                "shcoarse": (
                    "epoch",
                    np.arange(n_records, dtype=np.float64) * 15.0,
                ),
                "data": (
                    ("epoch", "samples"),
                    np.random.randint(100, 200, size=(n_records, 720), dtype=np.uint16),
                ),
            },
            coords={
                "epoch": met_to_ttj2000ns(met_times),
                "samples": np.arange(720),
            },
        )
        l1b_nhk = xr.Dataset(
            {
                "ifb_data_interval": ("epoch", [21.0] * n_records),
            },
            coords={"epoch": list(range(n_records))},
        )
        # Create spin data with known spin durations
        spin_data = xr.Dataset(
            {
                "acq_start_sec": ("epoch", [0, 15]),
                "acq_start_subsec": ("epoch", [0, 0]),
                "acq_end_sec": ("epoch", [420, 435]),  # 420s = 28 spins * 15s
                "acq_end_subsec": ("epoch", [0, 0]),
                "num_completed": ("epoch", [28, 28]),
            },
            coords={"epoch": [0, 1]},
        )
        sci_dependencies = {
            "imap_lo_l1a_star": l1a_star,
            "imap_lo_l1b_nhk": l1b_nhk,
            "imap_lo_l1a_spin": spin_data,
        }

        # Act
        l1b_star_ds = l1b_star(sci_dependencies, attr_mgr_l1b, group_size=64)

        # Assert
        assert l1b_star_ds.attrs["Logical_source"] == "imap_lo_l1b_prostar"
        assert "epoch" in l1b_star_ds.coords
        # 150 records / 64 group_size = 3 groups (64 + 64 + 22)
        assert len(l1b_star_ds.coords["epoch"]) == 3
        # spin_angle is now the coordinate (monotonically increasing)
        assert "spin_angle" in l1b_star_ds.coords
        assert len(l1b_star_ds.coords["spin_angle"]) == 720
        # spin_angle_bin is now a data variable
        assert "spin_angle_bin" in l1b_star_ds.data_vars
        assert "avg_amplitude" in l1b_star_ds.data_vars
        assert "count_per_bin" in l1b_star_ds.data_vars
        assert "pointing_mid_met" in l1b_star_ds.attrs
        # Check that spin_angle is monotonically increasing
        spin_angles = l1b_star_ds.coords["spin_angle"].values
        assert np.all(np.diff(spin_angles) > 0), (
            "spin_angle should be monotonically increasing"
        )
        assert spin_angles[0] >= 0.0
        assert spin_angles[-1] < 360.0
        # Check attributes
        assert "sampling_cadence_ms" in l1b_star_ds.attrs
        assert "spin_duration_sec" in l1b_star_ds.attrs
        assert "group_size" in l1b_star_ds.attrs
        assert l1b_star_ds.attrs["sampling_cadence_ms"] == 21.0
        assert l1b_star_ds.attrs["spin_duration_sec"] == 15.0
        assert l1b_star_ds.attrs["group_size"] == 64
        # Check data shapes - all variables have epoch as first dimension
        assert l1b_star_ds["spin_angle_bin"].shape == (720,)
        assert l1b_star_ds["avg_amplitude"].shape == (3, 720)
        assert l1b_star_ds["count_per_bin"].shape == (3, 720)
        # Check pointing_mid_met is a scalar with expected value
        assert float(l1b_star_ds.attrs["pointing_mid_met"]) == 1000.0

    @patch("imap_processing.lo.l1b.lo_l1b.get_pointing_mid_time")
    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_dataset_structure_and_attributes(
        self, mock_repoint, mock_pointing_mid, attr_mgr_l1b
    ):
        """Test that L1B star dataset has correct structure and attributes."""
        # Arrange
        mock_repoint.return_value = pd.DataFrame({"repoint_in_progress": [False]})
        mock_pointing_mid.return_value = 1000.0
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720]),
                "shcoarse": ("epoch", np.array([0.0], dtype=np.float64)),
                "data": (
                    ("epoch", "samples"),
                    np.ones((1, 720), dtype=np.uint16) * 150,
                ),
            },
            coords={"epoch": met_to_ttj2000ns([0.0]), "samples": np.arange(720)},
        )
        l1b_nhk = xr.Dataset(
            {
                "ifb_data_interval": ("epoch", [21.0]),
            },
            coords={"epoch": [0]},
        )
        spin_data = xr.Dataset(
            {
                "acq_start_sec": ("epoch", [0]),
                "acq_start_subsec": ("epoch", [0]),
                "acq_end_sec": ("epoch", [420]),  # 420s = 28 spins * 15s
                "acq_end_subsec": ("epoch", [0]),
                "num_completed": ("epoch", [28]),
            },
            coords={"epoch": [0]},
        )
        sci_dependencies = {
            "imap_lo_l1a_star": l1a_star,
            "imap_lo_l1b_nhk": l1b_nhk,
            "imap_lo_l1a_spin": spin_data,
        }

        # Act
        l1b_star_ds = l1b_star(sci_dependencies, attr_mgr_l1b)

        # Assert - Check spin_angle coordinate attributes
        # Check that resulting dataset is cdf-able by writing to file
        _ = write_cdf(l1b_star_ds)

        assert l1b_star_ds.coords["spin_angle"].attrs["UNITS"] == "deg"
        assert l1b_star_ds.coords["spin_angle"].attrs["VALIDMIN"] == 0.0
        assert l1b_star_ds.coords["spin_angle"].attrs["VALIDMAX"] == 360.0

        # Assert - Check spin_angle_bin variable attributes (now a data variable)
        assert (
            "Original spin angle bin index"
            in l1b_star_ds["spin_angle_bin"].attrs["CATDESC"]
        )
        assert l1b_star_ds["spin_angle_bin"].attrs["VALIDMIN"] == 0
        assert l1b_star_ds["spin_angle_bin"].attrs["VALIDMAX"] == 719

        assert l1b_star_ds["avg_amplitude"].attrs["UNITS"] == "mV"
        assert l1b_star_ds["avg_amplitude"].attrs["FILLVAL"] == -1.0e31

        assert l1b_star_ds["count_per_bin"].attrs["VALIDMIN"] == 0
        assert l1b_star_ds["count_per_bin"].attrs["VALIDMAX"] == 100000

        # Assert - Check processing parameter attributes
        assert "lo_angle_offset_deg" in l1b_star_ds.attrs
        assert "end_bins_excluded" in l1b_star_ds.attrs
        assert "min_count_threshold" in l1b_star_ds.attrs
        assert l1b_star_ds.attrs["lo_angle_offset_deg"] == 2.0
        assert l1b_star_ds.attrs["end_bins_excluded"] == 2
        assert l1b_star_ds.attrs["min_count_threshold"] == 700

    @patch("imap_processing.lo.l1b.lo_l1b.get_pointing_mid_time")
    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_start_and_end_doy_variables(
        self, mock_repoint, mock_pointing_mid, attr_mgr_l1b
    ):
        """Test that start_doy and end_doy variables are computed correctly."""
        # Arrange
        mock_pointing_mid.return_value = 1000.0  # Mock pointing mid time in MET
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False, False, False]}
        )
        np.random.seed(42)
        # Create epochs spanning 30 seconds
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720, 720, 720]),
                "shcoarse": ("epoch", np.array([0.0, 15.0, 30.0], dtype=np.float64)),
                "data": (
                    ("epoch", "samples"),
                    np.random.randint(100, 200, size=(3, 720), dtype=np.uint16),
                ),
            },
            coords={
                "epoch": met_to_ttj2000ns([0.0, 15.0, 30.0]),
                "samples": np.arange(720),
            },
        )
        l1b_nhk = xr.Dataset(
            {
                "ifb_data_interval": ("epoch", [21.0, 21.0, 21.0]),
            },
            coords={"epoch": [0, 1, 2]},
        )
        spin_data = xr.Dataset(
            {
                "acq_start_sec": ("epoch", [0, 15]),
                "acq_start_subsec": ("epoch", [0, 0]),
                "acq_end_sec": ("epoch", [420, 435]),
                "acq_end_subsec": ("epoch", [0, 0]),
                "num_completed": ("epoch", [28, 28]),
            },
            coords={"epoch": [0, 1]},
        )
        sci_dependencies = {
            "imap_lo_l1a_star": l1a_star,
            "imap_lo_l1b_nhk": l1b_nhk,
            "imap_lo_l1a_spin": spin_data,
        }

        # Act
        l1b_star_ds = l1b_star(sci_dependencies, attr_mgr_l1b)

        # Assert - Check that start_doy and end_doy exist as scalars (global values)
        assert "start_doy" in l1b_star_ds.attrs
        assert "end_doy" in l1b_star_ds.attrs

        # Assert - Check values are valid day of year (1.0 to 366.x for leap years)
        start_doy = float(l1b_star_ds.attrs["start_doy"])
        end_doy = float(l1b_star_ds.attrs["end_doy"])
        assert 1.0 <= start_doy <= 367.0
        assert 1.0 <= end_doy <= 367.0

        # Assert - end_doy should be >= start_doy (data spans 30 seconds)
        assert end_doy >= start_doy

    @patch("imap_processing.lo.l1b.lo_l1b.get_pointing_mid_time")
    @patch("imap_processing.lo.l1b.lo_l1b.interpolate_repoint_data")
    def test_multiple_groups_created(
        self, mock_repoint, mock_pointing_mid, attr_mgr_l1b
    ):
        """Test that multiple 64-spin groups are created correctly."""
        # Arrange - Create 150 records to produce 3 groups (64 + 64 + 22)
        n_records = 150
        mock_pointing_mid.return_value = 1000.0  # Mock pointing mid time in MET
        mock_repoint.return_value = pd.DataFrame(
            {"repoint_in_progress": [False] * n_records}
        )
        met_times = np.arange(n_records, dtype=np.float64) * 15.0
        l1a_star = xr.Dataset(
            {
                "count": ("epoch", [720] * n_records),
                "shcoarse": (
                    "epoch",
                    np.arange(n_records, dtype=np.float64) * 15.0,
                ),
                "data": (
                    ("epoch", "samples"),
                    np.ones((n_records, 720), dtype=np.uint16) * 100,
                ),
            },
            coords={
                "epoch": met_to_ttj2000ns(met_times),
                "samples": np.arange(720),
            },
        )
        l1b_nhk = xr.Dataset(
            {
                "ifb_data_interval": ("epoch", [21.0] * n_records),
            },
            coords={"epoch": list(range(n_records))},
        )
        spin_data = xr.Dataset(
            {
                "acq_start_sec": ("epoch", [0]),
                "acq_start_subsec": ("epoch", [0]),
                "acq_end_sec": ("epoch", [420]),
                "acq_end_subsec": ("epoch", [0]),
                "num_completed": ("epoch", [28]),
            },
            coords={"epoch": [0]},
        )
        sci_dependencies = {
            "imap_lo_l1a_star": l1a_star,
            "imap_lo_l1b_nhk": l1b_nhk,
            "imap_lo_l1a_spin": spin_data,
        }

        # Act
        l1b_star_ds = l1b_star(sci_dependencies, attr_mgr_l1b, group_size=64)

        # Assert
        assert len(l1b_star_ds.coords["epoch"]) == 3
        # Check pointing_mid_met is present (scalar value)
        assert "pointing_mid_met" in l1b_star_ds.attrs
        # First group epoch should be the first L1A epoch
        assert l1b_star_ds.coords["epoch"].values[0] == met_to_ttj2000ns([0.0])[0]
        # Second group epoch should be record 64
        assert l1b_star_ds.coords["epoch"].values[1] == met_to_ttj2000ns([64 * 15.0])[0]
        # Third group epoch should be record 128
        assert (
            l1b_star_ds.coords["epoch"].values[2] == met_to_ttj2000ns([128 * 15.0])[0]
        )


def test_get_pivot_angle_from_nhk():
    """Test get_pivot_angle_from_nhk function."""
    # Arrange - Create a mock NHK dataset with pivot angle information
    l1b_nhk = xr.Dataset(
        {
            # Previous 90 degrees at the beginning, then shifted to 75 degrees
            "pcc_cumulative_cnt_pri": ("epoch", [90, 90, 75, 75, 75, 75, 75]),
        },
        coords={"epoch": [0, 1, 2, 3, 4, 5, 6]},
    )
    expected_pivot_angle = 75

    # Act
    pivot_angle = get_pivot_angle_from_nhk(l1b_nhk)

    # Assert
    assert pivot_angle == expected_pivot_angle


def test_l1b_bgrates_and_goodtimes_basic(anc_dependencies, attr_mgr_l1b):
    """Test basic functionality of l1b_bgrates_and_goodtimes."""
    # Arrange - Create a simple L1B histogram rates dataset
    # with enough data points to create goodtime intervals
    num_epochs = 100
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Low counts to keep background rates below threshold
    h_counts = np.ones((num_epochs, 7, 60)) * 0.00028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert - Should return a list with two datasets
    assert isinstance(result, list)
    assert len(result) == 2

    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Check that bgrates dataset is cdf-able by writing to file
    _ = write_cdf(l1b_bgrates_ds)
    assert "epoch" in l1b_bgrates_ds.coords
    assert "esa_step" in l1b_bgrates_ds.coords

    # Check bgrates dataset structure (BACKGROUND_RATE_FIELDS)
    assert "h_background_rates" in l1b_bgrates_ds.data_vars
    assert "h_background_variance" in l1b_bgrates_ds.data_vars
    assert "h_synthetic_floor" in l1b_bgrates_ds.data_vars
    assert "h_proxy_floor" in l1b_bgrates_ds.data_vars
    assert "o_background_rates" in l1b_bgrates_ds.data_vars
    assert "o_background_variance" in l1b_bgrates_ds.data_vars
    assert "o_synthetic_floor" in l1b_bgrates_ds.data_vars
    assert "o_proxy_floor" in l1b_bgrates_ds.data_vars

    # Check that goodtimes dataset is cdf-able by writing to file
    _ = write_cdf(l1b_goodtimes_ds)
    assert "epoch" in l1b_goodtimes_ds.coords

    # Check goodtimes dataset structure (GOODTIMES_FIELDS)
    assert "gt_start_met" in l1b_goodtimes_ds.data_vars
    assert "gt_end_met" in l1b_goodtimes_ds.data_vars
    assert "pivot" in l1b_goodtimes_ds.data_vars
    assert "pivot_de" in l1b_goodtimes_ds.data_vars

    # Check that goodtime intervals were created
    assert len(l1b_goodtimes_ds["gt_start_met"]) > 0
    assert len(l1b_goodtimes_ds["gt_end_met"]) > 0

    # Check that start times are before end times
    assert np.all(
        l1b_goodtimes_ds["gt_start_met"].values <= l1b_goodtimes_ds["gt_end_met"].values
    )


def test_l1b_bgrates_and_goodtimes_with_gap(anc_dependencies, attr_mgr_l1b):
    """Test l1b_bgrates_and_goodtimes handles data gaps correctly."""
    # Arrange - Create dataset with a large gap in the middle
    num_epochs_first = 50
    num_epochs_second = 50
    met_start = 473389200
    met_spacing = 42
    gap_size = 10000  # Large gap (> delay_max + interval_nom)

    # First segment
    met_times_first = np.arange(
        met_start, met_start + num_epochs_first * met_spacing, met_spacing
    )
    # Second segment after gap
    met_times_second = np.arange(
        met_start + num_epochs_first * met_spacing + gap_size,
        met_start
        + num_epochs_first * met_spacing
        + gap_size
        + num_epochs_second * met_spacing,
        met_spacing,
    )

    met_times = np.concatenate([met_times_first, met_times_second])
    epoch_times = met_to_ttj2000ns(met_times)

    # Low background counts (below threshold)
    h_counts = np.ones((len(met_times), 7, 60)) * 0.00028
    o_counts = np.ones((len(met_times), 7, 60)) * 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Should create at least 2 separate goodtime intervals (before and after gap)
    assert len(l1b_goodtimes_ds["gt_start_met"]) >= 2

    # Check that intervals don't span across the gap
    for i in range(len(l1b_goodtimes_ds["gt_start_met"])):
        interval_duration = (
            l1b_goodtimes_ds["gt_end_met"].values[i]
            - l1b_goodtimes_ds["gt_start_met"].values[i]
        )
        # No interval should be as large as the gap
        assert interval_duration < gap_size


def test_l1b_bgrates_and_goodtimes_high_rate(anc_dependencies, attr_mgr_l1b):
    """Test l1b_bgrates_and_goodtimes handles high count rates correctly."""
    # Arrange - Create dataset with high rates that exceed threshold
    num_epochs = 100
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Create high counts (above threshold)
    # h_bg_rate_nom = 0.0014925, exposure = 420*7*0.5 = 1470 seconds
    # To be above threshold: rate > 0.0014925
    # Use 10x threshold for high rate periods: 0.014925 counts/sec
    h_counts = np.ones((num_epochs, 7, 60)) * 0.014925  # High rate (10x threshold)
    o_counts = np.ones((num_epochs, 7, 60)) * 0.0014925

    # Make first 20 epochs low (below threshold)
    h_counts[:20, :, :] = 0.00014925
    o_counts[:20, :, :] = 0.000014925

    # Make last 20 epochs low
    h_counts[80:, :, :] = 0.00014925
    o_counts[80:, :, :] = 0.000014925

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Should create at least 2 intervals (before and after high rate period)
    assert len(l1b_goodtimes_ds["gt_start_met"]) >= 2

    # Check that background rates were calculated
    assert np.all(l1b_bgrates_ds["h_background_rates"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_rates"].values > 0)


def test_l1b_bgrates_and_goodtimes_no_goodtimes(anc_dependencies, attr_mgr_l1b):
    """When no goodtimes are detected the function should still return datasets."""
    num_epochs = 50
    met_start = 473389200
    met_spacing = 42
    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Make counts high everywhere so no low-rate goodtime intervals are found
    h_counts = np.ones((num_epochs, 7, 60)) * 0.1
    o_counts = np.ones((num_epochs, 7, 60)) * 0.01

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        _, goodtimes_ds = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # When no goodtimes are detected a single fallback row (0, 0) is used.
    # The padding loop runs before the fallback is inserted, so the zeros are unchanged.
    assert int(goodtimes_ds["gt_start_met"].values[0]) == 0
    assert int(goodtimes_ds["gt_end_met"].values[0]) == 0


def test_l1b_bgrates_and_goodtimes_empty_dataset(anc_dependencies, attr_mgr_l1b):
    """Test l1b_bgrates_and_goodtimes handles edge case with minimal data."""
    # Arrange - Create minimal dataset (just enough for one cycle)
    num_epochs = 10
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Low counts (below threshold)
    h_counts = np.ones((num_epochs, 7, 60)) * 0.00028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert - Should still create valid datasets even with minimal data
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    assert "h_background_rates" in l1b_bgrates_ds.data_vars
    assert "gt_start_met" in l1b_goodtimes_ds.data_vars


def test_split_backgrounds_and_goodtimes_dataset(attr_mgr_l1b):
    """Test split_backgrounds_and_goodtimes_dataset separates fields correctly."""
    # Arrange - Create a combined dataset matching the structure produced by
    # l1b_bgrates_and_goodtimes: 1-D (esa_step) background rate fields and
    # epoch-indexed goodtime interval fields.
    num_records = 3
    n_esa = 7
    met_starts = np.arange(473389200, 473389200 + num_records * 420, 420)
    epoch_times = met_to_ttj2000ns(met_starts)
    pointing_start_epoch = met_to_ttj2000ns(met_starts[0]).item()

    combined_ds = xr.Dataset(
        coords={"epoch": epoch_times},
    )
    combined_ds["gt_start_met"] = xr.DataArray(
        met_starts.astype(np.int64), dims=["epoch"]
    )
    combined_ds["gt_end_met"] = xr.DataArray(
        (met_starts + 400).astype(np.int64), dims=["epoch"]
    )
    combined_ds["pivot"] = xr.DataArray(np.float32(90.0))
    combined_ds["pivot_de"] = xr.DataArray(np.float32(89.5))
    combined_ds["h_background_rates"] = xr.DataArray(
        np.full(n_esa, np.float32(0.01)), dims=["esa_step"]
    )
    combined_ds["h_background_variance"] = xr.DataArray(
        np.full(n_esa, np.float32(0.001)), dims=["esa_step"]
    )
    combined_ds["o_background_rates"] = xr.DataArray(
        np.full(n_esa, np.float32(0.002)), dims=["esa_step"]
    )
    combined_ds["o_background_variance"] = xr.DataArray(
        np.full(n_esa, np.float32(0.0002)), dims=["esa_step"]
    )
    combined_ds["h_synthetic_floor"] = xr.DataArray(np.float32(5.0))
    combined_ds["h_proxy_floor"] = xr.DataArray(np.float32(4.0))
    combined_ds["o_synthetic_floor"] = xr.DataArray(np.float32(0.5))
    combined_ds["o_proxy_floor"] = xr.DataArray(np.float32(0.4))

    # Act
    bgrates_ds, goodtimes_ds = split_backgrounds_and_goodtimes_dataset(
        combined_ds, attr_mgr_l1b, pointing_start_epoch
    )

    # Assert - _background_rates/_background_variance are (epoch, esa_step)
    for field in [
        "h_background_rates",
        "h_background_variance",
        "o_background_rates",
        "o_background_variance",
    ]:
        assert field in bgrates_ds.data_vars
        assert bgrates_ds[field].dims == ("epoch", "esa_step")
        assert bgrates_ds[field].shape == (1, n_esa)

    for field in [
        "h_synthetic_floor",
        "h_proxy_floor",
        "o_synthetic_floor",
        "o_proxy_floor",
    ]:
        assert field in bgrates_ds.data_vars
        assert bgrates_ds[field].dims == ("epoch",)
        assert bgrates_ds[field].shape == (1,)

    assert len(bgrates_ds["epoch"]) == 1
    np.testing.assert_array_equal(bgrates_ds["epoch"].values, pointing_start_epoch)

    # Assert - goodtimes dataset contains the expected fields
    assert "gt_start_met" in goodtimes_ds.data_vars
    assert "gt_end_met" in goodtimes_ds.data_vars
    assert "pivot" in goodtimes_ds.data_vars
    assert "pivot_de" in goodtimes_ds.data_vars

    # Assert - goodtime intervals were created and are valid
    assert len(goodtimes_ds["gt_start_met"]) > 0
    assert np.all(
        goodtimes_ds["gt_start_met"].values <= goodtimes_ds["gt_end_met"].values
    )

    # Assert - pivot fields are scalar
    assert goodtimes_ds["pivot"].dims == ()
    assert goodtimes_ds["pivot_de"].dims == ()


def test_l1b_bgrates_and_goodtimes_ram_and_anti_ram_bins(
    anc_dependencies, attr_mgr_l1b
):
    """Test that the function correctly uses bins anti-RAM 20-50 and RAM 0-20/50-60."""
    # Arrange - Create dataset with specific counts in different azimuth bins
    num_epochs = 30
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # High counts everywhere by default
    h_counts = np.ones((num_epochs, 7, 60)) * 0.028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.0028

    # Anti-RAM bins (20-50): set low across all ESA steps (below anti-RAM threshold)
    h_counts[:, :, 20:50] = 0.00028
    o_counts[:, :, 20:50] = 0.000028

    # RAM bins (0-20, 50-60) for RAM ESA steps (0-indexed 5, 6):
    # set low (below RAM threshold)
    h_counts[:, 5:7, 0:20] = 0.00028
    h_counts[:, 5:7, 50:60] = 0.00028
    o_counts[:, 5:7, 0:20] = 0.000028
    o_counts[:, 5:7, 50:60] = 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    # Required dependencies added in the updated function signature
    cdf_de = xr.Dataset({"pivot_angle": xr.DataArray(90.0)})
    cdf_hk = xr.Dataset()  # No pcc_coarse_pot_pri; pivot defaults to 90.0

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": cdf_de,
        "imap_lo_l1b_nhk": cdf_hk,
    }

    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Should create goodtime intervals because RAM and anti-RAM bins have low counts
    assert len(l1b_goodtimes_ds["gt_start_met"]) > 0
    # h_synthetic_floor accumulates the modeled background during good times
    assert l1b_bgrates_ds["h_synthetic_floor"].values > 0


def test_l1b_bgrates_and_goodtimes_variance_calculation(anc_dependencies, attr_mgr_l1b):
    """Test that variance is calculated correctly and handles edge cases."""
    # Arrange
    num_epochs = 30
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Use very low counts to test zero variance handling
    h_counts = np.zeros((num_epochs, 7, 60))
    o_counts = np.zeros((num_epochs, 7, 60))

    # Add some small counts (below threshold)
    h_counts[:, :, 20:50] = 0.00001  # Very low but non-zero
    o_counts[:, :, 20:50] = 0.000001

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Variance should never be zero (fallback logic should apply)
    assert np.all(l1b_bgrates_ds["h_background_variance"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_variance"].values > 0)

    # Background rates should also never be zero (fallback logic should apply)
    assert np.all(l1b_bgrates_ds["h_background_rates"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_rates"].values > 0)


def test_l1b_bgrates_and_goodtimes_offset_application(anc_dependencies, attr_mgr_l1b):
    """Test that padding is applied to goodtime intervals."""
    # Arrange
    num_epochs = 30
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Low counts (below threshold)
    h_counts = np.ones((num_epochs, 7, 60)) * 0.00028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # All epochs are below threshold, so one goodtime interval spanning the full
    # dataset is expected. GOODTIME_PADDING is subtracted from begin and added to end.
    assert len(l1b_goodtimes_ds["gt_start_met"]) == 1

    raw_begin = met_times[0]
    raw_end = met_times[-1]

    assert l1b_goodtimes_ds["gt_start_met"].values[0] <= raw_begin
    assert l1b_goodtimes_ds["gt_end_met"].values[0] >= raw_end


def test_l1b_bgrates_and_goodtimes_rate_transition_low_to_high(
    anc_dependencies, attr_mgr_l1b
):
    """Test interval closure when transitioning from low to high rate
    (covers begin > 0.0 block)."""
    # Arrange - Create dataset that transitions from LOW to HIGH rates
    # This specifically tests the "if begin > 0.0:" code path at line ~2787
    num_epochs = 50  # Need at least 5 cycles (50 epochs / 10 per cycle)
    met_start = 473389200
    met_spacing = 42

    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Start with LOW rates for first 30 epochs (3 cycles)
    # Then switch to HIGH rates for last 20 epochs (2 cycles)
    h_counts = np.ones((num_epochs, 7, 60)) * 0.00028  # Low (below threshold)
    o_counts = np.ones((num_epochs, 7, 60)) * 0.000028

    # Make last 20 epochs HIGH (above threshold) to trigger interval closure
    h_counts[30:, :, :] = 0.028  # High (10x threshold)
    o_counts[30:, :, :] = 0.0028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Should create goodtime interval that gets closed when rate goes high
    # The interval should span the first 3 cycles (epochs 0-29)
    assert len(l1b_goodtimes_ds["gt_start_met"]) >= 1

    # First interval should start around epoch 0's time
    first_start = l1b_goodtimes_ds["gt_start_met"].values[0]
    first_end = l1b_goodtimes_ds["gt_end_met"].values[0]

    # Verify interval was created
    assert first_start < first_end

    # Background rates should be calculated from the low-rate period
    assert np.all(l1b_bgrates_ds["h_background_rates"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_rates"].values > 0)

    # Variance should also be positive
    assert np.all(l1b_bgrates_ds["h_background_variance"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_variance"].values > 0)


def test_l1b_bgrates_and_goodtimes_rate_transition_high_to_low_to_high(
    anc_dependencies, attr_mgr_l1b
):
    """Test multiple intervals created by multiple rate transitions."""
    # Arrange - Create dataset with HIGH -> LOW -> HIGH -> LOW pattern
    # This tests multiple calls to the "if begin > 0.0:" code path
    num_epochs = 80
    met_start = 473389200
    met_spacing = 42
    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Initialize with HIGH rates
    h_counts = np.ones((num_epochs, 7, 60)) * 0.028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.0028

    # Pattern: HIGH(0-9), LOW(10-29), HIGH(30-39), LOW(40-59), HIGH(60-79)
    # Epochs 10-29 (2 cycles): LOW - should create interval 1
    h_counts[10:30, :, :] = 0.00028
    o_counts[10:30, :, :] = 0.000028

    # Epochs 40-59 (2 cycles): LOW - should create interval 2
    h_counts[40:60, :, :] = 0.00028
    o_counts[40:60, :, :] = 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Act
    with patch(
        "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
        return_value=(met_start, met_start + 1),
    ):
        result = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # Assert
    l1b_bgrates_ds, l1b_goodtimes_ds = result

    # Should create at least 2 goodtime intervals (one for each LOW period)
    assert len(l1b_goodtimes_ds["gt_start_met"]) >= 2

    # All intervals should have valid start < end
    for i in range(len(l1b_goodtimes_ds["gt_start_met"])):
        assert (
            l1b_goodtimes_ds["gt_start_met"].values[i]
            < l1b_goodtimes_ds["gt_end_met"].values[i]
        )

    # Background rates should be positive for all intervals
    assert np.all(l1b_bgrates_ds["h_background_rates"].values > 0)
    assert np.all(l1b_bgrates_ds["o_background_rates"].values > 0)


def test_l1b_bgrates_when_synthetic_floor_is_zero(anc_dependencies, attr_mgr_l1b):
    num_epochs = 100
    met_start = 473389200
    met_spacing = 42
    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # Low counts so that goodtime intervals are found
    h_counts = np.ones((num_epochs, 7, 60)) * 0.00028
    o_counts = np.ones((num_epochs, 7, 60)) * 0.000028

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    patched_bg_rates = dict(LoConstants.BG_RATES)
    patched_bg_rates["H"] = 0.0
    with (
        patch(
            "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
            return_value=(met_start, met_start + 1),
        ),
        patch.object(LoConstants, "BG_RATES", patched_bg_rates),
    ):
        bgrates_ds, _ = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    # After the floor: bg_rate = anti_ram_nominal / BG_RATE_FLOOR_DIVISOR["H"] > 0
    assert np.all(bgrates_ds["h_background_rates"].values > 0)


def test_l1b_bgrates_sigma_when_anti_ram_nominal_is_zero(
    anc_dependencies, attr_mgr_l1b
):
    num_epochs = 50
    met_start = 473389200
    met_spacing = 42
    met_times = np.arange(met_start, met_start + num_epochs * met_spacing, met_spacing)
    epoch_times = met_to_ttj2000ns(met_times)

    # High counts everywhere so no goodtime intervals are found
    h_counts = np.ones((num_epochs, 7, 60)) * 0.1
    o_counts = np.ones((num_epochs, 7, 60)) * 0.01

    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "spin_bin_6"), h_counts),
            "o_counts": (("epoch", "esa_step", "spin_bin_6"), o_counts),
        },
        coords={
            "epoch": epoch_times,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
        attrs={"Repointing": "repoint00001"},
    )

    sci_dependencies = {
        "imap_lo_l1b_histrates": l1b_histrates,
        "imap_lo_l1b_de": xr.Dataset(),
        "imap_lo_l1b_nhk": xr.Dataset(),
    }

    # Zero the anti-RAM threshold so bg_rate_anti_ram_nominal = 0 for any pivot angle
    with (
        patch(
            "imap_processing.lo.l1b.lo_l1b.get_pointing_times_from_id",
            return_value=(met_start, met_start + 1),
        ),
        patch.object(LoConstants, "PIVOT_ANGLE_THRESHOLDS", {}),
        patch.object(LoConstants, "THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT", 0.0),
    ):
        bgrates_ds, _ = l1b_bgrates_and_goodtimes(
            sci_dependencies, anc_dependencies, attr_mgr_l1b, delay_max=840
        )

    assert np.all(bgrates_ds["h_background_rates"].values == 0.0)
    assert np.all(bgrates_ds["h_background_variance"].values == 0.0)
