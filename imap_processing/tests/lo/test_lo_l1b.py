from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1b.lo_l1b import (
    calculate_histogram_rates,
    calculate_tof1_for_golden_triples,
    convert_start_end_acq_times,
    convert_tofs_to_eu,
    create_badtimes_dataset,
    create_datasets,
    get_avg_spin_durations_per_cycle,
    get_spin_start_times,
    identify_species,
    initialize_l1b_de,
    lo_l1b,
    resweep_histogram_data,
    set_avg_spin_durations_per_event,
    set_bad_or_goodtimes,
    set_bad_times,
    set_coincidence_type,
    set_each_event_epoch,
    set_esa_mode,
    set_event_met,
    set_pointing_bin,
    set_pointing_direction,
    set_spin_cycle,
    set_spin_cycle_from_spin_data,
)
from imap_processing.lo.lo_ancillary import read_ancillary_file
from imap_processing.spice.spin import get_spin_data
from imap_processing.spice.time import (
    et_to_met,
    et_to_ttj2000ns,
    met_to_ttj2000ns,
    str_to_et,
    ttj2000ns_to_met,
)


@pytest.fixture
def dependencies():
    return {
        "imap_lo_l1a_de": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_de_20241022_v002.cdf"
        ),
        "imap_lo_l1a_spin": load_cdf(
            imap_module_directory
            / "tests/lo/test_cdfs/imap_lo_l1a_spin_20241022_v002.cdf"
        ),
    }


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
    l1b_histrates = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
            "o_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "azimuth_6": np.arange(60),
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
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "azimuth_6": np.arange(60),
        },
        attrs={"Logical_source": "imap_lo_l1a_histogram"},
    )
    return l1a_hist


@patch(
    "imap_processing.lo.l1b.lo_l1b.frame_transform",
    return_value=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
)
@patch(
    "imap_processing.lo.l1b.lo_l1b.instrument_pointing",
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
def test_lo_l1b_de(
    mock_frame_transform,
    mock_instrument_pointing,
    mocked_get_pointing_times,
    mock_spin_number,
    mock_cartesian_to_latitudinal,
    anc_dependencies,
):
    # Arrange
    de_file = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1a_de_20241022_v002.cdf"
    )
    spin_file = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1a_spin_20241022_v002.cdf"
    )
    data = {}
    for file in [de_file, spin_file]:
        dataset = load_cdf(file)
        data[dataset.attrs["Logical_source"]] = dataset

    expected_logical_source_de = "imap_lo_l1b_de"

    # Act
    output_files = lo_l1b(data, anc_dependencies)

    # Assert
    assert expected_logical_source_de == output_files[-1].attrs["Logical_source"]


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
    l1b_datasets = lo_l1b(sci_dependencies, anc_dependencies)

    # Assert
    assert "h_rates" in l1b_datasets[-1].data_vars
    assert "o_rates" in l1b_datasets[-1].data_vars
    assert "exposure_time" in l1b_datasets[-1].data_vars
    assert "h_counts" in l1b_datasets[-1].data_vars
    assert "o_counts" in l1b_datasets[-1].data_vars


# @pytest.mark.external_kernel
# @pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
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
    assert len(l1b_de.data_vars) == 4
    assert len(l1b_de.coords) == 0
    for l1b_name, l1a_name in {
        "pos": "pos",
        "mode_bit": "mode",
        "absent": "coincidence_type",
        "esa_step": "esa_step",
    }.items():
        assert l1b_name in l1b_de.data_vars
        np.testing.assert_array_equal(l1b_de[l1b_name], l1a_de[l1a_name])


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
    acq_start = xr.DataArray([0, 423, 846.2], dims="epoch")
    acq_end = xr.DataArray([422.8, 846, 1269.7], dims="epoch")
    expected_avg_spin_durations = np.array([422.8, 423, 423.5]) / 28

    # Act
    avg_spin_durations = get_avg_spin_durations_per_cycle(acq_start, acq_end)

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


def test_get_spin_start_times():
    # Arrange
    l1b_de = xr.Dataset(
        {
            "spin_cycle": ("direct_event", [0, 1, 2, 3, 4]),
        },
        coords={
            "direct_event": [
                0,
                1,
                2,
                3,
                4,
            ]
        },
    )
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "met": ("epoch", [0, 1]),  # MET per time epoch, not per direct event
            "de_time": ("direct_event", [0000, 1000, 2000, 3000, 4000]),
        },
        coords={"epoch": [0, 1], "direct_event": [0, 1, 2, 3, 4]},
    )
    spin = xr.Dataset(
        {
            "start_sec_spin": (
                ["epoch", "spin"],
                [[20, 25, 30, 35, 40], [45, 50, 55, 60, 65]],
            ),
            "start_subsec_spin": (
                ["epoch", "spin"],
                [[2000, 3000, 4000, 5000, 6000], [1000, 1500, 2000, 3000, 4000]],
            ),
        }
    )

    end_acq = xr.DataArray([0, 1], dims="epoch")
    spin_start_times_expected = np.array([20.002, 25.003, 55.002, 60.003, 65.004])
    spin_start_times = get_spin_start_times(l1a_de, l1b_de, spin, end_acq)

    np.testing.assert_allclose(
        spin_start_times,
        spin_start_times_expected,
        atol=1e-4,
    )


def test_set_event_met():
    # Arrange
    l1b_de = xr.Dataset()
    l1a_de = xr.Dataset(
        {
            "de_count": ("epoch", [2, 3]),
            "de_time": ("direct_event", [0000, 1000, 2000, 3000, 4000]),
        },
        coords={
            "epoch": [0, 1],
            "direct_event": [
                0,
                1,
                2,
                3,
                4,
            ],
        },
    )
    avg_spin_durations = xr.DataArray([5, 10])
    spin_start_times = xr.DataArray([10, 20, 30, 40, 50])
    expected_event_met = np.array([10, 21.2207, 34.8828, 47.3242, 59.7656])

    # Act
    l1b_de = set_event_met(l1a_de, l1b_de, spin_start_times, avg_spin_durations)

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
            "tof1": ("epoch", [42, 36, 0]),
            "tof2": ("epoch", [2, 6, 2]),
            "tof3": ("epoch", [2, 8, 2]),
            "cksm": ("epoch", [2, 12, 2]),
        }
    )

    # Act
    l1a_de = calculate_tof1_for_golden_triples(l1a_de)

    # Assert
    assert l1a_de_expected.equals(l1a_de)


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
    tof3_expected = np.array([3.88245, 1.818162])

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


def test_set_bad_times(anc_dependencies):
    # Arrange
    l1b_de = xr.Dataset(
        {
            "esa_step": ("epoch", [1, 1, 3, 1]),
            "spin_bin": ("epoch", [1900, 2000, 3000, 2]),
        },
        coords={
            "epoch": met_to_ttj2000ns([473385599, 473385600, 473385601, 473385602]),
        },
    )

    expected_bad_times = np.array([0, 1, 0, 0])

    # Act
    l1b_de = set_bad_times(l1b_de, anc_dependencies)

    # Assert
    np.testing.assert_array_equal(l1b_de["badtimes"], expected_bad_times)


def test_set_bad_or_goodtimes(anc_dependencies):
    # Arrange
    # badtimes ancillary
    df = read_ancillary_file(anc_dependencies[1])

    epoch = met_to_ttj2000ns([473385599, 473385600, 473385601, 473385602])
    esa_step = np.array([1, 1, 3, 1])
    spin_bin = np.array([1900, 2000, 3000, 2])

    expected_bad_times = np.array([0, 1, 0, 0])

    # Act
    badtimes = set_bad_or_goodtimes(df, epoch, esa_step, spin_bin)

    # Assert
    np.testing.assert_array_equal(badtimes, expected_bad_times)


@patch(
    "imap_processing.lo.l1b.lo_l1b.instrument_pointing",
    return_value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
)
def test_set_direction(imap_ena_sim_metakernel):
    # Arrange
    l1b_de = xr.Dataset(
        {},
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


@patch(
    "imap_processing.lo.l1b.lo_l1b.frame_transform",
    return_value=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
)
@patch(
    "imap_processing.lo.l1b.lo_l1b.cartesian_to_latitudinal",
    return_value=np.array([[0, -180, -2], [0, 0, 0], [0, 90, 1], [0, 180, 2]]),
)
def test_pointing_bins(mock_cartesian_to_latitudinal, mock_frame_transform):
    # Arrange
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
        },
    )

    expected_pointing_lats = np.array([0, 20, 30, 40])
    expected_pointing_lons = np.array([0, 1800, 2700, 3600])

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


def test_resweep_histogram_success(anc_dependencies):
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-15T02:00:00", "2025-04-15T03:00:00"])
    )
    l1b_histrate = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
            "o_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
    )
    exposure_factor_expected = np.full((2, 7, 60), 1)
    exposure_factor_expected[:, 0, :] = 2

    l1b_histrate.h_counts[0, 0, 0] = 5
    l1b_histrate.h_counts[0, 1, 0] = 10
    l1b_histrate.h_counts[0, 2, 0] = 2

    l1b_histrate.o_counts[1, 0, 0] = 2
    l1b_histrate.o_counts[1, 1, 0] = 3
    l1b_histrate.o_counts[1, 2, 0] = 4

    l1b_histrates, exposure_factor = resweep_histogram_data(
        l1b_histrate, anc_dependencies
    )

    assert l1b_histrates.h_counts[0, 0, 0] == 15
    assert l1b_histrates.h_counts[0, 1, 0] == 0
    assert l1b_histrates.h_counts[0, 2, 0] == 2

    assert l1b_histrates.o_counts[1, 0, 0] == 5
    assert l1b_histrates.o_counts[1, 1, 0] == 0
    assert l1b_histrates.o_counts[1, 2, 0] == 4

    assert np.array_equal(exposure_factor, exposure_factor_expected)


def test_resweep_histogram_no_date(anc_dependencies):
    # Arrange
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-25T02:00:00", "2025-04-25T03:00:00"])
    )
    l1b_histrate = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
            "o_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
    )

    l1b_histrate.h_counts[0, 0, 0] = 5
    l1b_histrate.h_counts[0, 1, 0] = 10
    l1b_histrate.h_counts[0, 2, 0] = 2

    with pytest.raises(
        ValueError,
        match="No sweep table entry found for date "
        "2025-04-25T02:00:00.000 at epoch idx 0",
    ):
        resweep_histogram_data(l1b_histrate, anc_dependencies)


def test_resweep_histogram_multiple_lut(anc_dependencies):
    epoch_date = et_to_ttj2000ns(
        str_to_et(["2025-04-16T02:00:00", "2025-04-16T03:00:00"])
    )
    l1b_histrate = xr.Dataset(
        {
            "h_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
            "o_counts": (("epoch", "esa_step", "azimuth_6"), np.zeros((2, 7, 60))),
        },
        coords={
            "epoch": epoch_date,
            "esa_step": np.arange(1, 8),
            "spin_bin_6": np.arange(60),
        },
    )

    with pytest.raises(
        ValueError,
        match=f"Expected exactly 1 unique LUT_table "
        f"value for date 2025-04-16, but found 2:{[1, 2]}",
    ):
        resweep_histogram_data(l1b_histrate, anc_dependencies)


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
    exposure_factor = np.zeros((2, 7, 60))
    exposure_factor[0, 0, 0] = 1
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
        l1b_histrates, acq_start, acq_end, avg_spin_durations_per_cycle, exposure_factor
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
    exposure_factor = np.zeros((2, 7, 60))
    l1b_histrate = calculate_histogram_rates(
        l1b_histrates, acq_start, acq_end, avg_spin_durations_per_cycle, exposure_factor
    )

    np.testing.assert_array_equal(l1b_histrate["h_rates"], np.full((2, 7, 60), np.nan))
    np.testing.assert_array_equal(l1b_histrate["o_rates"], np.full((2, 7, 60), np.nan))


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
    exposure_factor = np.zeros((2, 7, 60))
    l1b_histrate = calculate_histogram_rates(
        l1b_histrates, acq_start, acq_end, avg_spin_durations_per_cycle, exposure_factor
    )

    np.testing.assert_array_equal(l1b_histrate["h_rates"], np.full((2, 7, 60), np.nan))
    np.testing.assert_array_equal(l1b_histrate["o_rates"], np.full((2, 7, 60), np.nan))


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
        "imap_processing.lo.l1b.lo_l1b.get_spin_number", return_value=np.array([28])
    ):
        result = set_spin_cycle_from_spin_data(l1a_hist, l1b_hist, spin_data)

    # Assert - Only epoch 1 (science_met[1]=200) should remain
    # (matched to spin with 28 spins)
    assert len(result["epoch"]) == 1
    expected_epochs = met_to_ttj2000ns([200])
    np.testing.assert_array_equal(result["epoch"].values, expected_epochs)

    # Verify spin_cycle shape matches filtered data
    assert result["spin_cycle"].shape == (1, 7)
