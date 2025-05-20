import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1c.lo_l1c import (
    FilterType,
    calculate_exposure_times,
    create_pset_counts,
    filter_goodtimes,
    initialize_pset,
    lo_l1c,
)


@pytest.fixture
def l1b_de():
    l1b_de = xr.Dataset(
        {
            "pointing_bin_lon": ("epoch", [20, 0, 20, 2000, 3500]),
            "pointing_bin_lat": ("epoch", [20, 20, 20, 20, 20]),
            "esa_step": ("epoch", [1, 2, 1, 4, 5]),
            "coincidence_type": (
                "epoch",
                [
                    "111111",
                    "111100",
                    "111000",
                    "110100",
                    "110000",
                ],
            ),
            "species": ("epoch", ["h", "o", "h", "h", "o"]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9]),
        },
        coords={
            "epoch": [
                7.9794907049e17,
                7.9794907153e17,
                7.9794907254e17,
                7.9794907354e17,
                7.9794907454e17,
            ],
        },
    )
    return l1b_de


@pytest.fixture
def anc_dependencies():
    anc_dependencies_path = (
        imap_module_directory / "tests/lo/test_anc/imap_lo_goodtimes_20250415_v001.csv"
    )
    return [str(anc_dependencies_path)]


@pytest.fixture
def attr_mgr():
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1c")
    return attr_mgr_l1b


@pytest.fixture
def counts():
    """Fixture for initial counts."""
    return np.zeros((1, 3600, 40, 7))


@pytest.fixture
def h_counts(counts):
    h = counts.copy()
    h[0, 20, 20, 1] = 2
    h[0, 2000, 20, 4] = 1
    return h


@pytest.fixture
def o_counts(counts):
    o = counts.copy()
    o[0, 3500, 20, 5] = 1
    o[0, 0, 20, 2] = 1
    return o


@pytest.fixture
def triples_counts(counts):
    triples = counts.copy()
    triples[0, 20, 20, 1] = 2
    triples[0, 0, 20, 2] = 1
    return triples


@pytest.fixture
def doubles_counts(counts):
    doubles = counts.copy()
    doubles[0, 2000, 20, 4] = 1
    doubles[0, 3500, 20, 5] = 1
    return doubles


def test_lo_l1c(l1b_de, anc_dependencies):
    # Arrange
    data = {"imap_lo_l1b_de": l1b_de}

    expected_logical_source = "imap_lo_l1c_pset"
    # Act
    output_dataset = lo_l1c(data, anc_dependencies)

    # Assert
    assert expected_logical_source == output_dataset[0].attrs["Logical_source"]


def test_initialize_pset(l1b_de, attr_mgr):
    # Arrange
    logical_source = "imap_lo_l1c_pset"
    expected_epoch = 7.9794907049e17

    # Act
    pset = initialize_pset(l1b_de, attr_mgr, logical_source)

    # Assert
    assert pset.attrs["Logical_source"] == logical_source
    np.testing.assert_array_equal(pset["epoch"], expected_epoch)


def test_filter_goodtimes(l1b_de, anc_dependencies):
    # Arrange
    l1b_de_with_badtimes = xr.Dataset(
        {
            "pointing_bin_lon": ("epoch", [20, 0, 20, 2000, 3500, 200]),
            "pointing_bin_lat": ("epoch", [20, 20, 20, 20, 20, 40]),
            "esa_step": ("epoch", [1, 2, 1, 4, 5, 2]),
            "coincidence_type": (
                "epoch",
                ["111111", "111100", "111000", "110100", "110000", "000000"],
            ),
            "species": ("epoch", ["h", "o", "h", "h", "o", "u"]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5, 12]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9, 50]),
        },
        coords={
            "epoch": [
                7.9794907049e17,
                7.9794907153e17,
                7.9794907254e17,
                7.9794907354e17,
                7.9794907454e17,
                8.74117692184e17,
            ],
        },
    )
    l1b_de_no_badtimes_expected = l1b_de.copy()

    # Act
    l1b_no_badtimes = filter_goodtimes(l1b_de_with_badtimes, anc_dependencies)

    # Assert
    xr.testing.assert_equal(l1b_no_badtimes, l1b_de_no_badtimes_expected)


def test_create_pset_counts(l1b_de):
    # Arrange
    expected_counts = np.zeros((1, 3600, 40, 7))
    expected_counts[0, 20, 20, 1] = 2
    expected_counts[0, 2000, 20, 4] = 1
    expected_counts[0, 3500, 20, 5] = 1
    expected_counts[0, 0, 20, 2] = 1

    # Act
    counts = create_pset_counts(l1b_de)

    # Assert
    np.testing.assert_array_equal(counts, expected_counts)


def test_create_h_pset_counts(l1b_de, h_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.HYDROGEN)

    # Assert
    np.testing.assert_array_equal(counts, h_counts)


def test_create_o_pset_counts(l1b_de, o_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.OXYGEN)

    # Assert
    np.testing.assert_array_equal(counts, o_counts)


def test_create_triples_pset_counts(l1b_de, triples_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.TRIPLES)

    # Assert
    np.testing.assert_array_equal(counts, triples_counts)


def test_create_doubles_pset_counts(l1b_de, doubles_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.DOUBLES)

    # Assert
    np.testing.assert_array_equal(counts, doubles_counts)


def test_calculate_exposure_times(l1b_de):
    # Arrange
    counts = create_pset_counts(l1b_de)
    expected_exposure_times = np.full((1, 3600, 40, 7), np.nan)
    # Average of the exposure times for each bin
    expected_exposure_times[0, 20, 20, 1] = 4 * np.mean([15.2, 14.9]) / 3600
    expected_exposure_times[0, 2000, 20, 4] = 4 * 15 / 3600
    expected_exposure_times[0, 3500, 20, 5] = 4 * 14.9 / 3600
    expected_exposure_times[0, 0, 20, 2] = 4 * 15.2 / 2600
    # Act
    exposure_times = calculate_exposure_times(counts, l1b_de)

    # Assert
    np.testing.assert_allclose(
        exposure_times,
        expected_exposure_times,
        atol=1e-2,
    )
