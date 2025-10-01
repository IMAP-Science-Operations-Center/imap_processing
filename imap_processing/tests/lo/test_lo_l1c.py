from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1c.lo_l1c import (
    PSET_SHAPE,
    FilterType,
    calculate_exposure_times,
    create_pset_counts,
    filter_goodtimes,
    lo_l1c,
    set_background_rates,
)
from imap_processing.spice.time import met_to_ttj2000ns


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
            "species": ("epoch", ["H", "O", "H", "H", "O"]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9]),
            "spin_bin": ("epoch", [1900, 2000, 3000, 3000, 3000]),
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
def repoint_met():
    met = np.arange(511000000, 511000000 + 86400 * 5, 86400)
    return met


@pytest.fixture
def l1b_de_spin():
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
            "species": ("epoch", ["H", "O", "H", "H", "O"]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9]),
            "spin_bin": ("epoch", [1900, 2000, 3000, 3000, 3000]),
        },
        coords={
            "epoch": met_to_ttj2000ns(np.arange(511000000, 511000000 + 200, 40) + 902),
        },
    )
    return l1b_de


@pytest.fixture
def anc_dependencies():
    anc_dependencies_path = [
        str(
            imap_module_directory
            / "tests/lo/test_anc/imap_lo_good-times-small_20250101_20270101_v001.csv"
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/"
            / "imap_lo_hydrogen-background-small_20250101_20270101_v001.csv"
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/"
            / "imap_lo_oxygen-background-small_20250101_20270101_v001.csv"
        ),
    ]
    return anc_dependencies_path


@pytest.fixture
def attr_mgr():
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1c")
    return attr_mgr_l1b


@pytest.fixture
def counts():
    """Fixture for initial counts."""
    return np.zeros(PSET_SHAPE)


@pytest.fixture
def h_counts(counts):
    h = counts.copy()
    h[0, 1, 20, 20] = 2
    h[0, 4, 2000, 20] = 1
    return h


@pytest.fixture
def o_counts(counts):
    o = counts.copy()
    o[0, 5, 3500, 20] = 1
    o[0, 2, 0, 20] = 1
    return o


@pytest.fixture
def triples_counts(counts):
    triples = counts.copy()
    triples[0, 1, 20, 20] = 2
    triples[0, 2, 0, 20] = 1
    return triples


@pytest.fixture
def doubles_counts(counts):
    doubles = counts.copy()
    doubles[0, 4, 2000, 20] = 1
    doubles[0, 5, 3500, 20] = 1
    return doubles


@pytest.fixture
def expected_bg():
    expected_rates = np.array(
        [
            np.full((3600, 40), 0.0098),
            np.full((3600, 40), 0.0089),
            np.full((3600, 40), 0.0118),
            np.full((3600, 40), 0.0113),
            np.full((3600, 40), 0.0056),
            np.full((3600, 40), 0.0008),
            np.full((3600, 40), 0.0),
        ],
        dtype=np.float16,
    )

    expected_uncert = np.array(
        [
            np.full((3600, 40), 0.0025),
            np.full((3600, 40), 0.002),
            np.full((3600, 40), 0.0015),
            np.full((3600, 40), 0.0015),
            np.full((3600, 40), 0.001),
            np.full((3600, 40), 0.0008),
            np.full((3600, 40), 0.0),
        ],
        dtype=np.float16,
    )

    expected_err = np.zeros((7, 3600, 40), dtype=np.float16)

    expected_bg = (expected_rates, expected_uncert, expected_err)
    return expected_bg


@patch("imap_processing.lo.l1c.lo_l1c.set_background_rates")
@patch("imap_processing.lo.l1c.lo_l1c.filter_goodtimes")
def test_lo_l1c(
    mock_filter_goodtimes,
    mock_set_background_rates,
    l1b_de_spin,
    anc_dependencies,
    use_fake_repoint_data_for_time,
    use_fake_spin_data_for_time,
    repoint_met,
):
    # Arrange
    data = {"imap_lo_l1b_de": l1b_de_spin}
    use_fake_spin_data_for_time(511000000)
    use_fake_repoint_data_for_time(np.arange(511000000, 511000000 + 86400 * 5, 86400))
    mock_set_background_rates.return_value = (None, None, None)
    mock_filter_goodtimes.return_value = l1b_de_spin
    expected_logical_source = "imap_lo_l1c_pset"

    # Act
    output_dataset = lo_l1c(data, anc_dependencies)

    # Assert
    assert expected_logical_source == output_dataset[0].attrs["Logical_source"]


def test_filter_goodtimes(l1b_de, anc_dependencies):
    # Arrange
    l1b_de_all = xr.Dataset(
        {
            "esa_step": ("epoch", [1, 2, 1, 4, 5, 2]),
            "spin_bin": ("epoch", [1900, 2000, 3000, 3000, 3000, 3000]),
        },
        coords={
            "epoch": met_to_ttj2000ns(
                [
                    473389199,
                    473389200,
                    473389201,
                    473389202,
                    473389203,
                    473407619,
                ]
            )
        },
    )
    expected_goodtimes_mask = [False, False, True, False, True, False]

    l1b_goodtimes_onl_expected = l1b_de_all.isel(epoch=expected_goodtimes_mask)

    # Act
    l1b_goodtimes_only = filter_goodtimes(l1b_de_all, anc_dependencies)

    # Assert
    xr.testing.assert_equal(l1b_goodtimes_only, l1b_goodtimes_onl_expected)


def test_create_pset_counts(l1b_de):
    # Arrange
    expected_counts = np.zeros((1, 7, 3600, 40))
    expected_counts[0, 1, 20, 20] = 2
    expected_counts[0, 4, 2000, 20] = 1
    expected_counts[0, 5, 3500, 20] = 1
    expected_counts[0, 2, 0, 20] = 1

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
    expected_exposure_times = np.full(PSET_SHAPE, np.nan)
    # Average of the exposure times for each bin
    expected_exposure_times[0, 1, 20, 20] = 4 * np.mean([15.2, 14.9]) / 3600
    expected_exposure_times[0, 4, 2000, 20] = 4 * 15 / 3600
    expected_exposure_times[0, 5, 3500, 20] = 4 * 14.9 / 3600
    expected_exposure_times[0, 2, 0, 20] = 4 * 15.2 / 2600
    # Act
    exposure_times = calculate_exposure_times(counts, l1b_de)

    # Assert
    np.testing.assert_allclose(
        exposure_times,
        expected_exposure_times,
        atol=1e-2,
    )


@pytest.mark.parametrize("species", [FilterType.HYDROGEN, FilterType.OXYGEN])
def test_set_background_rates(
    l1b_de_spin, anc_dependencies, attr_mgr, species, expected_bg
):
    # Arrange
    pointing_start_met = 473389100.0
    pointing_end_met = 473472100.0

    # Act
    rates, uncert, err = set_background_rates(
        pointing_start_met, pointing_end_met, species, anc_dependencies, attr_mgr
    )

    # Assert
    np.testing.assert_array_equal(
        rates.values,
        expected_bg[0],
    )
    np.testing.assert_array_equal(
        uncert.values,
        expected_bg[1],
    )
    np.testing.assert_array_equal(
        err.values,
        expected_bg[2],
    )


def test_set_background_rates_species_error(anc_dependencies, attr_mgr):
    # Arrange
    pointing_start_met = 473389100.0
    pointing_end_met = 473472100.0
    species = FilterType.DOUBLES

    # Act
    with pytest.raises(
        ValueError, match="Species must be 'h' or 'o', but got doubles."
    ):
        rates, uncert, err = set_background_rates(
            pointing_start_met, pointing_end_met, species, anc_dependencies, attr_mgr
        )
