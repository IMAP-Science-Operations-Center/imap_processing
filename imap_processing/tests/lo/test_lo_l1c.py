import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1c.lo_l1c import (
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
