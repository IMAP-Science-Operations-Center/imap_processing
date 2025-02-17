from collections import namedtuple

import numpy as np

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1c.lo_l1c import create_datasets, lo_l1c


def test_lo_l1c():
    # Arrange
    de_file = (
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1b_de_20100101_v001.cdf"
    )
    data = {}
    dataset = load_cdf(de_file)
    data[dataset.attrs["Logical_source"]] = dataset

    expected_logical_source = "imap_lo_l1c_pset"
    # Act
    output_dataset = lo_l1c(data, "001")

    # Assert
    assert expected_logical_source == output_dataset.attrs["Logical_source"]


def test_create_dataset():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1c")

    logical_source = "imap_lo_l1c_pset"

    data_field_tup = namedtuple("data_field_tup", ["name"])
    data_fields = [
        data_field_tup("POINTING_START"),
        data_field_tup("POINTING_END"),
        data_field_tup("MODE"),
        data_field_tup("PIVOT_ANGLE"),
        data_field_tup("TRIPLES_COUNTS"),
        data_field_tup("TRIPLES_RATES"),
        data_field_tup("DOUBLES_COUNTS"),
        data_field_tup("DOUBLES_RATES"),
        data_field_tup("HYDROGEN_COUNTS"),
        data_field_tup("HYDROGEN_RATES"),
        data_field_tup("OXYGEN_COUNTS"),
        data_field_tup("OXYGEN_RATES"),
        data_field_tup("EXPOSURE_TIME"),
    ]

    dataset = create_datasets(attr_mgr, logical_source, data_fields)

    np.testing.assert_array_equal(dataset.pointing_start, np.ones(1))
    np.testing.assert_array_equal(dataset.pointing_end, np.ones(1))
    np.testing.assert_array_equal(dataset.mode, np.ones(1))
    np.testing.assert_array_equal(dataset.pivot_angle, np.ones(1))
    np.testing.assert_array_equal(dataset.triples_counts, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.triples_rates, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.doubles_counts, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.doubles_rates, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.hydrogen_counts, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.hydrogen_rates, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.oxygen_counts, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.oxygen_rates, np.ones((1, 3600, 7)))
    np.testing.assert_array_equal(dataset.exposure_time, np.ones((1, 7)))
