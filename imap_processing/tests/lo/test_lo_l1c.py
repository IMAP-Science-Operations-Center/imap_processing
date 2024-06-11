from collections import namedtuple

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

    expected_out = "imap_lo_l1c_pset_20100101_v001.cdf"
    # Act
    output_file = lo_l1c(data)

    # Assert
    assert expected_out == output_file.name


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

    assert len(dataset.pointing_start.shape) == 1
    assert dataset.pointing_start.shape[0] == 1
    assert len(dataset.pointing_end.shape) == 1
    assert dataset.pointing_end.shape[0] == 1
    assert len(dataset.mode.shape) == 1
    assert dataset.mode.shape[0] == 1
    assert len(dataset.pivot_angle.shape) == 1
    assert dataset.pivot_angle.shape[0] == 1
    assert len(dataset.triples_counts.shape) == 3
    assert dataset.triples_counts.shape[0] == 1
    assert dataset.triples_counts.shape[1] == 3600
    assert dataset.triples_counts.shape[2] == 7
    assert len(dataset.triples_rates.shape) == 3
    assert dataset.triples_rates.shape[0] == 1
    assert dataset.triples_rates.shape[1] == 3600
    assert dataset.triples_rates.shape[2] == 7
    assert len(dataset.doubles_counts.shape) == 3
    assert dataset.doubles_counts.shape[0] == 1
    assert dataset.doubles_counts.shape[1] == 3600
    assert dataset.doubles_counts.shape[2] == 7
    assert len(dataset.doubles_rates.shape) == 3
    assert dataset.doubles_rates.shape[0] == 1
    assert dataset.doubles_rates.shape[1] == 3600
    assert dataset.doubles_rates.shape[2] == 7
    assert len(dataset.hydrogen_counts.shape) == 3
    assert dataset.hydrogen_counts.shape[0] == 1
    assert dataset.hydrogen_counts.shape[1] == 3600
    assert dataset.hydrogen_counts.shape[2] == 7
    assert len(dataset.hydrogen_rates.shape) == 3
    assert dataset.hydrogen_rates.shape[0] == 1
    assert dataset.hydrogen_rates.shape[1] == 3600
    assert dataset.hydrogen_rates.shape[2] == 7
    assert len(dataset.oxygen_counts.shape) == 3
    assert dataset.oxygen_counts.shape[0] == 1
    assert dataset.oxygen_counts.shape[1] == 3600
    assert dataset.oxygen_counts.shape[2] == 7
    assert len(dataset.oxygen_rates.shape) == 3
    assert dataset.oxygen_rates.shape[0] == 1
    assert dataset.oxygen_rates.shape[1] == 3600
    assert dataset.oxygen_rates.shape[2] == 7
    assert len(dataset.exposure_time.shape) == 2
    assert dataset.exposure_time.shape[0] == 1
    assert dataset.exposure_time.shape[1] == 7
