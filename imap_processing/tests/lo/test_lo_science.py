import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.lo_science import (
    combine_segmented_packets,
    parse_de_bin,
    parse_events,
    parse_fixed_fields,
    parse_variable_fields,
)
from imap_processing.utils import convert_to_binary_string, packet_file_to_datasets


@pytest.fixture()
def fake_de_dataset():
    # binary packet fields
    count = "0000000000000010"  # 2
    passes = "00000000000000000000000000000001"  # 1
    # DE One
    absent_1 = "0000"  # case 0
    time_1 = "000001100100"  # 100
    energy_1 = "010"  # 2
    mode_1 = "1"
    tof0_1 = "0000000000"
    # TOF1 not transmitted
    tof2_1 = "000000010"  # 2
    tof3_1 = "000011"  # 3
    cksm_1 = "0000"  # 0
    # POS not transmitted

    # DE Two
    absent_2 = "1010"  # case 10
    time_2 = "000001100100"  # 100
    energy_2 = "010"  # 2
    mode_2 = "1"
    # TOF0 not transmitted
    tof1_2 = "000000001"  # 1
    # TOF2, TOF3, CKSM not transmitted
    pos_2 = "00"  # 0

    de_data = (
        count
        + passes
        + absent_1
        + time_1
        + energy_1
        + mode_1
        + tof0_1
        + tof2_1
        + tof3_1
        + cksm_1
        + absent_2
        + time_2
        + energy_2
        + mode_2
        + tof1_2
        + pos_2
    )
    dataset = xr.Dataset(
        data_vars=dict(
            count=(["time"], np.array([2])),
            events=(["time"], np.array([de_data])),
        )
    )

    return dataset


@pytest.fixture()
def sample_data():
    xtce_file = imap_module_directory / "lo/packet_definitions/lo_xtce.xml"
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
        use_derived_value=False,
    )
    return datasets_by_apid


@pytest.fixture()
def segmented_pkts_fake_data():
    dataset = xr.Dataset(
        data_vars=dict(
            seq_flgs=(["epoch"], np.array([1, 0, 0, 2, 3, 1, 0, 2, 1, 2])),
            src_seq_ctr=(["epoch"], np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])),
            shcoarse=(["epoch"], np.array([0, 0, 0, 0, 10, 20, 20, 20, 30, 30])),
            data=(
                ["epoch"],
                np.array(
                    [
                        "0000000001",
                        "0000000010",
                        "0000000100",
                        "0000001000",
                        "0000010000",
                        "0000100000",
                        "0001000000",
                        "0010000000",
                        "0100000000",
                        "1000000000",
                    ]
                ),
            ),
        ),
        coords=dict(epoch=(["epoch"], np.array([0, 0, 0, 0, 10, 20, 20, 20, 30, 30]))),
    )
    return dataset


@pytest.fixture()
def attr_mgr():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", "v000")
    return attr_mgr


@pytest.fixture()
def initialized_dataset(fake_de_dataset, attr_mgr):
    fake_de_dataset.attrs["bit_pos"] = 0
    de_fields = [
        "coincidence_type",
        "de_time",
        "esa_step",
        "mode",
        "tof0",
        "tof1",
        "tof2",
        "tof3",
        "cksm",
        "pos",
    ]
    for field in de_fields:
        fake_de_dataset[field] = xr.DataArray(
            np.full(2, attr_mgr.get_variable_attributes(field)["FILLVAL"]),
            dims="direct_events",
        )
    return fake_de_dataset


def test_parse_events(fake_de_dataset, attr_mgr):
    # Act
    dataset = parse_events(fake_de_dataset, attr_mgr)

    # Assert
    np.testing.assert_array_equal(dataset["coincidence_type"].values, np.array([0, 10]))
    np.testing.assert_array_equal(dataset["de_time"].values, np.array([100, 100]))
    np.testing.assert_array_equal(dataset["esa_step"].values, np.array([2, 2]))
    np.testing.assert_array_equal(dataset["mode"].values, np.array([1, 1]))
    np.testing.assert_array_equal(dataset["tof0"].values, np.array([0 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof1"].values, np.array([65535, 1 << 1]))
    np.testing.assert_array_equal(dataset["tof2"].values, np.array([2 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof3"].values, np.array([3 << 1, 65535]))
    np.testing.assert_array_equal(dataset["cksm"].values, np.array([0 << 1, 255]))
    np.testing.assert_array_equal(dataset["pos"].values, np.array([255, 0]))


def test_parse_fixed_fields(initialized_dataset):
    # Arrange
    initialized_dataset.attrs["bit_pos"] = 48

    # Act
    dataset = parse_fixed_fields(initialized_dataset, 0, 0)

    # Assert
    np.testing.assert_array_equal(
        dataset["coincidence_type"].values, np.array([0, 255])
    )
    np.testing.assert_array_equal(dataset["de_time"].values, np.array([100, 65535]))
    np.testing.assert_array_equal(dataset["esa_step"].values, np.array([2, 255]))
    np.testing.assert_array_equal(dataset["mode"].values, np.array([1, 255]))


def test_parse_variable_fields(initialized_dataset):
    # Arrange
    initialized_dataset["coincidence_type"].values = np.array([0, 255])
    initialized_dataset["mode"].values = np.array([1, 255])
    initialized_dataset.attrs["bit_pos"] = 68

    # Act
    dataset = parse_variable_fields(initialized_dataset, 0, 0)

    # Assert
    np.testing.assert_array_equal(dataset["tof0"].values, np.array([0 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof1"].values, np.array([65535, 65535]))
    np.testing.assert_array_equal(dataset["tof2"].values, np.array([2 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof3"].values, np.array([3 << 1, 65535]))
    np.testing.assert_array_equal(dataset["cksm"].values, np.array([0 << 1, 255]))
    np.testing.assert_array_equal(dataset["pos"].values, np.array([255, 255]))


def test_parse_de_bin(initialized_dataset):
    # Act
    parsed_int = parse_de_bin(initialized_dataset, 0, 4, 0)
    # Assert
    assert parsed_int == 0


def test_combine_segmented_packets(segmented_pkts_fake_data):
    dataset = combine_segmented_packets(segmented_pkts_fake_data)

    np.testing.assert_array_equal(
        dataset["seq_flgs"].values, np.array([1, 0, 0, 2, 3, 1, 0, 2, 1, 2])
    )
    np.testing.assert_array_equal(
        dataset["src_seq_ctr"].values, np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    )
    np.testing.assert_array_equal(
        dataset["shcoarse"].values, np.array([0, 0, 0, 0, 10, 20, 20, 20, 30, 30])
    )
    np.testing.assert_array_equal(
        dataset["events"].values,
        np.array(
            [
                "0000000001000000001000000001000000001000",
                "0000010000",
                "01000000001000000000",
            ]
        ),
    )
    np.testing.assert_array_equal(dataset["epoch"].values, np.array([0, 10, 30]))


def test_validate_parse_events(sample_data, attr_mgr):
    de_data = sample_data[LoAPID.ILO_SCI_DE]
    validation_path = (
        imap_module_directory / "tests/lo/validation_data/"
        "Instrument_FM1_T104_R129_20240803_ILO_SCI_DE_dec_DN_with_fills.csv"
    )

    validation_data = pd.read_csv(validation_path)
    de_fields = [
        "coincidence_type",
        "de_time",
        "esa_step",
        "mode",
        "tof0",
        "tof1",
        "tof2",
        "tof3",
        "cksm",
        "pos",
    ]

    de_data["data"] = xr.DataArray(
        [convert_to_binary_string(data) for data in de_data["data"].values],
        dims=de_data["data"].dims,
        attrs=de_data["data"].attrs,
    )
    de_data = combine_segmented_packets(de_data)
    dataset = parse_events(de_data, attr_mgr)

    for field in de_fields:
        np.testing.assert_array_equal(
            dataset[field].values, validation_data[field.upper()].values
        )

    assert dataset["de_count"].values == 1998
    assert dataset["passes"].values == 8
