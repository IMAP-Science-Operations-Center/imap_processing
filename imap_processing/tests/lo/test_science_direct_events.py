from collections import namedtuple

import numpy as np
import pytest

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.lo.l0.data_classes.science_direct_events import (
    ScienceDirectEvents,
)


@pytest.fixture()
def fake_packet_data():
    fake_data_type = namedtuple("fake_data_cats", ["header", "data"])
    fake_data_field = namedtuple("fake_packet", ["raw_value", "derived_value"])
    return fake_data_type(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(0, 0),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        },
        {
            "SHCOARSE": fake_data_field(0, 0),
            "COUNT": fake_data_field(0, 0),
            "DATA": fake_data_field("00", "00"),
            "CHKSUM": fake_data_field(0, 0),
        },
    )


@pytest.fixture()
def single_de(fake_packet_data):
    de = ScienceDirectEvents(fake_packet_data, "0", "fakepacketname")
    de.COUNT = 1
    de.TIME = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.ENERGY = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.MODE = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF0 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF1 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF2 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF3 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.CKSM = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.POS = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    return de


@pytest.fixture()
def multi_de(fake_packet_data):
    de = ScienceDirectEvents(fake_packet_data, "0", "fakepacketname")
    de.COUNT = 2
    de.TIME = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.ENERGY = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.MODE = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF0 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF1 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF2 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.TOF3 = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.CKSM = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    de.POS = np.ones(de.COUNT) * GlobalConstants.DOUBLE_FILLVAL
    return de


@pytest.fixture()
def tof_data():
    TOFData = namedtuple(
        "TOFData", ["ENERGY", "POS", "TOF0", "TOF1", "TOF2", "TOF3", "CKSM", "TIME"]
    )
    return TOFData


def test_parse_data_case_0(single_de):
    # Arrange
    absent = "0000"  # case 0
    time = "000001100100"  # 100
    energy = "010"  # 2
    mode = "1"
    tof0 = "0000000000"
    # TOF1 not transmitted
    tof2 = "000000010"  # 2
    tof3 = "000011"  # 3
    cksm = "000"  # 0
    # POS not transmitted
    single_de.DATA = absent + time + energy + mode + tof0 + tof2 + tof3 + cksm

    expected_time = np.array([100])
    expected_energy = np.array([2])
    expected_mode = np.array([1])
    # tofs and cksm are bit shifted to the left by 1 during decompression
    expected_tof0 = np.array([0 << 1])
    expected_tof1 = np.array([GlobalConstants.DOUBLE_FILLVAL])
    expected_tof2 = np.array([2 << 1])
    expected_tof3 = np.array([3 << 1])
    expected_cksm = np.array([0 << 1])
    expected_pos = np.array([GlobalConstants.DOUBLE_FILLVAL])

    # Act
    single_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(single_de.TIME, expected_time)
    np.testing.assert_array_equal(single_de.ENERGY, expected_energy)
    np.testing.assert_array_equal(single_de.MODE, expected_mode)
    np.testing.assert_array_equal(single_de.TOF0, expected_tof0)
    np.testing.assert_array_equal(single_de.TOF1, expected_tof1)
    np.testing.assert_array_equal(single_de.TOF2, expected_tof2)
    np.testing.assert_array_equal(single_de.TOF3, expected_tof3)
    np.testing.assert_array_equal(single_de.CKSM, expected_cksm)
    np.testing.assert_array_equal(single_de.POS, expected_pos)


def test_parse_data_case_10(single_de):
    # Arrange
    absent = "1010"  # case 10
    time = "000001100100"  # 100
    energy = "010"  # 2
    mode = "1"
    # TOF0 not transmitted
    tof1 = "000000001"  # 1
    # TOF2, TOF3, CKSM not transmitted
    pos = "00"  # 0
    single_de.DATA = absent + time + energy + mode + tof1 + pos

    expected_time = np.array([100])
    expected_energy = np.array([2])
    expected_mode = np.array([1])
    expected_tof0 = np.array([GlobalConstants.DOUBLE_FILLVAL])
    # tofs and cksm are bit shifted to the left by 1 during decompression
    expected_tof1 = np.array([1 << 1])
    expected_tof2 = np.array([GlobalConstants.DOUBLE_FILLVAL])
    expected_tof3 = np.array([GlobalConstants.DOUBLE_FILLVAL])
    expected_cksm = np.array([GlobalConstants.DOUBLE_FILLVAL])
    expected_pos = np.array([0])

    # Act
    single_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(single_de.TIME, expected_time)
    np.testing.assert_array_equal(single_de.ENERGY, expected_energy)
    np.testing.assert_array_equal(single_de.MODE, expected_mode)
    np.testing.assert_array_equal(single_de.TOF0, expected_tof0)
    np.testing.assert_array_equal(single_de.TOF1, expected_tof1)
    np.testing.assert_array_equal(single_de.TOF2, expected_tof2)
    np.testing.assert_array_equal(single_de.TOF3, expected_tof3)
    np.testing.assert_array_equal(single_de.CKSM, expected_cksm)
    np.testing.assert_array_equal(single_de.POS, expected_pos)


def test_decompress_data_multi_de(multi_de):
    # Arrange

    # DE One
    absent_1 = "0000"  # case 0
    time_1 = "000001100100"  # 100
    energy_1 = "010"  # 2
    mode_1 = "1"
    tof0_1 = "0000000000"
    # TOF1 not transmitted
    tof2_1 = "000000010"  # 2
    tof3_1 = "000011"  # 3
    cksm_1 = "000"  # 0
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

    multi_de.DATA = (
        absent_1
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

    expected_time = np.array([100, 100])
    expected_energy = np.array([2, 2])
    expected_mode = np.array([1, 1])
    # tofs and cksm are bit shifted to the left by 1 during decompression
    expected_tof0 = np.array([0 << 1, GlobalConstants.DOUBLE_FILLVAL])
    expected_tof1 = np.array([GlobalConstants.DOUBLE_FILLVAL, 1 << 1])
    expected_tof2 = np.array([2 << 1, GlobalConstants.DOUBLE_FILLVAL])
    expected_tof3 = np.array([3 << 1, GlobalConstants.DOUBLE_FILLVAL])
    expected_cksm = np.array([0 << 1, GlobalConstants.DOUBLE_FILLVAL])
    expected_pos = np.array([GlobalConstants.DOUBLE_FILLVAL, 0])

    # Act
    multi_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(multi_de.TIME, expected_time)
    np.testing.assert_array_equal(multi_de.ENERGY, expected_energy)
    np.testing.assert_array_equal(multi_de.MODE, expected_mode)
    np.testing.assert_array_equal(multi_de.TOF0, expected_tof0)
    np.testing.assert_array_equal(multi_de.TOF1, expected_tof1)
    np.testing.assert_array_equal(multi_de.TOF2, expected_tof2)
    np.testing.assert_array_equal(multi_de.TOF3, expected_tof3)
    np.testing.assert_array_equal(multi_de.CKSM, expected_cksm)
    np.testing.assert_array_equal(multi_de.POS, expected_pos)
