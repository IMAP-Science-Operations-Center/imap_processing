from collections import namedtuple

import numpy as np
import pytest

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
            "DE_COUNT": fake_data_field(0, 0),
            "DATA": fake_data_field("00", "00"),
            "CHKSUM": fake_data_field(0, 0),
        },
    )


@pytest.fixture()
def single_de(fake_packet_data):
    de = ScienceDirectEvents(fake_packet_data, "0", "fakepacketname")
    de.DE_COUNT = 1
    de.DE_TIME = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.ESA_STEP = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.MODE = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF0 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF1 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF2 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF3 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.CKSM = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.POS = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    return de


@pytest.fixture()
def multi_de(fake_packet_data):
    de = ScienceDirectEvents(fake_packet_data, "0", "fakepacketname")
    de.DE_COUNT = 2
    de.DE_TIME = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.ESA_STEP = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.MODE = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF0 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF1 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF2 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.TOF3 = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.CKSM = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    de.POS = np.ones(de.DE_COUNT) * np.float64(-1.0e31)
    return de


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
    expected_tof1 = np.array([np.float64(-1.0e31)])
    expected_tof2 = np.array([2 << 1])
    expected_tof3 = np.array([3 << 1])
    expected_cksm = np.array([0 << 1])
    expected_pos = np.array([np.float64(-1.0e31)])

    # Act
    single_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(single_de.DE_TIME, expected_time)
    np.testing.assert_array_equal(single_de.ESA_STEP, expected_energy)
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
    expected_tof0 = np.array([np.float64(-1.0e31)])
    # tofs and cksm are bit shifted to the left by 1 during decompression
    expected_tof1 = np.array([1 << 1])
    expected_tof2 = np.array([np.float64(-1.0e31)])
    expected_tof3 = np.array([np.float64(-1.0e31)])
    expected_cksm = np.array([np.float64(-1.0e31)])
    expected_pos = np.array([0])

    # Act
    single_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(single_de.DE_TIME, expected_time)
    np.testing.assert_array_equal(single_de.ESA_STEP, expected_energy)
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
    expected_tof0 = np.array([0 << 1, np.float64(-1.0e31)])
    expected_tof1 = np.array([np.float64(-1.0e31), 1 << 1])
    expected_tof2 = np.array([2 << 1, np.float64(-1.0e31)])
    expected_tof3 = np.array([3 << 1, np.float64(-1.0e31)])
    expected_cksm = np.array([0 << 1, np.float64(-1.0e31)])
    expected_pos = np.array([np.float64(-1.0e31), 0])

    # Act
    multi_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(multi_de.DE_TIME, expected_time)
    np.testing.assert_array_equal(multi_de.ESA_STEP, expected_energy)
    np.testing.assert_array_equal(multi_de.MODE, expected_mode)
    np.testing.assert_array_equal(multi_de.TOF0, expected_tof0)
    np.testing.assert_array_equal(multi_de.TOF1, expected_tof1)
    np.testing.assert_array_equal(multi_de.TOF2, expected_tof2)
    np.testing.assert_array_equal(multi_de.TOF3, expected_tof3)
    np.testing.assert_array_equal(multi_de.CKSM, expected_cksm)
    np.testing.assert_array_equal(multi_de.POS, expected_pos)
