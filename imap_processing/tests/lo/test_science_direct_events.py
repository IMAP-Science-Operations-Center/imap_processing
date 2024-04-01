from collections import namedtuple

import numpy as np
import pytest

from imap_processing.lo.l0.data_classes.science_direct_events import (
    ScienceDirectEventsPacket,
)
from imap_processing.lo.l0.utils.binary_string import BinaryString


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
    de = ScienceDirectEventsPacket(fake_packet_data, "0", "fakepacketname")
    de.COUNT = 1
    return de


@pytest.fixture()
def multi_de(fake_packet_data):
    de = ScienceDirectEventsPacket(fake_packet_data, "0", "fakepacketname")
    de.COUNT = 2
    return de


@pytest.fixture()
def tof_data():
    TOFData = namedtuple(
        "TOFData", ["ENERGY", "POS", "TOF0", "TOF1", "TOF2", "TOF3", "CKSM", "TIME"]
    )
    return TOFData


def test_decompression_case(single_de):
    # Arrange
    single_de.DATA = "000100010101"
    case_number_expected = 1
    data = BinaryString(single_de.DATA)

    # Act
    case_number = single_de._decompression_case(data)

    # Assert
    assert case_number == case_number_expected


def test_case_decoder_variant_1(single_de, tof_data):
    # Arrange
    single_de.DATA = "000010010101"
    data = BinaryString(single_de.DATA)
    # at this point, the first 4 bits will have already been read
    # to get the case number
    data.bit_pos = 4
    tof_decoder_expected = tof_data(3, 0, 10, 0, 9, 6, 3, 12)
    case_number = 0

    # Act
    tof_decoder = single_de._case_decoder(case_number, data)

    # Assert
    assert tof_decoder == tof_decoder_expected


def test_case_decoder_variant_0(single_de, tof_data):
    # Arrange
    single_de.DATA = "000000010101"
    data = BinaryString(single_de.DATA)
    # at this point, the first 4 bits will have already been read
    # to get the case number
    data.bit_pos = 4
    tof_decoder_expected = tof_data(3, 0, 10, 9, 9, 6, 0, 12)
    case_number = 0

    # Act
    tof_decoder = single_de._case_decoder(case_number, data)

    # Assert
    assert tof_decoder == tof_decoder_expected


def test_decompress_existing_field(single_de):
    # Arrange
    single_de.DATA = "00001001"
    data = BinaryString(single_de.DATA)
    data.bit_pos = 5
    # the energy field is packed immediately after the case and variant bits
    # so the bit_pos will be at index 5
    energy_field_length = 3
    sig_bits = np.array([0, 0.32, 0.16])
    decompressed_field_expected = 0.16

    # Act
    decompressed_field = single_de._decompress_field(
        data, energy_field_length, sig_bits
    )

    # Assert
    assert decompressed_field == decompressed_field_expected


def test_decompress_non_existing_field(single_de):
    # Arrange
    single_de.DATA = "00001001"
    data = BinaryString(single_de.DATA)
    # this is testing the position field in the binary, so the bit_pos
    # will be at position 7. Note: the bit pos is different than the position field.
    # the position field is a direct event field and the bit_pos is the current bit
    # position during the parsing of the binary string.
    data.bit_pos = 7
    pos_field_length = 0
    sig_bits = np.array([0.32, 0.16])
    decompressed_field_expected = np.float64(-1.0e31)

    # Act
    decompressed_field = single_de._decompress_field(data, pos_field_length, sig_bits)

    # Assert
    assert decompressed_field == decompressed_field_expected


def test_single_de_parse_case(single_de, tof_data):
    # Arrange
    single_de.DATA = "000010010000000011000000011000001001000000000001"
    data = BinaryString(single_de.DATA)
    # parse_case is run after the case and variant are checked,
    # so the bit_pos will be at index 5
    data.bit_pos = 5
    case_decoder = tof_data(
        3,
        0,
        10,
        0,
        9,
        6,
        3,
        12,
    )
    energy_expected = np.array([0.16])
    pos_expected = np.array([np.float64(-1.0e31)])
    tof0_expected = np.array([0.96])
    tof1_expected = np.array([np.float64(-1.0e31)])
    tof2_expected = np.array([0.96])
    tof3_expected = np.array([0.32])
    cksm_expected = np.array([0.32])
    time_expected = np.array([0.16])

    # Act
    single_de._parse_case(data, case_decoder)

    # Assert
    np.testing.assert_array_equal(single_de.ENERGY, energy_expected)
    np.testing.assert_array_equal(single_de.POS, pos_expected)
    np.testing.assert_array_equal(single_de.TOF0, tof0_expected)
    np.testing.assert_array_equal(single_de.TOF1, tof1_expected)
    np.testing.assert_array_equal(single_de.TOF2, tof2_expected)
    np.testing.assert_array_equal(single_de.TOF3, tof3_expected)
    np.testing.assert_array_equal(single_de.CKSM, cksm_expected)
    np.testing.assert_array_equal(single_de.TIME, time_expected)


def test_decompress_data(multi_de, tof_data):
    # Arrange
    multi_de.DATA = (
        "0000100100000000110000000110000010010"
        + "00000000001000010010000000011000000011000001001000000000001"
    )
    energy_expected = np.array([0.16, 0.16])
    pos_expected = np.array([np.float64(-1.0e31), np.float64(-1.0e31)])
    tof0_expected = np.array([0.96, 0.96])
    tof1_expected = np.array([np.float64(-1.0e31), np.float64(-1.0e31)])
    tof2_expected = np.array([0.96, 0.96])
    tof3_expected = np.array([0.32, 0.32])
    cksm_expected = np.array([0.32, 0.32])
    time_expected = np.array([0.16, 0.16])

    # Act
    multi_de._decompress_data()

    # Assert
    np.testing.assert_array_equal(multi_de.ENERGY, energy_expected)
    np.testing.assert_array_equal(multi_de.POS, pos_expected)
    np.testing.assert_array_equal(multi_de.TOF0, tof0_expected)
    np.testing.assert_array_equal(multi_de.TOF1, tof1_expected)
    np.testing.assert_array_equal(multi_de.TOF2, tof2_expected)
    np.testing.assert_array_equal(multi_de.TOF3, tof3_expected)
    np.testing.assert_array_equal(multi_de.CKSM, cksm_expected)
    np.testing.assert_array_equal(multi_de.TIME, time_expected)
