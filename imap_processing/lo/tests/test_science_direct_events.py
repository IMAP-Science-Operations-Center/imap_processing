import bitstring
import pytest

from imap_processing.lo.l0.science_direct_events import ScienceDirectEvents

# TODO: Because I currently don't have any compressed DE data, the decompress method
# needs to be commented out and the private methods need to be called directly for
# testing. When DE data does become available, these tests will be updated and
# the need for the bitstring import will also go away.


@pytest.fixture()
def de():
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    return de


def test_find_decompression_case(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    case_number_expected = 1

    # Act
    de._find_decompression_case()

    # Assert
    assert de.case_number == case_number_expected


def test_find_tof_decoder_for_case(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    tof_decoder_expected = {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 10,
        "TOF1": 9,
        "TOF2": 9,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    }
    de._find_decompression_case()

    # Act
    de._find_tof_decoder_for_case()

    # Assert
    assert de.tof_decoder == tof_decoder_expected


def test_read_tof_calculation_table(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    de._find_decompression_case()
    binary_strings_expected = {
        "TIME": bitstring.Bits(bin="0000111111111111"),
        "ENERGY": bitstring.Bits(bin="0000000000000011"),
        "TOF0": bitstring.Bits(bin="0000011111111110"),
        "TOF1": bitstring.Bits(bin="0000001111111110"),
        "TOF2": bitstring.Bits(bin="0000001111111110"),
        "TOF3": bitstring.Bits(bin=""),
        "POS": bitstring.Bits(bin=""),
        "CKSM": bitstring.Bits(bin=""),
    }

    # Act
    de._read_tof_calculation_table()

    # Assert
    assert de.tof_calculation_binary == binary_strings_expected


def test_find_remaining_bits(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    remaining_bits_expected = {
        "TIME": [
            327.68,
            163.84,
            81.82,
            40.96,
            20.48,
            10.24,
            5.12,
            2.56,
            1.28,
            0.64,
            0.32,
            0.16,
        ],
        "ENERGY": [
            0.32,
            0.16,
        ],
        "TOF0": [
            163.84,
            81.82,
            40.96,
            20.48,
            10.24,
            5.12,
            2.56,
            1.28,
            0.64,
            0.32,
        ],
        "TOF1": [
            81.82,
            40.96,
            20.48,
            10.24,
            5.12,
            2.56,
            1.28,
            0.64,
            0.32,
        ],
        "TOF2": [
            81.82,
            40.96,
            20.48,
            10.24,
            5.12,
            2.56,
            1.28,
            0.64,
            0.32,
        ],
        "TOF3": [],
        "POS": [],
        "CKSM": [],
    }

    # Act
    de._find_remaining_bits()

    # Assert
    assert de.remaining_bits == remaining_bits_expected


def test_parse_binary_for_gold_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000010010101001101011100111100111011101001111101")
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    parsed_bits_expected = {
        "ENERGY": bitstring.Bits(bin="001"),
        "POS": bitstring.Bits(bin=""),
        "TOF0": bitstring.Bits(bin="0101001101"),
        "TOF1": bitstring.Bits(bin=""),
        "TOF2": bitstring.Bits(bin="011100111"),
        "TOF3": bitstring.Bits(bin="100111"),
        "CKSM": bitstring.Bits(bin="011"),
        "TIME": bitstring.Bits(bin="101001111101"),
    }

    # Act
    de._parse_binary()

    # Assert
    assert de.parsed_bits == parsed_bits_expected


def test_parse_binary_for_silver_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(
        bin="000000010101001101011100111100111011101001111101101101"
    )
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    parsed_bits_expected = {
        "ENERGY": bitstring.Bits(bin="001"),
        "POS": bitstring.Bits(bin=""),
        "TOF0": bitstring.Bits(bin="0101001101"),
        "TOF1": bitstring.Bits(bin="011100111"),
        "TOF2": bitstring.Bits(bin="100111011"),
        "TOF3": bitstring.Bits(bin="101001"),
        "CKSM": bitstring.Bits(bin=""),
        "TIME": bitstring.Bits(bin="111101101101"),
    }

    # Act
    de._parse_binary()

    # Assert
    assert de.parsed_bits == parsed_bits_expected


def test_parse_binary_for_bronze_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="0100100101010011010111001111001")
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    parsed_bits_expected = {
        "ENERGY": bitstring.Bits(bin="001"),
        "POS": bitstring.Bits(bin="01"),
        "TOF0": bitstring.Bits(bin="0100110101"),
        "TOF1": bitstring.Bits(bin=""),
        "TOF2": bitstring.Bits(bin=""),
        "TOF3": bitstring.Bits(bin=""),
        "CKSM": bitstring.Bits(bin=""),
        "TIME": bitstring.Bits(bin="11001111001"),
    }

    # Act
    de._parse_binary()

    # Assert
    assert de.parsed_bits == parsed_bits_expected


def test_parse_binary_for_not_bronze_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="010000010101001101011100111100101110")
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    parsed_bits_expected = {
        "ENERGY": bitstring.Bits(bin="001"),
        "POS": bitstring.Bits(bin=""),
        "TOF0": bitstring.Bits(bin="0101001101"),
        "TOF1": bitstring.Bits(bin=""),
        "TOF2": bitstring.Bits(bin=""),
        "TOF3": bitstring.Bits(bin="011100"),
        "CKSM": bitstring.Bits(bin=""),
        "TIME": bitstring.Bits(bin="111100101110"),
    }

    # Act
    de._parse_binary()

    # Assert
    assert de.parsed_bits == parsed_bits_expected


def test_decode_fields(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000010010101001101011100111100111011101001111101")
    de._find_decompression_case()
    de._find_tof_decoder_for_case()
    de._read_tof_calculation_table()
    de._find_remaining_bits()
    de._parse_binary()

    energy_expected = 0.16
    position_expected = 0
    tof0_expected = 106.46
    tof1_expected = 0
    tof2_expected = 73.92
    tof3_expected = 12.48
    time_expected = 429.5

    # Act
    de._decode_fields()

    # Assert
    assert de.ENERGY == energy_expected
    assert de.POS == position_expected
    assert de.TOF0 == tof0_expected
    assert de.TOF1 == tof1_expected
    assert de.TOF2 == tof2_expected
    assert de.TOF3 == tof3_expected
    assert de.TIME == time_expected
