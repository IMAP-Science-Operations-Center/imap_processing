from collections import namedtuple

import bitstring
import pytest

from imap_processing.lo.l0.science_direct_events import ScienceDirectEvents

# TODO: Because I currently don't have any compressed DE data, the decompress method
# needs to be commented out and the private methods need to be called directly for
# testing. When DE data does become available, these tests will be updated and
# the need for the bitstring import will also go away.


@pytest.mark.skip(reason="no data to initialize with")
@pytest.fixture()
def de():
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    return de


@pytest.mark.skip(reason="no data to initialize with")
@pytest.fixture()
def tof_data():
    TOFData = namedtuple(
        "TOFData", ["ENERGY", "POS", "TOF0", "TOF1", "TOF2", "TOF3", "CKSM", "TIME"]
    )
    return TOFData


@pytest.mark.skip(reason="no data to initialize with")
def test_find_decompression_case(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    case_number_expected = 1

    # Act
    case_number = de._find_decompression_case()

    # Assert
    assert case_number == case_number_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_find_tof_decoder_for_case(de, tof_data):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    tof_decoder_expected = tof_data(3, 0, 10, 9, 9, 0, 0, 12)

    case_number = de._find_decompression_case()

    # Act
    tof_decoder = de._find_tof_decoder_for_case(case_number)

    # Assert
    assert tof_decoder == tof_decoder_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_read_tof_calculation_table(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    case_number = de._find_decompression_case()
    binary_strings_expected = {
        "ENERGY": bitstring.Bits(bin="0000000000000011"),
        "POS": bitstring.Bits(bin=""),
        "TOF0": bitstring.Bits(bin="0000011111111110"),
        "TOF1": bitstring.Bits(bin="0000001111111110"),
        "TOF2": bitstring.Bits(bin="0000001111111110"),
        "TOF3": bitstring.Bits(bin=""),
        "CKSM": bitstring.Bits(bin=""),
        "TIME": bitstring.Bits(bin="0000111111111111"),
    }

    # Act
    tof_calc_bin = de._read_tof_calculation_table(case_number)

    # Assert
    assert tof_calc_bin == binary_strings_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_find_remaining_bits(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000100010101")
    case_number = de._find_decompression_case()
    tof_calc = de._read_tof_calculation_table(case_number)
    remaining_coeff_expected = {
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
    remaining_coeff = de._find_remaining_bit_coefficients(tof_calc)

    # Assert
    assert remaining_coeff == remaining_coeff_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_parse_binary_for_gold_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000010010101001101011100111100111011101001111101")
    case_number = de._find_decompression_case()
    tof_decoder = de._find_tof_decoder_for_case(case_number)
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
    parsed_bits = de._parse_binary(case_number, tof_decoder)

    # Assert
    assert parsed_bits == parsed_bits_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_parse_binary_for_silver_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(
        bin="000000010101001101011100111100111011101001111101101101"
    )
    case_number = de._find_decompression_case()
    tof_decoder = de._find_tof_decoder_for_case(case_number)
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
    parsed_bits = de._parse_binary(case_number, tof_decoder)

    # Assert
    assert parsed_bits == parsed_bits_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_parse_binary_for_bronze_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="0100100101010011010111001111001")
    case_number = de._find_decompression_case()
    tof_decoder = de._find_tof_decoder_for_case(case_number)
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
    parsed_bits = de._parse_binary(case_number, tof_decoder)

    # Assert
    assert parsed_bits == parsed_bits_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_parse_binary_for_not_bronze_triple(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="010000010101001101011100111100101110")
    case_number = de._find_decompression_case()
    tof_decoder = de._find_tof_decoder_for_case(case_number)
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
    parsed_bits = de._parse_binary(case_number, tof_decoder)

    # Assert
    assert parsed_bits == parsed_bits_expected


@pytest.mark.skip(reason="no data to initialize with")
def test_decode_fields(de):
    # Arrange
    de.DATA = bitstring.Bits(bin="000010010101001101011100111100111011101001111101")
    case_number = de._find_decompression_case()
    tof_decoder = de._find_tof_decoder_for_case(case_number)
    tof_calc = de._read_tof_calculation_table(case_number)
    remaining_coeffs = de._find_remaining_bit_coefficients(tof_calc)
    parsed_bits = de._parse_binary(case_number, tof_decoder)

    energy_expected = 0.16
    position_expected = 0
    tof0_expected = 106.46
    tof1_expected = 0
    tof2_expected = 73.92
    tof3_expected = 12.48
    time_expected = 429.5

    # Act
    de._decode_fields(remaining_coeffs, parsed_bits)

    # Assert
    assert de.ENERGY == energy_expected
    assert de.POS == position_expected
    assert de.TOF0 == tof0_expected
    assert de.TOF1 == tof1_expected
    assert de.TOF2 == tof2_expected
    assert de.TOF3 == tof3_expected
    assert de.TIME == time_expected
