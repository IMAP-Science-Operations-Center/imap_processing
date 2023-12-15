import bitstring

from imap_processing.lo.l0.science_direct_events import ScienceDirectEvents

# TODO: These unit tests don't call the public function for the ScienceDirectEvents
# class and are also only able to run if the contects of __init__ are commented out
# because I currently don't have any compressed DE data. As a workaround, I am
# directly setting the DATA attribute and calling and testing the private methods
# as needed. These tests will be updated when data becomes available. When that happens
# the need for the bitstring import will also go away.


def test_find_decompression_case():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = bitstring.Bits(bin="000100010101")
    case_number_expected = 1

    # Act
    de._find_decompression_case()
    case_number_true = de.case_number

    # Assert
    assert case_number_true == case_number_expected


def test_find_tof_decoder_for_case():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    tof_decoder_true = de.tof_decoder

    # Assert
    assert tof_decoder_true == tof_decoder_expected


def test_read_tof_calculation_table():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    binary_strings_true = de.tof_calculation_values

    # Assert
    assert binary_strings_true == binary_strings_expected


def test_find_remaining_bits():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    remaining_bits_true = de.remaining_bits

    # Assert
    assert remaining_bits_true == remaining_bits_expected


def test_parse_binary_for_gold_triple():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    parsed_bits_true = de.parsed_bits

    # Assert
    assert parsed_bits_true == parsed_bits_expected


def test_parse_binary_for_silver_triple():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    parsed_bits_true = de.parsed_bits

    # Assert
    assert parsed_bits_true == parsed_bits_expected


def test_parse_binary_for_bronze_triple():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    parsed_bits_true = de.parsed_bits

    # Assert
    assert parsed_bits_true == parsed_bits_expected


def test_parse_binary_for_not_bronze_triple():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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
    parsed_bits_true = de.parsed_bits

    # Assert
    assert parsed_bits_true == parsed_bits_expected


def test_decode_fields():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
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

    energy_true = de.ENERGY
    position_true = de.POS
    tof0_true = de.TOF0
    tof1_true = de.TOF1
    tof2_true = de.TOF2
    tof3_true = de.TOF3
    time_true = de.TIME

    # Assert
    assert energy_true == energy_expected
    assert position_true == position_expected
    assert tof0_true == tof0_expected
    assert tof1_true == tof1_expected
    assert tof2_true == tof2_expected
    assert tof3_true == tof3_expected
    assert time_true == time_expected
