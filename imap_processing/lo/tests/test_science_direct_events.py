from imap_processing.lo.l0.science_direct_events import ScienceDirectEvents

# from imap_processing import imap_module_directory


def test_find_decompression_case():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "000100010101"
    case_number_expected = 1

    # Act
    de._find_decompression_case()
    case_number_true = de.case_number

    # Assert
    assert case_number_true == case_number_expected


def test_hexadecimal_to_binary():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "000100010101"
    binary_string_expected = "0000011111111110"

    # Act
    binary_string_true = de._hexadecimal_to_binary("0x07FE")

    # Assert
    assert binary_string_true == binary_string_expected


def test_find_bit_length_for_case():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "000100010101"
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
    de._find_bit_length_for_case()
    tof_decoder_true = de.tof_decoder

    # Assert
    assert tof_decoder_true == tof_decoder_expected


def test_find_binary_strings():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "000100010101"
    de._find_decompression_case()
    binary_strings_expected = {
        "TIME": "0000111111111111",
        "ENERGY": "0000000000000011",
        "TOF0": "0000011111111110",
        "TOF1": "0000001111111110",
        "TOF2": "0000001111111110",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    }

    # Act
    de._find_binary_strings()
    binary_strings_true = de.binary_strings

    # Assert
    assert binary_strings_true == binary_strings_expected


def test_find_remaining_bits():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "000100010101"
    de._find_decompression_case()
    de._find_binary_strings()
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


def test_parse_binary():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "0001000101010011010111001111001110111010011"
    de._find_decompression_case()
    de._find_bit_length_for_case()
    de._find_binary_strings()
    parsed_bits_expected = {
        "ENERGY": "000",
        "POS": "",
        "TOF0": "1000101010",
        "TOF1": "011010111",
        "TOF2": "001111001",
        "TOF3": "",
        "CKSM": "",
        "TIME": "110111010011",
    }

    # Act
    de._parse_binary()
    parsed_bits_true = de.parsed_bits

    # Assert
    assert parsed_bits_true == parsed_bits_expected


def test_decode_fields():
    # Arrange
    de = ScienceDirectEvents("fake_packet", "0", "fakepacketname")
    de.DATA = "00010001010100110101110011110011101110100110100"
    de._find_decompression_case()
    de._find_bit_length_for_case()
    de._find_binary_strings()
    de._parse_binary()

    # Act

    # Assert
