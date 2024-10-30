from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imap_processing.mag.l0.decom_mag import decom_packets
from imap_processing.mag.l1a.mag_l1a import mag_l1a, process_packets
from imap_processing.mag.l1a.mag_l1a_data import (
    MAX_FINE_TIME,
    MagL1a,
    MagL1aPacketProperties,
    TimeTuple,
)
from imap_processing.spice.time import met_to_j2000ns


@pytest.fixture()
def uncompressed_vector_bytearray():
    input_data = np.array(
        [
            2,
            4,
            8,
            16,
            16,
            32,
            192,
            129,
            194,
            7,
            68,
            14,
            176,
            32,
            160,
            130,
            161,
            5,
            76,
            8,
            52,
            32,
            220,
            65,
            191,
            2,
            17,
            8,
            68,
            16,
            137,
            192,
            133,
            2,
            20,
            132,
            41,
            48,
            33,
            112,
            133,
            241,
            11,
            236,
            8,
            108,
            33,
            176,
            67,
            99,
            2,
            30,
            8,
            121,
            16,
            243,
            192,
            136,
            66,
            33,
            132,
            67,
            112,
            34,
            64,
            137,
            49,
            18,
            124,
            8,
            160,
            34,
            132,
            69,
            11,
            2,
            43,
            8,
            174,
            17,
            92,
            192,
            139,
            130,
            46,
            196,
            93,
            176,
            35,
            32,
            140,
            129,
            25,
            28,
            8,
            212,
            35,
            84,
            70,
            175,
            2,
            3,
            8,
            15,
            16,
            31,
            192,
            129,
            194,
            7,
            4,
            14,
            112,
            32,
            160,
            130,
            161,
            5,
            76,
            8,
            52,
            32,
            220,
            65,
            187,
            2,
            17,
            8,
            68,
            16,
            136,
            192,
            133,
            2,
            20,
            68,
            40,
            240,
            33,
            112,
            133,
            225,
            11,
            220,
            8,
            104,
            33,
            172,
            67,
            95,
            2,
            30,
            8,
            121,
            16,
            242,
            192,
            136,
            66,
            33,
            132,
            67,
            48,
            34,
            64,
            137,
            49,
            18,
            124,
            8,
            160,
            34,
            128,
            69,
            7,
            2,
            43,
            8,
            173,
            17,
            91,
            192,
            139,
            130,
            46,
            196,
            93,
            176,
            35,
            32,
            140,
            129,
            25,
            12,
            8,
            212,
            35,
            84,
            70,
            171,
        ],
        dtype=np.uint32,
    )
    return input_data


@pytest.fixture()
def expected_vectors():
    primary_expected = np.array(
        [
            [516, 2064, 4128, 3],
            [519, 2077, 4154, 3],
            [522, 2090, 4180, 3],
            [525, 2103, 4207, 3],
            [529, 2116, 4233, 3],
            [532, 2130, 4260, 3],
            [535, 2143, 4286, 3],
            [539, 2156, 4312, 3],
            [542, 2169, 4339, 3],
            [545, 2182, 4365, 3],
            [548, 2195, 4391, 3],
            [552, 2209, 4418, 3],
            [555, 2222, 4444, 3],
            [558, 2235, 4470, 3],
            [562, 2248, 4497, 3],
            [565, 2261, 4523, 3],
        ]
    )

    secondary_expected = np.array(
        [
            [515, 2063, 4127, 3],
            [519, 2076, 4153, 3],
            [522, 2090, 4180, 3],
            [525, 2103, 4206, 3],
            [529, 2116, 4232, 3],
            [532, 2129, 4259, 3],
            [535, 2142, 4285, 3],
            [538, 2155, 4311, 3],
            [542, 2169, 4338, 3],
            [545, 2182, 4364, 3],
            [548, 2195, 4391, 3],
            [552, 2208, 4417, 3],
            [555, 2221, 4443, 3],
            [558, 2235, 4470, 3],
            [562, 2248, 4496, 3],
            [565, 2261, 4522, 3],
        ]
    )

    return (primary_expected, secondary_expected)


@pytest.fixture()
def raw_compressed_vectors():
    # compressed vectors, without the first starting uncompressed vector.
    # 15 primary vectors and 15 secondary vectors, corresponding to most of
    # the vectors in expected_vectors.
    primary_compressed = (
        "0101110010"
        "011100101011010111001001110010101101011100100110000000011100"
        "011100100111001010110101100001011000000001101011100100111001"
        "010111000111001001110010101101011100100110000000011010111001"
        "001110010101101011100100111001010111000110000101100000000110"
        "101110010011100101011010111001001110010101110001110010011000"
        "00000110101110010011100101011"
    )

    secondary_compressed = (
        "10001110"
        "0100111001010110101100001011000000001101011100100111001010"
        "1110001110010011100101011010111001001100000000110101110010"
        "0111001010110101110010011100101011100011000010110000000011"
        "0101110010011100101011010111001001100000000111000111001001"
        "1100101011010111001001110010101101011000010110000000011100"
        "011100100111001010110101110010011100101011"
    )
    return primary_compressed, secondary_compressed


def test_different_vector_rates(
    uncompressed_vector_bytearray, expected_vectors, raw_compressed_vectors
):
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l0 = decom_packets(test_file)["norm"][0]

    # overwrite vectors and different vector rates

    l0.PRI_VECSEC = 4  # twice as many primary vectors as secondary vectors - 32 vectors
    l0.VECTORS = np.concatenate(
        (
            uncompressed_vector_bytearray[:100],
            uncompressed_vector_bytearray[:100],
            uncompressed_vector_bytearray[100:],
        )
    )
    l1 = process_packets([l0])
    expected_day = np.datetime64("2023-11-30")

    assert len(l1["magi"][expected_day].vectors) == 16
    assert len(l1["mago"][expected_day].vectors) == 32

    assert np.array_equal(
        l1["mago"][expected_day].vectors[:, :4],
        np.concatenate((expected_vectors[0], expected_vectors[0])),
    )
    assert np.array_equal(l1["magi"][expected_day].vectors[:, :4], expected_vectors[1])

    # compressed data
    # Compression headers - indicating a 16 bit width and no range section
    headers = "01000000"

    # 50 bits each - 16 bits per vector value, 2 bits for range.
    first_primary_vector = "00000010000001000000100000010000000100000010000011"
    first_secondary_vector = "00000010000000110000100000001111000100000001111111"

    primary_compressed = (
        first_primary_vector + raw_compressed_vectors[0] + raw_compressed_vectors[0]
    )
    secondary_compressed = first_secondary_vector + raw_compressed_vectors[1]

    input_data = np.array(
        [int(i) for i in headers + primary_compressed + secondary_compressed],
        dtype=np.uint8,
    )

    # Will be the input data format
    input_data = np.packbits(input_data)

    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 31, 16)

    assert (primary > 0).all()
    assert np.array_equal(primary[:16], expected_vectors[0])
    assert np.array_equal(secondary, expected_vectors[1])


def test_padding_uncompressed(expected_vectors):
    # Test if the padding falls directly on a byte boundary
    headers = "01000000"

    # 50 bits each - 16 bits per vector value, 2 bits for range.
    first_primary_vector = "00000010000001000000100000010000000100000010000011"
    first_secondary_vector = "00000010000000110000100000001111000100000001111111"

    primary_vectors = "11111111111011"
    secondary_vectors = "11111111111011"

    last_vector = expected_vectors[0][0].copy()
    last_vector[2] = last_vector[2] - 2

    input_data = np.array(
        [
            int(i)
            for i in headers
            + first_primary_vector
            + primary_vectors
            + first_secondary_vector
            + secondary_vectors
        ],
        dtype=np.uint8,
    )

    # Will be the input data format
    input_data = np.packbits(input_data)

    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 3, 3)

    assert np.array_equal(primary[0], expected_vectors[0][0])
    assert np.array_equal(primary[1], expected_vectors[0][0])
    assert np.array_equal(primary[2], last_vector)

    last_vector = expected_vectors[1][0].copy()
    last_vector[2] = last_vector[2] - 2

    assert np.array_equal(secondary[0], expected_vectors[1][0])
    assert np.array_equal(secondary[1], expected_vectors[1][0])
    assert np.array_equal(secondary[2], last_vector)


def test_compare_validation_data():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l0 = decom_packets(test_file)
    l1 = process_packets(l0["norm"])
    # Should have one day of data
    expected_day = np.datetime64("2023-11-30")
    l1_mago = l1["mago"][expected_day]
    l1_magi = l1["magi"][expected_day]

    assert len(l1_mago.vectors) == 96
    assert len(l1_magi.vectors) == 96

    validation_data = pd.read_csv(current_directory / "mag_l1a_test_output.csv")

    # Validation data does not have differing timestamps
    for index in validation_data.index:
        # Sequence in validation data starts at 5
        # Mago is primary, Magi is secondary in test data

        assert l1_mago.vectors[index][0] == validation_data["x_pri"][index]
        assert l1_mago.vectors[index][1] == validation_data["y_pri"][index]
        assert l1_mago.vectors[index][2] == validation_data["z_pri"][index]
        assert l1_mago.vectors[index][3] == validation_data["rng_pri"][index]

        assert l1_magi.vectors[index][0] == validation_data["x_sec"][index]
        assert l1_magi.vectors[index][1] == validation_data["y_sec"][index]
        assert l1_magi.vectors[index][2] == validation_data["z_sec"][index]
        assert l1_magi.vectors[index][3] == validation_data["rng_sec"][index]


def test_compressed_vector_data(expected_vectors, raw_compressed_vectors):
    # Values from test packet
    primary_expected = expected_vectors[0]
    secondary_expected = expected_vectors[1]

    # Compression headers - indicating a 16 bit width and no range section
    headers = "01000000"

    # 50 bits each - 16 bits per vector value, 2 bits for range.
    first_primary_vector = "00000010000001000000100000010000000100000010000011"
    first_secondary_vector = "00000010000000110000100000001111000100000001111111"

    primary_compressed = first_primary_vector + raw_compressed_vectors[0]
    secondary_compressed = first_secondary_vector + raw_compressed_vectors[1]

    padding = "00000"  # Pad to byte boundary
    input_data = np.array(
        [int(i) for i in headers + primary_compressed + secondary_compressed + padding],
        dtype=np.uint8,
    )

    # Will be the input data format
    input_data = np.packbits(input_data)

    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 16, 16)

    assert np.array_equal(primary[0], primary_expected[0])
    assert np.array_equal(secondary[0], secondary_expected[0])

    # There should be 16 vectors for both primary and secondary, with 4 values per
    # vector.
    assert primary.shape[0] == 16
    assert secondary.shape[0] == 16

    assert primary.shape[1] == 4
    assert secondary.shape[1] == 4

    assert np.array_equal(primary, primary_expected)
    assert np.array_equal(secondary, secondary_expected)

    # range data has 2 bits per vector, primary and then secondary in sequence.
    # It excludes the first vector, making the length (primary_count - 1) * 2
    range_primary = "000000000000000101101011111111"

    # This includes the first range value and all the ranges from range_primary
    expected_range_primary = [3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
    range_secondary = "000000000000000101101011111101"
    expected_range_secondary = [3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 1]
    # 16 bit width with range section
    headers = "01000010"
    input_data = np.array(
        [
            int(i)
            for i in headers
            + primary_compressed
            + secondary_compressed
            + padding
            + range_primary
            + range_secondary
        ],
        dtype=np.uint8,
    )

    # In this step, input_data is automatically padded to a byte boundary by adding
    # zeros to the end
    input_data = np.packbits(input_data)

    for i in range(len(expected_range_primary)):
        primary_expected[i][3] = expected_range_primary[i]
        secondary_expected[i][3] = expected_range_secondary[i]

    (primary_with_range, secondary_with_range) = MagL1a.process_compressed_vectors(
        input_data, 16, 16
    )

    assert primary_with_range.shape[0] == 16
    assert secondary_with_range.shape[0] == 16

    assert np.array_equal(primary_with_range, primary_expected)
    assert np.array_equal(secondary_with_range, secondary_expected)


def test_switch_to_uncompressed_vector_data(
    expected_vectors, uncompressed_vector_bytearray
):
    primary_compressed = (
        "000000100000010000001000000100000001000000100000110101110010"
        "011100101011010111001001110010101101011100100110000000011100"
        "011100100111001010110101100001011000000001101011100100111001"
        "010111000111001001110010101101011100100110000000011010111001"
        "001110010101101011100100111001010111000110000101100000000110"
        "101110010011100101011010111001001110010101110001110010011000"
        "000001100000000000000000000010111000000000000000000000010011"
        "000000000000000000000000100101011"
    )

    # 4 uncompressed vectors from uncompressed_vector_bytearray
    uncompressed_bits = (
        "000000100000010000001000000100000001000000100000"
        "000000100000011100001000000111010001000000111010"
        "000000100000101000001000001010100001000001010100"
        "000000100000110100001000001101110001000001101111"
    )

    secondary_compressed = (
        "0000001000000011000010000000111100010000000111111110001110"
        "0100111001010110101100001011000000001101011100100111001010"
        "1110001110010011100101011010111001001100000000110101110010"
        "0111001010110101110010011100101011100011000010110000000011"
        "0101110010011100101011010111001001100000000111000111001001"
        "1100101011010111001001110010101101011000010110000000011100"
        "0111001001110010101100000000000000101110000000000000000000"
        "00010011100101000000000011"
    )

    uncompressed_expected_vectors = expected_vectors[0][:4]

    headers = "01000000"

    input_data = np.array(
        [
            int(i)
            for i in headers
            + primary_compressed
            + uncompressed_bits
            + secondary_compressed
            + uncompressed_bits
        ],
        dtype=np.uint8,
    )

    input_data = np.packbits(input_data)
    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 20, 20)

    # The 16th compressed vector is bad because it needs to be >60 bits
    assert np.array_equal(primary[:15], expected_vectors[0][:-1])
    assert np.array_equal(primary[16:], uncompressed_expected_vectors)

    assert np.array_equal(secondary[:15], expected_vectors[1][:-1])
    assert np.array_equal(secondary[16:], uncompressed_expected_vectors)

    # Test if first primary vector is too long
    primary_first_vector = "00000010000001000000100000010000000100000010000011"
    primary_long_second_vector = (
        "0000000000000000000001011100000000000000000000001"
        "0011000000000000000000000000100101011"
    )

    input_data = np.array(
        [
            int(i)
            for i in headers
            + primary_first_vector
            + primary_long_second_vector
            + uncompressed_bits
            + secondary_compressed
        ],
        dtype=np.uint8,
    )
    input_data = np.packbits(input_data)

    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 6, 16)
    assert len(primary) == 6
    assert np.array_equal(primary[0], expected_vectors[0][0])
    assert np.array_equal(primary[2:], uncompressed_expected_vectors)


def test_different_compression_width(raw_compressed_vectors):
    # Compression headers - indicating a 12 bit width and no range section
    headers = "00110000"

    first_primary_vector = "00100000010010000001000000000010000011"
    first_secondary_vector = "00000001011000000000000011111111111101"

    primary_compressed = raw_compressed_vectors[0]
    secondary_compressed = raw_compressed_vectors[1]

    expected_first_vector = [516, -2032, 32, 3]
    expected_second_vector = [22, 0, -1, 1]

    padding = "00000"  # Pad to byte boundary

    input_data = np.array(
        [
            int(i)
            for i in headers
            + first_primary_vector
            + primary_compressed
            + first_secondary_vector
            + secondary_compressed
            + padding
        ],
        dtype=np.uint8,
    )

    input_data = np.packbits(input_data)
    (primary, secondary) = MagL1a.process_compressed_vectors(input_data, 16, 16)

    assert np.array_equal(primary[0], expected_first_vector)
    assert np.array_equal(secondary[0], expected_second_vector)

    assert sum(primary[-1]) != 0
    assert sum(secondary[-1]) != 0

    assert len(primary) == 16
    assert len(secondary) == 16


def test_real_uncompressed_vector_data(uncompressed_vector_bytearray, expected_vectors):
    primary_expected = expected_vectors[0]
    secondary_expected = expected_vectors[1]

    (primary, secondary) = MagL1a.process_uncompressed_vectors(
        uncompressed_vector_bytearray, 16, 16
    )
    assert np.array_equal(primary_expected, primary)
    assert np.array_equal(secondary_expected, secondary)


def test_accumulate_vectors():
    range = 4
    start_vector = np.array([1, 2, 3, range], dtype=np.uint)

    diff_vectors = [1, 1, 1, 3, 0, -3, -1, -10, 1]

    expected_vectors = np.array(
        [[1, 2, 3, range], [2, 3, 4, range], [5, 3, 1, range], [4, -7, 2, range]]
    )

    test_vectors = MagL1a.convert_diffs_to_vectors(start_vector, diff_vectors, 4)

    assert np.array_equal(test_vectors, expected_vectors)


sixteen_bits = "00000010000001000000100000010000000100000010000011"
twelve_bits = "000000010110000000000000111111111111"
eighteenbits = "000000100000010010000010000001000011000100000010000001"
twentybits = "00000010000001000010000010000001000000000000000000100000000101"


@pytest.mark.parametrize(
    "vector_string, expected_vectors, width, include_range",
    [
        (sixteen_bits, [516, 2064, 4128, 3], 16, 1),
        (twelve_bits, [22, 0, -1, 0], 12, 0),
        (eighteenbits, [2066, 8259, 16513, 0], 18, 0),
        (twentybits, [8258, 33024, 513, 1], 20, 1),
    ],
    ids=["16bit", "12bit", "18bit", "20bit"],
)
def test_unpack_one_vector(
    vector_string, expected_vectors, uncompressed_vector_bytearray, width, include_range
):
    test_vector = np.array([int(i) for i in vector_string], dtype=np.uint8)
    test_output = MagL1a.unpack_one_vector(test_vector, width, include_range)
    assert all(test_output == expected_vectors)


def test_twos_complement():
    # -19 in binary
    input_test = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    input_test_uint = np.packbits(input_test)

    twos_complement = MagL1a.twos_complement(input_test_uint, 8)
    assert twos_complement == -19
    assert twos_complement.dtype == np.int32

    # In 12 bits, the number is 237
    twos_complement = MagL1a.twos_complement(input_test_uint, 12)
    assert twos_complement == 237

    # Higher bit number
    # -19001 in 16 bits
    input_test = np.array(
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8
    )
    input_test_uint = np.packbits(input_test)
    twos_complement = MagL1a.twos_complement(input_test_uint, 16)

    assert twos_complement == -19001


def test_decode_fib_zig_zag():
    test_values = np.array([1, 0, 0, 1, 0, 0, 1, 1])
    assert MagL1a.decode_fib_zig_zag(test_values) == 13

    test_values = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    assert MagL1a.decode_fib_zig_zag(test_values) == -138


def test_process_uncompressed_vector_data():
    expected_vector_data = [[1001, 1002, -3001, 3], [2001, -2002, -3333, 1]]

    # 100 bits, created by hand by appending all bits from expected_vector_data into one
    # hex string, with range being 2 bits (so the second half is offset from the hex
    # values)
    hex_string = "03E903EAF447C1F47E0BBCBED0"
    input_data = np.frombuffer(bytes.fromhex(hex_string), dtype=np.dtype(">b"))
    total_primary_vectors = 1
    total_secondary_vectors = 1

    # 36 bytes
    (primary_vectors, secondary_vectors) = MagL1a.process_vector_data(
        input_data, total_primary_vectors, total_secondary_vectors, 0
    )

    assert primary_vectors[0][0] == expected_vector_data[0][0]
    assert primary_vectors[0][1] == expected_vector_data[0][1]
    assert primary_vectors[0][2] == expected_vector_data[0][2]
    assert primary_vectors[0][3] == expected_vector_data[0][3]

    assert secondary_vectors[0][0] == expected_vector_data[1][0]
    assert secondary_vectors[0][1] == expected_vector_data[1][1]
    assert secondary_vectors[0][2] == expected_vector_data[1][2]
    assert secondary_vectors[0][3] == expected_vector_data[1][3]


def test_time_tuple():
    example_time_tuple = TimeTuple(439067318, 64618)

    test_add = example_time_tuple + 2

    assert test_add == TimeTuple(439067320, 64618)

    # 1 / MAX_FINE_TIME
    test_add = example_time_tuple + 1 / MAX_FINE_TIME

    assert test_add == TimeTuple(439067318, 64619)

    test_add = example_time_tuple + (1000 / MAX_FINE_TIME)

    assert test_add == TimeTuple(439067319, 83)


def test_calculate_vector_time():
    test_vectors = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint
    )
    test_vecsec = 2
    start_time = TimeTuple(10000, 0)

    test_data = MagL1a.calculate_vector_time(test_vectors, test_vecsec, start_time)

    converted_start_time_ns = met_to_j2000ns(start_time.to_seconds())

    skips_ns = np.timedelta64(int(1 / test_vecsec * 1e9), "ns")
    expected_data = np.array(
        [
            [1, 2, 3, int(converted_start_time_ns)],
            [4, 5, 6, int(converted_start_time_ns + skips_ns)],
            [7, 8, 9, int(converted_start_time_ns + skips_ns * 2)],
            [10, 11, 12, int(converted_start_time_ns + skips_ns * 3)],
        ]
    )
    assert (test_data == expected_data).all()


def test_mag_l1a_data():
    test_vectors = np.array(
        [
            [1, 2, 3, 4, 10000],
            [5, 6, 7, 8, 10050],
            [9, 10, 11, 12, 10100],
            [13, 13, 11, 12, 10150],
        ],
        dtype=np.uint,
    )
    test_vecsec = 2
    start_time = TimeTuple(10000, 0)

    packet_properties = MagL1aPacketProperties(
        435954628, start_time, test_vecsec, 1, 0, 0, 1, 0
    )
    mag_l1a = MagL1a(True, True, 10000, test_vectors, packet_properties)

    new_vectors = np.array(
        [[13, 14, 15, 16, 10400], [16, 17, 18, 19, 10450]], dtype=np.uint
    )

    new_seq = 5
    compression = 1
    # equal to 01001011, should resolve to a width of 18 (first 6 bits)
    compression_width_header = 75
    new_properties = MagL1aPacketProperties(
        435954628,
        TimeTuple(10400, 0),
        test_vecsec,
        0,
        new_seq,
        compression,
        1,
        compression_width_header,
    )
    mag_l1a.append_vectors(new_vectors, new_properties)

    assert np.array_equal(
        mag_l1a.vectors,
        np.array(
            [
                [1, 2, 3, 4, 10000],
                [5, 6, 7, 8, 10050],
                [9, 10, 11, 12, 10100],
                [13, 13, 11, 12, 10150],
                [13, 14, 15, 16, 10400],
                [16, 17, 18, 19, 10450],
            ],
            dtype=np.uint,
        ),
    )
    assert mag_l1a.missing_sequences == [1, 2, 3, 4]

    assert mag_l1a.compression_flags.shape == (6, 2)
    assert np.array_equal(
        mag_l1a.compression_flags,
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 18], [1, 18]], dtype=np.uint8),
    )


def test_mag_l1a():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"

    output_data = mag_l1a(test_file, "v001")

    # Test data is one day's worth of NORM data, so it should return one raw, one MAGO
    # and one MAGI dataset
    assert len(output_data) == 3
    expected_logical_source = [
        "imap_mag_l1a_norm-raw",
        "imap_mag_l1a_norm-mago",
        "imap_mag_l1a_norm-magi",
    ]

    for data_type in [data.attrs["Logical_source"] for data in output_data]:
        assert data_type in expected_logical_source

    for data in output_data:
        assert data.attrs["Data_version"] == "v001"

    assert "vectors" in output_data[-1].variables.keys()
    assert "compression_flags" in output_data[-1].variables.keys()

    assert output_data[-1]["vectors"].data.shape[1] == 4
    assert output_data[-1]["compression_flags"].data.shape[1] == 2
    assert (
        output_data[-1]["vectors"].data.shape[0]
        == output_data[-1]["compression_flags"].data.shape[0]
    )


def test_mag_packet_properties():
    # equal to 01001011
    test_byte = 75
    packet_properties = MagL1aPacketProperties(
        435954628, TimeTuple(10000, 0), 2, 1, 0, 1, 1, test_byte
    )

    assert packet_properties.compression_width == 18
