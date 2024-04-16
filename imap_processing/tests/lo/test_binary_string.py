from imap_processing.lo.l0.utils.binary_string import BinaryString


def test_binary_string_init():
    binary = "00101"

    data = BinaryString(binary)

    assert data.bin == "00101"
    assert data.bit_pos == 0


def test_get_bits():
    binary = "0001110"
    data = BinaryString(binary)

    first_three_bits = data.next_bits(3)
    next_four_bits = data.next_bits(4)

    assert first_three_bits == "000"
    assert next_four_bits == "1110"
