from dataclasses import dataclass
from itertools import compress

import imap_processing.lo.l0.compression_tables as ct
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.lol0 import LoL0

# TODO: Talk to Colin to get better names for tables


@dataclass
class ScienceDirectEvents(LoL0):
    """L1A Science Drect Events data.

    The Science Direct Events class handles the parsing and
    decompression of L0 to L1A data.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time
    COUNT: int
        Number of direct events
    DATA: str
        Compressed TOF Direct Event time tagged data.
    TOF0: int
        Time of Flight 0 value for direct event.
    TOF1: int
        Time of Flight 1 value for direct event.
    TOF2: int
        Time of Flight 2 value for direct event.
    TOF3: int
        Time of Flight 3 value for direct event.
    TIME: int
        time tag for the direct event
        #TODO: Is this different than the SHCOARSE?
    ENERGY: int
        energy of the direct event ENA.
    POS: int
        Position of the direct event ENA
        #TODO: Is this the position on the final detector?
    CKSM: int
        #TODO: There's a checksum in the packet and another in
        the dedcompressed data? Are these different?

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Constructor, uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Direct Event packet data.
    decompress_data():
        Decompresses the Science Direct Event TOF  data.

    """

    SHCOARSE: int
    COUNT: int
    DATA: str
    CHKSUM: int
    TOF0: int
    TOF1: int
    TOF2: int
    TOF3: int
    TIME: int
    ENERGY: int
    POS: int
    # TODO: Maybe all the below variables should be local variables instead of
    # instance variables for the class to keep all the instance variables data
    # fields from the packet?
    case_number: int
    binary_strings: list
    tof_decoder: list
    remaining_bits: dict
    parsed_bits: list

    def __init__(self, packet, software_version: str, packet_file_name: str):
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)

    def decompress_data(self):
        """Decompress the Lo Science Direct Events data."""
        self._find_decompression_case()
        self._find_bit_length_for_case()
        self._find_binary_strings()
        self._find_remaining_bits()
        self._parse_binary()
        self._decode_fields()

    def _find_bit_length_for_case(self):
        # The case number determines which bit length table to use.
        # This table shows how the TOF bits should be parsed in the
        # binary data.
        self.tof_decoder = ct.tof_bit_length_table[self.case_number]

    def _find_binary_strings(self):
        # TODO: Not sure how I can describe the "hex_table" better.
        # Ask Colin for a better name suggestions for this table.
        hex_strings = ct.hex_table[self.case_number]
        self.binary_strings = {
            field: self._hexadecimal_to_binary(hex_string) if hex_string else ""
            for field, hex_string in hex_strings.items()
        }

    def _hexadecimal_to_binary(self, hex_string: str):
        # TODO: Is 16 bits accurate here? Will this change depending
        # on the Row/Column of the hex?
        return bin(int(hex_string, 16))[2:].zfill(16)

    def _find_decompression_case(self):
        # TODO: Do I read the binary left to right or right to left?
        self.case_number = ct.tof_case_table[self.DATA[0:4]]

    def _find_remaining_bits(self):
        self.remaining_bits = {}
        for field, binary_string in self.binary_strings.items():
            bit_list = [bool(int(bit)) for bit in list(binary_string)]
            second_tof_table = ct.another_tof_table
            # TODO: Temporary: Need to figure out how to handle different lengths
            if bit_list != []:
                bit_list = bit_list[-12:]
            ########
            self.remaining_bits[field] = list(compress(second_tof_table, bit_list))

    def _parse_binary(self):
        # separate data bits into its parts
        self.parsed_bits = {}
        bit_start = 0
        for field, bit_length in self.tof_decoder.items():
            self.parsed_bits[field] = self.DATA[bit_start : bit_start + bit_length]
            bit_start = bit_start + bit_length

    def _decode_fields(self):
        for field, bits in self.parsed_bits.items():
            # TODO: Temporary - figure out how to handled lengths not matching
            # TODO: What about TIME with 16bits and there are only 12 possible remaining
            # bits columns in the second table?
            remaining_bits = self.remaining_bits[field][-len(bits) :]
            bit_list = [int(bit) for bit in list(bits)]
            decompressed_data = sum(
                bit[0] * bit[1] for bit in zip(bit_list, remaining_bits)
            )
            # TODO: Is it better to explicitly set the TOF fields? There
            # shouldn't be any invalid names because the field names are
            # coming from compression_tables.py
            setattr(self, field, decompressed_data)
