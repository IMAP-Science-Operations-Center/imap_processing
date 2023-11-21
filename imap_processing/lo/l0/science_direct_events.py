from dataclasses import dataclass

import imap_processing.lo.l0.compression_tables as ct
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.lol0 import LoL0


@dataclass
class ScienceDirectEvents(LoL0):
    SHCOARSE: int
    COUNT: int
    DATA: str
    CHKSUM: int
    TOF0: int
    TOF1: int
    TOF2: int
    TOF3: int
    case_number: int
    DATA_DECOMPRESSED: str
    binary_strings: list
    tof_decoder: list

    def __init__(self, packet, software_version: str, packet_file_name: str):
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)

    def decompress_data(self):
        self._find_decompression_case()
        self._find_bit_length_for_case()
        self._find_binary_strings()

        # separate data bits into its parts
        parse_bits = {}
        bit_start = 0
        for field, bit_length in self.tof_decoder.items():
            parse_bits[field] = self.DATA[bit_start:bit_length]
            bit_start = bit_start + bit_length

    def _find_bit_length_for_case(self):
        self.tof_decoder = ct.tof_bit_length_table[self.case_number]

    def _find_binary_strings(self):
        hex_strings = ct.other_decoder[self.case_number]
        self.binary_strings = [
            self._hexadecimal_to_binary(hex_string)
            for hex_string in hex_strings
            if hex_string != ""
        ]

    def _hexadecimal_to_binary(self, hex_string: str):
        return bin(int(hex_string))[2:]

    def _find_decompression_case(self):
        # TODO: Do I read the binary left to right or right to left?
        self.case_number = ct.tof_case_table[self.DATA[0:4]]
