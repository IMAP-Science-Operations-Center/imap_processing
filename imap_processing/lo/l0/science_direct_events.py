from dataclasses import dataclass
from itertools import compress

import imap_processing.lo.l0.compression_tables as ct
from imap_processing.lo.l0.lol0 import LoL0

# TODO: Talk to Colin to get better names for tables

# 2nd table could be called TOF calculation table
# direct event unpacking scheme table for hex table
# unpack into arrays for repeating patterns

# if Bronze = 1, use table as is
# if Bronze = 0, position not transmitted, TOF3 is transmitted
# ENERGY first 3, TOF0 next 10, TOF3 next 6, TIME next 12


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
        # This is related to the spin. Colin will double check
    ENERGY: int
        energy of the direct event ENA.
    POS: int
        Position of the direct event ENA
        #TODO: Is this the position on the final detector?
        # stop position. there's 4 quadrants on the stop
        # it's compressing the TOF3 value into 2 bits
    CKSM: int
        #TODO: There's a checksum in the packet and another in
        the dedcompressed data? Are these different?
        # This is checksum defined relative to the TOFs
        # condition for golden triples. If golden triples are below
        # threshold in checksum it's considered golden, otherwise,
        # it's considered a silver triple. Important for the compression
        # for golden triples because it's used to recover TOF1 because
        # compression scheme to save space on golden triples doesn't send
        # down TOF1 so it's recovered on the ground using the checksum

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
        # super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        # self.parse_data(packet)
        pass

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

        # Case 0 can either be a gold or silver triple. Gold triples do
        # not send down the TOF1 value and instead recover the TOF1 value
        # on the ground using the decompressed checksum
        if self.case_number == 0:
            gold_or_silver = self.DATA[4]
            self.tof_decoder = ct.tof_bit_length_table[self.case_number][gold_or_silver]

        # Cases 4, 6, 10, 12, 13 may be Bronze triples. If it's not a bronze triple,
        # the Position is not transmitted, but TOF3 is.
        elif self.case_number in [4, 6, 10, 12, 13]:
            is_bronze = self.DATA[4]
            self.tof_decoder = ct.tof_bit_length_table[self.case_number]
            # TODO: maybe I should just add a sub-dictionary to the tof decoder table
            # instead of changing the values here?
            if is_bronze:
                self.tof_decoder["POS"] = 0
                self.tof_decoder["TOF3"] = 6
        # We're not expecting to recieve data for the rest of the cases, but need
        # to handle them in case things change in the future.
        else:
            self.tof_decoder = ct.tof_bit_length_table[self.case_number]

    def _find_binary_strings(self):
        # The case numebr determines what hex value is needed to help
        # calculate the TOF values
        hex_strings = ct.hex_table[self.case_number]
        self.binary_strings = {
            field: self._hexadecimal_to_binary(hex_string) if hex_string else ""
            for field, hex_string in hex_strings.items()
        }

    def _hexadecimal_to_binary(self, hex_string: str):
        # convert the hexadecimal table to binary
        return bin(int(hex_string, 16))[2:].zfill(16)

    def _find_decompression_case(self):
        # The first 4 bits of the binary data are used to
        # determine which case number we are working with.
        # The case number is used to determine how to
        # decompress the TOF values.
        self.case_number = ct.tof_case_table[self.DATA[0:4]]

    def _find_remaining_bits(self):
        self.remaining_bits = {}
        for field, binary_string in self.binary_strings.items():
            bit_list = [bool(int(bit)) for bit in list(binary_string)]
            second_tof_table = ct.tof_coefficient_table
            # We only need the last 12 bits from the converted hex values
            bit_list = bit_list[-12:]
            # the converted hex values (bit list) are used to determine which values
            # from the tof calculate table should be used when
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
            remaining_bits = self.remaining_bits[field][-len(bits) :]
            bit_list = [int(bit) for bit in list(bits)]
            decompressed_data = sum(
                bit[0] * bit[1] for bit in zip(bit_list, remaining_bits)
            )
            # TODO: Is it better to explicitly set the TOF fields? There
            # shouldn't be any invalid names because the field names are
            # coming from compression_tables.py
            setattr(self, field, decompressed_data)
