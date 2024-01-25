"""L1A Science Direct Events data class."""
from dataclasses import dataclass
from itertools import compress

import bitstring
from bitarray import bitarray

import imap_processing.lo.l0.decompression_tables as decompress_tables
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.lo_base import LoBase


@dataclass
class ScienceDirectEvents(LoBase):
    """L1A Science Direct Events data.

    The Science Direct Events class handles the parsing and
    decompression of L0 to L1A data.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    COUNT: int
        Number of direct events.
    CHKSUM: int
        Checksum for the packet.
    DATA: bitstring.Bits
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
    ENERGY: int
        energy of the direct event ENA.
    POS: int
        Stop position for the direct event. There are 4 quadrants
        on the at the stop position.
    CKSM: int
        This is checksum defined relative to the TOFs
        condition for golden triples. If golden triples are below
        a certain threshold in checksum it's considered golden, otherwise,
        it's considered a silver triple. This is important for the compression
        for golden triples because it's used to recover TOF1 because
        compression scheme to save space on golden triples doesn't send
        down TOF1 so it's recovered on the ground using the checksum
    case_number: int
        The compression case number for the direct event. The case number
        determines how the bits are arranged in the compressed data.
    tof_calculation_binary: dict
        Binary used to determine which TOF coefficients should be used
        for decompressing the binary.
    tof_decoder: list
        Shows how the fields in the binary are split up by bit length and order.
    remaining_bits: dict
        The TOF coefficients that should be used for decompression
    parsed_bits: dict
        The binary bits split up by TOF field.

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Direct Event packet data.
    decompress_data():
        Decompresses the Science Direct Event TOF data.

    """

    SHCOARSE: int
    COUNT: int
    DATA: bitstring.Bits
    CHKSUM: int
    TOF0: float
    TOF1: float
    TOF2: float
    TOF3: float
    TIME: float
    ENERGY: float
    POS: float

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Intialization method for Science Direct Events Data class."""
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
        self._decompress_data()

    def _decompress_data(self):
        """Decompress the Lo Science Direct Events data."""
        case_number = self._find_decompression_case()
        tof_decoder = self._find_tof_decoder_for_case(case_number)
        tof_calculation_binary = self._read_tof_calculation_table(case_number)
        remaining_bit_coefficients = self._find_remaining_bit_coefficients(
            tof_calculation_binary
        )
        parsed_bits = self._parse_binary(case_number, tof_decoder)
        # decode_fields will set the TOF class variables
        self._decode_fields(remaining_bit_coefficients, parsed_bits)

    def _find_decompression_case(self):
        """Find the decompression case for this DE.

        The first 4 bits of the binary data are used to
        determine which case number we are working with.
        The case number is used to determine how to
        decompress the TOF values.
        """
        return int(self.DATA[0:4].bin, 2)

    def _find_tof_decoder_for_case(self, case_number):
        """Get the TOF decoder for this DE's case number.

        The case number determines wich TOF decoder to use.
        The TOF decoder table shows how the TOF bits should be
        parsed in the binary data.

        Case 0 can either be a gold or silver triple. Gold triples do
        not send down the TOF1 value and instead recover the TOF1 value
        on the ground using the decompressed checksum.
        Cases 4, 6, 10, 12, 13 may be Bronze triples. If it's not a bronze triple,
        the Position is not transmitted, but TOF3 is. If it is bronze, the table
        should be used as is. If it's not bronze, position was not transmitted,
        but TOF3 was transmitted.
        """
        if case_number == 0:
            gold_or_silver = int(self.DATA.bin[4])
            tof_decoder = decompress_tables.tof_decoder_table[case_number][
                gold_or_silver
            ]

        elif case_number in [4, 6, 10, 12, 13]:
            is_bronze = int(self.DATA.bin[4])
            tof_decoder = decompress_tables.tof_decoder_table[case_number]

            if not is_bronze:
                # if Bronze = 1, use table as is
                # if Bronze = 0, position not transmitted, TOF3 is transmitted
                # and will be six bits long
                tof_decoder = decompress_tables.TOFData(
                    tof_decoder.ENERGY,
                    0,
                    tof_decoder.TOF0,
                    tof_decoder.TOF1,
                    tof_decoder.TOF2,
                    6,
                    tof_decoder.CKSM,
                    tof_decoder.TIME,
                )
        # We're not expecting to recieve data for the rest of the cases, but need
        # to handle them in case things change in the future.
        else:
            tof_decoder = decompress_tables.tof_decoder_table[case_number]

        return tof_decoder

    def _read_tof_calculation_table(self, case_number):
        """Get the TOF calculation values for this DE's case number.

        The case number determines what calculation value is needed to help
        calculate the TOF values.
        """
        tof_calculation_values = decompress_tables.tof_calculation_table[case_number]
        tof_calculation_binary = {
            field: bitstring.Bits(calculation_value)
            for field, calculation_value in tof_calculation_values._asdict().items()
        }
        return tof_calculation_binary

    def _find_remaining_bit_coefficients(self, tof_calculation_binary):
        """Find which TOF coefficients are needed for a data field and case number."""
        remaining_bit_coefficients = {}
        for field, tof_binary in tof_calculation_binary.items():
            # get a list of tof calculation values as integers
            # We only need the last 12 bits from the tof calculation binary
            tof_calculation_array = bitarray(tof_binary).tolist()[-12:]

            # the tof calculation table binary are used to determine which values
            # from the tof coefficient table should be used in combination with
            # the tof decoder.
            remaining_bit_coefficients[field] = list(
                compress(decompress_tables.tof_coefficient_table, tof_calculation_array)
            )

        return remaining_bit_coefficients

    def _parse_binary(self, case_number, tof_decoder):
        """Use the TOF decoder to split up the data bits into its TOF fields.

        The first few binary bits are only used for determining the case number
        so either the first 5 or 6 bits need to be ignored depending the kind
        of DE.
        """
        # separate the binary data into its parts
        parsed_bits = {}
        # check whether the first 4 or 5 bits were needed for the
        # case determination
        if case_number in [0, 4, 6, 10, 12, 14]:
            bit_start = 5
        else:
            bit_start = 4
        # Use the TOF decoder to chunk the data binary into its componenets
        # TOF0, TOF1, TOF2, etc.
        for field, bit_length in tof_decoder._asdict().items():
            parsed_bits[field] = self.DATA[bit_start : bit_start + bit_length]
            bit_start = bit_start + bit_length

        return parsed_bits

    def _decode_fields(self, remaining_bit_coefficients, parsed_bits):
        """Use the parsed data and TOF coefficients to decode the binary."""
        for field, tof_bits in parsed_bits.items():
            needed_bits = tof_bits[-len(remaining_bit_coefficients[field]) :]
            # Use the TOF coefficients and the bits for the current field to
            # calculate the decompressed value. Also round to 2 because the TOF
            # coefficients only have 2 decimals.
            decompressed_data = round(
                sum(
                    bit[0] * bit[1]
                    for bit in zip(
                        bitarray(needed_bits).tolist(),
                        remaining_bit_coefficients[field],
                    )
                ),
                2,
            )
            setattr(self, field, decompressed_data)
