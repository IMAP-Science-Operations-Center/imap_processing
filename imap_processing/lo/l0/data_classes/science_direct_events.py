"""L1A Science Direct Events data class."""
from dataclasses import dataclass

import numpy as np

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.lo.l0.decompression_tables.decompression_tables import (
    CASE_DECODER,
    SIGNIFICANT_BITS,
)
from imap_processing.lo.l0.utils.binary_string import BinaryString
from imap_processing.lo.l0.utils.set_dataclass_attr import set_attributes


@dataclass
class ScienceDirectEventsPacket:
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
    DATA: str
        Compressed TOF Direct Event time tagged data.
    TOF0: numpy.ndarray
        Time of Flight 0 value for direct event.
    TOF1: numpy.ndarray
        Time of Flight 1 value for direct event.
    TOF2: numpy.ndarray
        Time of Flight 2 value for direct event.
    TOF3: numpy.ndarray
        Time of Flight 3 value for direct event.
    TIME: numpy.ndarray
        time tag for the direct event
    ENERGY: numpy.ndarray
        energy of the direct event ENA.
    POS: numpy.ndarray
        Stop position for the direct event. There are 4 quadrants
        on the at the stop position.
    CKSM: numpy.ndarray
        This is checksum defined relative to the TOFs
        condition for golden triples. If golden triples are below
        a certain threshold in checksum it's considered golden, otherwise,
        it's considered a silver triple. This is important for the compression
        for golden triples because it's used to recover TOF1 because
        compression scheme to save space on golden triples doesn't send
        down TOF1 so it's recovered on the ground using the checksum

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Direct Event packet data.
    """

    SHCOARSE: int
    COUNT: int
    DATA: str
    CHKSUM: int
    TOF0: np.ndarray
    TOF1: np.ndarray
    TOF2: np.ndarray
    TOF3: np.ndarray
    TIME: np.ndarray
    ENERGY: np.ndarray
    POS: np.ndarray
    CKSM: np.ndarray
    software_version: str
    packet_file_name: str
    ccsds_header: CcsdsData

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Intialization method for Science Direct Events Data class."""
        self.software_version = software_version
        self.packet_file_name = packet_file_name
        self.ccsds_header = CcsdsData(packet.header)
        # TODO: Is there a better way to do this?
        self.TOF0 = np.array([])
        self.TOF1 = np.array([])
        self.TOF2 = np.array([])
        self.TOF3 = np.array([])
        self.TIME = np.array([])
        self.ENERGY = np.array([])
        self.POS = np.array([])
        self.CKSM = np.array([])
        set_attributes(self, packet)
        self._decompress_data()

    def _decompress_data(self):
        """Decompress the Lo Science Direct Events data."""
        data = BinaryString(self.DATA)
        for _ in range(self.COUNT):
            case_number = self._decompression_case(data)
            case_decoder = self._case_decoder(case_number, data)
            self._parse_case(data, case_decoder)

    def _decompression_case(self, data: BinaryString):
        """Find the decompression case for this DE.

        The first 4 bits of the binary data are used to
        determine which case number we are working with.
        The case number is used to determine how to
        decompress the TOF values.
        """
        return int(data.next_bits(4), 2)

    def _case_decoder(self, case_number, data):
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
        case_variant = 0
        # the other cases do no have variants
        if case_number in [0, 4, 6, 10, 12, 13]:
            case_variant = int(data.next_bits(1))
        return CASE_DECODER[(case_number, case_variant)]

    def _parse_case(self, data, case_decoder):
        """Parse out the values for each DE data field."""
        self.ENERGY = np.append(
            self.ENERGY,
            self._decompress_field(data, case_decoder.ENERGY, SIGNIFICANT_BITS.ENERGY),
        )

        self.POS = np.append(
            self.POS,
            self._decompress_field(data, case_decoder.POS, SIGNIFICANT_BITS.ENERGY),
        )

        self.TOF0 = np.append(
            self.TOF0,
            self._decompress_field(data, case_decoder.TOF0, SIGNIFICANT_BITS.TOF0),
        )

        self.TOF1 = np.append(
            self.TOF1,
            self._decompress_field(data, case_decoder.TOF1, SIGNIFICANT_BITS.TOF1),
        )

        self.TOF2 = np.append(
            self.TOF2,
            self._decompress_field(data, case_decoder.TOF2, SIGNIFICANT_BITS.TOF2),
        )

        self.TOF3 = np.append(
            self.TOF3,
            self._decompress_field(data, case_decoder.TOF3, SIGNIFICANT_BITS.TOF3),
        )

        self.CKSM = np.append(
            self.CKSM,
            self._decompress_field(data, case_decoder.CKSM, SIGNIFICANT_BITS.CKSM),
        )

        self.TIME = np.append(
            self.TIME,
            self._decompress_field(data, case_decoder.TIME, SIGNIFICANT_BITS.TIME),
        )

    def _decompress_field(self, data, field_length, sig_bits):
        """Decompress the DE field."""
        # If the field length is 0 then that field is not
        # present in this type of Direct Event case
        if field_length == 0:
            return GlobalConstants.DOUBLE_FILLVAL
        field_bits = data.next_bits(field_length)
        field_array = np.array([int(bit) for bit in field_bits])
        # The dot product of the binary and the significant bits arrays
        # is used to calculate the decompressed field value
        return np.dot(sig_bits, field_array)
