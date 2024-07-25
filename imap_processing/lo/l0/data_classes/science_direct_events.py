"""L1A Science Direct Events data class."""

from dataclasses import dataclass

import numpy as np
from space_packet_parser.parser import Packet

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.decompression_tables.decompression_tables import (
    CASE_DECODER,
    DATA_BITS,
    DE_BIT_SHIFT,
)
from imap_processing.lo.l0.utils.binary_string import BinaryString
from imap_processing.lo.l0.utils.lo_base import LoBase


@dataclass
class ScienceDirectEvents(LoBase):
    """
    L1A Science Direct Events data.

    The Science Direct Events class handles the parsing and
    decompression of L0 to L1A data.

    The TOF data in the binary is in the following order:
    ABSENT, TIME, ENERGY, MODE, TOF0, TOF1, TOF2, TOF3, CKSM, POS

    ABSENT, TIME, ENERGY, and MODE will be present for every type of DE.

    ABSENT: signals the case number for the DE (4 bits).
    TIME: the time of the DE (12 bits).
    ENERGY: Energy step (3 bits).
    MODE: Signals how the data is packed. If MODE is 1, then the TOF1
    (for case 1a) will need to be calculated using the checksum and other TOFs
    in the L1B data product.
    If MODE is 0, then there was no compression and all TOFs are transmitted.

    The presence of TOF0, TOF1, TOF2, TOF3, CKSM, and POS depend on the
    case number.

    - Case 0 can either be a gold or silver triple. Gold triples do
    not send down the TOF1 value and instead recover the TOF1 value
    on the ground using the decompressed checksum.

    - Cases 4, 6, 10, 12, 13 may be Bronze. If it's not a bronze,
    the Position is not transmitted, but TOF3 is. If it is bronze, the table
    should be used as is. If it's not bronze, position was not transmitted,
    but TOF3 was transmitted.

    - Cases 1, 2, 3, 5, 7, 9, 13 will always have a MODE of 0, so the same
    fields will always be transmitted.

    Bit Shifting:
    TOF0, TOF1, TOF2, TOF3, and CKSM all must be shifted by one bit to the
    left. All other fields do not need to be bit shifted.

    The raw values are computed for L1A and will be converted to
    engineering units in L1B.

    Parameters
    ----------
    packet : dict
        Single packet from space_packet_parser.
    software_version : str
        Current version of IMAP-Lo processing.
    packet_file_name : str
        Name of the CCSDS file where the packet originated.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    DE_COUNT: int
        Number of direct events.
    DATA: str
        Compressed TOF Direct Event time tagged data.
    DE_TIME: numpy.ndarray
        Time tag for the direct event.
    ESA_STEP: numpy.ndarray
        Energy of the direct event ENA.
    MODE: numpy.ndarray
        Indication of how the data is packed.
    TOF0: numpy.ndarray
        Time of Flight 0 value for direct event.
    TOF1: numpy.ndarray
        Time of Flight 1 value for direct event.
    TOF2: numpy.ndarray
        Time of Flight 2 value for direct event.
    TOF3: numpy.ndarray
        Time of Flight 3 value for direct event.
    CKSM: numpy.ndarray
        This is checksum defined relative to the TOFs
        condition for golden triples. If golden triples are below
        a certain threshold in checksum it's considered golden, otherwise,
        it's considered a silver triple. This is important for the compression
        for golden triples because it's used to recover TOF1 because
        compression scheme to save space on golden triples doesn't send
        down TOF1 so it's recovered on the ground using the checksum.
    POS: numpy.ndarray
        Stop position for the direct event. There are 4 quadrants
        on the at the stop position.

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Direct Event packet data.
    """

    SHCOARSE: int
    DE_COUNT: int
    DATA: str
    DE_TIME: np.ndarray
    ESA_STEP: np.ndarray
    MODE: np.ndarray
    TOF0: np.ndarray
    TOF1: np.ndarray
    TOF2: np.ndarray
    TOF3: np.ndarray
    CKSM: np.ndarray
    POS: np.ndarray

    def __init__(
        self,
        packet: Packet,
        software_version: str,
        packet_file_name: str,
    ) -> None:
        """
        Initialize Science Direct Events Data class.

        Parameters
        ----------
        packet : space_packet_parser.parser.Packet
            Single packet from space_packet_parser.
        software_version : str
            Current version of IMAP-Lo processing.
        packet_file_name : str
            Name of the CCSDS file where the packet originated.
        """
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.set_attributes(packet)
        # TOF values are not transmitted for certain
        # cases, so these can be initialized to the
        # CDF fill val and stored with this value for
        # those cases.
        self.DE_TIME = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.ESA_STEP = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.MODE = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.TOF0 = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.TOF1 = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.TOF2 = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.TOF3 = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.CKSM = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self.POS = np.ones(self.DE_COUNT) * np.float64(-1.0e31)
        self._decompress_data()

    def _decompress_data(self) -> None:
        """
        Will decompress the Lo Science Direct Events data.

        TOF data is decompressed and the direct event data class
        attributes are set.
        """
        data = BinaryString(self.DATA)
        for de_idx in range(self.DE_COUNT):
            # The first 4 bits of the binary data are used to
            # determine which case number we are working with.
            # The case number is used to determine how to
            # decompress the TOF values.
            case_number = int(data.next_bits(4), 2)

            # time, ESA_STEP, and mode are always transmitted.
            self.DE_TIME[de_idx] = int(data.next_bits(DATA_BITS.DE_TIME), 2)
            self.ESA_STEP[de_idx] = int(data.next_bits(DATA_BITS.ESA_STEP), 2)
            self.MODE[de_idx] = int(data.next_bits(DATA_BITS.MODE), 2)

            # Case decoder indicates which parts of the data
            # are transmitted for each case.
            case_decoder = CASE_DECODER[(case_number, self.MODE[de_idx])]

            # Check the case decoder to see if the TOF field was
            # transmitted for this case. Then grab the bits from
            # the binary turn these into an integer, and perform
            # a bit shift to the left on that integer value. The
            # data was packed using a right bit shift (1 bit), so
            # needs to be bit shifted to the left (1 bit) during
            # unpacking.
            if case_decoder.TOF0:
                self.TOF0[de_idx] = (
                    int(data.next_bits(DATA_BITS.TOF0), 2) << DE_BIT_SHIFT
                )
            if case_decoder.TOF1:
                self.TOF1[de_idx] = (
                    int(data.next_bits(DATA_BITS.TOF1), 2) << DE_BIT_SHIFT
                )
            if case_decoder.TOF2:
                self.TOF2[de_idx] = (
                    int(data.next_bits(DATA_BITS.TOF2), 2) << DE_BIT_SHIFT
                )
            if case_decoder.TOF3:
                self.TOF3[de_idx] = (
                    int(data.next_bits(DATA_BITS.TOF3), 2) << DE_BIT_SHIFT
                )
            if case_decoder.CKSM:
                self.CKSM[de_idx] = (
                    int(data.next_bits(DATA_BITS.CKSM), 2) << DE_BIT_SHIFT
                )
            if case_decoder.POS:
                # no bit shift for POS
                self.POS[de_idx] = int(data.next_bits(DATA_BITS.POS), 2)
