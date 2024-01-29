from dataclasses import dataclass
import numpy as np
from bitstring import BitArray

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.lol0 import LoBase
from imap_processing.lo.l0.utils.loApid import LoAPID
from imap_processing.lo.l0.utils.bit_decompression import Decompress, decompress_int


@dataclass
class ScienceCounts(LoBase):
    SHCOARSE: int
    START: np.array
    STOP: np.array
    TOF0: np.array
    TOF1: np.array
    TOF2: np.array
    TOF3: np.array
    TOF0_TOF1: np.array
    TOF0_TOF2: np.array
    TOF1_TOF2: np.array
    SILVER: np.array
    DISC_TOF0: np.array
    DISC_TOF1: np.array
    DISC_TOF2: np.array
    DISC_TOF3: np.array
    POS0: np.array
    POS1: np.array
    POS2: np.array
    POS3: np.array
    HYDROGEN: np.array
    OXYGEN: np.array

    def __init__(self, packet, software_version: str, packet_file_name: str):
        super().__init__(
            software_version,
            packet_file_name,
            CcsdsData(packet.header),
            LoAPID.ILO_STAR,
        )
        self.parse_data(packet)
        self._decompress_data()