"""L1A Science Counts data class."""
from dataclasses import dataclass

import numpy as np
from bitstring import ConstBitStream

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.utils.bit_decompression import Decompress, decompress_int
from imap_processing.lo.l0.utils.lo_base import LoBase


@dataclass
class ScienceCounts(LoBase):
    """L1A Science Count data class.

    The Science Counts class handles the parsing
    and decompression of L0 to L1A data.

    Attributes
    ----------
    SHCOARSE: int
        Spacecraft time.
    SCI_CNT: str
        science count compressed binary data
    START_A: numpy.array
        Single rates for electon, anode A. 2D array Azimuth(6), Energy(7)
    START_C: numpy.array
        Single rates for electron, anode C. 2D array Azimuth(6), Energy(7)
    STOP_B0: numpy.array
        Single rates for Ion, anode B0. 2D array Azimuth(6), Energy(7)
    STOP_B3: numpy.array
        Single rates for Ion, anode B3. 2D array Azimuth(6), Energy(7)
    TOF0: numpy.array
        TOF rates for Electron anode A / Ion anode B0. 2D array Azimuth(6), Energy(7)
    TOF1: numpy.array
        TOF rates for Electron anode C / Ion anode B3. 2D array Azimuth(6), Energy(7)
    TOF2: numpy.array
        TOF rates for Electron anode A / Ion anode C. 2D array Azimuth(6), Energy(7)
    TOF3: numpy.array
        TOF Rates for Ion anode B0 / Ion anode B. 2D array Azimuth(60), Energy(7)
    TOF0_TOF1: numpy.array
        Triple coincidence rates for TOF0 and TOF1. 2D array Azimuth(60), Energy(7)
    TOF0_TOF2: numpy.array
        Triple coincidence rates for TOF0 and TOF2. 2D array Azimuth(60), Energy(7)
    TOF1_TOF2: numpy.array
        Triple coincidence rates for TOF1 and TOF2. 2D array Azimuth(60), Energy(7)
    SILVER: numpy.array
        Triple coincidence rates for TOF0, TOF1, TOF2 and TOF3.
        2D array Azimuth(60), Energy(7)
    DISC_TOF0: numpy.array
        Discarded rates for TOF0 value less than TOF0 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF1: numpy.array
        Discarded rates for TOF1 value less than TOF1 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF2: numpy.array
        Discarded rates for TOF2 value less than TOF2 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF3: numpy.array
        Discarded rates for TOF3 value less than TOF3 threshold setting.
        2D array Azimuth(6), Energy(7)
    POS0: numpy.array
        Postition rate counts for Ion anode B0. 2D array Azimuth(6), Energy(7)
    POS1: numpy.array
        Position rate counts for Ion anode B1. 2D array Azimuth(6), Energy(7)
    POS2: numpy.array
        Position rate counts for Ion anode B2. 2D array Azimuth(6), Energy(7)
    POS3: numpy.array
        Position rate counts for Ion anode B3. 2D array Azimuth(6), Energy(7)
    HYDROGEN: numpy.array
        Hydrogen species histogram. 2D array Azimuth(60), Energy(7)
    OXYGEN: numpy.array
        Oxygen species histogram. 2D array Azimuth(60), Energy(7)

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store data for
        the Science Count packet.
    """

    SHCOARSE: int
    SCI_CNT: str
    START_A: np.array
    START_C: np.array
    STOP_B0: np.array
    STOP_B3: np.array
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
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
        self._parse_binary()
        pass

    def _parse_binary(self):
        """Parse the science count binary chunk for each section of data."""
        # make a bit stream containing the binary for the entire
        # chunk of Science Count packet data
        bitstream = ConstBitStream(bin=self.SCI_CNT)

        # The START_A data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_A is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.START_A = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The START_C data in the binary is 492 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_C is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.START_C = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The STOP_B0 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_B0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.STOP_B0 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The STOP_B3 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_B3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.STOP_B3 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The TOF0 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF0 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF1 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF1 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF2 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF2 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF3 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF3 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF0_TOF1 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0_TOF1 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF0_TOF1 = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The TOF0_TOF2 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0_TOF2 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF0_TOF2 = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The TOF1_TOF2 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF1_TOF2 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF1_TOF2 = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The SILVER data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # SILVER is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.SILVER = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The DISC_TOF0 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF0 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF1 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF1 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF2 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF2 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF3 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF3 = self._parse_section(
            bitstream, 336, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The POS0 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS0 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS1 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS1 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS2 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS2 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS3 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS3 = self._parse_section(
            bitstream, 504, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The HYDROGEN data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # HYDROGEN is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.HYDROGEN = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The OXYGEN data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # OXYGEN is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.OXYGEN = self._parse_section(
            bitstream, 3360, Decompress.DECOMPRESS8TO16, (60, 7)
        )

    def _parse_section(self, bitstream, section_length, decompression, data_shape):
        """Parse a single section of data in the science counts data binary."""
        # Use the decompression method to get the bit length
        # for this section.
        if decompression == Decompress.DECOMPRESS8TO16:
            bit_length = 8
        elif decompression == Decompress.DECOMPRESS12TO16:
            bit_length = 12
        else:
            raise ValueError(
                "Science Counts only use 8 to 16 or 12 to 16 decompression"
            )

        # Extract the section of binary for this data
        data_array = self._extract_binary(
            bitstream, section_length, bit_length, decompression
        )
        # Reshape the data array. Data shapes are specified in the
        # telemetry definition sheet.
        return data_array.reshape(data_shape[0], data_shape[1])

    def _extract_binary(self, bitstream, section_length, bit_length, decompression):
        """Extract and decompress science count binary data section."""
        data_list = list()
        # stop the bitstream once you reach the end of the data section.
        bit_stop = bitstream.pos + section_length
        while bitstream.pos < bit_stop:
            # Extract the 12 bit long field from the binary chunk and get the integer
            extracted_integer = bitstream.read(bit_length).uint
            # The Star Sensor packet uses a 12 to 8 bit compression
            decompressed_integer = decompress_int(extracted_integer, decompression)
            data_list.append(decompressed_integer)
        return np.array(data_list)
