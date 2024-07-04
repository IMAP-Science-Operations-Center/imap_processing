"""L1A Science Counts data class."""

from dataclasses import dataclass

import numpy as np
import space_packet_parser

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.utils.binary_string import BinaryString
from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
    decompress_int,
)
from imap_processing.lo.l0.utils.lo_base import LoBase


@dataclass
class ScienceCounts(LoBase):
    """
    L1A Science Count data class.

    The Science Counts class handles the parsing
    and decompression of L0 to L1A data.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        Single packet from space_packet_parser.
    software_version : str
        Current version of IMAP-Lo processing.
    packet_file_name : str
        Name of the CCSDS file where the packet originated.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    SCI_CNT : str
        science count compressed binary data
    START_A : numpy.ndarray
        Single rates for electron, anode A. 2D array Azimuth(6), Energy(7)
    START_C : numpy.ndarray
        Single rates for electron, anode C. 2D array Azimuth(6), Energy(7)
    STOP_B0 : numpy.ndarray
        Single rates for Ion, anode B0. 2D array Azimuth(6), Energy(7)
    STOP_B3 : numpy.ndarray
        Single rates for Ion, anode B3. 2D array Azimuth(6), Energy(7)
    TOF0 : numpy.ndarray
        TOF rates for Electron anode A / Ion anode B0. 2D array Azimuth(6), Energy(7)
    TOF1 : numpy.ndarray
        TOF rates for Electron anode C / Ion anode B3. 2D array Azimuth(6), Energy(7)
    TOF2 : numpy.ndarray
        TOF rates for Electron anode A / Ion anode C. 2D array Azimuth(6), Energy(7)
    TOF3 : numpy.ndarray
        TOF Rates for Ion anode B0 / Ion anode B. 2D array Azimuth(60), Energy(7)
    TOF0_TOF1 : numpy.ndarray
        Triple coincidence rates for TOF0 and TOF1. 2D array Azimuth(60), Energy(7)
    TOF0_TOF2 : numpy.ndarray
        Triple coincidence rates for TOF0 and TOF2. 2D array Azimuth(60), Energy(7)
    TOF1_TOF2 : numpy.ndarray
        Triple coincidence rates for TOF1 and TOF2. 2D array Azimuth(60), Energy(7)
    SILVER : numpy.ndarray
        Triple coincidence rates for TOF0, TOF1, TOF2 and TOF3.
        2D array Azimuth(60), Energy(7)
    DISC_TOF0 : numpy.ndarray
        Discarded rates for TOF0 value less than TOF0 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF1 : numpy.ndarray
        Discarded rates for TOF1 value less than TOF1 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF2 : numpy.ndarray
        Discarded rates for TOF2 value less than TOF2 threshold setting.
        2D array Azimuth(6), Energy(7)
    DISC_TOF3 : numpy.ndarray
        Discarded rates for TOF3 value less than TOF3 threshold setting.
        2D array Azimuth(6), Energy(7)
    POS0 : numpy.ndarray
        Position rate counts for Ion anode B0. 2D array Azimuth(6), Energy(7)
    POS1 : numpy.ndarray
        Position rate counts for Ion anode B1. 2D array Azimuth(6), Energy(7)
    POS2 : numpy.ndarray
        Position rate counts for Ion anode B2. 2D array Azimuth(6), Energy(7)
    POS3 : numpy.ndarray
        Position rate counts for Ion anode B3. 2D array Azimuth(6), Energy(7)
    HYDROGEN : numpy.ndarray
        Hydrogen species histogram. 2D array Azimuth(60), Energy(7)
    OXYGEN : numpy.ndarray
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
    START_A: np.ndarray
    START_C: np.ndarray
    STOP_B0: np.ndarray
    STOP_B3: np.ndarray
    TOF0: np.ndarray
    TOF1: np.ndarray
    TOF2: np.ndarray
    TOF3: np.ndarray
    TOF0_TOF1: np.ndarray
    TOF0_TOF2: np.ndarray
    TOF1_TOF2: np.ndarray
    SILVER: np.ndarray
    DISC_TOF0: np.ndarray
    DISC_TOF1: np.ndarray
    DISC_TOF2: np.ndarray
    DISC_TOF3: np.ndarray
    POS0: np.ndarray
    POS1: np.ndarray
    POS2: np.ndarray
    POS3: np.ndarray
    HYDROGEN: np.ndarray
    OXYGEN: np.ndarray

    # TODO: Because test data does not currently exist, the init function contents
    # must be commented out for the unit tests to run properly
    def __init__(
        self,
        packet: space_packet_parser.parser.Packet,
        software_version: str,
        packet_file_name: str,
    ) -> None:
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.set_attributes(packet)
        self._decompress_data()

    def _decompress_data(self) -> None:
        """Parse the science count binary chunk for each section of data."""
        # make a bit stream containing the binary for the entire
        # chunk of Science Count packet data
        binary_string = BinaryString(self.SCI_CNT)

        # The START_A data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_A is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.START_A = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The START_C data in the binary is 492 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_C is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.START_C = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The STOP_B0 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_B0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.STOP_B0 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The STOP_B3 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # START_B3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.STOP_B3 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The TOF0 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF0 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF1 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF1 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF2 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF2 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF3 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.TOF3 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The TOF0_TOF1 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0_TOF1 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF0_TOF1 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The TOF0_TOF2 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF0_TOF2 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF0_TOF2 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The TOF1_TOF2 data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # TOF1_TOF2 is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.TOF1_TOF2 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The SILVER data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # SILVER is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.SILVER = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The DISC_TOF0 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF0 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF1 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF1 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF2 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF2 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The DISC_TOF3 data in the binary is 336 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # DISC_TOF3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.DISC_TOF3 = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (6, 7)
        )
        # The POS0 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS0 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS0 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS1 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS1 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS1 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS2 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS2 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS2 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The POS3 data in the binary is 504 bits long and each field is
        # compressed to 12 bits. This data uses 12 to 16 bit decompression.
        # POS3 is a 6, 7 matrix containing Azimuth (6) and Energy (7)
        self.POS3 = self._parse_section(
            binary_string, Decompress.DECOMPRESS12TO16, (6, 7)
        )
        # The HYDROGEN data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # HYDROGEN is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.HYDROGEN = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )
        # The OXYGEN data in the binary is 3360 bits long and each field is
        # compressed to 8 bits. This data uses 8 to 16 bit decompression.
        # OXYGEN is a 60, 7 matrix containing Azimuth (60) and Energy (7)
        self.OXYGEN = self._parse_section(
            binary_string, Decompress.DECOMPRESS8TO16, (60, 7)
        )

    def _parse_section(
        self,
        binary_string: BinaryString,
        decompression: Decompress,
        data_shape: tuple[int, int],
    ) -> np.array:
        """
        Parse a single section of data in the science counts data binary.

        Parameters
        ----------
        binary_string : BinaryString
            Binary string.
        decompression : Decompress
            The decompressed integer.
        data_shape : list
            Shape of the data.

        Returns
        -------
        np.array
            Data array.
        """
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

        # calculate the length of this data section
        section_length = bit_length * data_shape[0] * data_shape[1]
        # Extract the section of binary for this data
        data_array = self._extract_binary(
            binary_string, section_length, bit_length, decompression
        )
        # Reshape the data array. Data shapes are specified in the
        # telemetry definition sheet.
        return data_array.reshape(data_shape[0], data_shape[1])

    def _extract_binary(
        self,
        binary_string: BinaryString,
        section_length: int,
        bit_length: int,
        decompression: Decompress,
    ) -> np.ndarray:
        """
        Extract and decompress science count binary data section.

        Parameters
        ----------
        binary_string : BinaryString
            Binary string.
        section_length : int
            Length of section.
        bit_length : int
            Length of the bit.
        decompression : Decompress
            The decompressed integer.

        Returns
        -------
        numpy.array
            Decompressed science count binary data.
        """
        data_list = list()
        # stop the binary_string once you reach the end of the data section.
        bit_stop = binary_string.bit_pos + section_length
        while binary_string.bit_pos < bit_stop:
            extracted_integer = int(binary_string.next_bits(bit_length), 2)
            # decompression look up is passed in rather than being accessed within this
            # function to avoid reading the csvs each time this function is used.
            decompressed_integer = decompress_int(
                extracted_integer, decompression, DECOMPRESSION_TABLES
            )
            data_list.append(decompressed_integer)
        return np.array(data_list)
