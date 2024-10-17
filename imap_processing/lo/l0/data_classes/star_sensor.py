"""L1A Star Sensor data class."""

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
class StarSensor(LoBase):
    """
    L1A Star Sensor data class.

    The Start Sensor class handles the parsing
    and decompression of L0 to L1A data.

    Parameters
    ----------
    packet : space_packet_parser.packets.CCSDSPacket
        The packet.
    software_version : str
        Software version.
    packet_file_name : str
        Name of packet file.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    COUNT : int
        Number of star sensor samples.
    DATA_COMPRESSED : str
        Star sensor compressed binary data.
    DATA : list(int)
        Decompressed star sensor data list.

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store data for
        the Star Sensor packet.
    """

    SHCOARSE: int
    COUNT: int
    DATA_COMPRESSED: str
    DATA: np.ndarray

    # TODO: Because test data does not currently exist, the init function contents
    # must be commented out for the unit tests to run properly
    def __init__(
        self,
        packet: space_packet_parser.packets.CCSDSPacket,
        software_version: str,
        packet_file_name: str,
    ) -> None:
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.set_attributes(packet)
        self._decompress_data()

    def _decompress_data(self) -> None:
        """
        Will decompress the Star Sensor packet data.

        The Star packet data is read in as one large binary chunk
        in the XTCE, but contains multiple data fields where each data field
        is compressed to 8 bits. The data fields need to be extracted from
        binary, decompressed, and stored in a list.
        """
        # Star Sensor data fields have a bit length of 8.
        # See Telem definition sheet for more information.
        bit_length = 8

        # make a bit stream containing the binary for the entire
        # chunk of Star packet data
        data_list = list()
        binary_string = BinaryString(self.DATA_COMPRESSED)
        while binary_string.bit_pos < len(binary_string.bin):
            # Extract the 8 bit long field from the binary chunk and get the integer
            extracted_integer = int(binary_string.next_bits(bit_length), 2)
            # The Star Sensor packet uses a 12 to 8 bit compression
            decompressed_integer = decompress_int(
                [extracted_integer], Decompress.DECOMPRESS8TO12, DECOMPRESSION_TABLES
            )
            # TODO: Need to update this to work with decompress_int outputting
            #  a list of ints. Remove function from loop during refactor
            data_list.append(decompressed_integer[0])
        self.DATA = np.array(data_list)
