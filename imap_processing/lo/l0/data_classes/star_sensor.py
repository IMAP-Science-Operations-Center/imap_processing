"""L1A Star Sensor data class."""
from dataclasses import dataclass

import numpy as np
from bitstring import ConstBitStream

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.utils.bit_decompression import (
    Decompress,
    decompress_int,
    decompression_tables,
)
from imap_processing.lo.l0.utils.lo_base import LoBase

decompression_lookup = decompression_tables()


@dataclass
class StarSensor(LoBase):
    """L1A Star Sensor data class.

    The Start Sensor class handles the parsing
    and decompression of L0 to L1A data.

    Attributes
    ----------
    SHCOARSE: int
        Spacecraft time.
    COUNT: int
        number of star sensor samples
    DATA_COMPRESSED: str
        star sensor compressed binary data
    DATA: list(int)
        decompressed star sensor data list

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
    DATA: np.array

    def __init__(self, packet, software_version: str, packet_file_name: str):
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
        self._decompress_data()

    def _decompress_data(self):
        """
        Decompress the Star Sensor packet data.

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
        bitstream = ConstBitStream(bin=self.DATA_COMPRESSED)
        while bitstream.pos < len(bitstream):
            # Extract the 8 bit long field from the binary chunk and get the integer
            extracted_integer = bitstream.read(bit_length).uint
            # The Star Sensor packet uses a 12 to 8 bit compression
            decompressed_integer = decompress_int(
                extracted_integer, Decompress.DECOMPRESS8TO12, decompression_lookup
            )
            data_list.append(decompressed_integer)
        self.DATA = np.array(data_list)
