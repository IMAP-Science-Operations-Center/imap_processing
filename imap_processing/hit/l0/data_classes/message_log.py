"""L1A HIT Message Log data class."""
from dataclasses import dataclass

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.hit.l0.utils.hit_base import HITBase


@dataclass
class MessageLog(HITBase):
    """L1A HIT Message Log data.

    The HIT Message Log data class handles the decommutation
    and parsing of L0 to L1A data.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    TEXT: str
        Message log left in binary from packet.

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Houskeeping packet data.
    """

    SHCOARSE: int
    TEXT: str

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Intialization method for Housekeeping Data class."""
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
