"""General HIT L0 data class used for parsing data and setting attributes."""

from dataclasses import dataclass, fields

import space_packet_parser

from imap_processing.ccsds.ccsds_data import CcsdsData


@dataclass
class HITBase:
    """
    Data structure for common values across HIT.

    Attributes
    ----------
    ground_sw_version : str
        Ground software version.
    packet_file_name : str
        File name of the source packet.
    ccsds_header : CcsdsData
        CCSDS header data.

    Methods
    -------
    parse_data(packet):
        Parse the packet and assign to class variable using the xtce defined named.
    """

    ground_sw_version: str
    packet_file_name: str
    ccsds_header: CcsdsData

    def parse_data(self, packet: space_packet_parser.parser.Packet) -> None:
        """
        Parse Lo L0 packet data.

        Parameters
        ----------
        packet : space_packet_parser.parser.Packet
            A single Lo L0 packet from space packet parser.
        """
        attributes = [field.name for field in fields(self)]

        # For each item in packet, assign it to the matching attribute in the class.
        for key, item in packet.data.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            if "SPARE" in key:
                continue
            if key not in attributes:
                raise KeyError(
                    f"Did not find matching attribute in {self.__class__} data class"
                    f"for {key}"
                )
            setattr(self, key, value)
