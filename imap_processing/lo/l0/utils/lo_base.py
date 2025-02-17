"""General Lo L0 data class used for parsing data and setting attributes."""

from dataclasses import dataclass, fields

from space_packet_parser import packets

from imap_processing.ccsds.ccsds_data import CcsdsData


@dataclass
class LoBase:
    """
    Data structure for common values across histogram and direct events data.

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

    def set_attributes(self, packet: packets.CCSDSPacket) -> None:
        """
        Set dataclass attributes with packet data.

        Parameters
        ----------
        packet : space_packet_parser.packets.CCSDSPacket
            A single Lo L0 packet from space packet parser.
        """
        attributes = [field.name for field in fields(self)]

        # For each item in packet, assign it to the matching attribute in the class.
        for key, item in packet.user_data.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            if "SPARE" in key or "CHKSUM" in key:
                continue
            if key not in attributes:
                raise KeyError(
                    f"Did not find matching attribute in {self.__class__} data class"
                    f"for {key}"
                )
            setattr(self, key, value)
