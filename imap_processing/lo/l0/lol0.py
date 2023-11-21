from dataclasses import dataclass, fields

from imap_processing.ccsds.ccsds_data import CcsdsData


@dataclass
class LoL0:
    """Data structure for common values across histogram and direct events data.

    Attributes
    ----------
    ground_sw_version : str
        Ground software version
    packet_file_name : str
        File name of the source packet
    ccsds_header : CcsdsData
        CCSDS header data
    """

    ground_sw_version: str
    packet_file_name: str
    ccsds_header: CcsdsData

    def parse_data(self, packet):
        attributes = [field.name for field in fields(self)]

        # For each item in packet, assign it to the matching attribute in the class.
        for key, item in packet.data.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            print(value)
            if key in attributes:
                setattr(self, key, value)
            else:
                raise KeyError(
                    f"Did not find matching attribute in {self.__class__} data class for "
                    f"{key}"
                )
