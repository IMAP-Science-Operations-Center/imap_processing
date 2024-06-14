"""Contain a data class for CCSDS data."""

from dataclasses import dataclass, fields


@dataclass
class CcsdsData:
    """
    Data class for CCSDS header.

    Parameters
    ----------
    packet_header : dict
        A dictionary containing the CCSDS packet header information.

    Attributes
    ----------
    VERSION: int
        CCSDS Packet Version Number
    TYPE: int
        CCSDS Packet Type Indicator
    SEC_HDR_FLG: int
        CCSDS Packet Secondary Header Flag
    PKT_APID: int
        CCSDS Packet Application Process ID
    SEQ_FLGS: int
        CCSDS Packet Grouping Flags
    SRC_SEQ_CTR: int
        CCSDS Packet Sequence Count
    PKT_LEN: int
        CCSDS Packet Length
    """

    VERSION: int
    TYPE: int
    SEC_HDR_FLG: int
    PKT_APID: int
    SEQ_FLGS: int
    SRC_SEQ_CTR: int
    PKT_LEN: int

    def __init__(self, packet_header: dict):
        """
        Initialize the CCSDSData object.

        Parameters
        ----------
        packet_header : dict
            A dictionary containing CCSDS packet header information.
        """
        attributes = [field.name for field in fields(self)]

        for key, item in packet_header.items():
            value = (
                item.derived_value if item.derived_value is not None else item.raw_value
            )
            if key in attributes:
                setattr(self, key, value)
            else:
                raise KeyError(
                    f"Did not find matching attribute in Histogram data class for "
                    f"{key}"
                )
