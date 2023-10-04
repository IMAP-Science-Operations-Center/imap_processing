from dataclasses import dataclass


@dataclass
class CcsdsData:
    """Data class for CCSDS header.

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
        print(packet_header)
