"""
A class representing CCSDS packet parameters.

This class defines the parameters commonly found in CCSDS telemetry packets.
Each parameter consists of a name, parameter type reference, and description.
"""


class CCSDSParameters:
    """The defined class for CCSDS parameters. Contains the object itself."""

    def __init__(self) -> None:
        self.parameters = [
            {
                "name": "VERSION",
                "parameterTypeRef": "UINT3",
                "description": "CCSDS Packet Version Number (always 0)",
            },
            {
                "name": "TYPE",
                "parameterTypeRef": "UINT1",
                "description": "CCSDS Packet Type Indicator (0=telemetry)",
            },
            {
                "name": "SEC_HDR_FLG",
                "parameterTypeRef": "UINT1",
                "description": "CCSDS Packet Secondary Header Flag (always 1)",
            },
            {
                "name": "PKT_APID",
                "parameterTypeRef": "UINT11",
                "description": "CCSDS Packet Application Process ID",
            },
            {
                "name": "SEQ_FLGS",
                "parameterTypeRef": "UINT2",
                "description": "CCSDS Packet Grouping Flags (3=not part of group)",
            },
            {
                "name": "SRC_SEQ_CTR",
                "parameterTypeRef": "UINT14",
                "description": "CCSDS Packet Sequence Count "
                "(increments with each new packet)",
            },
            {
                "name": "PKT_LEN",
                "parameterTypeRef": "UINT16",
                "description": "CCSDS Packet Length "
                "(number of bytes after Packet length minus 1)",
            },
        ]


# Other utility functions related to CCSDS parameters can also be added here
