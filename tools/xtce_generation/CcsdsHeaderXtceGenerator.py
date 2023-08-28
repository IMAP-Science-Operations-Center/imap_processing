class CCSDSParameters:
    def __init__(self):
        self.parameters = [
            {
                "name": "VERSION",
                "parameterTypeRef": "uint3",
                "description": "CCSDS Packet Version Number (always 0)",
            },
            {
                "name": "TYPE",
                "parameterTypeRef": "uint1",
                "description": "CCSDS Packet Type Indicator (0=telemetry)",
            },
            {
                "name": "SEC_HDR_FLG",
                "parameterTypeRef": "uint1",
                "description": "CCSDS Packet Secondary Header Flag (always 1)",
            },
            {
                "name": "PKT_APID",
                "parameterTypeRef": "uint11",
                "description": "CCSDS Packet Application Process ID",
            },
            {
                "name": "SEG_FLGS",
                "parameterTypeRef": "uint2",
                "description": "CCSDS Packet Grouping Flags (3=not part of group)",
            },
            {
                "name": "SRC_SEQ_CTR",
                "parameterTypeRef": "uint14",
                "description": "CCSDS Packet Sequence Count (increments with each new packet)",
            },
            {
                "name": "PKT_LEN",
                "parameterTypeRef": "uint16",
                "description": "CCSDS Packet Length (number of bytes after Packet length minus 1)",
            },
            {
                "name": "SHCOARSE",
                "parameterTypeRef": "uint32",
                "description": "CCSDS Packet Time Stamp (coarse time)",
            },
        ]


# Other utility functions related to CCSDS parameters can also be added here
