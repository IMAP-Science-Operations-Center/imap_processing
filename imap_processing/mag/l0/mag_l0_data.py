from dataclasses import dataclass
from enum import Enum


class MagApid(Enum):
    """Enum class for Glows packet data.

    Attributes
    ----------
    HIST_APID : int
        Histogram packet APID
    DE_APID : int
        Direct event APID
    """

    BURST_APID = 1068
    NORM_APID = 1052


@dataclass
class MagL0:
    # TODO: Are burst and norm similar enough to share dataclass?
    """Data class for MAG Level 0 data.

    Attributes
    ----------
    PHVERNO: int
        CCSDS Packet Version Number
    PHTYPE: int
        CCSDS Packet Type Indicator
    PHSHF: int
        CCSDS Packet Secondary Header Flag
    PHAPID: int
        CCSDS Packet Application Process ID
    PHGROUPF: int
        CCSDS Packet Grouping Flags
    PHSEQCNT: int
        CCSDS Packet Sequence Count
    PHDLEN: int
        CCSDS Packet Length
    SHCOARSE: int
        Mission elapsed time
    PUS_SPARE1: int
        PUS Spare 1
    PUS_VERSION: int
        PUS Version Number
    PUS_SPARE2: int
        PUS Spare 2
    PUS_STYPE: int
        PUS Service Type
    PUS_SSUBTYPE: int
        PUS Service Subtype
    COMPRESSION: int
        Science Data Compression Flag
    FOB_ACT: int
        FOB Active Status
    FIB_ACT: int
        FIB Active Status
    PRI_SENS: int
        Primary Sensor
    SPARE1: int
        Spare
    PRI_VECSEC: int
        Primary Vectors per Second
    SEC_VECSEC: int
        Secondary Vectors per second
    SPARE2: int
        Spare
    PRI_COARSETM: int
        Primary Coarse Time
    PRI_FNTM: int
        Primary Fine Time
    SEC_COARSETM: int
        Secondary Coarse Time
    SEC_FNTM: int
        Secondary Fine Time
    VECTORS: int
        MAG Science Vectors
    FILL: int
        Filler byte
    packet_type: MagApid
        Indicates whether the packet is a Burst or Norm packet.
    """

    PHVERNO: int
    PHTYPE: int
    PHSHF: int
    PHAPID: int
    PHGROUPF: int
    PHSEQCNT: int
    PHDLEN: int
    SHCOARSE: int
    PUS_SPARE1: int
    PUS_VERSION: int
    PUS_SPARE2: int
    PUS_STYPE: int
    PUS_SSUBTYPE: int
    COMPRESSION: int
    FOB_ACT: int
    FIB_ACT: int
    PRI_SENS: int
    SPARE1: int
    PRI_VECSEC: int
    SEC_VECSEC: int
    SPARE2: int
    PRI_COARSETM: int
    PRI_FNTM: int
    SEC_COARSETM: int
    SEC_FNTM: int
    VECTORS: bin
    FILL: int
    packet_type: MagApid
