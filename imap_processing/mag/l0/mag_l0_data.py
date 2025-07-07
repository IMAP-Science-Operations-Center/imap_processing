"""Dataclasses for Level 0 MAG data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from imap_processing.ccsds.ccsds_data import CcsdsData


class Mode(IntEnum):
    """
    Enum class for MAG mode.

    Attributes
    ----------
    BURST : int
        APID for Burst mode data.
    NORMAL : int
        ApID for Normal mode data.
    """

    BURST = 1068
    NORMAL = 1052


@dataclass
class MagL0:
    """
    Data class for MAG Level 0 data.

    No attributes should be updated after creation. This class acts as a snapshot of the
    packet data, and is read-only.

    Attributes
    ----------
    ccsds_header: CcsdsData
        CCSDS Header data.
    SHCOARSE: int
        Mission elapsed time.
    PUS_SPARE1: int
        ESA standard headers, PUS Spare 1.
    PUS_VERSION: int
        PUS Version Number.
    PUS_SPARE2: int
        PUS Spare 2.
    PUS_STYPE: int
        PUS Service Type.
    PUS_SSUBTYPE: int
        PUS Service Subtype - tells number of seconds of data.
    COMPRESSION: int
        Science Data Compression Flag - indicates if the data compressed - throw error
        if 1.
    MAGO_ACT: int
        MAGO Active Status - if MAGo is active. May also be referred to as "FOB".
    MAGI_ACT: int
        MAGI Active Status - if MAGi is active.  May also be referred to as "FIB".
    PRI_SENS: int
        Primary Sensor - 0 is MAGo, 1 is MAGi.
    SPARE1: int
        Spare.
    PRI_VECSEC: int
        Primary Vectors per Second - lookup for L1b.
    SEC_VECSEC: int
        Secondary Vectors per second.
    SPARE2: int
        Spare.
    PRI_COARSETM: int
        Primary Coarse Time for first vector, seconds.
    PRI_FNTM: int
        Primary Fine Time for first vector, subseconds.
    SEC_COARSETM: int
        Secondary Coarse Time for first vector, seconds.
    SEC_FNTM: int
        Secondary Fine Time for first vector, subseconds.
    VECTORS: numpy.ndarray or str
        MAG Science Vectors - divide based on PRI_VECSEC and PUS_SSUBTYPE for vector
        counts. There is a post init call to convert a string into a numpy array -
        the only place it is a string is in the class initialization.
    """

    ccsds_header: CcsdsData
    SHCOARSE: int
    PUS_SPARE1: int
    PUS_VERSION: int
    PUS_SPARE2: int
    PUS_STYPE: int
    PUS_SSUBTYPE: int
    COMPRESSION: int
    MAGO_ACT: int
    MAGI_ACT: int
    PRI_SENS: int
    SPARE1: int
    PRI_VECSEC: int
    SEC_VECSEC: int
    SPARE2: int
    PRI_COARSETM: int
    PRI_FNTM: int
    SEC_COARSETM: int
    SEC_FNTM: int
    VECTORS: np.ndarray | bytes

    def __post_init__(self) -> None:
        """
        Convert Vectors attribute from string to bytearray if needed.

        Also convert encoded "VECSEC" (vectors per second) into proper vectors per
        second values.
        """
        if isinstance(self.VECTORS, bytes):
            # Convert byte output from space_packet_parser to numpy array of
            # big-endian bytes
            self.VECTORS = np.frombuffer(
                self.VECTORS,
                dtype=np.dtype(">B"),
            )

        self.PRI_VECSEC = 2**self.PRI_VECSEC
        self.SEC_VECSEC = 2**self.SEC_VECSEC

    def __eq__(self, other: object) -> bool:
        """
        Compare two MagL0 objects for equality.

        Two objects are said to be equal if the SHCOARSE, APID, and SRC_SEQ_CTR are
        equal.

        Parameters
        ----------
        other : object
            The other MagL0 object to compare against.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, MagL0):
            return NotImplemented

        return (
            self.SHCOARSE == other.SHCOARSE
            and self.ccsds_header.PKT_APID == other.ccsds_header.PKT_APID
            and self.ccsds_header.SRC_SEQ_CTR == other.ccsds_header.SRC_SEQ_CTR
        )

    def __hash__(self) -> int:
        """
        Return a hash of the MagL0 object for use in sets.

        Returns
        -------
        int
            The hash value of the MagL0 object.
        """
        return hash(
            (
                self.SHCOARSE,
                self.ccsds_header.PKT_APID,
                self.ccsds_header.SRC_SEQ_CTR,
            )
        )
