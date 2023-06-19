"""
Quality flag example for Level 2 developers

Authors: Laura Sandoval, Gavin Medley, Matt Watwood
"""
import libera_utils.quality_flags as qf


@qf.with_all_none
class L2QualityFlag(qf.QualityFlag, metaclass=qf.FrozenFlagMeta):
    """Quality flag for L2 observation"""
    INF = qf.FlagBit(
        2**0,  # bit 0
        message="Infinite value.")
    MISSING_TELEM = qf.FlagBit(
        2**1,  # bit 1
        message="Missing telemetry.")
    NEG = qf.FlagBit(
        2**2,  # bit 2
        message="Negative value.")
    UNEXPECTED_TELEM_VALUE_CHANGE = qf.FlagBit(
        2**3,  # bit 3
        message="Value changed within the observation that should not have.")
