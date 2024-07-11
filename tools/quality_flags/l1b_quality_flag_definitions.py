"""L1b quality flag definitions."""

from tools.quality_flags import quality_flags as qf


@qf.with_all_none
class L1bQualityFlags(qf.QualityFlag):
    """Base Quality flags."""

    INF = qf.FlagBit(
        2**0,  # bit 0
        message="Infinite value.",
    )
    MISSING_TELEM = qf.FlagBit(
        2**1,  # bit 1
        message="Missing telemetry.",
    )
    NEG = qf.FlagBit(
        2**2,  # bit 2
        message="Negative value.",
    )
    RES1 = qf.FlagBit(
        2**3,  # bit 3
        message="Reserved 1.",
    )

    # ENA L1b specific flags
    ENA_SPECIFIC = qf.FlagBit(
        2**4,  # bit 4
        message="ENA specific flag.",
    )

    # Ultra L1b specific flags
    ULTRA1 = qf.FlagBit(
        2**5,  # bit 5
        message="Ultra 1.",
    )
    ULTRA2 = qf.FlagBit(
        2**6,  # bit 6
        message="Ultra 2.",
    )
