"""Test coverage for quality flags """
import pytest

from tools.quality_flags import quality_flags as qf

def test_quality_flag():
    """Test our ability to create summary messages from a quality flag"""
    @qf.with_all_none
    class TestFlag(qf.QualityFlag):
        A = qf.FlagBit(0b1, message="Bit 0 - A")
        B = qf.FlagBit(0b10, message="Bit 1 - B")
        C = qf.FlagBit(0b100, message="Bit 2 - C")
        D = qf.FlagBit(0b100000000, message="D")
        # INF = qf.FlagBit(
        #     2**0,  # bit 0
        #     message="Infinite value.")
        # MISSING_TELEM = qf.FlagBit(
        #     2**1,  # bit 1
        #     message="Missing telemetry.")
        # NEG = qf.FlagBit(
        #     2**2,  # bit 2
        #     message="Negative value.")
        # UNEXPECTED_TELEM_VALUE_CHANGE = qf.FlagBit(
        #     2**3,  # bit 3
        #     message="Value changed within the observation that should not have.")

    assert (TestFlag.A | TestFlag.B).decompose() == ([TestFlag.B, TestFlag.A], 0)

    assert TestFlag.A & TestFlag.B == TestFlag.NONE

    assert bool(TestFlag.NONE) is False

    assert bool(TestFlag.A) is True

    f0 = TestFlag.A | TestFlag.B
    f1 = TestFlag.A | TestFlag.B | TestFlag.C
    assert f1 & f0 == TestFlag.A | TestFlag.B
    assert f1 & TestFlag.A & TestFlag.B == TestFlag.NONE

    f2 = TestFlag.A | TestFlag.C
    assert f2 & f0 == TestFlag.A

    assert TestFlag.D.decompose() == ([TestFlag.D], 0)

    f = TestFlag.A | TestFlag.B
    assert set(f.summary[1]) == {"Bit 0 - A", "Bit 1 - B"}

    all = TestFlag.ALL
    assert set(all.summary[1]) == {"Bit 0 - A", "Bit 1 - B", "Bit 2 - C", "D"}




