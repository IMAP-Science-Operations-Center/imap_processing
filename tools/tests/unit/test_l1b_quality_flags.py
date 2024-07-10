"""Test coverage for quality flags. """
import pytest

from tools.quality_flags import quality_flags as qf
from tools.quality_flags import quality_flag_definitions as definitions


def test_quality_flags():
    """Test behavior of the QualityFlags classes."""

    expected_base_mapping = {
        1: "Infinite value.",
        2: "Missing telemetry.",
        4: "Negative value.",
        8: "Reserved 1.",
        16: "ENA specific flag.",
        32: "Ultra 1.",
        64: "Ultra 2."
    }

    for f in definitions.L1bQualityFlags:
        assert f.value in expected_base_mapping
        assert f.value.message == expected_base_mapping[f.value]

    assert definitions.L1bQualityFlags.ALL.summary[1].sort() == list(expected_base_mapping.values()).sort()
    assert definitions.L1bQualityFlags.ALL.summary[0] == sum(expected_base_mapping.keys())


def test_example_usage():
    """Test example usage."""
    ultra_flag = definitions.L1bQualityFlags.MISSING_TELEM | \
                 definitions.L1bQualityFlags.ULTRA1 | \
                 definitions.L1bQualityFlags.ENA_SPECIFIC

    decomposed = ultra_flag.decompose()

    expected_mapping = {
        2: ("MISSING_TELEM", "Missing telemetry."),
        32: ("ULTRA1", "Ultra 1."),
        16: ("ENA_SPECIFIC", "ENA specific flag.")
    }

    for flag in decomposed[0]:
        assert flag.value in expected_mapping
        expected_name, expected_message = expected_mapping[flag.value]
        assert flag.name == expected_name
        assert flag.value.message == expected_message


def test_quality_flag():
    """Test our ability to create summary messages from a quality flag"""
    @qf.with_all_none
    class TestFlag(qf.QualityFlag):
        A = qf.FlagBit(0b1, message="Bit 0 - A")
        B = qf.FlagBit(0b10, message="Bit 1 - B")
        C = qf.FlagBit(0b100, message="Bit 2 - C")
        D = qf.FlagBit(0b100000000, message="D")

    assert (TestFlag.A | TestFlag.B).decompose() == ([TestFlag.B, TestFlag.A], 0)

    # Checks that there are no bits set in both A and B at the same positions
    assert TestFlag.A & TestFlag.B == TestFlag.NONE

    # ensuring that NONE behaves correctly as a flag representing "no flags set"
    assert bool(TestFlag.NONE) is False
    # ensuring that flags are set
    assert bool(TestFlag.A) is True

    # f0 is a combination of TestFlag.A and TestFlag.B
    # f1 is a combination of TestFlag.A, TestFlag.B, and TestFlag.C
    f0 = TestFlag.A | TestFlag.B
    f1 = TestFlag.A | TestFlag.B | TestFlag.C

    # Checks the common flags between f1 and f0
    assert f1 & f0 == TestFlag.A | TestFlag.B

    # f2 is a combination of TestFlag.A and TestFlag.C
    f2 = TestFlag.A | TestFlag.C
    # Checks the common flags between f2 and f0
    assert f2 & f0 == TestFlag.A

    assert TestFlag.D.decompose() == ([TestFlag.D], 0)

    f = TestFlag.A | TestFlag.B
    assert set(f.summary[1]) == {"Bit 0 - A", "Bit 1 - B"}

    all = TestFlag.ALL
    assert set(all.summary[1]) == {"Bit 0 - A", "Bit 1 - B", "Bit 2 - C", "D"}


def test_strict_quality_flags():
    """Test that quality flags are STRICT and raise errors for invalid values"""
    class TestFlag(qf.QualityFlag):
        BIT_0 = 0b001
        BIT_2 = 0b100
        # Note: there is no way to represent the number 2 or 3 with this quality flag since it is missing bit 1

    TestFlag(1)  # bit 0
    TestFlag(4)  # bit 2
    TestFlag(5)  # bit 0 and 2

    with pytest.raises(ValueError):
        TestFlag(2)
    with pytest.raises(ValueError):
        TestFlag(3)
    with pytest.raises(ValueError):
        TestFlag(6)