"""Test coverage for the static.l2_quality_flags module"""

import src.static.l2_quality_flags as qf


def test_L1QualityFlag():
    """Test behavior of the L2QualityFlag class"""
    for f in qf.L2QualityFlag:
        assert f.value
        assert f.value.message
        assert f.summary

    assert len(qf.L2QualityFlag.ALL.summary[1]) == 4
